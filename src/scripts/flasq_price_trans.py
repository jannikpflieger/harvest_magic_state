#!/usr/bin/env python3
"""Price a benchmark with the FLASQ cost model.

Runs inside an isolated environment that has Qualtran with
``qualtran.surface_code.flasq`` (Huggins et al., arXiv:2511.08508); it imports
nothing from HARVEST, so HARVEST's own environment is never modified.

Inputs (data qubits always sit on a row-major ``ceil(sqrt(n))``-wide grid, the
matched-IR default layout, so FLASQ's span term sees the same geometry):

``.qasm`` (recommended)
    The exact Clifford+T source circuit every matched-IR system started from
    -- FLASQ's native input.  Only ``h s sdg t tdg x y z cx`` are accepted.
``.trans``
    The PureMagic export of the matched Pauli-product stream.  FLASQ has no
    multi-qubit rotation primitive, so each ``±P<T>`` is lowered with the
    textbook construction (basis change, CNOT star, T/T†, mirror).  That
    lowering charges FLASQ for long-range CNOT spans a lattice-surgery merge
    does not need, so it overstates FLASQ's cost and is kept only for
    diagnosis.  Any operation, count, span, or depth that
FLASQ cannot price makes the run fail instead of being silently dropped.

Two parameter sets (``--assumptions``):

``matched`` (default)
    The assumptions the matched-IR schedulers are evaluated under: magic
    states are free (cultivation volume 0, as HARVEST/Silva's static,
    always-ready storage), no reaction-time wait is charged, and each layer of
    T measurements takes one logical timestep (one cycle per rotation).  The
    reaction-limited floor is then exactly the dependency critical path.
``cultivation``
    Like ``matched`` but every T gate pays ``--v-cult`` blocks of cultivation
    volume, mirroring the common cultivation model the schedulers run under
    (harvest.baselines.cultivation: mean cultivation cycles per magic state).
``paper``
    FLASQ's physical defaults as used by RushHour: cultivation volume from
    FLASQ's simulation data and ``t_react = t_r / (d·t_cyc)``.

For ``A = 1, 2, …`` fluid ancillas the FLASQ depth is
``L(A) = max(t_react·D, V/A)`` and the logical footprint is ``N = Q + A``.  The
sweep stops at the first ``A`` that is reaction-limited, because more ancilla
cannot lower ``L`` further.  Output is JSON on stdout.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import cirq
from frozendict import frozendict
from qualtran.resource_counting import QubitCount, get_cost_value
from qualtran.surface_code.flasq import (
    FLASQGateTotals,
    TotalMeasurementDepth,
    TotalSpanCost,
    apply_flasq_cost_model,
    conservative_FLASQ_costs,
    convert_circuit_for_flasq_analysis,
    optimistic_FLASQ_costs,
)
from qualtran.surface_code.flasq.cultivation_analysis import find_best_cultivation_parameters
from qualtran.surface_code.flasq.symbols import ROTATION_ERROR, T_REACT, V_CULT_FACTOR

_LINE = re.compile(r"^([+-])([XYZ_]+)<([A-Za-z]+)>$")
MODELS = {"optimistic": optimistic_FLASQ_costs, "conservative": conservative_FLASQ_costs}


def trans_to_cirq(text: str) -> tuple[cirq.Circuit, int]:
    ops = []
    width = None
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        match = _LINE.match(line)
        if match is None:
            raise ValueError(f".trans line {lineno} is malformed: {raw!r}")
        sign, paulis, tag = match.groups()
        if tag != "T":
            raise ValueError(f".trans line {lineno}: <{tag}> has no FLASQ lowering here")
        width = len(paulis) if width is None else width
        if len(paulis) != width:
            raise ValueError(f".trans line {lineno}: inconsistent register width")
        ops.append((sign, paulis))
    if not ops:
        raise ValueError("empty .trans stream")

    cols = max(1, math.ceil(math.sqrt(width)))
    qubits = [cirq.GridQubit(i // cols, i % cols) for i in range(width)]
    circuit = cirq.Circuit()
    for sign, paulis in ops:
        support = [i for i, p in enumerate(paulis) if p != "_"]
        if not support:
            raise ValueError("identity Pauli product")
        basis = []
        for i in support:
            if paulis[i] == "X":
                basis.append(cirq.H(qubits[i]))
            elif paulis[i] == "Y":
                basis += [cirq.S(qubits[i]) ** -1, cirq.H(qubits[i])]
        target = qubits[support[-1]]
        star = [cirq.CNOT(qubits[i], target) for i in support[:-1]]
        rotation = cirq.T(target) if sign == "+" else cirq.T(target) ** -1
        circuit += basis + star + [rotation] + star[::-1] + [cirq.inverse(op) for op in basis[::-1]]
    return circuit, width


_QASM_GATE = re.compile(r"^([a-z]+)\s+([^;]+);$")
_QASM_ARG = re.compile(r"^(\w+)\[(\d+)\]$")
_QASM_1Q = {
    "h": cirq.H, "s": cirq.S, "sdg": cirq.S**-1, "t": cirq.T, "tdg": cirq.T**-1,
    "x": cirq.X, "y": cirq.Y, "z": cirq.Z,
}


def qasm_to_cirq(text: str) -> tuple[cirq.Circuit, int]:
    """Strict reader for Clifford+T OpenQASM 2 (as written by qiskit.qasm2.dumps)."""
    offsets: dict[str, int] = {}
    width = 0
    gates = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.split("//")[0].strip()
        # Measurements are free in FLASQ (and removed by HARVEST's front end).
        if not line or line.startswith(("OPENQASM", "include", "creg", "barrier", "measure")):
            continue
        reg = re.match(r"^qreg\s+(\w+)\[(\d+)\];$", line)
        if reg:
            offsets[reg.group(1)] = width
            width += int(reg.group(2))
            continue
        match = _QASM_GATE.match(line)
        if match is None or match.group(1) not in set(_QASM_1Q) | {"cx"}:
            raise ValueError(f"qasm line {lineno} is not Clifford+T: {raw!r}")
        args = []
        for arg in match.group(2).split(","):
            m = _QASM_ARG.match(arg.strip())
            if m is None or m.group(1) not in offsets:
                raise ValueError(f"qasm line {lineno}: bad operand {arg!r}")
            args.append(offsets[m.group(1)] + int(m.group(2)))
        gates.append((match.group(1), args))
    if not gates:
        raise ValueError("empty qasm circuit")
    cols = max(1, math.ceil(math.sqrt(width)))
    qubits = [cirq.GridQubit(i // cols, i % cols) for i in range(width)]
    ops = []
    for name, args in gates:
        if name == "cx":
            ops.append(cirq.CNOT(qubits[args[0]], qubits[args[1]]))
        else:
            ops.append(_QASM_1Q[name](qubits[args[0]]))
    return cirq.Circuit(ops), width


def cultivation_factor(physical_error_rate: float, target_t_error: float, distance: int):
    row = find_best_cultivation_parameters(physical_error_rate, target_t_error)
    if row.empty:
        raise ValueError("no FLASQ cultivation data for these parameters")
    # Same normalization as flasq.optimization.configs (blocks at distance d).
    v_cult = float(row["expected_volume"]) / (2 * (distance + 1) ** 2 * distance)
    return v_cult, row


def price(path: Path, args) -> dict:
    text = path.read_text()
    circuit, n = qasm_to_cirq(text) if path.suffix == ".qasm" else trans_to_cirq(text)
    cbloq, _ = convert_circuit_for_flasq_analysis(circuit)
    counts = get_cost_value(cbloq, FLASQGateTotals())
    span = get_cost_value(cbloq, TotalSpanCost())
    # FLASQ's QubitCount only sees qubits the stream touches; every matched-IR
    # system still holds all n register qubits in data patches, so Q = n.
    touched = int(get_cost_value(cbloq, QubitCount()))
    if touched > n:
        raise ValueError(f"FLASQ counted {touched} qubits for an {n}-qubit register")
    q = n
    depth = get_cost_value(cbloq, TotalMeasurementDepth())
    if counts.bloqs_with_unknown_cost or span.uncounted_bloqs or depth.bloqs_with_unknown_depth:
        raise ValueError(
            f"FLASQ could not price every bloq: counts={counts.bloqs_with_unknown_cost} "
            f"span={span.uncounted_bloqs} depth={depth.bloqs_with_unknown_depth}"
        )
    if args.assumptions == "paper":
        t_react = args.reaction_time_us / (args.code_distance * args.cycle_time_us)
        v_cult, cult_row = cultivation_factor(
            args.physical_error_rate, args.target_t_error, args.code_distance
        )
        steps_per_measurement = t_react
        cultivation = {
            "cultivation_distance": int(cult_row["cultivation_distance"]),
            "cultivation_error_rate": float(cult_row["t_gate_cultivation_error_rate"]),
        }
    elif args.assumptions == "cultivation":
        t_react, v_cult, steps_per_measurement, cultivation = 0.0, args.v_cult, 1.0, {}
    else:
        t_react, v_cult, steps_per_measurement, cultivation = 0.0, 0.0, 1.0, {}
    assumptions = frozendict({T_REACT: t_react, V_CULT_FACTOR: v_cult, ROTATION_ERROR: 1e-3})
    model = MODELS[args.cost_model]

    frontier = []
    for a in range(1, args.max_fluid_ancilla + 1):
        s = apply_flasq_cost_model(
            model=model,
            n_total_logical_qubits=q + a,
            qubit_counts=q,
            counts=counts,
            span_info=span,
            measurement_depth=depth,
            logical_timesteps_per_measurement=steps_per_measurement,
            assumptions=assumptions,
        )
        point = {
            "fluid_ancilla": a,
            "logical_qubits": q + a,
            "total_depth": float(s.total_depth),
            "volume_limited_depth": float(s.volume_limited_depth),
            "reaction_limited_depth": float(s.scaled_measurement_depth),
            "total_spacetime_volume": float(s.total_spacetime_volume),
        }
        if not frontier or point["total_depth"] < frontier[-1]["total_depth"]:
            frontier.append(point)
        if s.is_reaction_limited or s.volume_limited_depth <= s.scaled_measurement_depth:
            break
    else:
        raise ValueError(
            f"still volume-limited at A={args.max_fluid_ancilla}; raise --max-fluid-ancilla"
        )

    computational = float(s.total_computational_volume)  # independent of A
    return {
        "input_kind": "clifford_t_source" if path.suffix == ".qasm" else "ppr_lowered",
        "num_qubits": n,
        "algorithmic_qubits": q,
        "touched_qubits": touched,
        "measurement_depth": float(depth.depth),
        "gate_counts": {k: v for k, v in counts.asdict().items() if k != "bloqs_with_unknown_cost"},
        "connect_span": float(span.connect_span),
        "compute_span": float(span.compute_span),
        "computational_volume": computational,
        "t_react_logical_timesteps": t_react,
        "logical_timesteps_per_measurement": steps_per_measurement,
        "v_cult_blocks": v_cult,
        "volume_breakdown": {
            "clifford": float(s.clifford_computational_volume),
            "non_clifford_lattice_surgery": float(s.non_clifford_lattice_surgery_volume),
            "cultivation": float(s.cultivation_volume),
        },
        **cultivation,
        "frontier": frontier,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("inputs", nargs="+", type=Path, help=".qasm (Clifford+T) or .trans")
    parser.add_argument("--assumptions", choices=["matched", "cultivation", "paper"], default="matched")
    parser.add_argument("--v-cult", type=float, default=3.0,
                        help="Cultivation volume per T in blocks (cultivation assumptions).")
    parser.add_argument("--cost-model", choices=sorted(MODELS), default="optimistic")
    parser.add_argument("--code-distance", type=int, default=13)
    parser.add_argument("--physical-error-rate", type=float, default=1e-3)
    parser.add_argument("--target-t-error", type=float, default=1e-6)
    parser.add_argument("--reaction-time-us", type=float, default=10.0)
    parser.add_argument("--cycle-time-us", type=float, default=1.0)
    parser.add_argument("--max-fluid-ancilla", type=int, default=10000)
    args = parser.parse_args(argv)
    import qualtran

    qualtran_dir = Path(qualtran.__file__).resolve().parent.parent
    proc = subprocess.run(
        ["git", "-C", str(qualtran_dir), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=False,
    )
    out = {
        "qualtran_commit": proc.stdout.strip() if proc.returncode == 0 else "unknown",
        "qualtran_path": str(qualtran_dir),
        "parameters": {k: v for k, v in vars(args).items() if k != "inputs"},
        "circuits": {p.stem: price(p, args) for p in args.inputs},
    }
    json.dump(out, sys.stdout, indent=2, sort_keys=True)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
