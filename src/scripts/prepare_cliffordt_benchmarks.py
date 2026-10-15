#!/usr/bin/env python3
"""Exactly lower benchmark circuits to the Clifford+T gate set.

Qiskit's ``LitinskiTransformation`` (used by HARVEST's PCB conversion) only
accepts Clifford gates plus ``t/tdg/rz``; a circuit containing e.g. ``ccx`` is
returned unchanged, so it would never reach the matched Pauli-product IR.
This script rewrites each input with Qiskit's standard equivalence library
into ``{h, s, sdg, t, tdg, x, y, z, cx}`` -- an exact rewrite (Toffoli uses the
textbook 7-T circuit), never an approximate synthesis -- and refuses any
circuit that would need one.

Each output is checked:

* only Clifford+T gates remain;
* the T count equals ``t + tdg + 7·ccx`` of the source plus one per phase
  rotation (``rz``/``p``/``u1``) at an odd multiple of π/4 -- such rotations at
  exact multiples of π/4 are rewritten to T/S/Z (up to global phase), any
  other angle is refused;
* for <= ``--unitary-check-max-qubits`` qubits, the output's unitary equals the
  source's up to global phase.

A ``manifest.json`` records source/output SHA-256, gate counts, and checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import math

from qiskit import QuantumCircuit, qasm2, transpile
from qiskit.quantum_info import Operator

from harvest.compilation.circuit_analysis import pre_prep_circuit
from harvest.compilation.qasm_loader import qasm_to_circuit

CLIFFORD_T = ["h", "s", "sdg", "t", "tdg", "x", "y", "z", "cx"]
# Gates with an exact Clifford+T rewrite and the T gates each one introduces.
T_PER_GATE = {"t": 1, "tdg": 1, "ccx": 7}
EXACT_SOURCE_GATES = set(CLIFFORD_T) | {"ccx", "cz", "swap", "id"}
PHASE_ROTATIONS = {"rz", "p", "u1"}
# k (mod 8) for a phase rotation by k·π/4  ->  equivalent Clifford+T gates.
_PHASE_GATES = {0: [], 1: ["t"], 2: ["s"], 3: ["s", "t"], 4: ["z"], 5: ["z", "t"],
                6: ["sdg"], 7: ["tdg"]}


def rewrite_phase_rotations(circuit: QuantumCircuit) -> QuantumCircuit:
    """Replace rz/p/u1 by exact multiples of π/4 with T/S/Z (up to global phase)."""
    out = circuit.copy_empty_like()
    for instruction in circuit.data:
        op = instruction.operation
        if op.name not in PHASE_ROTATIONS:
            out.append(instruction)
            continue
        angle = float(op.params[0])
        k = angle / (math.pi / 4)
        if abs(k - round(k)) > 1e-9:
            raise ValueError(f"{op.name}({angle}) is not an exact multiple of pi/4")
        for name in _PHASE_GATES[round(k) % 8]:
            getattr(out, name)(instruction.qubits[0])
    return out


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lower(source: Path, out_dir: Path, unitary_max_qubits: int) -> dict:
    original = pre_prep_circuit(qasm_to_circuit(str(source)))
    circuit = rewrite_phase_rotations(original)
    ops = dict(circuit.count_ops())
    inexact = sorted(set(ops) - EXACT_SOURCE_GATES)
    if inexact:
        raise ValueError(f"{source.name}: no exact Clifford+T rewrite for {inexact}")
    lowered = transpile(circuit, basis_gates=CLIFFORD_T, optimization_level=0)
    out_ops = dict(lowered.count_ops())
    stray = sorted(set(out_ops) - set(CLIFFORD_T))
    if stray:
        raise ValueError(f"{source.name}: lowering left non-Clifford+T gates {stray}")
    # ``ops`` already contains the T gates introduced by the phase rewrite.
    expected_t = sum(T_PER_GATE.get(g, 0) * n for g, n in ops.items())
    got_t = out_ops.get("t", 0) + out_ops.get("tdg", 0)
    if got_t != expected_t:
        raise ValueError(f"{source.name}: T count {got_t} != expected {expected_t}")
    unitary_checked = circuit.num_qubits <= unitary_max_qubits
    if unitary_checked and not Operator(lowered).equiv(Operator(original)):
        raise ValueError(f"{source.name}: lowered unitary differs from the source")

    target = out_dir / f"{source.stem}.qasm"
    target.write_text(qasm2.dumps(lowered))
    return {
        "source": str(source.relative_to(PROJECT_ROOT) if source.is_relative_to(PROJECT_ROOT) else source),
        "source_sha256": sha256(source),
        "output": target.name,
        "output_sha256": sha256(target),
        "num_qubits": circuit.num_qubits,
        "source_ops": dict(original.count_ops()),
        "rewritten_ops": ops,
        "output_ops": out_ops,
        "t_count": got_t,
        "unitary_checked": unitary_checked,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("sources", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--unitary-check-max-qubits", type=int, default=10)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    # The manifest accumulates: re-lowering an unchanged source is skipped, so
    # circuits can be added to an existing input directory over time.
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    by_output = {record["output"]: record for record in manifest}
    for source in sorted(p.resolve() for p in args.sources):
        previous = by_output.get(f"{source.stem}.qasm")
        target = args.output_dir / f"{source.stem}.qasm"
        if (
            previous is not None
            and previous.get("source_sha256") == sha256(source)
            and target.exists()
            and previous.get("output_sha256") == sha256(target)
        ):
            print(f"{source.stem}: already lowered (unchanged source), skipped")
            continue
        record = lower(source, args.output_dir, args.unitary_check_max_qubits)
        by_output[record["output"]] = record
        print(
            f"{source.stem}: n={record['num_qubits']} T={record['t_count']} "
            f"unitary_checked={record['unitary_checked']}"
        )
    manifest_path.write_text(
        json.dumps(sorted(by_output.values(), key=lambda r: r["output"]), indent=2) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
