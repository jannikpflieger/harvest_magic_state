"""TACO (transpiler + architecture co-design) as a native end-to-end baseline.

Meng Wang et al., "Transpiler-Architecture Co-Design to Curb Clifford Costs in
Fault-Tolerant Quantum Computing", ISCA 2026, arXiv:2412.15434v2.

Two halves, treated differently because only one of them is released:

* **Transpiler (released, run as is).**  The Clifford-reduction pass of
  pnnl/nwqec (MIT; PyPI ``nwqec``; Zenodo 10.5281/zenodo.19449157) is executed
  as an external process from ``config.executable``: either the ``nwqec-cli``
  binary (``nwqec-cli in.qasm --cr -o out.qasm``) or the Python interpreter of
  an isolated environment that has ``nwqec`` installed (a bundled driver calls
  ``nwqec.to_clifford_reduction``).  Both paths produce byte-identical QASM for
  nwqec 0.1.2.  The output is gate-level Clifford+T (CX kept, weight <= 2, H
  mostly absorbed into ``rx(±pi/4)``, S largely removed via ``TS = T†Z``).
* **Architecture / cycle-accurate simulator (not released, reimplemented).**
  §IV-D/§V-B: a compact memory block (Litinski compact layout, "1.5n tiles")
  plus ``k`` compute blocks of 4 tiles, each with one pi/4 site exposing X and
  Z edges.  The simulator processes the dependency DAG; "local operations are
  applied directly, non-local gates (cnot) and state injections (T, S) are
  routed to available compute blocks ... first-come, first-served".  RushHour
  (arXiv:2608.18985, App. A) sweeps the published single block over k = 1, 2,
  4 (``1.5n + 4k`` tiles); the variants here follow that sweep.

Gate model (latencies in QEC rounds, §II-B, identical to nwqec's
``Circuit::get_gate_duration``):

=====================  ============  ===========================================
gate                   rounds        resource
=====================  ============  ===========================================
cx                     3d + 4        one compute block (routing region), §V-B
t, tdg, rx(±pi/4)      2.5d + 4      pi/4 site of a compute block + |m> state
s, sdg, sx, sxdg       1.5d + 3      pi/4 site of a compute block (|Y> inject.)
h                      3d + 4        local (memory or wherever the patch sits)
x, y, z, id            0             Pauli frame (§II-B "no cost"; nwqec: d)
memory <-> compute     1             "a single code cycle" (§IV-D4), see below
=====================  ============  ===========================================

Time is simulated exactly in QEC rounds (half-round integer arithmetic for the
``1.5d`` and ``2.5d`` terms); the makespan is converted once at the end to
logical time steps of ``d`` rounds: ``schedule_length = ceil(rounds / d)``.

Residency: a qubit stays on the pi/4 site of the block that last injected into
it (the paper's locality argument, §IV-A5/§IV-D1).  A later injection on the
same qubit reuses that block without a transfer; otherwise the qubit is moved
in (one transfer) and any idle resident is first returned to memory (one more
transfer).  The paper defines a "cycle" as one QEC round (§II-B, Table III
"magic states / round"), so a transfer costs 1 round by default
(``options["transfer_rounds"]``).  This reading reproduces the published
18-qubit QFT cycle counts within 1-3 % after scaling for the T-count of the
released transpiler (k = 2, 3, 4: 1.04M / 711k / 556k rounds vs. the paper's
1.14M / 761k / 596k at 40,777 instead of 37,725 T gates); charging Litinski's
d-round code cycle instead (``transfer_rounds = d``) gives 1.69M / 1.12M /
835k, i.e. 1.4-1.6x the published numbers.  CNOTs use the block of a resident operand when it is free,
otherwise any free block, and move nothing (every compute block reaches every
memory qubit through the shared ancilla row, §IV-E).  Patch rotations (§IV-D3)
are not charged separately: they occur only in memory for CNOT/H, whose
§II-B latencies already describe the full lattice-surgery/deformation gate,
and the paper gives no rule for when they occur.

Magic states default to the matched assumption shared with the other
baselines: free and always ready, one magic port tile per compute block.  With
``options["magic_model"] = "distillation"`` each block is instead fed by
``distillation_units_per_block`` (default 11, Table II) 15-to-1 units of 11
tiles, each producing one state per 11 code cycles (§VI-C3).
"""

from __future__ import annotations

import csv
import heapq
import json
import math
import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .base import BaselineAdapter, BaselineResult, BaselineRunConfig
from .metrics import space_time_volume

VARIANTS = ("taco_k1", "taco_k2", "taco_k4")
DEFAULT_CODE_DISTANCE = 13
COMPUTE_BLOCK_TILES = 4  # Fig. 12: two tiles for the pi/4-site qubit + two ancilla
MAGIC_PORT_TILES_PER_BLOCK = 1
DISTILLATION_UNIT_TILES = 11  # 15-to-1, §VI-C3
DISTILLATION_PERIOD_CYCLES = 11
DISTILLATION_UNITS_PER_BLOCK = 11  # Table II: 3 blocks x 11 units = 33
TRANSFER_ROUNDS = 1  # §IV-D4 'single code cycle'; cycle = QEC round (§II-B)

# Gate kinds of the transpiled circuit.
CNOT, T_TYPE, S_TYPE, LOCAL_H, PAULI = "cnot", "t", "s", "h", "pauli"
GATE_KIND = {
    "cx": CNOT,
    "t": T_TYPE,
    "tdg": T_TYPE,
    "rx(pi/4)": T_TYPE,
    "rx(-pi/4)": T_TYPE,
    "s": S_TYPE,
    "sdg": S_TYPE,
    "sx": S_TYPE,
    "sxdg": S_TYPE,
    "h": LOCAL_H,
    "x": PAULI,
    "y": PAULI,
    "z": PAULI,
    "id": PAULI,
}
COMPUTE_KINDS = (CNOT, T_TYPE, S_TYPE)
# Gates accepted in the source circuit: Clifford+T only, plus exactly
# decomposable CZ/CCX.  Anything needing approximate synthesis is refused.
INPUT_GATES = {"x", "y", "z", "id", "h", "s", "sdg", "sx", "sxdg", "t", "tdg", "cx", "cz", "ccx"}

_HEADER = re.compile(r'^(OPENQASM\s+2\.0|include\s+"qelib1\.inc")$')
_QREG = re.compile(r"^qreg\s+([A-Za-z_]\w*)\s*\[\s*(\d+)\s*\]$")
_INPUT_OP = re.compile(r"^([a-z_][a-z0-9_]*)(\([^)]*\))?\s+(.+)$")
_OUTPUT_OP = re.compile(
    r"^(?P<name>[a-z]+(?:\((?:-?pi/4)\))?)\s+q\[(?P<a>\d+)\](?:\s*,\s*q\[(?P<b>\d+)\])?$"
)

# Executed with the isolated environment's interpreter (python -I driver ...).
DRIVER_SOURCE = '''\
"""HARVEST driver: run nwqec's released TACO Clifford-reduction pass."""
import json, sys, time
from importlib import metadata
import nwqec

src, out_qasm, out_json, d = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
circuit = nwqec.load_qasm(src)
started = time.perf_counter()
reduced = nwqec.to_clifford_reduction(circuit)
elapsed = time.perf_counter() - started
reduced.save_qasm(out_qasm)
json.dump({
    "nwqec_version": metadata.version("nwqec"),
    "nwqec_module_version": getattr(nwqec, "__version__", None),
    "with_gridsynth_cpp": bool(nwqec.WITH_GRIDSYNTH_CPP),
    "num_qubits": reduced.num_qubits(),
    "input_counts": {k: int(v) for k, v in circuit.count_ops().items()},
    "output_counts": {k: int(v) for k, v in reduced.count_ops().items()},
    "input_depth": circuit.depth(),
    "output_depth": reduced.depth(),
    "input_duration_rounds": circuit.duration(d),
    "output_duration_rounds": reduced.duration(d),
    "code_distance": d,
    "transpile_seconds": elapsed,
}, open(out_json, "w"), indent=2, sort_keys=True)
'''


class UnsupportedTACOInput(ValueError):
    """Raised when a circuit cannot be transpiled/scheduled without loss."""


# --------------------------------------------------------------------------
# Latencies and resources
# --------------------------------------------------------------------------


def gate_rounds(kind: str, code_distance: int) -> float:
    """§II-B latency of one gate kind in QEC rounds (Pauli frame = 0)."""
    d = code_distance
    return {
        CNOT: 3 * d + 4,
        LOCAL_H: 3 * d + 4,
        S_TYPE: 1.5 * d + 3,
        T_TYPE: 2.5 * d + 4,
        PAULI: 0,
    }[kind]


def _half_rounds(kind: str, code_distance: int) -> int:
    value = 2 * gate_rounds(kind, code_distance)
    assert value == int(value)
    return int(value)


def nwqec_gate_rounds(name: str, code_distance: int) -> float:
    """nwqec ``Circuit::get_gate_duration`` (Pauli/default gates cost ``d``)."""
    kind = GATE_KIND[name]
    return code_distance if kind == PAULI else gate_rounds(kind, code_distance)


def rounds_to_steps(rounds: float, code_distance: int) -> int:
    """Logical time steps of ``d`` rounds, rounded up once for the whole makespan."""
    if code_distance < 1:
        raise ValueError("code_distance must be positive")
    return math.ceil(rounds / code_distance - 1e-12)


def memory_tiles(num_qubits: int) -> int:
    """§IV-D2 headline "1.5n tiles" (the paper's 18-qubit example uses 29 = 1.5n+2)."""
    if num_qubits < 1:
        raise ValueError("num_qubits must be positive")
    return math.ceil(1.5 * num_qubits)


def tile_breakdown(
    num_qubits: int,
    compute_blocks: int,
    magic_model: str = "free",
    units_per_block: int = DISTILLATION_UNITS_PER_BLOCK,
) -> Dict[str, int]:
    if compute_blocks < 1:
        raise ValueError("compute_blocks must be positive")
    breakdown = {
        "memory_tiles": memory_tiles(num_qubits),
        "compute_tiles": COMPUTE_BLOCK_TILES * compute_blocks,
        "magic_port_tiles": 0,
        "distillation_tiles": 0,
    }
    if magic_model == "free":
        breakdown["magic_port_tiles"] = MAGIC_PORT_TILES_PER_BLOCK * compute_blocks
    elif magic_model == "distillation":
        breakdown["distillation_tiles"] = (
            DISTILLATION_UNIT_TILES * units_per_block * compute_blocks
        )
    else:
        raise ValueError(f"unknown magic_model {magic_model!r}")
    breakdown["total"] = sum(breakdown.values())
    return breakdown


def variant_blocks(variant: str) -> int:
    if variant not in VARIANTS:
        raise ValueError(f"unknown TACO variant {variant!r}")
    return int(variant.rsplit("_k", 1)[1])


# --------------------------------------------------------------------------
# QASM parsing
# --------------------------------------------------------------------------


def _statements(text: str) -> List[str]:
    text = re.sub(r"//[^\n]*", "", text)
    return [s.strip() for s in text.split(";") if s.strip()]


def inspect_input_qasm(text: str) -> Dict[str, Any]:
    """Validate a Clifford+T source; return qubit count and gate counts."""
    num_qubits = 0
    counts: Dict[str, int] = {}
    for stmt in _statements(text):
        stmt = " ".join(stmt.split())
        if _HEADER.match(stmt):
            continue
        if match := _QREG.match(stmt):
            num_qubits += int(match.group(2))
            continue
        match = _INPUT_OP.match(stmt)
        if match is None or match.group(1) not in INPUT_GATES or match.group(2):
            raise UnsupportedTACOInput(
                f"unsupported statement for the Clifford+T TACO input: {stmt!r}"
            )
        counts[match.group(1)] = counts.get(match.group(1), 0) + 1
    if num_qubits < 1:
        raise UnsupportedTACOInput("input declares no quantum register")
    if not counts:
        raise UnsupportedTACOInput("input contains no gates")
    return {"num_qubits": num_qubits, "counts": counts}


def parse_transpiled_qasm(text: str) -> Tuple[int, List[Tuple[str, Tuple[int, ...]]]]:
    """Parse nwqec's output QASM into ``(num_qubits, [(gate, qubits), ...])``.

    Every statement must be a gate of :data:`GATE_KIND` on the single ``q``
    register; anything else raises instead of being skipped.
    """
    num_qubits: Optional[int] = None
    gates: List[Tuple[str, Tuple[int, ...]]] = []
    for stmt in _statements(text):
        stmt = " ".join(stmt.split())
        if _HEADER.match(stmt):
            continue
        if match := _QREG.match(stmt):
            if num_qubits is not None or match.group(1) != "q":
                raise UnsupportedTACOInput(f"unexpected register declaration {stmt!r}")
            num_qubits = int(match.group(2))
            continue
        match = _OUTPUT_OP.match(stmt)
        if match is None or match.group("name") not in GATE_KIND:
            raise UnsupportedTACOInput(f"cannot schedule transpiled statement {stmt!r}")
        name = match.group("name")
        qubits = tuple(int(match.group(g)) for g in ("a", "b") if match.group(g) is not None)
        expected = 2 if GATE_KIND[name] == CNOT else 1
        if len(qubits) != expected or len(set(qubits)) != len(qubits):
            raise UnsupportedTACOInput(f"wrong operands in {stmt!r}")
        gates.append((name, qubits))
    if num_qubits is None:
        raise UnsupportedTACOInput("transpiled circuit declares no register")
    for name, qubits in gates:
        if max(qubits) >= num_qubits:
            raise UnsupportedTACOInput(f"{name} on qubit {max(qubits)} outside q[{num_qubits}]")
    return num_qubits, gates


def gate_counts(gates: Sequence[Tuple[str, Tuple[int, ...]]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for name, _ in gates:
        counts[name] = counts.get(name, 0) + 1
    return counts


# --------------------------------------------------------------------------
# Cycle-accurate compute-block simulator (§V-B reimplementation)
# --------------------------------------------------------------------------


@dataclass
class TACOSchedule:
    makespan_rounds: float
    schedule_length: int
    critical_path_rounds: float
    nwqec_style_critical_path_rounds: float
    transfers: int
    evictions: int
    block_busy_rounds: List[float]
    kind_counts: Dict[str, int]
    magic_wait_rounds: float = 0.0
    records: List[Tuple[int, str, Tuple[int, ...], int, float, float]] = field(
        default_factory=list
    )


def critical_path_rounds(
    gates: Sequence[Tuple[str, Tuple[int, ...]]],
    num_qubits: int,
    code_distance: int,
    nwqec_style: bool = False,
) -> float:
    """Architecture-independent critical path (§V-B), as in ``Circuit::duration``."""
    clock = [0.0] * num_qubits
    for name, qubits in gates:
        cost = (
            nwqec_gate_rounds(name, code_distance)
            if nwqec_style
            else gate_rounds(GATE_KIND[name], code_distance)
        )
        start = max(clock[q] for q in qubits)
        for q in qubits:
            clock[q] = start + cost
    return max(clock) if clock else 0.0


def simulate(
    gates: Sequence[Tuple[str, Tuple[int, ...]]],
    num_qubits: int,
    compute_blocks: int,
    code_distance: int = DEFAULT_CODE_DISTANCE,
    *,
    magic_model: str = "free",
    units_per_block: int = DISTILLATION_UNITS_PER_BLOCK,
    distillation_period_cycles: int = DISTILLATION_PERIOD_CYCLES,
    transfer_rounds: Optional[float] = None,
    record: bool = False,
) -> TACOSchedule:
    """Greedy first-come-first-served list scheduling on ``k`` compute blocks.

    Event-driven: at every event time the ready gates (all predecessors
    finished) are visited in FCFS order ``(ready time, program index)``; a gate
    starts when its qubits and a suitable compute block are free.  A gate that
    cannot start does not block later ready gates.  Internally all times are
    integer half-rounds.  ``transfer_rounds`` is the cost of one
    memory<->compute move (default 1 round, §IV-D4 with §II-B's cycle = round).
    """
    k = int(compute_blocks)
    d = int(code_distance)
    if k < 1:
        raise ValueError("compute_blocks must be positive")
    if d < 1:
        raise ValueError("code_distance must be positive")
    if magic_model not in ("free", "distillation"):
        raise ValueError(f"unknown magic_model {magic_model!r}")
    n_gates = len(gates)
    if n_gates == 0:
        raise UnsupportedTACOInput("nothing to schedule")

    kinds: List[str] = []
    for index, (name, qubits) in enumerate(gates):
        if name not in GATE_KIND:
            raise UnsupportedTACOInput(f"gate #{index} {name!r} has no TACO resource model")
        kind = GATE_KIND[name]
        if len(qubits) != (2 if kind == CNOT else 1) or any(
            not 0 <= q < num_qubits for q in qubits
        ):
            raise UnsupportedTACOInput(f"gate #{index} {name!r} has invalid operands {qubits}")
        kinds.append(kind)
    latency = {kind: _half_rounds(kind, d) for kind in (CNOT, T_TYPE, S_TYPE, LOCAL_H, PAULI)}
    if transfer_rounds is None:
        transfer_rounds = TRANSFER_ROUNDS
    transfer = int(round(2 * float(transfer_rounds)))
    if transfer < 0 or transfer != 2 * float(transfer_rounds):
        raise ValueError("transfer_rounds must be a non-negative multiple of 0.5")
    period = 2 * d * distillation_period_cycles

    # Dependency DAG: per-qubit program order (gates on a qubit never commute here).
    last = [-1] * num_qubits
    npred = [0] * n_gates
    succs: List[List[int]] = [[] for _ in range(n_gates)]
    for i, (_, qubits) in enumerate(gates):
        preds = {last[q] for q in qubits if last[q] >= 0}
        npred[i] = len(preds)
        for p in preds:
            succs[p].append(i)
        for q in qubits:
            last[q] = i

    ready_time = [0] * n_gates
    ready: List[Tuple[int, int]] = [(0, i) for i in range(n_gates) if npred[i] == 0]
    events: List[Tuple[int, int]] = []  # (time, gate index or -1 for a resource wake-up)
    block_free = [0] * k
    resident: List[Optional[int]] = [None] * k
    home: List[Optional[int]] = [None] * num_qubits
    qubit_free = [0] * num_qubits
    units = [[0] * units_per_block for _ in range(k)] if magic_model == "distillation" else None
    block_busy = [0] * k
    transfers = evictions = magic_wait = 0
    records = []

    def start_injection(i: int, q: int, t: int) -> Optional[int]:
        nonlocal transfers, evictions, magic_wait
        options: List[Tuple[int, int, Optional[int]]] = []  # (block, move time, evicted)
        b = home[q]
        if b is not None and block_free[b] <= t:
            options.append((b, -1, None))  # -1: no transfer
        else:
            candidates = []
            for j in range(k):
                if block_free[j] > t:
                    continue
                r = resident[j]
                if r is not None and qubit_free[r] > t:
                    continue
                candidates.append((r is not None, j))
            for occupied, j in sorted(candidates):
                options.append((j, transfer * (2 if occupied else 1), resident[j]))
        if not options:
            return None
        chosen = None
        for j, move, evicted in options:
            move = max(move, 0)
            if units is None or kinds[i] != T_TYPE:
                chosen = (j, move, evicted, None)
                break
            unit = min(range(units_per_block), key=lambda u: (units[j][u], u))
            if units[j][unit] <= t + move:
                chosen = (j, move, evicted, unit)
                break
        if chosen is None:
            wake = min(
                min(units[j]) - max(move, 0) for j, move, _ in options  # type: ignore[index]
            )
            heapq.heappush(events, (max(wake, t + 1), -1))
            return None
        j, move, evicted, unit = chosen
        end = t + move + latency[kinds[i]]
        if unit is not None:
            magic_wait += max(0, t + move - ready_time[i])
            units[j][unit] = t + move + period
        if evicted is not None:
            evictions += 1
            home[evicted] = None
            qubit_free[evicted] = t + transfer
            heapq.heappush(events, (t + transfer, -1))
        if j != home[q]:
            transfers += 1
            if home[q] is not None:
                resident[home[q]] = None
        resident[j] = q
        home[q] = j
        block_free[j] = end
        block_busy[j] += end - t
        qubit_free[q] = end
        return j

    def try_start(i: int, t: int) -> Optional[Tuple[int, Optional[int]]]:
        qubits = gates[i][1]
        if any(qubit_free[q] > t for q in qubits):
            return None
        kind = kinds[i]
        if kind in (LOCAL_H, PAULI):
            end = t + latency[kind]
            qubit_free[qubits[0]] = end
            return end, None
        if kind == CNOT:
            own = sorted(
                j for j in {home[q] for q in qubits} if j is not None and block_free[j] <= t
            )
            if own:
                j = own[0]
            else:
                free = [(resident[j] is not None, j) for j in range(k) if block_free[j] <= t]
                if not free:
                    return None
                j = min(free)[1]
            end = t + latency[CNOT]
            block_free[j] = end
            block_busy[j] += end - t
            for q in qubits:
                qubit_free[q] = end
            return end, j
        j = start_injection(i, qubits[0], t)
        if j is None:
            return None
        return block_free[j], j

    t = 0
    done = 0
    finish = [0] * n_gates
    while done < n_gates:
        new_ready = []
        while events and events[0][0] <= t:
            time, i = heapq.heappop(events)
            if i < 0:
                continue
            done += 1
            for s in succs[i]:
                npred[s] -= 1
                ready_time[s] = max(ready_time[s], time)
                if npred[s] == 0:
                    new_ready.append((ready_time[s], s))
        if new_ready:
            ready.extend(new_ready)
            ready.sort()
        remaining = []
        instant = False
        for item in ready:
            i = item[1]
            started = try_start(i, t)
            if started is None:
                remaining.append(item)
                continue
            end, block = started
            finish[i] = end
            heapq.heappush(events, (end, i))
            instant |= end == t
            if record:
                records.append((i, gates[i][0], gates[i][1], -1 if block is None else block, t / 2, end / 2))
        ready = remaining
        if instant or done == n_gates:
            continue
        if not events:
            raise RuntimeError("TACO simulator deadlocked with ready gates and no events")
        # Every event pushed in this pass lies strictly after t (zero-latency
        # finishes are handled by ``instant``), so advance to the next one.
        t = events[0][0]

    makespan = max(finish) / 2
    cp = critical_path_rounds(gates, num_qubits, d)
    if makespan + 1e-9 < cp:
        raise AssertionError("schedule shorter than the dependency critical path")
    kind_counts: Dict[str, int] = {}
    for kind in kinds:
        kind_counts[kind] = kind_counts.get(kind, 0) + 1
    return TACOSchedule(
        makespan_rounds=makespan,
        schedule_length=rounds_to_steps(makespan, d),
        critical_path_rounds=cp,
        nwqec_style_critical_path_rounds=critical_path_rounds(gates, num_qubits, d, nwqec_style=True),
        transfers=transfers,
        evictions=evictions,
        block_busy_rounds=[b / 2 for b in block_busy],
        kind_counts=kind_counts,
        magic_wait_rounds=magic_wait / 2,
        records=records,
    )


# --------------------------------------------------------------------------
# Adapter
# --------------------------------------------------------------------------


def _resolve_executable(path: Optional[Path]) -> Optional[str]:
    if path is None:
        return None
    candidate = str(path)
    if Path(candidate).is_file():
        return str(Path(candidate).absolute())
    return shutil.which(candidate)


def _is_python(executable: str) -> bool:
    return Path(executable).name.startswith("python")


def _write_schedule_csv(path: Path, records) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["gate_index", "gate", "qubits", "block", "start_round", "end_round"])
        for i, name, qubits, block, start, end in sorted(records, key=lambda r: (r[4], r[0])):
            writer.writerow([i, name, " ".join(map(str, qubits)), block, start, end])


class TACOAdapter(BaselineAdapter):
    """Released nwqec TACO pass + reimplemented §V-B compute-block simulator."""

    name = "taco"

    def run(self, config: BaselineRunConfig, **kwargs: Any) -> BaselineResult:
        input_repr = "OpenQASM 2 Clifford+T -> nwqec TACO Clifford reduction (gate-level, CX kept)"
        if config.comparison_mode != "native_end_to_end":
            return BaselineResult.failed(
                config,
                input_representation=input_repr,
                error="TACO is only qualified for native_end_to_end comparisons "
                "(its IR is gate-level Clifford+T, not a Pauli-product stream)",
                seed=None,
            )
        try:
            k = variant_blocks(config.variant)
        except ValueError as exc:
            return BaselineResult.failed(config, input_representation=input_repr, error=str(exc), seed=None)
        options = dict(config.options or {})
        d = int(options.get("code_distance", DEFAULT_CODE_DISTANCE))
        magic_model = str(options.get("magic_model", "free"))
        units_per_block = int(options.get("distillation_units_per_block", DISTILLATION_UNITS_PER_BLOCK))
        transfer_rounds = float(options.get("transfer_rounds", TRANSFER_ROUNDS))

        executable = _resolve_executable(config.executable)
        if executable is None:
            return BaselineResult.failed(
                config,
                input_representation=input_repr,
                error=f"nwqec executable not found: {config.executable!r}",
                notes=[
                    "Pass the nwqec-cli binary or the python of an isolated env with "
                    "`pip install nwqec` (e.g. /tmp/harvest-nwqec/bin/python)."
                ],
                seed=None,
            )

        work_dir = config.work_dir()
        work_dir.mkdir(parents=True, exist_ok=True)
        work_dir = work_dir.resolve()
        out_qasm = work_dir / "taco_transpiled.qasm"
        stats_path = work_dir / "nwqec_stats.json"
        for stale in (out_qasm, stats_path):
            if stale.exists():
                stale.unlink()
        input_path = Path(config.input_path).resolve()
        command: List[str] = []
        started = perf_counter()
        try:
            source = inspect_input_qasm(input_path.read_text())
            if _is_python(executable):
                driver = work_dir / "nwqec_taco_driver.py"
                driver.write_text(DRIVER_SOURCE)
                command = [executable, "-I", str(driver), str(input_path), str(out_qasm), str(stats_path), str(d)]
                mode = "python"
            else:
                command = [executable, str(input_path), "--cr", "-o", str(out_qasm)]
                mode = "cli"
            (work_dir / "command.txt").write_text(shlex.join(command) + "\n")
            proc = subprocess.run(
                command, cwd=work_dir, text=True, capture_output=True,
                timeout=config.timeout_s, check=False,
            )
            transpile_s = perf_counter() - started
            (work_dir / "stdout.txt").write_text(proc.stdout)
            (work_dir / "stderr.txt").write_text(proc.stderr)
            if proc.returncode != 0 or not out_qasm.exists():
                return BaselineResult.failed(
                    config,
                    input_representation=input_repr,
                    error=f"nwqec exited with code {proc.returncode}",
                    command=shlex.join(command),
                    runtime_s=transpile_s,
                    notes=[proc.stderr.strip()[-2000:]] if proc.stderr.strip() else [],
                    seed=None,
                )
            stats = json.loads(stats_path.read_text()) if mode == "python" else {}

            num_qubits, gates = parse_transpiled_qasm(out_qasm.read_text())
            counts = gate_counts(gates)
            self._check_lossless(source, num_qubits, counts, stats)

            sched_started = perf_counter()
            schedule = simulate(
                gates, num_qubits, k, d,
                magic_model=magic_model, units_per_block=units_per_block,
                transfer_rounds=transfer_rounds, record=True,
            )
            scheduler_s = perf_counter() - sched_started
            native_duration = stats.get("output_duration_rounds")
            if native_duration is not None and abs(
                float(native_duration) - schedule.nwqec_style_critical_path_rounds
            ) > 1e-6:
                raise UnsupportedTACOInput(
                    f"latency model disagrees with nwqec duration({d}): "
                    f"{schedule.nwqec_style_critical_path_rounds} != {native_duration}"
                )
            _write_schedule_csv(work_dir / "schedule.csv", schedule.records)
        except subprocess.TimeoutExpired:
            return BaselineResult.failed(
                config,
                input_representation=input_repr,
                error=f"nwqec timed out after {config.timeout_s}s",
                command=shlex.join(command) if command else None,
                timed_out=True,
                runtime_s=perf_counter() - started,
                seed=None,
            )
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            return BaselineResult.failed(
                config,
                input_representation=input_repr,
                error=str(exc),
                command=shlex.join(command) if command else None,
                runtime_s=perf_counter() - started,
                notes=["Unsupported operations were not discarded or rewritten."],
                seed=None,
            )

        tiles = tile_breakdown(num_qubits, k, magic_model, units_per_block)
        steps = schedule.schedule_length
        version = stats.get("nwqec_version")
        upstream = f"nwqec {version}" if version else config.upstream_version
        notes = [
            "Transpiler: released nwqec TACO Clifford-reduction pass, run unmodified; "
            "architecture and cycle-accurate simulator are a reimplementation (not released).",
            f"k={k} compute blocks (paper publishes one; RushHour App. A sweep 1/2/4, 1.5n+4k tiles).",
            f"Latencies in QEC rounds from §II-B at d={d} (CX/H 3d+4, S-type 1.5d+3, "
            "T/Tdg/Rx(pi/4) 2.5d+4); makespan converted once as ceil(rounds/d) logical steps.",
            "Pauli gates cost 0 (§II-B); nwqec's own duration() charges them d — reported in metadata only.",
            "CNOT, T-type and S-type gates each occupy one compute block (§V-B); H is local and "
            "occupies no block.",
            "Greedy event-driven FCFS (ready time, program order); a blocked gate does not block "
            "later ready gates.  The paper's 'layer by layer' loop is not further specified.",
            f"Memory<->compute transfer = {transfer_rounds} QEC round(s) per move (§IV-D4 'single code "
            "cycle' with §II-B's cycle = round; reproduces the paper's QFT-18 cycle counts), eviction "
            "of an idle resident = 1 more; qubits stay resident on the pi/4 site until evicted.",
            "Patch rotations (§IV-D3, 3 cycles) are not charged separately; the paper gives no "
            "trigger rule and the §II-B CX/H latencies are full gate costs.",
            "Memory tiles = ceil(1.5n) (§IV-D2 headline); the paper's 18-qubit example uses 29 "
            "= 1.5n+2 and Table II lists 41 data tiles for that design.",
        ]
        if magic_model == "free":
            notes.append(
                "Magic states free and always ready (matched assumption); one |m> port tile per "
                "compute block; distillation tiles/latency not charged."
            )
        else:
            notes.append(
                f"Magic states from {units_per_block} 15-to-1 units per block (11 tiles, one state "
                "per 11 code cycles of d rounds each, Litinski; the paper's cycle=round reading would "
                "be 11 rounds — non-binding at 11 units/block either way, §VI-C3); S-type |Y> preparation is inside its 1.5d+3 latency."
            )
        return BaselineResult(
            baseline=config.baseline,
            variant=config.variant,
            circuit=config.circuit,
            num_qubits=num_qubits,
            input_representation=input_repr,
            comparison_mode=config.comparison_mode,
            schedule_length=steps,
            logical_patches=tiles["total"],
            space_time_volume=space_time_volume(tiles["total"], steps),
            routing_wirelength=None,
            compiler_runtime_s=transpile_s + scheduler_s,
            completed=True,
            trial=config.trial,
            seed=None,
            upstream_version=upstream,
            command=shlex.join(command),
            notes=notes,
            initial_logical_patches=tiles["total"],
            metadata={
                "compute_blocks": k,
                "code_distance": d,
                "magic_model": magic_model,
                "transfer_rounds": transfer_rounds,
                "tiles": tiles,
                "nwqec_invocation": mode,
                "nwqec_version": version,
                "nwqec_stats": stats,
                "input_gate_counts": source["counts"],
                "transpiled_gate_counts": counts,
                "transpiled_gate_total": len(gates),
                "kind_counts": schedule.kind_counts,
                "makespan_rounds": schedule.makespan_rounds,
                "critical_path_rounds": schedule.critical_path_rounds,
                "critical_path_steps": rounds_to_steps(schedule.critical_path_rounds, d),
                "nwqec_duration_rounds": schedule.nwqec_style_critical_path_rounds,
                "transfers": schedule.transfers,
                "evictions": schedule.evictions,
                "block_busy_rounds": schedule.block_busy_rounds,
                "magic_wait_rounds": schedule.magic_wait_rounds,
                "transpiler_runtime_s": transpile_s,
                "scheduler_runtime_s": scheduler_s,
                "time_unit": "logical step = d QEC rounds; schedule_length = ceil(makespan_rounds/d)",
                "space_time_definition": "(ceil(1.5n) memory + 4k compute + magic tiles) * steps",
                "work_dir": str(work_dir),
                "transpiled_qasm": str(out_qasm),
            },
        )

    @staticmethod
    def _check_lossless(source, num_qubits, counts, stats) -> None:
        """Refuse outputs that are not an exact, complete Clifford reduction."""
        if num_qubits != source["num_qubits"]:
            raise UnsupportedTACOInput(
                f"nwqec changed the register size {source['num_qubits']} -> {num_qubits}"
            )
        src = source["counts"]
        t_in = src.get("t", 0) + src.get("tdg", 0) + 7 * src.get("ccx", 0)
        t_out = sum(v for name, v in counts.items() if GATE_KIND[name] == T_TYPE)
        if t_in != t_out:
            raise UnsupportedTACOInput(f"non-Clifford count changed {t_in} -> {t_out}")
        if "ccx" not in src and counts.get("cx", 0) != src.get("cx", 0) + src.get("cz", 0):
            raise UnsupportedTACOInput(
                f"CNOT count changed {src.get('cx', 0) + src.get('cz', 0)} -> {counts.get('cx', 0)}"
            )
        if stats:
            native = {k: int(v) for k, v in stats["output_counts"].items() if v}
            if native != counts:
                raise UnsupportedTACOInput(
                    f"parsed QASM {counts} disagrees with nwqec count_ops {native}"
                )
