"""Space/time lower bounds and normalization for the matched-IR trade-off plot.

Everything here is derived from raw SOTA comparison records; no baseline
algorithm or comparison semantics are changed.  Only ``matched_ir`` records
are ever admitted, because only that regime gives every system the same
post-transformation Pauli-product stream and therefore the same bounds.

Lower bounds (per benchmark, on the matched Pauli-product DAG):

``time_lb``
    Length of the longest dependency chain when every schedulable
    Pauli-product operation (``PauliEvolution`` or
    ``PauliProductMeasurement``) takes exactly one logical cycle.  Two
    operations depend on each other iff they act on a common logical qubit,
    i.e. the same "trivial" dependency semantics HARVEST and Silva EAF
    schedule against (see ``docs/sota_baselines.md``).  A system that
    additionally exploits commutation may legitimately beat this bound; such
    points are reported, never clamped.

``space_lb``
    ``num_qubits + has_magic_state_operation``: every logical qubit needs at
    least one data patch, and a stream that consumes a magic state
    (``PauliEvolution``, see ``harvest.compilation.utils.node_needs_magic_state``)
    needs at least one patch to hold that magic state.  Routing space is
    deliberately *not* counted, so the bound is generous and not attainable
    by any lattice-surgery layout that needs a bus.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .base import _file_sha256, safe_name, stable_payload_hash
from .metrics import geometric_mean

MATCHED_IR = "matched_ir"
SCHEDULABLE_OPS = frozenset({"PauliEvolution", "PauliProductMeasurement"})
MAGIC_OPS = frozenset({"PauliEvolution"})
# DASCOT has no qualified matched-IR path; it never enters this plot.
EXCLUDED_BASELINES = frozenset({"dascot"})
_TRANS_LINE = re.compile(r"^([+-])([XYZ_]+)<([A-Za-z]+)>$")
_TRANS_MAGIC_TAGS = frozenset({"T"})


class TradeoffDataError(ValueError):
    """Raised whenever the data cannot support the plot without guessing."""


# --------------------------------------------------------------------------
# Lower bounds
# --------------------------------------------------------------------------


def _longest_chain(ops: Sequence[Sequence[int]]) -> int:
    """Longest chain of unit-cost operations ordered by shared qubits."""
    frontier: Dict[int, int] = {}  # qubit -> finish cycle of last op on it
    longest = 0
    for qubits in ops:
        if not qubits:
            raise TradeoffDataError("a Pauli-product operation acts on no qubit")
        finish = 1 + max(frontier.get(q, 0) for q in qubits)
        for q in qubits:
            frontier[q] = finish
        longest = max(longest, finish)
    return longest


def dag_operations(dag) -> List[Tuple[str, List[int]]]:
    """Return ``(op_name, qubit_indices)`` in topological order.

    Fails loudly on any operation that is not a schedulable Pauli product,
    instead of guessing its cost.
    """
    ops = []
    for index, node in enumerate(dag.topological_op_nodes()):
        name = node.op.name
        if name not in SCHEDULABLE_OPS:
            raise TradeoffDataError(
                f"operation #{index} ({name}) is not a schedulable Pauli product; "
                "refusing to guess its logical-cycle cost"
            )
        ops.append((name, [dag.find_bit(q).index for q in node.qargs]))
    if not ops:
        raise TradeoffDataError("matched DAG contains no schedulable operation")
    return ops


def dag_time_lower_bound(dag) -> int:
    """Dependency critical path of the matched DAG at one cycle per operation."""
    return _longest_chain([qubits for _, qubits in dag_operations(dag)])


def dag_has_magic_operation(dag) -> bool:
    return any(name in MAGIC_OPS for name, _ in dag_operations(dag))


def parse_trans(text: str) -> List[Tuple[str, List[int]]]:
    """Parse PureMagic ``.trans`` lines into ``(tag, qubit_indices)``."""
    ops = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        match = _TRANS_LINE.match(line)
        if match is None:
            raise TradeoffDataError(f".trans line {lineno} is malformed: {raw!r}")
        paulis, tag = match.group(2), match.group(3)
        ops.append((tag, [i for i, p in enumerate(paulis) if p != "_"]))
    if not ops:
        raise TradeoffDataError(".trans stream contains no operation")
    return ops


def trans_time_lower_bound(text: str) -> int:
    return _longest_chain([qubits for _, qubits in parse_trans(text)])


def trans_has_magic_operation(text: str) -> bool:
    return any(tag in _TRANS_MAGIC_TAGS for tag, _ in parse_trans(text))


_CLIFFORD_GATES = frozenset(
    {"id", "x", "y", "z", "h", "s", "sdg", "sx", "sxdg", "cx", "cz", "cy", "swap"}
)
_T_GATES = frozenset({"t", "tdg"})


def circuit_t_depth(circuit) -> int:
    """T-depth of a Clifford+T circuit: longest chain of T/Tdg gates.

    Cliffords cost zero but carry dependencies, exactly like FLASQ's
    measurement depth.  Any other gate raises instead of being guessed.
    """
    from qiskit.converters import circuit_to_dag

    dag = circuit_to_dag(circuit)
    depth: Dict[Any, int] = {}
    for node in dag.topological_op_nodes():
        name = node.op.name
        if name not in _CLIFFORD_GATES and name not in _T_GATES and name != "barrier":
            raise TradeoffDataError(f"gate {name} is not Clifford+T; no T-depth defined")
        own = 1 if name in _T_GATES else 0
        depth[node] = own + max((depth[p] for p in dag.op_predecessors(node)), default=0)
    if not depth or max(depth.values()) == 0:
        raise TradeoffDataError("circuit contains no T gate")
    return max(depth.values())


def space_lower_bound(num_qubits: int, has_magic_state_operation: bool) -> int:
    """One data patch per logical qubit, plus one magic patch if any is consumed."""
    if num_qubits is None or num_qubits < 1:
        raise TradeoffDataError(f"invalid num_qubits {num_qubits!r}")
    return int(num_qubits) + int(bool(has_magic_state_operation))


@dataclass(frozen=True)
class CircuitBounds:
    circuit: str
    num_qubits: int
    num_operations: int
    time_lb: int
    space_lb: int
    has_magic_state_operation: bool
    source: str
    input_path: Optional[str] = None  # QASM the matched DAG was built from


# --------------------------------------------------------------------------
# Records and points
# --------------------------------------------------------------------------


@dataclass
class TradeoffPoint:
    """One (compiler configuration, circuit) observation, trials aggregated."""

    circuit: str
    baseline: str
    variant: str
    compiler: str
    config_id: str
    source_dir: str
    num_qubits: int
    logical_patches: float
    schedule_length: float
    schedule_length_min: int
    schedule_length_max: int
    trials_successful: int
    trials_total: int
    logical_patches_min: Optional[float] = None
    logical_patches_max: Optional[float] = None
    space_lb: int = 0
    time_lb: float = 0
    normalized_space: float = 0.0
    normalized_time: float = 0.0
    normalized_space_time: float = 0.0
    pareto_optimal: bool = True
    in_shared_subset: bool = False

    def to_row(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FailureRecord:
    circuit: str
    compiler: str
    config_id: str
    source_dir: str
    trials_failed: int
    trials_total: int
    errors: List[str] = field(default_factory=list)


def load_raw_payloads(result_dir: Path) -> List[Dict[str, Any]]:
    raw_dir = Path(result_dir) / "raw"
    if not raw_dir.is_dir():
        raise TradeoffDataError(f"{result_dir} has no raw/ directory")
    payloads = []
    for path in sorted(raw_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise TradeoffDataError(f"malformed raw result {path}: {exc}") from exc
        if "result" not in payload or "config" not in payload:
            raise TradeoffDataError(f"raw result {path} lacks result/config")
        payload["raw_file"] = str(path)
        payloads.append(payload)
    if not payloads:
        raise TradeoffDataError(f"{raw_dir} contains no raw results")
    return payloads


def _config_id(payload: Dict[str, Any]) -> str:
    """Identify a compiler configuration; trials differ only by trial/seed."""
    config = payload["config"]
    options = dict(config.get("options") or {})
    return f"{config['variant']}@{stable_payload_hash(options)}"


def aggregate_points(
    payloads: Sequence[Dict[str, Any]],
    source_dir: str,
    display: Callable[[str, str], str],
    mode: str = MATCHED_IR,
) -> Tuple[List[TradeoffPoint], List[FailureRecord], List[str]]:
    """Group trials into points, validating every metric the plot needs.

    Returns ``(points, failures, notes)``.  ``failures`` lists every
    configuration with at least one failed trial (fully or partially failed);
    nothing is dropped silently.
    """
    notes: List[str] = []
    groups: Dict[Tuple[str, str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    excluded_modes: Dict[str, int] = defaultdict(int)
    excluded_baselines: Dict[str, int] = defaultdict(int)
    for payload in payloads:
        result = payload["result"]
        if result.get("comparison_mode") != mode:
            excluded_modes[str(result.get("comparison_mode"))] += 1
            continue
        if mode == MATCHED_IR and result.get("baseline") in EXCLUDED_BASELINES:
            excluded_baselines[str(result.get("baseline"))] += 1
            continue
        key = (
            result["circuit"],
            result["baseline"],
            result["variant"],
            _config_id(payload),
        )
        groups[key].append(payload)
    for mode, count in sorted(excluded_modes.items()):
        notes.append(f"{source_dir}: excluded {count} record(s) of another comparison mode ({mode})")
    for baseline, count in sorted(excluded_baselines.items()):
        notes.append(f"{source_dir}: excluded {count} {baseline} record(s) (not matched_ir-qualified)")

    points: List[TradeoffPoint] = []
    failures: List[FailureRecord] = []
    for (circuit, baseline, variant, config_id), trials in sorted(groups.items()):
        compiler = display(baseline, variant)
        ok = [t["result"] for t in trials if t["result"].get("completed") is True]
        bad = [t["result"] for t in trials if t["result"].get("completed") is not True]
        if bad:
            errors = sorted({str(r.get("error") or "unknown error") for r in bad})
            failures.append(
                FailureRecord(circuit, compiler, config_id, source_dir, len(bad), len(trials), errors)
            )
        if not ok:
            continue
        for r in ok:
            for metric in ("schedule_length", "logical_patches", "num_qubits"):
                if r.get(metric) is None:
                    raise TradeoffDataError(
                        f"{compiler} on {circuit} completed but '{metric}' is missing "
                        f"(trial {r.get('trial')}, {source_dir})"
                    )
            if r["logical_patches"] < r["num_qubits"]:
                raise TradeoffDataError(
                    f"{compiler} on {circuit}: logical_patches={r['logical_patches']} "
                    f"< num_qubits={r['num_qubits']}"
                )
        patches = [int(r["logical_patches"]) for r in ok]
        qubits = {r["num_qubits"] for r in ok}
        if len(qubits) != 1:
            raise TradeoffDataError(
                f"{compiler} on {circuit} ({config_id}): num_qubits {sorted(qubits)} vary "
                "across trials of one configuration"
            )
        # Seeded stochastic trials (PureMagic; in-tree models under cultivation)
        # may differ in footprint too (e.g. HARVEST's post-scheduling pruning):
        # like the schedule length, the point uses the mean over trials.
        lengths = [int(r["schedule_length"]) for r in ok]
        points.append(
            TradeoffPoint(
                circuit=circuit,
                baseline=baseline,
                variant=variant,
                compiler=compiler,
                config_id=config_id,
                source_dir=source_dir,
                num_qubits=qubits.pop(),
                logical_patches=mean(patches),
                logical_patches_min=min(patches),
                logical_patches_max=max(patches),
                schedule_length=mean(lengths),
                schedule_length_min=min(lengths),
                schedule_length_max=max(lengths),
                trials_successful=len(ok),
                trials_total=len(trials),
            )
        )
    return points, failures, notes


# --------------------------------------------------------------------------
# Bounds reconstruction
# --------------------------------------------------------------------------


def _resolve_qasm(path_text: str, project_root: Path) -> Path:
    path = Path(path_text)
    if path.exists():
        return path
    marker = "benchmark_circuits"
    if marker in path.parts:
        candidate = project_root.joinpath(*path.parts[path.parts.index(marker):])
        if candidate.exists():
            return candidate
    raise TradeoffDataError(f"benchmark input {path_text} no longer exists")


def compute_bounds(
    payloads: Sequence[Dict[str, Any]],
    result_dir: Path,
    project_root: Path,
    load_dag: Callable[[Path], Tuple[Any, Any]],
    circuits: Iterable[str],
    mode: str = MATCHED_IR,
    time_bound: str = "ppr",
    load_source: Optional[Callable[[Path], Any]] = None,
    cache: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, CircuitBounds]:
    """Reconstruct the matched DAG per circuit and derive its lower bounds.

    The DAG is rebuilt from the QASM input recorded by HARVEST/Silva (with a
    SHA-256 check against the recorded input).  When the PureMagic ``.trans``
    export of that DAG exists it is used as an independent cross-check; if no
    QASM record exists, the ``.trans`` export (the matched IR itself) is the
    sole source.

    ``cache`` (a dict the caller persists) memoizes bounds per circuit, keyed by
    the bound kind and the SHA-256 of the QASM and ``.trans`` inputs, so adding
    circuits never recomputes the existing ones and changed inputs never reuse
    stale bounds.

    ``time_bound="tdepth"`` instead uses the T-depth of the Clifford+T source
    circuit (loaded with ``load_source``), the dependency floor shared by
    systems with different IRs (native_end_to_end) and by FLASQ.
    """
    if time_bound not in ("ppr", "tdepth"):
        raise TradeoffDataError(f"unknown time bound {time_bound!r}")
    if time_bound == "tdepth" and load_source is None:
        raise TradeoffDataError("time_bound='tdepth' needs load_source")
    qasm_inputs: Dict[str, Dict[str, Any]] = {}
    num_qubits: Dict[str, int] = {}
    for payload in payloads:
        result, config = payload["result"], payload["config"]
        if result.get("comparison_mode") != mode:
            continue
        if result.get("num_qubits") is not None:
            num_qubits.setdefault(result["circuit"], int(result["num_qubits"]))
        # DASCOT may receive a pre-synthesized copy; bounds always come from
        # the original source recorded by HARVEST, PureMagic native or Silva.
        if str(config.get("input_path", "")).endswith(".qasm") and result.get("baseline") != "dascot":
            qasm_inputs.setdefault(result["circuit"], config)

    bounds: Dict[str, CircuitBounds] = {}
    for circuit in sorted(set(circuits)):
        # The driver writes the export under the filesystem-safe circuit name.
        trans_path = Path(result_dir) / "inputs" / f"{safe_name(circuit)}.trans"
        trans_text = trans_path.read_text() if trans_path.exists() else None
        config = qasm_inputs.get(circuit)
        qasm = actual = None
        if config is not None:
            qasm = _resolve_qasm(config["input_path"], project_root)
            expected = config.get("input_sha256")
            stat = qasm.stat()
            actual = _file_sha256(str(qasm.resolve()), stat.st_mtime_ns, stat.st_size)
            if expected and expected != actual:
                raise TradeoffDataError(
                    f"{qasm} changed since the experiment (sha256 {actual[:12]} != "
                    f"recorded {expected[:12]}); cannot reconstruct the same DAG"
                )
        trans_sha = (
            hashlib.sha256(trans_text.encode()).hexdigest() if trans_text is not None else "-"
        )
        cache_key = f"{circuit}|{time_bound}|{actual or '-'}|{trans_sha}"
        if cache is not None and cache_key in cache:
            cached = CircuitBounds(**{**cache[cache_key], "input_path": str(qasm) if qasm else None})
            if circuit in num_qubits and num_qubits[circuit] != cached.num_qubits:
                raise TradeoffDataError(
                    f"{circuit}: cached bounds have {cached.num_qubits} qubits, records report "
                    f"{num_qubits[circuit]}"
                )
            bounds[circuit] = cached
            continue
        if config is not None:
            qc, dag = load_dag(qasm)
            ops = dag_operations(dag)
            time_lb = dag_time_lower_bound(dag)
            has_magic = dag_has_magic_operation(dag)
            n = int(qc.num_qubits)
            source = f"DAG from {qasm.name}"
            input_path = str(qasm)
            if time_bound == "tdepth":
                time_lb = circuit_t_depth(load_source(qasm))
                source = f"T-depth of {qasm.name}"
                trans_text = None  # the PPR export is not this bound's IR
            if trans_text is not None:
                if len(parse_trans(trans_text)) != len(ops):
                    raise TradeoffDataError(
                        f"{circuit}: .trans has {len(parse_trans(trans_text))} ops but the "
                        f"reconstructed DAG has {len(ops)}"
                    )
                if trans_time_lower_bound(trans_text) != time_lb:
                    raise TradeoffDataError(
                        f"{circuit}: .trans critical path {trans_time_lower_bound(trans_text)} "
                        f"!= DAG critical path {time_lb}"
                    )
                source += " (cross-checked against .trans)"
        elif trans_text is not None and time_bound == "ppr":
            ops = parse_trans(trans_text)
            time_lb = trans_time_lower_bound(trans_text)
            has_magic = trans_has_magic_operation(trans_text)
            if circuit not in num_qubits:
                raise TradeoffDataError(f"{circuit}: num_qubits unknown")
            n = num_qubits[circuit]
            source = f".trans export {trans_path.name}"
            input_path = None
        else:
            raise TradeoffDataError(
                f"{circuit}: neither a QASM input record nor a .trans export is "
                "available to reconstruct the matched DAG"
            )
        if circuit in num_qubits and num_qubits[circuit] != n:
            raise TradeoffDataError(
                f"{circuit}: DAG has {n} qubits, records report {num_qubits[circuit]}"
            )
        bounds[circuit] = CircuitBounds(
            circuit=circuit,
            num_qubits=n,
            num_operations=len(ops),
            time_lb=time_lb,
            space_lb=space_lower_bound(n, has_magic),
            has_magic_state_operation=has_magic,
            source=source,
            input_path=input_path,
        )
        if cache is not None:
            cache[cache_key] = asdict(bounds[circuit])
    return bounds


# --------------------------------------------------------------------------
# Normalization, frontiers, summaries
# --------------------------------------------------------------------------


def normalize_point(point: TradeoffPoint, bounds: CircuitBounds) -> TradeoffPoint:
    if bounds.time_lb <= 0 or bounds.space_lb <= 0:
        raise TradeoffDataError(f"{point.circuit}: non-positive lower bound")
    if point.num_qubits != bounds.num_qubits:
        raise TradeoffDataError(
            f"{point.compiler} on {point.circuit}: num_qubits {point.num_qubits} "
            f"!= DAG qubits {bounds.num_qubits}"
        )
    point.space_lb = bounds.space_lb
    point.time_lb = bounds.time_lb
    point.normalized_space = point.logical_patches / bounds.space_lb
    point.normalized_time = point.schedule_length / bounds.time_lb
    point.normalized_space_time = point.normalized_space * point.normalized_time
    return point


def pareto_mask(points: Sequence[Tuple[float, float]]) -> List[bool]:
    """Non-dominated mask for minimizing both coordinates.

    A point is dominated if another is no worse in both and strictly better in
    one.  Exact duplicates are both kept (neither dominates the other).
    """
    mask = []
    for i, (xi, yi) in enumerate(points):
        dominated = any(
            xj <= xi and yj <= yi and (xj < xi or yj < yi)
            for j, (xj, yj) in enumerate(points)
            if j != i
        )
        mask.append(not dominated)
    return mask


def mark_pareto(points: Sequence[TradeoffPoint]) -> None:
    """Mark Pareto-optimal points per (compiler, circuit) in raw units."""
    groups: Dict[Tuple[str, str], List[TradeoffPoint]] = defaultdict(list)
    for p in points:
        groups[(p.compiler, p.circuit)].append(p)
    for group in groups.values():
        mask = pareto_mask([(p.logical_patches, p.schedule_length) for p in group])
        for p, keep in zip(group, mask):
            p.pareto_optimal = keep


def shared_circuits(points: Sequence[TradeoffPoint], compilers: Iterable[str]) -> List[str]:
    by_compiler: Dict[str, set] = defaultdict(set)
    for p in points:
        by_compiler[p.compiler].add(p.circuit)
    sets = [by_compiler.get(c, set()) for c in compilers]
    return sorted(set.intersection(*sets)) if sets else []


def representative_points(points: Sequence[TradeoffPoint]) -> Dict[Tuple[str, str], TradeoffPoint]:
    """One point per (compiler, circuit): the minimum normalized space-time.

    With a single configuration per compiler this is simply that point.  With
    a resource sweep it selects one measured Pareto point (never a blend).
    """
    chosen: Dict[Tuple[str, str], TradeoffPoint] = {}
    for p in points:
        key = (p.compiler, p.circuit)
        best = chosen.get(key)
        if best is None or (p.normalized_space_time, p.normalized_space) < (
            best.normalized_space_time,
            best.normalized_space,
        ):
            chosen[key] = p
    return chosen


def compiler_geomeans(
    points: Sequence[TradeoffPoint], circuits: Sequence[str]
) -> Dict[str, Dict[str, float]]:
    """Geometric-mean normalized space/time per compiler over ``circuits``."""
    reps = representative_points(points)
    out: Dict[str, Dict[str, float]] = {}
    for compiler in sorted({p.compiler for p in points}):
        chosen = [reps[(compiler, c)] for c in circuits if (compiler, c) in reps]
        if len(chosen) != len(circuits) or not chosen:
            continue
        out[compiler] = {
            "normalized_space": geometric_mean(p.normalized_space for p in chosen),
            "normalized_time": geometric_mean(p.normalized_time for p in chosen),
            "normalized_space_time": geometric_mean(p.normalized_space_time for p in chosen),
            "n": len(chosen),
        }
    return out


# --------------------------------------------------------------------------
# FLASQ reference model
# --------------------------------------------------------------------------

FLASQ_LABEL = "FLASQ model"


def flasq_points(
    flasq: Dict[str, Any],
    bounds: Dict[str, CircuitBounds],
    circuits: Sequence[str],
    source: str,
) -> List[TradeoffPoint]:
    """Turn ``flasq_price_trans.py`` output into normalized frontier points.

    FLASQ is an idealized resource model (Huggins et al., arXiv:2511.08508),
    not a scheduler: each point is the model's depth ``max(t_react·D, V/A)``
    for ``A`` fluid ancillas.  When FLASQ priced the Clifford+T source circuit
    (its native input), ``D`` is that circuit's T-depth and may lie below
    ``time_lb``.  When it priced the lowered Pauli-product stream, ``D`` must
    equal ``time_lb`` (both are the one-rotation-per-layer critical path),
    which is checked.
    """
    model = flasq.get("parameters", {}).get("cost_model", "unknown")
    out: List[TradeoffPoint] = []
    for circuit in circuits:
        records = flasq.get("circuits", {})
        record = records.get(circuit, records.get(safe_name(circuit)))
        if record is None:
            raise TradeoffDataError(f"FLASQ output has no entry for {circuit}")
        b = bounds[circuit]
        if record["num_qubits"] != b.num_qubits:
            raise TradeoffDataError(
                f"{circuit}: FLASQ register {record['num_qubits']} != {b.num_qubits}"
            )
        lowered = record.get("input_kind", "ppr_lowered") == "ppr_lowered"
        tdepth_bound = b.source.startswith("T-depth")
        if (lowered or tdepth_bound) and abs(record["measurement_depth"] - b.time_lb) > 1e-9:
            raise TradeoffDataError(
                f"{circuit}: FLASQ measurement depth {record['measurement_depth']} "
                f"!= dependency bound {b.time_lb}"
            )
        if not record.get("frontier"):
            raise TradeoffDataError(f"{circuit}: empty FLASQ frontier")
        for f in record["frontier"]:
            point = TradeoffPoint(
                circuit=circuit,
                baseline="flasq",
                variant=model,
                compiler=FLASQ_LABEL,
                config_id=f"A={f['fluid_ancilla']}",
                source_dir=source,
                num_qubits=record["num_qubits"],
                logical_patches=int(f["logical_qubits"]),
                schedule_length=float(f["total_depth"]),
                schedule_length_min=float(f["total_depth"]),
                schedule_length_max=float(f["total_depth"]),
                trials_successful=1,
                trials_total=1,
                in_shared_subset=True,
            )
            out.append(normalize_point(point, b))
    mark_pareto(out)
    return out


def geomean_attainable_curve(
    points: Sequence[TradeoffPoint], circuits: Sequence[str]
) -> List[Tuple[float, float]]:
    """Geomean over ``circuits`` of the best normalized time within ``x``× space.

    For every distinct normalized-space value ``x`` where each circuit has at
    least one point with ``normalized_space <= x``, take each circuit's
    minimum ``normalized_time`` among those points and return
    ``(x, geomean)``.  The result is a non-increasing staircase built only
    from evaluated points; nothing is interpolated.
    """
    by_circuit: Dict[str, List[TradeoffPoint]] = defaultdict(list)
    for p in points:
        by_circuit[p.circuit].append(p)
    missing = [c for c in circuits if c not in by_circuit]
    if missing:
        raise TradeoffDataError(f"no points for {missing}")
    curve: List[Tuple[float, float]] = []
    for x in sorted({p.normalized_space for c in circuits for p in by_circuit[c]}):
        best = []
        for c in circuits:
            feasible = [p.normalized_time for p in by_circuit[c] if p.normalized_space <= x + 1e-12]
            if not feasible:
                break
            best.append(min(feasible))
        else:
            y = geometric_mean(best)
            if not curve or y < curve[-1][1]:
                curve.append((x, y))
    return curve


def config_geomeans(
    points: Sequence[TradeoffPoint], circuits: Sequence[str]
) -> Dict[str, List[Dict[str, Any]]]:
    """Per compiler, one geomean point per configuration over ``circuits``.

    A configuration is a ``(source_dir, config_id)`` pair, i.e. one fixed
    architecture setting run on the whole suite.  Configurations missing any
    of ``circuits`` are skipped (reported via ``complete=False``) rather than
    averaged over a different subset.  Each entry carries ``pareto`` computed
    among that compiler's complete configurations.  A configuration whose
    per-circuit results are identical to an earlier one of the same compiler
    (e.g. a knob the tool ignores) is marked ``duplicate_of`` and excluded
    from the frontier instead of being drawn twice.
    """
    groups: Dict[Tuple[str, str, str], Dict[str, TradeoffPoint]] = defaultdict(dict)
    for p in points:
        groups[(p.compiler, p.source_dir, p.config_id)][p.circuit] = p
    out: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    seen: Dict[Tuple[str, tuple], str] = {}
    for (compiler, source_dir, config_id), by_circuit in sorted(groups.items()):
        chosen = [by_circuit[c] for c in circuits if c in by_circuit]
        signature = tuple(
            (p.circuit, p.logical_patches, round(float(p.schedule_length), 9)) for p in chosen
        )
        entry: Dict[str, Any] = {
            "source_dir": source_dir,
            "config_id": config_id,
            "n": len(chosen),
            "complete": len(chosen) == len(circuits) and bool(circuits),
        }
        duplicate = seen.get((compiler, signature))
        if duplicate is not None:
            entry["duplicate_of"] = duplicate
            entry["complete"] = False
        else:
            seen[(compiler, signature)] = f"{source_dir} [{config_id}]"
        if entry["complete"]:
            entry["normalized_space"] = geometric_mean(p.normalized_space for p in chosen)
            entry["normalized_time"] = geometric_mean(p.normalized_time for p in chosen)
            entry["normalized_space_time"] = geometric_mean(
                p.normalized_space_time for p in chosen
            )
        out[compiler].append(entry)
    for entries in out.values():
        complete = [e for e in entries if e["complete"]]
        mask = pareto_mask([(e["normalized_space"], e["normalized_time"]) for e in complete])
        for e, keep in zip(complete, mask):
            e["pareto"] = keep
    return dict(out)


def flasq_time_floors(flasq_points_: Sequence[TradeoffPoint]) -> Dict[str, float]:
    """FLASQ's best attainable depth per circuit (its reaction-limited floor)."""
    floors: Dict[str, float] = {}
    for p in flasq_points_:
        floors[p.circuit] = min(floors.get(p.circuit, math.inf), float(p.schedule_length))
    if any(v <= 0 or not math.isfinite(v) for v in floors.values()):
        raise TradeoffDataError("FLASQ produced a non-positive or infinite depth floor")
    return floors


def rebase_time(points: Sequence[TradeoffPoint], floors: Dict[str, float]) -> None:
    """Renormalize ``normalized_time`` against per-circuit FLASQ floors (in place).

    After this, FLASQ's own best point is exactly 1x in time for every circuit;
    the space normalization (one data patch per qubit plus one magic patch,
    which is also FLASQ's smallest footprint ``Q + 1``) is unchanged.
    """
    missing = sorted({p.circuit for p in points} - set(floors))
    if missing:
        raise TradeoffDataError(f"no FLASQ floor for {missing}")
    for p in points:
        p.time_lb = floors[p.circuit]
        p.normalized_time = p.schedule_length / p.time_lb
        p.normalized_space_time = p.normalized_space * p.normalized_time
