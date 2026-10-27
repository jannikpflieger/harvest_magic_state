"""Independent validator for routed lattice-surgery schedules (EVAL_PLAN B.3 / E7).

A schedule is the result list of ``DAGProcessor.process_entire_dag`` (HARVEST
or the in-tree Silva EAF): one dict per operation with ``time_step``,
``magic_terminal``, ``qubit_terminals``, ``steiner_nodes`` and
``steiner_edges``.  The validator re-derives every requirement from the DAG
and the routing graph alone; it shares no scheduling code, so a scheduler bug
cannot hide behind the same helper.

Rules
-----
V1  dependencies: every operation is scheduled exactly once, inside
    ``[0, T)``, strictly after each of its DAG predecessors.
V2  exclusivity: within one time step no routing tile, port, or magic
    terminal is used by two operations.
V3  routes: every node/edge of a route exists in the routing graph, the route
    is connected and contains each required terminal (the X/Z port of every
    X/Z factor, both ports of a Y factor, one magic port iff the operation
    consumes a magic state) and no port of an unrelated patch.
V4  magic supply: replaying a fresh magic-state source (same model and seed)
    against the schedule, every consumed terminal holds a ready state, and
    the number of consumptions the scheduler performed equals the number of
    magic operations.
"""

from __future__ import annotations

from collections import defaultdict
from time import perf_counter
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set

RULES = ("V1", "V2", "V3", "V4")
_MAX_MESSAGES = 20


def _port_info(node) -> Optional[tuple]:
    """``(patch, port_type)`` of a port node ``P:<patch>:<port_id>:<type>``."""
    if isinstance(node, str) and node.startswith("P:"):
        parts = node.split(":")
        if len(parts) >= 4:
            return parts[1], parts[-1]
    return None


def needs_magic(node) -> bool:
    # After PCB conversion every non-Clifford rotation is a PauliEvolution.
    return node.op.name == "PauliEvolution"


def pauli_factors(dag, node) -> Dict[int, str]:
    """``{global qubit index: 'X'|'Y'|'Z'}`` of a Pauli-product operation."""
    qubits = [dag.find_bit(q).index for q in node.qargs]
    op = node.op
    operator = getattr(op, "operator", None)
    label_by_position: Dict[int, str] = {}
    if operator is not None and hasattr(operator, "to_sparse_list"):
        terms = operator.to_sparse_list()
        if len(terms) != 1:
            raise ValueError(f"{op.name}: expected one Pauli term, got {len(terms)}")
        label, indices = terms[0][0], terms[0][1]
        label_by_position = dict(zip(indices, label))
    elif operator is not None and hasattr(operator, "to_label"):  # qiskit Pauli
        label = operator.to_label().lstrip("+-i")
        label_by_position = {i: c for i, c in enumerate(reversed(label))}
    elif getattr(op, "pauli", None) is not None:
        label = op.pauli.to_label().lstrip("+-i")
        label_by_position = {i: c for i, c in enumerate(reversed(label))}
    else:
        raise ValueError(f"cannot read the Pauli product of operation {op.name}")
    return {
        qubits[position]: pauli
        for position, pauli in label_by_position.items()
        if pauli in "XYZ"
    }


class _Report:
    def __init__(self) -> None:
        self.by_rule: Dict[str, int] = {rule: 0 for rule in RULES}
        self.first_rule: Optional[str] = None
        self.messages: List[str] = []

    def add(self, rule: str, message: str) -> None:
        self.by_rule[rule] += 1
        if self.first_rule is None:
            self.first_rule = rule
        if len(self.messages) < _MAX_MESSAGES:
            self.messages.append(f"{rule}: {message}")

    @property
    def total(self) -> int:
        return sum(self.by_rule.values())


def _connected(nodes: Set[Any], edges: Iterable[tuple]) -> bool:
    if len(nodes) <= 1:
        return True
    adjacency: Dict[Any, List[Any]] = defaultdict(list)
    for a, b in edges:
        adjacency[a].append(b)
        adjacency[b].append(a)
    start = next(iter(nodes))
    seen = {start}
    stack = [start]
    while stack:
        u = stack.pop()
        for v in adjacency[u]:
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return nodes <= seen


def validate(
    processor,
    dag,
    results: Sequence[Dict[str, Any]],
    make_source: Optional[Callable[[List[Any]], Any]] = None,
    *,
    total_steps: Optional[int] = None,
    graph: Optional[Dict[Any, list]] = None,
    consumed: Optional[int] = None,
    rules: Sequence[str] = RULES,
) -> Dict[str, Any]:
    """Check ``results`` against rules V1-V4; returns a JSON-serializable report.

    ``graph`` defaults to ``processor.graph`` -- call once before pruning and
    once after to check that pruning kept every tile a route uses.
    ``make_source(terminals)`` builds a fresh magic-state source with the run's
    model and seed (``None``: magic states are free, V4 replay is skipped).
    ``consumed`` is the number of consumptions the scheduler's own source
    recorded.
    """
    started = perf_counter()
    graph = processor.graph if graph is None else graph
    adjacency = {u: {v for v, _ in nbrs} for u, nbrs in graph.items()}
    report = _Report()
    check = set(rules)
    routed = [r for r in results if r.get("success", True)]
    steps = total_steps
    if steps is None:
        steps = processor.__dict__.get("_scheduling_metadata", {}).get("total_elapsed_steps")
    if steps is None:
        steps = 1 + max((r["time_step"] for r in routed), default=-1)

    op_nodes = list(dag.op_nodes())
    step_of: Dict[int, int] = {}
    if "V1" in check:
        for r in routed:
            node_id = r["node"]._node_id
            if node_id in step_of:
                report.add("V1", f"operation {node_id} scheduled twice")
            step_of[node_id] = r["time_step"]
            if not 0 <= r["time_step"] < steps:
                report.add("V1", f"operation {node_id} at step {r['time_step']} outside [0, {steps})")
        missing = [n._node_id for n in op_nodes if n._node_id not in step_of]
        if missing:
            report.add("V1", f"{len(missing)} operation(s) never scheduled, e.g. {missing[:5]}")
        for node in op_nodes:
            t = step_of.get(node._node_id)
            if t is None:
                continue
            for pred in dag.predecessors(node):
                if not hasattr(pred, "op"):
                    continue
                tp = step_of.get(pred._node_id)
                if tp is not None and tp >= t:
                    report.add("V1", f"operation {node._node_id} at step {t} does not follow "
                                     f"predecessor {pred._node_id} at step {tp}")

    by_step: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for r in routed:
        by_step[r["time_step"]].append(r)

    if "V2" in check:
        for t, ops in by_step.items():
            owner: Dict[Any, int] = {}
            for r in ops:
                node_id = r["node"]._node_id
                used = set(r.get("steiner_nodes", ())) | set(r.get("qubit_terminals", ()))
                if r.get("magic_terminal") is not None:
                    used.add(r["magic_terminal"])
                for item in used:
                    other = owner.setdefault(item, node_id)
                    if other != node_id:
                        kind = "tile" if isinstance(item, tuple) else "port"
                        report.add("V2", f"step {t}: {kind} {item} used by operations "
                                         f"{other} and {node_id}")

    if "V3" in check:
        for r in routed:
            node = r["node"]
            node_id = node._node_id
            nodes = set(r.get("steiner_nodes", ()))
            edges = [tuple(e) for e in r.get("steiner_edges", ())]
            magic = r.get("magic_terminal")
            try:
                factors = pauli_factors(dag, node)
            except ValueError as exc:
                report.add("V3", f"operation {node_id}: {exc}")
                continue
            required = set()
            for q, pauli in factors.items():
                for port_type in (("X", "Z") if pauli == "Y" else (pauli,)):
                    required.add((f"q_{q}", port_type))
            present = {_port_info(n) for n in nodes} - {None}
            for patch, port_type in sorted(required - present):
                report.add("V3", f"operation {node_id}: route lacks a {port_type} port of {patch}")
            if needs_magic(node):
                info = _port_info(magic)
                if magic is None or info is None or info[1] != "M":
                    report.add("V3", f"operation {node_id}: no magic port recorded ({magic!r})")
                elif magic not in nodes:
                    report.add("V3", f"operation {node_id}: route does not reach magic port {magic}")
            elif magic is not None:
                report.add("V3", f"operation {node_id}: Clifford product holds magic port {magic}")
            allowed_patches = {f"q_{q}" for q in factors}
            if magic is not None and _port_info(magic) is not None:
                allowed_patches.add(_port_info(magic)[0])
            for n in nodes:
                info = _port_info(n)
                if info is not None and info[0] not in allowed_patches:
                    report.add("V3", f"operation {node_id}: route passes through port {n} "
                                     f"of unrelated patch {info[0]}")
                if n not in adjacency:
                    report.add("V3", f"operation {node_id}: node {n} is not in the routing graph")
            for a, b in edges:
                if a not in nodes or b not in nodes:
                    report.add("V3", f"operation {node_id}: edge {a}-{b} leaves the route's nodes")
                elif b not in adjacency.get(a, ()) and a not in adjacency.get(b, ()):
                    report.add("V3", f"operation {node_id}: edge {a}-{b} is not in the routing graph")
            if not _connected(nodes, edges):
                report.add("V3", f"operation {node_id}: route is not connected")

    magic_ops = sum(1 for r in routed if needs_magic(r["node"]))
    not_ready = 0
    delayed_steps = 0
    if "V4" in check:
        source = make_source(list(processor.magic_terminals)) if make_source else None
        if source is not None:
            for t in range(steps):
                source.tick()
                ready = source.get_ready_terminals()
                ready = None if ready is None else set(ready)
                step_delayed = False
                for r in by_step.get(t, ()):
                    if not needs_magic(r["node"]):
                        continue
                    terminal = r.get("magic_terminal")
                    if ready is not None and terminal not in ready:
                        not_ready += 1
                        step_delayed = True
                        report.add("V4", f"step {t}: operation {r['node']._node_id} consumes from "
                                         f"{terminal}, which holds no ready state")
                        continue
                    source.consume(terminal)
                    if ready is not None:
                        ready.discard(terminal)
                delayed_steps += step_delayed
        if consumed is not None and consumed != magic_ops:
            report.add("V4", f"scheduler consumed {consumed} magic states for {magic_ops} "
                             "magic operations")

    return {
        "ok": report.total == 0,
        "n_violations": report.total,
        "violations_by_rule": dict(report.by_rule),
        "first_violation_rule": report.first_rule,
        "messages": report.messages,
        "n_ops": len(op_nodes),
        "n_routed": len(routed),
        "n_steps": steps,
        "magic_ops": magic_ops,
        "magic_not_ready": not_ready,
        "steps_needing_delay": delayed_steps,
        "rules_checked": [r for r in RULES if r in check],
        "validate_time_s": perf_counter() - started,
    }
