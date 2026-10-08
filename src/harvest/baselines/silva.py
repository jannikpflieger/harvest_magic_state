"""Silva et al. (TQC 2024) earliest-available-first baseline.

This module is deliberately separate from HARVEST's negotiated-congestion
router.  It follows Algorithms 1 and 2 of Silva et al.: take the dependency
ready set, build a Steiner tree over data terminals, attach the nearest
available static magic-storage terminal, and greedily remove used bus cells.
"""

from __future__ import annotations

import logging
from math import isclose, pi
from time import perf_counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from harvest.compilation.utils import node_needs_magic_state
from harvest.routing.scheduler import get_ready_nodes

from .base import BaselineAdapter, BaselineResult, BaselineRunConfig
from .metrics import space_time_volume

logger = logging.getLogger("HarvestMagicState.SilvaEAF")


def _copy_graph(
    graph: Dict[Any, List[Tuple[Any, float]]],
) -> Dict[Any, List[Tuple[Any, float]]]:
    return {node: list(neighbors) for node, neighbors in graph.items()}


def _remove_routing_nodes(graph: Dict, nodes: Iterable[Any]) -> None:
    routing_nodes = {node for node in nodes if isinstance(node, tuple)}
    for node in routing_nodes:
        graph.pop(node, None)
    for node in list(graph):
        graph[node] = [
            (nbr, weight) for nbr, weight in graph[node] if nbr not in routing_nodes
        ]


def _edge_key(a: Any, b: Any) -> Tuple[Any, Any]:
    return (a, b) if str(a) <= str(b) else (b, a)


def _nearest_magic_path(
    engine,
    graph: Dict,
    tree_nodes: Set[Any],
    magic_terminals: Sequence[Any],
) -> Optional[Tuple[Any, List[Any]]]:
    """Return ``(magic_terminal, tree-to-magic path)`` with stable tie breaks."""
    best: Optional[Tuple[float, int, str, Any, List[Any]]] = None
    for magic_index, magic in enumerate(magic_terminals):
        if magic not in graph:
            continue
        dist, prev = engine._dijkstra(graph, magic, targets=set(tree_nodes))
        for tree_node in tree_nodes:
            if tree_node not in dist:
                continue
            path = engine._reconstruct_path(prev, magic, tree_node)
            if path is None:
                continue
            candidate = (
                dist[tree_node],
                magic_index,
                str(tree_node),
                magic,
                list(reversed(path)),
            )
            if best is None or candidate[:3] < best[:3]:
                best = candidate
    if best is None:
        return None
    return best[3], best[4]


def _route_candidate(
    processor,
    dag,
    node,
    working_graph: Dict,
    available_magic: Sequence[Any],
) -> Dict[str, Any]:
    """Attempt one Algorithm-2 candidate without mutating shared state."""
    # Orientation is deterministic and based on the first still-available
    # storage patch.  The paper's graph exposes Pauli boundary vertices rather
    # than HARVEST's directional ports; this is the closest representation in
    # the current layout model and is recorded as a deviation in the audit.
    orientation_target = available_magic[0] if available_magic else None
    if orientation_target is None and node_needs_magic_state(node):
        return {"success": False, "error": "no static magic-storage terminal available"}

    if orientation_target is not None:
        data_terminals = processor._get_qubit_terminals_with_magic_direction(
            dag, node, orientation_target
        )
    else:
        data_terminals = processor._get_qubit_terminals(dag, node)

    if not data_terminals:
        return {"success": False, "error": "operation has no routable data terminal"}

    trial_graph = _copy_graph(working_graph)
    try:
        if len(data_terminals) == 1:
            sol_nodes = {data_terminals[0]}
            sol_edges: Set[Tuple[Any, Any]] = set()
        else:
            sol_nodes, sol_edges = processor.eng.steiner_tree(
                trial_graph, data_terminals
            )
    except (KeyError, ValueError) as exc:
        return {"success": False, "error": f"data Steiner tree failed: {exc}"}

    magic_terminal = None
    all_terminals = list(data_terminals)
    if node_needs_magic_state(node):
        nearest = _nearest_magic_path(
            processor.eng, trial_graph, set(sol_nodes), available_magic
        )
        if nearest is None:
            return {
                "success": False,
                "error": "no magic-storage path reaches the data tree",
            }
        magic_terminal, magic_path = nearest
        sol_nodes.update(magic_path)
        for a, b in zip(magic_path[:-1], magic_path[1:]):
            sol_edges.add(_edge_key(a, b))
        all_terminals = [magic_terminal] + all_terminals

    return {
        "success": True,
        "magic_terminal": magic_terminal,
        "qubit_terminals": list(data_terminals),
        "all_terminals": all_terminals,
        "steiner_nodes": set(sol_nodes),
        "steiner_edges": set(sol_edges),
    }


def process_dag_with_silva_eaf(processor, dag, visualize_each_step: bool = False):
    """Schedule a DAG with the Silva EAF/static-storage baseline.

    Ready candidates retain their stable circuit order.  Silva et al.'s text
    describes random candidate selection, but a deterministic order is used
    here so a reported trial is exactly reproducible and regression-testable.
    There is no negotiated congestion and no pruning.
    """
    del visualize_each_step  # no baseline-specific visualization at present
    validate_silva_ir(dag)
    all_nodes = list(dag.op_nodes())
    processed = set()
    results: List[Dict[str, Any]] = []
    timestep = 0
    failure_reasons: Dict[str, str] = {}

    if processor.magic_source is not None:
        logger.warning(
            "silva_eaf uses Silva's static, continuously replenished magic-storage "
            "assumption; the configured dynamic magic source is ignored"
        )

    while len(processed) < len(all_nodes):
        ready = get_ready_nodes(dag, all_nodes, processed)
        if not ready:
            break

        working_graph = _copy_graph(processor.graph)
        available_magic = list(processor.magic_terminals)
        scheduled_this_step = []

        for order_index, node in enumerate(ready):
            node_key = str(getattr(node, "_node_id", order_index))
            routed = _route_candidate(
                processor, dag, node, working_graph, available_magic
            )
            if not routed.get("success"):
                failure_reasons[node_key] = routed.get("error", "unroutable")
                continue

            failure_reasons.pop(node_key, None)
            magic_terminal = routed.get("magic_terminal")
            if magic_terminal in available_magic:
                available_magic.remove(magic_terminal)
            _remove_routing_nodes(working_graph, routed["steiner_nodes"])

            result = {
                "node": node,
                "gate_name": node.op.name,
                "qubits": [dag.find_bit(q).index for q in node.qargs],
                "time_step": timestep,
                "candidate_order": order_index,
                "algorithm": "silva_eaf",
                "success": True,
                **{k: v for k, v in routed.items() if k != "success"},
            }
            results.append(result)
            scheduled_this_step.append(node)

        if not scheduled_this_step:
            logger.warning(
                "Silva EAF made no progress at timestep %d; %d ready operations are unroutable",
                timestep,
                len(ready),
            )
            break

        processed.update(scheduled_this_step)
        timestep += 1

    processor.used_magic_terminals = {
        result["magic_terminal"]
        for result in results
        if result.get("time_step") == timestep - 1
        and result.get("magic_terminal") is not None
    }
    processor._scheduling_metadata = {
        "total_elapsed_steps": timestep,
        "num_nodes_completed": len(processed),
        "num_nodes_total": len(all_nodes),
        "completed": len(processed) == len(all_nodes),
        "unroutable_nodes": len(all_nodes) - len(processed),
        "final_unroutable_reasons": failure_reasons,
        "dependency_policy": "DAG roots (trivial/no-shared-data dependency semantics)",
        "candidate_order": "stable input order",
        "magic_state_model": "static storage, replenished each logical timestep",
    }
    return results


def _logical_patch_count(engine) -> int:
    blocked = sum(1 for value in engine.occ.values() if value == "BLOCKED")
    return engine.W * engine.H - blocked


def validate_silva_ir(dag) -> None:
    """Reject operations outside the post-transpilation ±π/8 PPR model."""
    unsupported = []
    for index, node in enumerate(dag.op_nodes()):
        if node.op.name != "PauliEvolution":
            unsupported.append(f"#{index}:{node.op.name}")
            continue
        if not node.op.params:
            unsupported.append(f"#{index}:PauliEvolution(no angle)")
            continue
        try:
            angle = float(node.op.params[0])
        except (TypeError, ValueError):
            unsupported.append(f"#{index}:PauliEvolution(symbolic angle)")
            continue
        if not isclose(abs(angle), pi / 8, rel_tol=1e-9, abs_tol=1e-10):
            unsupported.append(f"#{index}:PauliEvolution(angle={angle})")
    if unsupported:
        preview = ", ".join(unsupported[:8])
        if len(unsupported) > 8:
            preview += f", ... ({len(unsupported)} total)"
        raise ValueError(
            "Silva matched-IR requires a post-transpilation stream of ±pi/8 "
            f"Pauli rotations; unsupported operations: {preview}"
        )


class SilvaEAFAdapter(BaselineAdapter):
    """In-tree adapter exposing ``silva_eaf`` through the common schema."""

    name = "silva"

    def run(self, config: BaselineRunConfig, **kwargs: Any) -> BaselineResult:
        dag = kwargs.get("dag")
        layout_engine = kwargs.get("layout_engine")
        if dag is None or layout_engine is None:
            raise ValueError("SilvaEAFAdapter.run requires dag= and layout_engine=")

        if config.comparison_mode != "matched_ir":
            return BaselineResult.failed(
                config,
                input_representation="Pauli-product rotations",
                error="silva_eaf is only qualified for matched_ir comparisons",
            )

        try:
            validate_silva_ir(dag)
        except ValueError as exc:
            return BaselineResult.failed(
                config,
                input_representation="Pauli-product rotations",
                error=str(exc),
                notes=["Unsupported operations were not discarded or rewritten."],
            )

        from harvest.routing.processor import DAGProcessor

        start = perf_counter()
        processor = DAGProcessor(layout_engine=layout_engine)
        results = processor.process_entire_dag(dag, mode="silva_eaf")
        runtime = perf_counter() - start
        metadata = dict(getattr(processor, "_scheduling_metadata", {}))
        completed = bool(metadata.get("completed", False))
        length = metadata.get("total_elapsed_steps") if completed else None
        patches = _logical_patch_count(layout_engine)
        wirelength = sum(len(result.get("steiner_edges", ())) for result in results)
        schedule = [
            {
                "gate_name": result.get("gate_name"),
                "qubits": result.get("qubits", []),
                "time_step": result.get("time_step"),
                "candidate_order": result.get("candidate_order"),
                "magic_terminal": result.get("magic_terminal"),
                "routing_edges": len(result.get("steiner_edges", ())),
            }
            for result in results
        ]
        notes = [
            "No HARVEST negotiated-congestion optimization or pruning was used.",
            "Static magic-storage patches are available again each logical timestep.",
            "Stable input ordering replaces the paper's unspecified random candidate draw.",
            "Peak memory is null because in-process memory instrumentation is not enabled.",
        ]
        if not completed:
            notes.append("At least one dependency-ready operation was unroutable.")
        return BaselineResult(
            baseline=config.baseline,
            variant=config.variant,
            circuit=config.circuit,
            num_qubits=config.num_qubits,
            input_representation="Pauli-product rotations (±pi/8)",
            comparison_mode=config.comparison_mode,
            schedule_length=length,
            logical_patches=patches,
            space_time_volume=space_time_volume(patches, length),
            routing_wirelength=wirelength if completed else None,
            compiler_runtime_s=runtime,
            completed=completed,
            trial=config.trial,
            seed=None,
            upstream_version="Silva et al., TQC 2024 (Algorithms 1–2)",
            command="DAGProcessor.process_entire_dag(mode='silva_eaf')",
            notes=notes,
            error=None if completed else "one or more ready operations were unroutable",
            initial_logical_patches=patches,
            metadata={
                "scheduler": metadata,
                "schedule": schedule,
                "space_time_definition": "full_unpruned_layout_patches * schedule_length",
            },
        )
