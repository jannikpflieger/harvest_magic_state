"""
Adapters bridging HARVEST's DAGProcessor / layout objects to the
ILP solver's neutral data classes.

This module is intentionally *not* imported at the top level of the routing
package; it is imported lazily inside the scheduling functions that need it,
so that missing OR-Tools does not break the existing greedy schedulers.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("HarvestMagicState.ILPAdapters")


# ---------------------------------------------------------------------------
# ProductRequest construction
# ---------------------------------------------------------------------------


def prepare_ilp_product_requests(
    processor: Any,
    dag: Any,
    ready_nodes: List,
    ready_terminals: Optional[List[str]] = None,
) -> Tuple[List, Dict[int, Any]]:
    """Convert ready DAG nodes into ILP ``ProductRequest`` objects.

    Unlike :func:`~harvest.routing.scheduler._prepare_terminal_sets`, this
    function does **not** pre-select a magic terminal.  It only extracts the
    data-qubit port nodes and logical qubit indices for each product, leaving
    magic-root selection entirely to the ILP solver.

    A single *dummy* magic terminal (the first in the layout's list) is used
    solely to drive the directional port-orientation heuristic of
    :meth:`~harvest.routing.processor.DAGProcessor._get_qubit_terminals_with_magic_direction`.
    This is a first-version approximation; later versions could enumerate
    multiple port orientations per product.

    Args:
        processor: :class:`~harvest.routing.processor.DAGProcessor` instance.
        dag: The Qiskit ``DAGCircuit`` being scheduled.
        ready_nodes: Dependency-ready DAG operation nodes for this cycle.
        ready_terminals: If given, only these magic terminals are considered
            ready (used downstream for candidate pruning; not used here).

    Returns:
        ``(product_requests, pid_to_node)`` where *product_requests* is a
        list of :class:`~harvest.routing.ilp_steiner_packing.ProductRequest`
        (one per valid node) and *pid_to_node* maps the integer ``pid``
        assigned to each product back to its original DAG node.
    """
    from .ilp_steiner_packing import ProductRequest

    # Use the first available magic terminal as a dummy for direction hints.
    dummy_magic: Optional[str] = (
        processor.magic_terminals[0] if processor.magic_terminals else None
    )

    product_requests: List[ProductRequest] = []
    pid_to_node: Dict[int, Any] = {}

    for pid, node in enumerate(ready_nodes):
        if dummy_magic is None:
            logger.debug(
                "ILP adapters: no magic terminals in layout — cannot build product requests."
            )
            break

        # Extract data-qubit port terminals using a neutral dummy direction.
        qubit_terminals: List[str] = (
            processor._get_qubit_terminals_with_magic_direction(dag, node, dummy_magic)
        )

        if not qubit_terminals:
            logger.debug(
                "ILP adapters: node '%s' (pid=%d) has no qubit terminals — skipping.",
                node.op.name,
                pid,
            )
            continue

        # Logical qubit indices for the conflict constraint.
        if node.qargs:
            logical_qubits = frozenset(dag.find_bit(q).index for q in node.qargs)
        else:
            logical_qubits = frozenset()

        product_requests.append(
            ProductRequest(
                pid=pid,
                data_terminals=qubit_terminals,
                logical_qubits=logical_qubits,
                priority=1,
            )
        )
        pid_to_node[pid] = node

    return product_requests, pid_to_node


# ---------------------------------------------------------------------------
# Magic-candidate pruning
# ---------------------------------------------------------------------------


def prune_magic_candidates(
    product_requests: List,
    ready_magic_nodes: List[str],
    pos: Dict,
    max_k: int,
) -> Dict[int, List[str]]:
    """Prune magic candidate lists to at most *max_k* entries per product.

    Uses Manhattan distance from the centroid of a product's data-terminal
    positions to each ready magic node (via the ``pos`` dict from
    :meth:`~harvest.layout.engine.LayoutEngine.build_routing_graph`).

    If positional information is unavailable for a product's terminals, all
    ready magic nodes are kept as candidates (no pruning for that product).

    Args:
        product_requests: Products whose candidate lists should be pruned.
        ready_magic_nodes: All magic-terminal port node IDs ready this cycle.
        pos: Node-to-position mapping ``{node: (x, y)}``.
        max_k: Maximum number of candidate magic nodes per product.

    Returns:
        ``{pid: [magic_node, ...]}`` with at most *max_k* entries per product,
        sorted by ascending Manhattan distance.
    """
    if max_k <= 0 or not ready_magic_nodes:
        return {p.pid: list(ready_magic_nodes) for p in product_requests}

    candidate_roots: Dict[int, List[str]] = {}

    for p in product_requests:
        if not ready_magic_nodes:
            candidate_roots[p.pid] = []
            continue

        # Compute centroid of data-terminal positions.
        terminal_coords: List[Tuple[float, float]] = []
        for t in p.data_terminals:
            pt = pos.get(t)
            if pt is not None:
                terminal_coords.append(pt)

        if not terminal_coords:
            # No position data: keep up to max_k magic nodes unranked.
            candidate_roots[p.pid] = list(ready_magic_nodes[:max_k])
            continue

        cx = sum(c[0] for c in terminal_coords) / len(terminal_coords)
        cy = sum(c[1] for c in terminal_coords) / len(terminal_coords)

        def _dist(m: str) -> float:
            mp = pos.get(m)
            if mp is None:
                return float("inf")
            return abs(mp[0] - cx) + abs(mp[1] - cy)

        ranked = sorted(ready_magic_nodes, key=_dist)
        candidate_roots[p.pid] = ranked[:max_k]

    return candidate_roots


# ---------------------------------------------------------------------------
# Result translation
# ---------------------------------------------------------------------------


def routed_products_to_results(
    routed_products: List,
    pid_to_node: Dict[int, Any],
    product_requests: List,
    time_step: int,
) -> List[dict]:
    """Translate ILP :class:`~harvest.routing.ilp_steiner_packing.RoutedProduct`
    objects back to the standard scheduler result-dict format.

    The returned dicts are compatible with the existing downstream pipeline
    (wirelength evaluation, metadata collection, magic-state consumption).

    Args:
        routed_products: Scheduled products from
            :class:`~harvest.routing.ilp_steiner_packing.CycleRoutingResult`.
        pid_to_node: ``{pid: DAGOpNode}`` mapping from
            :func:`prepare_ilp_product_requests`.
        product_requests: Original product request list (used to recover
            ``data_terminals`` and ``logical_qubits``).
        time_step: Current scheduling time step index.

    Returns:
        List of result dicts.  Only products present in *routed_products* are
        included; un-scheduled products are omitted (they remain in the
        ready queue and will be retried in a later cycle).
    """
    pid_to_req: Dict[int, Any] = {p.pid: p for p in product_requests}
    results: List[dict] = []

    for rp in routed_products:
        node = pid_to_node.get(rp.pid)
        req = pid_to_req.get(rp.pid)
        if node is None or req is None:
            logger.warning(
                "ILP result: pid=%d has no matching DAG node or ProductRequest — skipping.",
                rp.pid,
            )
            continue

        # Reconstruct qubit index list in the order the DAG stores them.
        if node.qargs:
            # Use the same index computation as the existing scheduler.
            from qiskit.dagcircuit import DAGCircuit  # noqa: F401 – type hint only
            # ``dag`` is not available here; recover from node.qargs if possible.
            # In practice the caller already has the dag; but pid_to_node stores
            # the raw node, so we fall back to sorted logical_qubits.
            qubits = sorted(req.logical_qubits)
        else:
            qubits = []

        result: dict = {
            "node": node,
            "gate_name": node.op.name,
            "qubits": qubits,
            "time_step": time_step,
            "success": True,
            "magic_wait_cycles": 0,
            "magic_terminal": rp.magic_root,
            "qubit_terminals": req.data_terminals,
            "all_terminals": [rp.magic_root] + req.data_terminals,
            "steiner_nodes": rp.used_nodes,
            "steiner_edges": rp.used_edges,
            # ILP-specific metadata (does not break existing consumers).
            "ilp_routed": True,
        }
        results.append(result)

    return results
