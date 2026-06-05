"""
ILP-based per-cycle Steiner forest packing scheduler.

Solves a mixed-integer program (via OR-Tools CP-SAT) to jointly:
  1. Select a subset of products to schedule this cycle (maximise priority-weighted count).
  2. Assign each scheduled product a ready magic-state root node.
  3. Route node-disjoint Steiner trees connecting each magic root to all
     data-qubit terminals of that product.
  4. Minimise secondary route length as a tie-breaker.

Only routing cells (``(x, y)`` tuples) are subject to the node-disjoint
constraint, matching the semantics of the existing greedy packing algorithm.

OR-Tools is an optional dependency.  If it is not installed the ILP scheduler
will raise ``ImportError`` when invoked; callers should catch this and fall
back to the greedy packing scheduler.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("HarvestMagicState.ILPSteiner")

# ---------------------------------------------------------------------------
# OR-Tools optional import
# ---------------------------------------------------------------------------

try:
    from ortools.sat.python import cp_model as _cp_model  # type: ignore[import]

    _ORTOOLS_AVAILABLE = True
except ImportError:
    _cp_model = None  # type: ignore[assignment]
    _ORTOOLS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Configuration / data classes
# ---------------------------------------------------------------------------


@dataclass
class ILPConfig:
    """Configuration parameters for :class:`ILPSteinerForestRouter`.

    Attributes:
        time_limit_ms: Wall-clock budget for the CP-SAT solver in
            milliseconds (default 3000).
        max_ready_products: If the number of ready products exceeds this
            threshold the ILP is skipped and the greedy fallback is used
            (default 20).
        max_magic_candidates: Keep at most this many magic-terminal
            candidates per product after distance pruning (default 8).
        max_graph_nodes: If the number of relevant graph nodes exceeds this
            limit the ILP is skipped (default 500).
        count_weight: Reward for scheduling one product.  Must dominate the
            total route-length penalty so that scheduling more products
            always beats shorter routes (default 100 000).
        edge_cost: Per routing-cell-to-routing-cell directed arc penalty
            (default 1).
        node_cost: Per routing-cell usage penalty (default 0; set > 0 for
            sparse routing).
        fallback_to_greedy: When ``True``, fall back to the greedy packing
            scheduler if the ILP produces no scheduled products or errors
            (default ``True``).
    """

    time_limit_ms: int = 3000
    max_ready_products: int = 20
    max_magic_candidates: int = 8
    max_graph_nodes: int = 500
    count_weight: int = 100_000
    edge_cost: int = 1
    node_cost: int = 0
    fallback_to_greedy: bool = True


@dataclass
class ProductRequest:
    """Solver-facing representation of one Pauli product to be routed.

    Attributes:
        pid: Unique integer product id (index within the current cycle).
        data_terminals: Port node IDs (strings like ``"P:q_0:N0:X"``) for
            each required data-qubit contact.
        logical_qubits: Frozenset of qubit indices touched by this product.
            Products whose ``logical_qubits`` overlap cannot both be
            scheduled in the same cycle.
        priority: Scheduling priority weight (default 1).  Higher values
            make the product more attractive to the scheduler.
    """

    pid: int
    data_terminals: List[str]
    logical_qubits: frozenset
    priority: int = 1


@dataclass
class RoutedProduct:
    """Routing solution for one scheduled product.

    Attributes:
        pid: Product id matching the originating :class:`ProductRequest`.
        magic_root: Selected magic-terminal port node ID.
        used_nodes: All nodes in the Steiner tree (routing cells as
            ``(x, y)`` tuples *plus* the magic-root and data-terminal
            port string nodes).
        used_edges: Undirected edges in the Steiner tree represented as
            ``(u, v)`` tuples canonicalised so that ``str(u) <= str(v)``.
    """

    pid: int
    magic_root: str
    used_nodes: Set
    used_edges: Set


@dataclass
class CycleRoutingResult:
    """Output of one call to :meth:`ILPSteinerForestRouter.solve_cycle`.

    Attributes:
        scheduled: List of routed products for this cycle.
        objective_value: Raw solver objective value.
        status: Human-readable solver status string (``"OPTIMAL"``,
            ``"FEASIBLE"``, ``"INFEASIBLE"``, ``"UNKNOWN"``, ``"SKIPPED"``
            etc.).
        solver_wall_time_ms: Actual wall-clock time used by the solver.
        fallback_used: ``True`` when the greedy fallback was triggered.
    """

    scheduled: List[RoutedProduct]
    objective_value: float
    status: str
    solver_wall_time_ms: float
    fallback_used: bool = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _canon_edge(u: Any, v: Any) -> Tuple:
    """Canonical undirected-edge representation (same convention as
    :meth:`~harvest.layout.engine.LayoutEngine.steiner_tree`)."""
    return (u, v) if str(u) <= str(v) else (v, u)


# ---------------------------------------------------------------------------
# ILP solver
# ---------------------------------------------------------------------------


class ILPSteinerForestRouter:
    """Per-cycle ILP-based Steiner forest packing router.

    Usage::

        router = ILPSteinerForestRouter()
        result = router.solve_cycle(graph, products, ready_magic_nodes,
                                    candidate_roots=candidate_roots,
                                    config=ILPConfig(time_limit_ms=3000))
    """

    def solve_cycle(
        self,
        graph: Dict,
        products: List[ProductRequest],
        ready_magic_nodes: List[str],
        candidate_roots: Optional[Dict[int, List[str]]] = None,
        config: Optional[ILPConfig] = None,
    ) -> CycleRoutingResult:
        """Solve one scheduling cycle.

        Args:
            graph: Routing graph ``{node: [(neighbor, weight), ...]}``.
                Routing cells are ``(x, y)`` tuples; port nodes are strings.
            products: Products eligible for scheduling this cycle.
            ready_magic_nodes: Magic-terminal port node IDs ready for
                consumption.
            candidate_roots: Optional per-product magic candidate lists
                ``{pid: [magic_node, ...]}``.  If ``None``, all
                ``ready_magic_nodes`` are candidates for every product.
            config: Solver configuration.  Defaults to :class:`ILPConfig`.

        Returns:
            :class:`CycleRoutingResult` describing the scheduled subset.
        """
        if config is None:
            config = ILPConfig()

        if not _ORTOOLS_AVAILABLE:
            raise ImportError(
                "OR-Tools is required for the ILP Steiner packing scheduler. "
                "Install it with:  pip install ortools"
            )

        if not products:
            return CycleRoutingResult([], 0.0, "OPTIMAL", 0.0)

        # ------------------------------------------------------------------
        # Build per-product candidate root mapping
        # ------------------------------------------------------------------
        if candidate_roots is None:
            candidate_roots = {p.pid: list(ready_magic_nodes) for p in products}

        # ------------------------------------------------------------------
        # Determine relevant nodes: routing cells + used port nodes
        # ------------------------------------------------------------------
        routing_cells: List = [n for n in graph if isinstance(n, tuple)]
        relevant_ports: Set[str] = set()
        for p in products:
            for t in p.data_terminals:
                if t in graph:
                    relevant_ports.add(t)
            for m in candidate_roots.get(p.pid, []):
                if m in graph:
                    relevant_ports.add(m)

        relevant_nodes: Set = set(routing_cells) | relevant_ports

        if len(relevant_nodes) > config.max_graph_nodes:
            logger.warning(
                "ILP: %d relevant nodes exceeds limit %d — skipping.",
                len(relevant_nodes),
                config.max_graph_nodes,
            )
            return CycleRoutingResult([], 0.0, "SKIPPED", 0.0)

        # ------------------------------------------------------------------
        # Build directed arc list (only between relevant nodes)
        # ------------------------------------------------------------------
        arcs: List[Tuple] = []
        arc_set: Set[Tuple] = set()
        for u, neighbors in graph.items():
            if u not in relevant_nodes:
                continue
            for v, _w in neighbors:
                if v not in relevant_nodes:
                    continue
                arc = (u, v)
                if arc not in arc_set:
                    arcs.append(arc)
                    arc_set.add(arc)

        arc_idx: Dict[Tuple, int] = {arc: i for i, arc in enumerate(arcs)}

        # Build per-node incidence lists for the flow conservation pass
        arcs_out: Dict[Any, List[Tuple]] = {n: [] for n in relevant_nodes}
        arcs_in: Dict[Any, List[Tuple]] = {n: [] for n in relevant_nodes}
        for arc in arcs:
            u, v = arc
            if u in arcs_out:
                arcs_out[u].append(arc)
            if v in arcs_in:
                arcs_in[v].append(arc)

        # ------------------------------------------------------------------
        # Build CP-SAT model
        # ------------------------------------------------------------------
        model = _cp_model.CpModel()

        # y[p.pid]  —  product p is scheduled
        y: Dict[int, Any] = {
            p.pid: model.NewBoolVar(f"y_{p.pid}") for p in products
        }

        # r[p.pid][m]  —  product p selects magic root m
        r: Dict[int, Dict[str, Any]] = {}
        for p in products:
            r[p.pid] = {}
            for m in candidate_roots.get(p.pid, []):
                if m in graph:
                    r[p.pid][m] = model.NewBoolVar(f"r_{p.pid}_{m}")

        # x[p.pid][v]  —  product p uses routing cell v
        x: Dict[int, Dict[Any, Any]] = {}
        for p in products:
            x[p.pid] = {
                v: model.NewBoolVar(f"x_{p.pid}_{v[0]}_{v[1]}")
                for v in routing_cells
            }

        # a[p.pid][arc]  —  product p uses directed arc
        a: Dict[int, Dict[Tuple, Any]] = {}
        for p in products:
            a[p.pid] = {
                arc: model.NewBoolVar(f"a_{p.pid}_{arc_idx[arc]}")
                for arc in arcs
            }

        # f[p.pid][arc]  —  single-commodity flow for product p on arc
        # Value in [0, max_terminals]; max_t ≥ 1 to avoid degenerate bounds.
        f: Dict[int, Dict[Tuple, Any]] = {}
        for p in products:
            max_t = max(len(p.data_terminals), 1)
            f[p.pid] = {
                arc: model.NewIntVar(0, max_t, f"f_{p.pid}_{arc_idx[arc]}")
                for arc in arcs
            }

        # ------------------------------------------------------------------
        # Constraints
        # ------------------------------------------------------------------

        # 1. Root selection: exactly one magic root per scheduled product.
        for p in products:
            roots = list(r[p.pid].values())
            if roots:
                model.Add(sum(roots) == y[p.pid])
            else:
                # No reachable magic candidates → cannot schedule.
                model.Add(y[p.pid] == 0)

        # 2. Routing-cell usage only if product is scheduled.
        for p in products:
            for v in routing_cells:
                model.Add(x[p.pid][v] <= y[p.pid])

        # 3. Arc → endpoint node usage (routing-cell endpoints only).
        for p in products:
            for arc in arcs:
                u, v = arc
                if isinstance(u, tuple):
                    model.Add(a[p.pid][arc] <= x[p.pid][u])
                if isinstance(v, tuple):
                    model.Add(a[p.pid][arc] <= x[p.pid][v])

        # 4. Routing cell used only if at least one incident arc is used
        #    (prevents phantom cell reservations).
        for p in products:
            for v in routing_cells:
                incident = (
                    [a[p.pid][arc] for arc in arcs_out.get(v, []) if arc in a[p.pid]]
                    + [a[p.pid][arc] for arc in arcs_in.get(v, []) if arc in a[p.pid]]
                )
                if incident:
                    model.Add(x[p.pid][v] <= sum(incident))
                else:
                    model.Add(x[p.pid][v] == 0)

        # 5. Flow bounded by (max_terminals × arc_selected).
        for p in products:
            max_t = max(len(p.data_terminals), 1)
            for arc in arcs:
                model.Add(f[p.pid][arc] <= max_t * a[p.pid][arc])

        # 6. Single-commodity flow conservation.
        #
        #    For product p the magic root emits |T_p| units of flow.
        #    Each data terminal absorbs exactly 1 unit.
        #    All other nodes are balanced (net flow = 0).
        #
        #    Connectivity is guaranteed: if any terminal is unreachable, the
        #    balance constraint is violated, forcing y[p] = 0.
        for p in products:
            max_t = len(p.data_terminals)
            terminal_set = set(p.data_terminals)
            cand_magic_set = set(candidate_roots.get(p.pid, []))

            for node in relevant_nodes:
                # Build weighted-sum for net outflow (outflow − inflow).
                terms: List[Any] = []
                coeffs: List[int] = []
                for arc in arcs_out.get(node, []):
                    if arc in f[p.pid]:
                        terms.append(f[p.pid][arc])
                        coeffs.append(1)
                for arc in arcs_in.get(node, []):
                    if arc in f[p.pid]:
                        terms.append(f[p.pid][arc])
                        coeffs.append(-1)

                if not terms:
                    # Node has no flow arcs for this product — force
                    # any demand it would place to zero.
                    if node in cand_magic_set and node in r[p.pid]:
                        model.Add(r[p.pid][node] == 0)
                    elif node in terminal_set:
                        model.Add(y[p.pid] == 0)
                    # Internal isolated node: trivially balanced, no constraint.
                    continue

                net_out = _cp_model.LinearExpr.WeightedSum(terms, coeffs)

                if node in cand_magic_set and node in r[p.pid]:
                    # Source: net outflow = max_t × r[p, m].
                    model.Add(net_out == max_t * r[p.pid][node])
                elif node in terminal_set:
                    # Sink: net outflow = −y[p]  (absorbs 1 unit).
                    model.Add(net_out + y[p.pid] == 0)
                else:
                    # Internal pass-through: balanced.
                    model.Add(net_out == 0)

        # 7. Node-disjoint for routing cells: at most one product per cell.
        for v in routing_cells:
            users = [x[p.pid][v] for p in products]
            if len(users) > 1:
                model.Add(sum(users) <= 1)

        # 8. Each magic terminal used by at most one product per cycle.
        for m in ready_magic_nodes:
            users = [r[p.pid][m] for p in products if m in r[p.pid]]
            if len(users) > 1:
                model.Add(sum(users) <= 1)

        # 9. Logical-qubit conflict: conflicting product pairs cannot both
        #    be scheduled in the same cycle.
        for i, p1 in enumerate(products):
            for p2 in products[i + 1 :]:
                if p1.logical_qubits & p2.logical_qubits:
                    model.Add(y[p1.pid] + y[p2.pid] <= 1)

        # ------------------------------------------------------------------
        # Objective: maximise scheduled products, minimise routing cost.
        # ------------------------------------------------------------------
        obj_terms: List[Any] = []
        for p in products:
            obj_terms.append(config.count_weight * p.priority * y[p.pid])

        if config.edge_cost > 0:
            for p in products:
                for arc in arcs:
                    u, v = arc
                    # Penalise only routing-cell ↔ routing-cell arcs.
                    if isinstance(u, tuple) and isinstance(v, tuple):
                        obj_terms.append(-config.edge_cost * a[p.pid][arc])

        if config.node_cost > 0:
            for p in products:
                for v in routing_cells:
                    obj_terms.append(-config.node_cost * x[p.pid][v])

        if obj_terms:
            model.Maximize(sum(obj_terms))

        # ------------------------------------------------------------------
        # Solve
        # ------------------------------------------------------------------
        solver = _cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = config.time_limit_ms / 1000.0
        solver.parameters.log_search_progress = False

        t0 = time.monotonic()
        status_code = solver.Solve(model)
        wall_time_ms = (time.monotonic() - t0) * 1000.0

        status_name = solver.StatusName(status_code)
        logger.debug(
            "ILP: status=%s time=%.0fms products=%d arcs=%d",
            status_name,
            wall_time_ms,
            len(products),
            len(arcs),
        )

        if status_code not in (_cp_model.OPTIMAL, _cp_model.FEASIBLE):
            return CycleRoutingResult([], 0.0, status_name, wall_time_ms)

        # ------------------------------------------------------------------
        # Extract solution
        # ------------------------------------------------------------------
        obj_value = solver.ObjectiveValue()
        scheduled: List[RoutedProduct] = []

        for p in products:
            if solver.Value(y[p.pid]) != 1:
                continue

            # Selected magic root
            magic_root: Optional[str] = None
            for m, rv in r[p.pid].items():
                if solver.Value(rv) == 1:
                    magic_root = m
                    break
            if magic_root is None:
                logger.warning(
                    "ILP: product %d scheduled but no magic root selected — skipping.",
                    p.pid,
                )
                continue

            # Routing cells used by this product
            used_cells: Set = {
                v for v in routing_cells if solver.Value(x[p.pid][v]) == 1
            }

            # Undirected edges (canonicalised)
            used_edges: Set = set()
            for arc in arcs:
                if solver.Value(a[p.pid][arc]) == 1:
                    used_edges.add(_canon_edge(*arc))

            used_nodes = used_cells | {magic_root} | set(p.data_terminals)

            scheduled.append(
                RoutedProduct(
                    pid=p.pid,
                    magic_root=magic_root,
                    used_nodes=used_nodes,
                    used_edges=used_edges,
                )
            )

        return CycleRoutingResult(
            scheduled=scheduled,
            objective_value=obj_value,
            status=status_name,
            solver_wall_time_ms=wall_time_ms,
        )


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------


def validate_cycle_solution(
    result: CycleRoutingResult,
    products: List[ProductRequest],
    graph: Dict,
    ready_magic_nodes: List[str],
) -> None:
    """Validate a cycle solution against routing-graph and product constraints.

    Raises ``AssertionError`` with a descriptive message on the first
    violation found.  Intended for debug mode and unit tests.

    Checks:
      1. Every scheduled product has exactly one magic root.
      2. The magic root was ready at cycle start.
      3. No routing cell is used by two scheduled products.
      4. No two scheduled products share logical qubits.
      5. Every used edge exists in the routing graph.
      6. Every data terminal appears in ``used_nodes``.
      7. Each magic root is consumed at most once.
    """
    pid_to_product: Dict[int, ProductRequest] = {p.pid: p for p in products}
    used_routing_cells: Set = set()
    used_magic_roots: Set = set()

    graph_edge_set: Set = set()
    for u, neighbors in graph.items():
        for v, _w in neighbors:
            graph_edge_set.add(_canon_edge(u, v))

    for rp in result.scheduled:
        p = pid_to_product.get(rp.pid)
        assert p is not None, f"Routed pid={rp.pid} not in products list"

        assert rp.magic_root in ready_magic_nodes, (
            f"Product {rp.pid}: magic root '{rp.magic_root}' not in ready_magic_nodes"
        )
        assert rp.magic_root not in used_magic_roots, (
            f"Product {rp.pid}: magic root '{rp.magic_root}' already consumed by earlier product"
        )
        used_magic_roots.add(rp.magic_root)

        my_cells: Set = {n for n in rp.used_nodes if isinstance(n, tuple)}
        overlap = my_cells & used_routing_cells
        assert not overlap, (
            f"Product {rp.pid} shares routing cell(s) {overlap} with an earlier product"
        )
        used_routing_cells.update(my_cells)

        for e in rp.used_edges:
            assert e in graph_edge_set, (
                f"Product {rp.pid}: edge {e} is not present in the routing graph"
            )

        for t in p.data_terminals:
            assert t in rp.used_nodes, (
                f"Product {rp.pid}: data terminal '{t}' missing from used_nodes"
            )

    # Cross-product logical-qubit conflict check
    sched = result.scheduled
    for i in range(len(sched)):
        for j in range(i + 1, len(sched)):
            p1 = pid_to_product[sched[i].pid]
            p2 = pid_to_product[sched[j].pid]
            conflict = p1.logical_qubits & p2.logical_qubits
            assert not conflict, (
                f"Products {p1.pid} and {p2.pid} share logical qubits {conflict} "
                f"but are both scheduled"
            )
