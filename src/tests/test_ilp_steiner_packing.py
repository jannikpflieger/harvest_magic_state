"""
Unit tests for ILPSteinerForestRouter.

All tests use synthetic routing graphs (plain Python dicts) and require
OR-Tools to be installed.  If OR-Tools is absent, the whole module is skipped
via pytest.importorskip().

Graph convention (matches HARVEST layout):
    Routing cells  → (x, y) tuples
    Port nodes     → strings  (e.g. "M", "T", "T1", "T2")
Edge weights are irrelevant for the ILP (only topology matters).
"""

import pytest

# Skip entire module if OR-Tools is not installed.
cp_model = pytest.importorskip("ortools.sat.python.cp_model")

from harvest.routing.ilp_steiner_packing import (
    ILPConfig,
    ILPSteinerForestRouter,
    ProductRequest,
    validate_cycle_solution,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_router():
    return ILPSteinerForestRouter()


def quick_cfg(**kwargs):
    """ILPConfig with a short time limit suitable for unit tests."""
    defaults = dict(time_limit_ms=10_000, fallback_to_greedy=False)
    defaults.update(kwargs)
    return ILPConfig(**defaults)


# ---------------------------------------------------------------------------
# Test 1: Single product, single terminal — simplest possible routing
# ---------------------------------------------------------------------------
#
# Graph (line):  "M" -- (0,0) -- (1,0) -- "T"
#
def _line_graph():
    return {
        "M": [((0, 0), 0)],
        (0, 0): [("M", 0), ((1, 0), 1)],
        (1, 0): [((0, 0), 1), ("T", 0)],
        "T": [((1, 0), 0)],
    }


def test_single_product_single_terminal_scheduled():
    graph = _line_graph()
    products = [
        ProductRequest(pid=0, data_terminals=["T"], logical_qubits=frozenset([0]))
    ]
    router = make_router()
    result = router.solve_cycle(graph, products, ["M"], config=quick_cfg())

    assert len(result.scheduled) == 1
    rp = result.scheduled[0]
    assert rp.pid == 0
    assert rp.magic_root == "M"
    assert "T" in rp.used_nodes
    assert (0, 0) in rp.used_nodes
    assert (1, 0) in rp.used_nodes


def test_single_product_used_edges_in_graph():
    from harvest.routing.ilp_steiner_packing import _canon_edge

    graph = _line_graph()
    products = [
        ProductRequest(pid=0, data_terminals=["T"], logical_qubits=frozenset([0]))
    ]
    router = make_router()
    result = router.solve_cycle(graph, products, ["M"], config=quick_cfg())

    # All returned edges must exist in the graph.
    graph_edges = set()
    for u, nbrs in graph.items():
        for v, _ in nbrs:
            graph_edges.add(_canon_edge(u, v))

    rp = result.scheduled[0]
    for e in rp.used_edges:
        assert e in graph_edges, f"Edge {e} not in graph"


def test_single_product_validation():
    graph = _line_graph()
    products = [
        ProductRequest(pid=0, data_terminals=["T"], logical_qubits=frozenset([0]))
    ]
    router = make_router()
    result = router.solve_cycle(graph, products, ["M"], config=quick_cfg())
    # Should not raise.
    validate_cycle_solution(result, products, graph, ["M"])


# ---------------------------------------------------------------------------
# Test 2: Single product, two data terminals (Y-shaped tree)
# ---------------------------------------------------------------------------
#
# Graph:  "M" -- (0,0) -- (1,0) -- "T1"
#                              \
#                              (2,0) -- "T2"
#
def _y_graph():
    return {
        "M": [((0, 0), 0)],
        (0, 0): [("M", 0), ((1, 0), 1)],
        (1, 0): [((0, 0), 1), ((2, 0), 1), ("T1", 0)],
        (2, 0): [((1, 0), 1), ("T2", 0)],
        "T1": [((1, 0), 0)],
        "T2": [((2, 0), 0)],
    }


def test_y_graph_both_terminals_reached():
    graph = _y_graph()
    products = [
        ProductRequest(
            pid=0,
            data_terminals=["T1", "T2"],
            logical_qubits=frozenset([0, 1]),
        )
    ]
    router = make_router()
    result = router.solve_cycle(graph, products, ["M"], config=quick_cfg())

    assert len(result.scheduled) == 1
    rp = result.scheduled[0]
    assert "T1" in rp.used_nodes
    assert "T2" in rp.used_nodes
    validate_cycle_solution(result, products, graph, ["M"])


# ---------------------------------------------------------------------------
# Test 3: Two disjoint products — both should be scheduled
# ---------------------------------------------------------------------------
#
# Graph: two isolated corridors sharing no routing cells.
#   Corridor A:  "M1" -- (0,0) -- (1,0) -- "T1"
#   Corridor B:  "M2" -- (0,1) -- (1,1) -- "T2"
#
def _two_corridor_graph():
    return {
        "M1": [((0, 0), 0)],
        (0, 0): [("M1", 0), ((1, 0), 1)],
        (1, 0): [((0, 0), 1), ("T1", 0)],
        "T1": [((1, 0), 0)],
        "M2": [((0, 1), 0)],
        (0, 1): [("M2", 0), ((1, 1), 1)],
        (1, 1): [((0, 1), 1), ("T2", 0)],
        "T2": [((1, 1), 0)],
    }


def test_two_disjoint_products_both_scheduled():
    graph = _two_corridor_graph()
    products = [
        ProductRequest(pid=0, data_terminals=["T1"], logical_qubits=frozenset([0])),
        ProductRequest(pid=1, data_terminals=["T2"], logical_qubits=frozenset([1])),
    ]
    router = make_router()
    result = router.solve_cycle(graph, products, ["M1", "M2"], config=quick_cfg())

    assert len(result.scheduled) == 2
    pids = {rp.pid for rp in result.scheduled}
    assert pids == {0, 1}
    validate_cycle_solution(result, products, graph, ["M1", "M2"])


# ---------------------------------------------------------------------------
# Test 4: Logical-qubit conflict — at most one of the two products scheduled
# ---------------------------------------------------------------------------
def test_logical_qubit_conflict():
    graph = _two_corridor_graph()
    # Both products touch qubit 0 → cannot both be scheduled.
    products = [
        ProductRequest(pid=0, data_terminals=["T1"], logical_qubits=frozenset([0])),
        ProductRequest(pid=1, data_terminals=["T2"], logical_qubits=frozenset([0])),
    ]
    router = make_router()
    result = router.solve_cycle(graph, products, ["M1", "M2"], config=quick_cfg())

    assert len(result.scheduled) <= 1
    validate_cycle_solution(result, products, graph, ["M1", "M2"])


# ---------------------------------------------------------------------------
# Test 5: Routing bottleneck — two products require the same routing cell
# ---------------------------------------------------------------------------
#
# Graph: single shared corridor
#   "M1" -- (0,0) -- (1,0) -- "T1"
#   "M2" -- (0,0) -- (1,0) -- "T2"   ← same cells!
#
def _shared_corridor_graph():
    return {
        "M1": [((0, 0), 0)],
        "M2": [((0, 0), 0)],
        (0, 0): [("M1", 0), ("M2", 0), ((1, 0), 1)],
        (1, 0): [((0, 0), 1), ("T1", 0), ("T2", 0)],
        "T1": [((1, 0), 0)],
        "T2": [((1, 0), 0)],
    }


def test_routing_bottleneck_at_most_one_scheduled():
    graph = _shared_corridor_graph()
    products = [
        ProductRequest(pid=0, data_terminals=["T1"], logical_qubits=frozenset([0])),
        ProductRequest(pid=1, data_terminals=["T2"], logical_qubits=frozenset([1])),
    ]
    router = make_router()
    result = router.solve_cycle(graph, products, ["M1", "M2"], config=quick_cfg())

    # Node-disjoint constraint: cannot use (0,0) or (1,0) for both.
    assert len(result.scheduled) <= 1
    validate_cycle_solution(result, products, graph, ["M1", "M2"])


# ---------------------------------------------------------------------------
# Test 6: Route-length minimisation — shorter path preferred
# ---------------------------------------------------------------------------
#
# Graph: two routes from "M" to "T"
#   Short path:  "M" -- (0,0) -- "T"           (1 routing cell)
#   Long detour: "M" -- (0,0) -- (1,0) -- (2,0) -- (3,0) -- "T"  (4 cells)
#
def _two_path_graph():
    return {
        "M": [((0, 0), 0)],
        (0, 0): [("M", 0), ("T", 0), ((1, 0), 1)],
        (1, 0): [((0, 0), 1), ((2, 0), 1)],
        (2, 0): [((1, 0), 1), ((3, 0), 1)],
        (3, 0): [((2, 0), 1), ("T", 0)],
        "T": [((0, 0), 0), ((3, 0), 0)],
    }


def test_route_minimisation_prefers_short_path():
    graph = _two_path_graph()
    products = [
        ProductRequest(pid=0, data_terminals=["T"], logical_qubits=frozenset([0]))
    ]
    router = make_router()
    result = router.solve_cycle(
        graph, products, ["M"],
        config=quick_cfg(edge_cost=1, count_weight=100_000),
    )

    assert len(result.scheduled) == 1
    rp = result.scheduled[0]
    # Short path uses only (0,0); long path uses (0,0),(1,0),(2,0),(3,0).
    assert (1, 0) not in rp.used_nodes, (
        "ILP should prefer the shorter direct path via (0,0) only"
    )


# ---------------------------------------------------------------------------
# Test 7: No magic terminals — nothing can be scheduled
# ---------------------------------------------------------------------------
def test_no_magic_terminals_returns_empty():
    graph = _line_graph()
    products = [
        ProductRequest(pid=0, data_terminals=["T"], logical_qubits=frozenset([0]))
    ]
    router = make_router()
    result = router.solve_cycle(graph, products, [], config=quick_cfg())

    assert result.scheduled == []


# ---------------------------------------------------------------------------
# Test 8: max_ready_products limit is honoured by the caller
# ---------------------------------------------------------------------------
def test_solver_handles_max_ready_products_limit():
    """The ILP itself is allowed to schedule fewer products than max_ready_products;
    the caller (scheduler) is responsible for the cap.  Here we verify that
    passing more products than max_ready_products to solve_cycle still works
    (the ILP just becomes larger).  The scheduler test covers the capping logic."""
    graph = _two_corridor_graph()
    products = [
        ProductRequest(pid=0, data_terminals=["T1"], logical_qubits=frozenset([0])),
        ProductRequest(pid=1, data_terminals=["T2"], logical_qubits=frozenset([1])),
    ]
    # Providing a config with max_ready_products=1 does NOT limit the ILP directly;
    # that check lives in _process_time_step_with_ilp.  The ILP may schedule both.
    result = ILPSteinerForestRouter().solve_cycle(
        graph, products, ["M1", "M2"], config=quick_cfg()
    )
    assert len(result.scheduled) <= 2  # must not crash or violate disjointness
    validate_cycle_solution(result, products, graph, ["M1", "M2"])


# ---------------------------------------------------------------------------
# Test 9: Validate does not raise for a correct solution
# ---------------------------------------------------------------------------
def test_validate_passes_for_correct_solution():
    graph = _two_corridor_graph()
    products = [
        ProductRequest(pid=0, data_terminals=["T1"], logical_qubits=frozenset([0])),
        ProductRequest(pid=1, data_terminals=["T2"], logical_qubits=frozenset([1])),
    ]
    router = make_router()
    result = router.solve_cycle(graph, products, ["M1", "M2"], config=quick_cfg())
    # Must not raise.
    validate_cycle_solution(result, products, graph, ["M1", "M2"])


# ---------------------------------------------------------------------------
# Test 10: Empty products list → immediate OPTIMAL with no scheduled items
# ---------------------------------------------------------------------------
def test_empty_products_list():
    graph = _line_graph()
    router = make_router()
    result = router.solve_cycle(graph, [], ["M"], config=quick_cfg())

    assert result.scheduled == []
    assert result.status == "OPTIMAL"
