"""
Integration tests for the ILP Steiner packing scheduler.

These tests construct a real DAGProcessor with a small layout, run the ILP
scheduler over a small random circuit, and verify the fundamental invariants:

  * Every returned result has ``success=True``.
  * No routing cell is shared by two results in the same time step.
  * The ILP schedules at least as many nodes as the greedy scheduler (or
    exactly the same count if the circuit is so small that both are optimal).

OR-Tools is required; the test module is skipped if it is absent.
"""

import pytest
from collections import defaultdict

# Skip entire module if OR-Tools is not installed.
pytest.importorskip("ortools.sat.python.cp_model")

from harvest.compilation.pauli_block_conversion import (
    create_random_circuit,
    convert_to_PCB,
    create_dag,
)
from harvest.routing.processor import DAGProcessor
from harvest.routing.ilp_steiner_packing import ILPConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_dag():
    """A tiny 4-qubit random circuit — fast to schedule."""
    circ = create_random_circuit(num_qubits=4, depth=4, seed=42)
    pcb = convert_to_PCB(circ)
    return create_dag(pcb)


@pytest.fixture(scope="module")
def medium_dag():
    """A slightly larger 6-qubit circuit for more thorough checks."""
    circ = create_random_circuit(num_qubits=6, depth=6, seed=7)
    pcb = convert_to_PCB(circ)
    return create_dag(pcb)


def _ilp_cfg():
    return ILPConfig(
        time_limit_ms=15_000,
        max_ready_products=20,
        max_magic_candidates=8,
        fallback_to_greedy=True,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_node_disjoint(results):
    """Assert no routing cell is used by two results in the same time step."""
    cells_by_step = defaultdict(set)
    for r in results:
        ts = r["time_step"]
        for n in r.get("steiner_nodes", set()):
            if isinstance(n, tuple):
                assert n not in cells_by_step[ts], (
                    f"Routing cell {n} used twice at time step {ts}"
                )
                cells_by_step[ts].add(n)


# ---------------------------------------------------------------------------
# Test 1: ILP scheduler produces at least one result
# ---------------------------------------------------------------------------


def test_ilp_schedules_some_nodes(small_dag):
    proc = DAGProcessor(layout_rows=4, layout_cols=4)
    proc.set_ilp_config(_ilp_cfg())
    results = proc.process_entire_dag(small_dag, mode="ilp_steiner_packing")
    assert len(results) > 0, "ILP scheduler produced no results"


# ---------------------------------------------------------------------------
# Test 2: All returned results have success=True
# ---------------------------------------------------------------------------


def test_ilp_all_results_successful(small_dag):
    proc = DAGProcessor(layout_rows=4, layout_cols=4)
    proc.set_ilp_config(_ilp_cfg())
    results = proc.process_entire_dag(small_dag, mode="ilp_steiner_packing")
    for r in results:
        assert r["success"] is True, (
            f"Result for {r['gate_name']} at step {r['time_step']} has success=False"
        )


# ---------------------------------------------------------------------------
# Test 3: Routing cells are node-disjoint within each time step
# ---------------------------------------------------------------------------


def test_ilp_node_disjoint_per_time_step(small_dag):
    proc = DAGProcessor(layout_rows=4, layout_cols=4)
    proc.set_ilp_config(_ilp_cfg())
    results = proc.process_entire_dag(small_dag, mode="ilp_steiner_packing")
    _assert_node_disjoint(results)


# ---------------------------------------------------------------------------
# Test 4: magic_terminal field populated for every result
# ---------------------------------------------------------------------------


def test_ilp_results_have_magic_terminal(small_dag):
    proc = DAGProcessor(layout_rows=4, layout_cols=4)
    proc.set_ilp_config(_ilp_cfg())
    results = proc.process_entire_dag(small_dag, mode="ilp_steiner_packing")
    for r in results:
        assert r.get("magic_terminal"), (
            f"Result for {r['gate_name']} missing magic_terminal"
        )


# ---------------------------------------------------------------------------
# Test 5: ILP completes all DAG nodes (same count as greedy) for small circuits
# ---------------------------------------------------------------------------


def test_ilp_completion_matches_greedy(small_dag):
    proc_greedy = DAGProcessor(layout_rows=4, layout_cols=4)
    greedy_results = proc_greedy.process_entire_dag(small_dag, mode="steiner_packing")

    proc_ilp = DAGProcessor(layout_rows=4, layout_cols=4)
    proc_ilp.set_ilp_config(_ilp_cfg())
    ilp_results = proc_ilp.process_entire_dag(small_dag, mode="ilp_steiner_packing")

    assert len(ilp_results) == len(greedy_results), (
        f"ILP scheduled {len(ilp_results)} nodes but greedy scheduled {len(greedy_results)}"
    )


# ---------------------------------------------------------------------------
# Test 6: Node-disjoint invariant holds for a larger circuit too
# ---------------------------------------------------------------------------


def test_ilp_node_disjoint_medium_circuit(medium_dag):
    proc = DAGProcessor(layout_rows=5, layout_cols=5)
    proc.set_ilp_config(_ilp_cfg())
    results = proc.process_entire_dag(medium_dag, mode="ilp_steiner_packing")
    _assert_node_disjoint(results)


# ---------------------------------------------------------------------------
# Test 7: ILPConfig is respected — check _scheduling_metadata is populated
# ---------------------------------------------------------------------------


def test_ilp_scheduling_metadata(small_dag):
    proc = DAGProcessor(layout_rows=4, layout_cols=4)
    proc.set_ilp_config(_ilp_cfg())
    proc.process_entire_dag(small_dag, mode="ilp_steiner_packing")

    meta = getattr(proc, "_scheduling_metadata", None)
    assert meta is not None, "_scheduling_metadata not set after ILP run"
    assert "total_elapsed_steps" in meta
    assert "num_nodes_completed" in meta
    assert meta["num_nodes_completed"] > 0


# ---------------------------------------------------------------------------
# Test 8: set_ilp_config / ILP mode accessible from process_entire_dag
# ---------------------------------------------------------------------------


def test_process_entire_dag_ilp_mode(small_dag):
    """Verify the mode string 'ilp_steiner_packing' is wired into process_entire_dag."""
    proc = DAGProcessor(layout_rows=4, layout_cols=4)
    # Default ILPConfig should be used when no config is set explicitly.
    results = proc.process_entire_dag(small_dag, mode="ilp_steiner_packing")
    assert isinstance(results, list)
