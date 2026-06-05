"""
Integration tests for the HARVEST negotiated-congestion scheduler.

HARVEST is a Pathfinder-derived algorithm with two runtime-focused changes:
  1. Rerouting only nets that participate in congestion (selective rip-up).
  2. Early stopping when the conflict score does not improve for
     ``stagnation_limit`` consecutive iterations.

These tests verify:
  * Existing ``"steiner_pathfinder"`` still works and is unchanged.
  * ``"harvest"`` mode runs without errors.
  * HARVEST produces successfully routed nodes on a small DAG/layout where
    Pathfinder also succeeds.
  * HARVEST never commits routes with over-capacity routing-node conflicts.
  * HARVEST returns the same core result format expected by evaluation scripts.
  * HARVEST result dicts contain the ``"harvest_stats"`` metadata key with the
    expected sub-fields.
"""

import pytest
from collections import defaultdict

from harvest.compilation.pauli_block_conversion import (
    create_random_circuit,
    convert_to_PCB,
    create_dag,
)
from harvest.routing.processor import DAGProcessor

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_REQUIRED_RESULT_KEYS = {
    "node",
    "gate_name",
    "qubits",
    "time_step",
    "success",
    "magic_wait_cycles",
    "magic_terminal",
    "qubit_terminals",
    "all_terminals",
    "steiner_nodes",
    "steiner_edges",
}

_REQUIRED_HARVEST_STATS_KEYS = {
    "iterations",
    "initial_routes",
    "reroutes",
    "conflicted_reroutes",
    "dropped",
    "final_conflict_score",
    "stopped_reason",
}


@pytest.fixture(scope="module")
def small_dag():
    """Tiny 4-qubit random circuit — fast to schedule."""
    circ = create_random_circuit(num_qubits=4, depth=4, seed=42)
    pcb = convert_to_PCB(circ)
    return create_dag(pcb)


@pytest.fixture(scope="module")
def medium_dag():
    """Slightly larger 6-qubit circuit for more thorough checks."""
    circ = create_random_circuit(num_qubits=6, depth=6, seed=7)
    pcb = convert_to_PCB(circ)
    return create_dag(pcb)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_node_disjoint(results):
    """Assert no routing cell is shared by two successful results in the same time step."""
    cells_by_step = defaultdict(set)
    for r in results:
        if not r.get("success"):
            continue
        ts = r["time_step"]
        for n in r.get("steiner_nodes", set()):
            if isinstance(n, tuple):
                assert n not in cells_by_step[ts], (
                    f"Routing cell {n} used twice at time step {ts} — "
                    "HARVEST committed an over-capacity conflict."
                )
                cells_by_step[ts].add(n)


# ---------------------------------------------------------------------------
# Test 1: Existing steiner_pathfinder mode still works (regression guard)
# ---------------------------------------------------------------------------


def test_steiner_pathfinder_unchanged(small_dag):
    """Verify steiner_pathfinder is not broken by the HARVEST addition."""
    proc = DAGProcessor(layout_rows=4, layout_cols=4)
    results = proc.process_entire_dag(small_dag, mode="steiner_pathfinder")
    assert len(results) > 0, "steiner_pathfinder produced no results"
    for r in results:
        assert r["success"] is True, (
            f"steiner_pathfinder result for {r['gate_name']} has success=False"
        )


# ---------------------------------------------------------------------------
# Test 2: harvest mode runs without errors
# ---------------------------------------------------------------------------


def test_harvest_runs_without_error(small_dag):
    proc = DAGProcessor(layout_rows=4, layout_cols=4)
    # Must not raise.
    results = proc.process_entire_dag(small_dag, mode="harvest")
    assert isinstance(results, list)


# ---------------------------------------------------------------------------
# Test 3: HARVEST produces at least some successfully routed nodes
# ---------------------------------------------------------------------------


def test_harvest_schedules_some_nodes(small_dag):
    proc = DAGProcessor(layout_rows=4, layout_cols=4)
    results = proc.process_entire_dag(small_dag, mode="harvest")
    assert len(results) > 0, "HARVEST scheduled no nodes"


# ---------------------------------------------------------------------------
# Test 4: All HARVEST results have success=True (only successes are returned)
# ---------------------------------------------------------------------------


def test_harvest_all_results_successful(small_dag):
    proc = DAGProcessor(layout_rows=4, layout_cols=4)
    results = proc.process_entire_dag(small_dag, mode="harvest")
    for r in results:
        assert r["success"] is True, (
            f"HARVEST returned success=False for {r['gate_name']} at "
            f"time_step={r['time_step']}"
        )


# ---------------------------------------------------------------------------
# Test 5: No over-capacity routing-node conflicts in committed routes
# ---------------------------------------------------------------------------


def test_harvest_node_disjoint_per_time_step(small_dag):
    proc = DAGProcessor(layout_rows=4, layout_cols=4)
    results = proc.process_entire_dag(small_dag, mode="harvest")
    _assert_node_disjoint(results)


# ---------------------------------------------------------------------------
# Test 6: HARVEST result dicts contain all required core fields
# ---------------------------------------------------------------------------


def test_harvest_result_format(small_dag):
    proc = DAGProcessor(layout_rows=4, layout_cols=4)
    results = proc.process_entire_dag(small_dag, mode="harvest")
    assert results, "No results to inspect"
    for r in results:
        missing = _REQUIRED_RESULT_KEYS - r.keys()
        assert not missing, (
            f"Result for {r.get('gate_name', '?')} is missing keys: {missing}"
        )


# ---------------------------------------------------------------------------
# Test 7: Every HARVEST result has the harvest_stats metadata key
# ---------------------------------------------------------------------------


def test_harvest_stats_present(small_dag):
    proc = DAGProcessor(layout_rows=4, layout_cols=4)
    results = proc.process_entire_dag(small_dag, mode="harvest")
    for r in results:
        assert "harvest_stats" in r, (
            f"Result for {r['gate_name']} is missing 'harvest_stats'"
        )
        stats = r["harvest_stats"]
        missing = _REQUIRED_HARVEST_STATS_KEYS - stats.keys()
        assert not missing, (
            f"harvest_stats for {r['gate_name']} missing keys: {missing}"
        )


# ---------------------------------------------------------------------------
# Test 8: harvest_stats final_conflict_score is always 0 for committed results
# ---------------------------------------------------------------------------


def test_harvest_stats_zero_conflict_score(small_dag):
    proc = DAGProcessor(layout_rows=4, layout_cols=4)
    results = proc.process_entire_dag(small_dag, mode="harvest")
    for r in results:
        score = r["harvest_stats"]["final_conflict_score"]
        assert score == 0, (
            f"HARVEST committed routes with non-zero conflict score {score} "
            f"for {r['gate_name']} at time_step={r['time_step']}"
        )


# ---------------------------------------------------------------------------
# Test 9: HARVEST schedules the same number of nodes as Pathfinder
#         on a small circuit (both should be optimal / near-optimal)
# ---------------------------------------------------------------------------


def test_harvest_completion_matches_pathfinder(small_dag):
    proc_pf = DAGProcessor(layout_rows=4, layout_cols=4)
    pf_results = proc_pf.process_entire_dag(small_dag, mode="steiner_pathfinder")

    proc_h = DAGProcessor(layout_rows=4, layout_cols=4)
    h_results = proc_h.process_entire_dag(small_dag, mode="harvest")

    assert len(h_results) == len(pf_results), (
        f"HARVEST scheduled {len(h_results)} nodes but Pathfinder scheduled "
        f"{len(pf_results)} — they should agree on a small circuit."
    )


# ---------------------------------------------------------------------------
# Test 10: _scheduling_metadata is populated after a harvest run
# ---------------------------------------------------------------------------


def test_harvest_scheduling_metadata(small_dag):
    proc = DAGProcessor(layout_rows=4, layout_cols=4)
    proc.process_entire_dag(small_dag, mode="harvest")
    meta = getattr(proc, "_scheduling_metadata", None)
    assert meta is not None, "_scheduling_metadata not set after HARVEST run"
    for key in ("total_elapsed_steps", "num_nodes_completed", "num_nodes_total", "completed"):
        assert key in meta, f"_scheduling_metadata missing key '{key}'"
    assert meta["num_nodes_completed"] > 0


# ---------------------------------------------------------------------------
# Test 11: Node-disjoint invariant on medium circuit
# ---------------------------------------------------------------------------


def test_harvest_node_disjoint_medium(medium_dag):
    proc = DAGProcessor(layout_rows=5, layout_cols=5)
    results = proc.process_entire_dag(medium_dag, mode="harvest")
    _assert_node_disjoint(results)


# ---------------------------------------------------------------------------
# Test 12: algorithm field is set to "harvest" in every result
# ---------------------------------------------------------------------------


def test_harvest_algorithm_field(small_dag):
    proc = DAGProcessor(layout_rows=4, layout_cols=4)
    results = proc.process_entire_dag(small_dag, mode="harvest")
    for r in results:
        assert r.get("algorithm") == "harvest", (
            f"Expected algorithm='harvest', got {r.get('algorithm')!r}"
        )
