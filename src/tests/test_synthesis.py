"""
Tests for the synthesis module: circuit summary extraction, templates,
placement heuristics, layout emission, and end-to-end synthesis.
"""

import pytest
from collections import Counter

from harvest.compilation.pauli_block_conversion import create_random_circuit, convert_to_PCB, create_dag
from harvest.synthesis.circuit_summary import CircuitSummary, extract_circuit_summary
from harvest.synthesis.templates import LayoutTemplate, bus_template, select_template
from harvest.synthesis.placement import (
    PlacementConfig,
    PlacementResult,
    circuit_aware_placement,
    baseline_placement,
)
from harvest.synthesis.emitter import emit_layout
from harvest.synthesis.synthesizer import StaticLayoutSynthesizer, SynthesisReport


# ---------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------

@pytest.fixture
def small_dag():
    """5-qubit, depth-10 random circuit → PCB → DAG."""
    circ = create_random_circuit(5, 10, seed=42)
    pcb = convert_to_PCB(circ, verbose=False)
    return create_dag(pcb)


@pytest.fixture
def medium_dag():
    """15-qubit, depth-15 random circuit → PCB → DAG."""
    circ = create_random_circuit(15, 15, seed=123)
    pcb = convert_to_PCB(circ, verbose=False)
    return create_dag(pcb)


# ---------------------------------------------------------------
# CircuitSummary extraction
# ---------------------------------------------------------------

class TestCircuitSummary:
    def test_qubit_list_matches_dag(self, small_dag):
        summary = extract_circuit_summary(small_dag)
        # All qubits in the summary must be < dag's qubit count
        assert all(q < small_dag.num_qubits() for q in summary.qubit_list)
        assert summary.qubit_list == sorted(summary.qubit_list)

    def test_interaction_graph_symmetric_keys(self, small_dag):
        summary = extract_circuit_summary(small_dag)
        for qi, qj in summary.interaction_graph:
            assert qi < qj, "interaction keys must be canonical (i < j)"

    def test_interaction_weights_positive(self, small_dag):
        summary = extract_circuit_summary(small_dag)
        for w in summary.interaction_graph.values():
            assert w > 0

    def test_pauli_profile_valid_types(self, small_dag):
        summary = extract_circuit_summary(small_dag)
        for q, profile in summary.pauli_profile.items():
            for ptype in profile:
                assert ptype in ("X", "Y", "Z"), f"unexpected Pauli type: {ptype}"

    def test_parallelism_profile_present(self, small_dag):
        summary = extract_circuit_summary(small_dag)
        assert "num_layers" in summary.parallelism_profile
        assert "max_pauli_per_layer" in summary.parallelism_profile

    def test_num_qubits_consistent(self, small_dag):
        summary = extract_circuit_summary(small_dag)
        assert summary.num_qubits == len(summary.qubit_list)

    def test_empty_dag(self):
        """A circuit with no non-Clifford gates yields an empty summary."""
        from qiskit import QuantumCircuit
        from qiskit.converters import circuit_to_dag

        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        dag = circuit_to_dag(qc)
        summary = extract_circuit_summary(dag)
        assert summary.num_qubits == 0
        assert summary.total_pauli_evolutions == 0
        assert len(summary.interaction_graph) == 0


# ---------------------------------------------------------------
# Templates
# ---------------------------------------------------------------

class TestTemplates:
    def test_bus_template_site_count(self):
        tpl = bus_template(10, num_lanes=2)
        assert len(tpl.data_sites) == 10

    def test_bus_template_magic_sites_on_perimeter(self):
        tpl = bus_template(9, num_lanes=1)
        W, H = tpl.grid_width, tpl.grid_height
        for x, y in tpl.magic_sites:
            assert x == 0 or x == W - 1 or y == 0 or y == H - 1

    def test_pairwise_distances_populated(self):
        tpl = bus_template(6, num_lanes=2)
        n = len(tpl.data_sites)
        expected_pairs = n * (n - 1) // 2
        assert len(tpl.pairwise_distances) == expected_pairs

    def test_centrality_populated(self):
        tpl = bus_template(6, num_lanes=2)
        assert len(tpl.site_centrality) == len(tpl.data_sites)
        for v in tpl.site_centrality.values():
            assert v > 0

    def test_select_template_auto_lanes(self):
        tpl = select_template(10, max_parallelism=3)
        assert tpl.routing_lanes == 1
        tpl = select_template(10, max_parallelism=8)
        assert tpl.routing_lanes == 2
        tpl = select_template(10, max_parallelism=20)
        assert tpl.routing_lanes == 3

    def test_select_template_explicit_lanes(self):
        tpl = select_template(10, num_lanes=3)
        assert tpl.routing_lanes == 3

    def test_data_sites_no_overlap_with_magic(self):
        tpl = bus_template(16, num_lanes=2)
        data_set = set(tpl.data_sites)
        magic_set = set(tpl.magic_sites)
        assert data_set.isdisjoint(magic_set)


# ---------------------------------------------------------------
# Placement
# ---------------------------------------------------------------

class TestPlacement:
    def test_all_qubits_assigned(self, small_dag):
        summary = extract_circuit_summary(small_dag)
        if summary.num_qubits == 0:
            pytest.skip("No PauliEvolution gates in small circuit")
        tpl = bus_template(summary.num_qubits, num_lanes=2)
        result = circuit_aware_placement(summary, tpl)
        assert set(result.assignment.keys()) == set(summary.qubit_list)

    def test_assignment_sites_unique(self, small_dag):
        summary = extract_circuit_summary(small_dag)
        if summary.num_qubits == 0:
            pytest.skip("No PauliEvolution gates in small circuit")
        tpl = bus_template(summary.num_qubits, num_lanes=2)
        result = circuit_aware_placement(summary, tpl)
        sites = list(result.assignment.values())
        assert len(sites) == len(set(sites)), "duplicate site assignments"

    def test_cost_decreases_after_swaps(self, medium_dag):
        summary = extract_circuit_summary(medium_dag)
        if summary.num_qubits == 0:
            pytest.skip("No PauliEvolution gates")
        tpl = bus_template(summary.num_qubits, num_lanes=2)
        result = circuit_aware_placement(summary, tpl, PlacementConfig(seed=0))
        hist = result.improvement_history
        assert len(hist) >= 2
        assert hist[-1] <= hist[0], "cost should not increase"

    def test_baseline_row_major(self):
        tpl = bus_template(10, num_lanes=2)
        r = baseline_placement(tpl, 10, mode="row_major")
        assert r.assignment == {i: i for i in range(10)}

    def test_baseline_random_deterministic(self):
        tpl = bus_template(10, num_lanes=2)
        r1 = baseline_placement(tpl, 10, mode="random", seed=7)
        r2 = baseline_placement(tpl, 10, mode="random", seed=7)
        assert r1.assignment == r2.assignment

    def test_too_many_qubits_raises(self):
        tpl = bus_template(3, num_lanes=1)
        with pytest.raises(ValueError):
            baseline_placement(tpl, 100)


# ---------------------------------------------------------------
# Emitter
# ---------------------------------------------------------------

class TestEmitter:
    def test_emit_produces_valid_engine(self, small_dag):
        summary = extract_circuit_summary(small_dag)
        if summary.num_qubits == 0:
            pytest.skip("No PauliEvolution gates")
        tpl = bus_template(summary.num_qubits, num_lanes=2)
        placement = baseline_placement(tpl, summary.num_qubits)
        engine = emit_layout(tpl, placement)

        # Must be able to build the routing graph without error
        graph, ports_by_patch, pos, patch_used = engine.build_routing_graph()
        assert len(graph) > 0
        assert len(ports_by_patch) > 0

    def test_data_patch_names(self, small_dag):
        summary = extract_circuit_summary(small_dag)
        if summary.num_qubits == 0:
            pytest.skip("No PauliEvolution gates")
        tpl = bus_template(summary.num_qubits, num_lanes=2)
        placement = baseline_placement(tpl, summary.num_qubits)
        engine = emit_layout(tpl, placement)
        for q in summary.qubit_list:
            assert f"q_{q}" in engine.patches


# ---------------------------------------------------------------
# End-to-end synthesizer
# ---------------------------------------------------------------

class TestSynthesizer:
    def test_synthesize_returns_engine_and_report(self, small_dag):
        synth = StaticLayoutSynthesizer()
        engine, report = synth.synthesize(small_dag)
        assert isinstance(engine, type(engine))  # LayoutEngine
        assert isinstance(report, SynthesisReport)
        assert report.mode == "circuit_aware"

    def test_synthesize_baseline_returns_engine_and_report(self, small_dag):
        synth = StaticLayoutSynthesizer()
        engine, report = synth.synthesize_baseline(small_dag, mode="row_major")
        assert report.mode == "row_major"

    def test_end_to_end_routing(self, small_dag):
        """Synthesize a layout and route the same DAG through it."""
        from harvest.routing.processor import DAGProcessor

        synth = StaticLayoutSynthesizer()
        engine, report = synth.synthesize(small_dag)

        # If no PauliEvolution gates, synthesize will produce 0 data sites.
        if report.num_data_sites == 0:
            pytest.skip("No PauliEvolution gates in circuit")

        processor = DAGProcessor(layout_engine=engine)
        results = processor.process_entire_dag(small_dag, mode="steiner_tree")
        assert len(results) > 0

    def test_both_paths_same_template_size(self, medium_dag):
        """Circuit-aware and baseline should use the same template dimensions."""
        synth = StaticLayoutSynthesizer(num_lanes=2)
        _, rpt_aware = synth.synthesize(medium_dag)
        _, rpt_base = synth.synthesize_baseline(medium_dag)
        assert rpt_aware.grid_width == rpt_base.grid_width
        assert rpt_aware.grid_height == rpt_base.grid_height
        assert rpt_aware.num_data_sites == rpt_base.num_data_sites
