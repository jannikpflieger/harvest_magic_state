"""
Top-level orchestrator for circuit-aware static layout synthesis.

Usage::

    from harvest.synthesis import StaticLayoutSynthesizer

    synth = StaticLayoutSynthesizer()
    engine, report = synth.synthesize(dag)
    # engine is a LayoutEngine — pass it straight to DAGProcessor
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from qiskit.dagcircuit import DAGCircuit

from harvest.layout.engine import LayoutEngine
from .circuit_summary import CircuitSummary, extract_circuit_summary
from .templates import LayoutTemplate, select_template
from .placement import (
    PlacementConfig,
    PlacementResult,
    circuit_aware_placement,
    baseline_placement,
)
from .emitter import emit_layout


@dataclass
class SynthesisReport:
    """Diagnostic output from a synthesis run."""

    template_name: str
    grid_width: int
    grid_height: int
    num_data_sites: int
    num_magic_sites: int
    assignment: Dict[int, int]
    placement_cost: float
    cost_breakdown: Dict[str, float] = field(default_factory=dict)
    improvement_history: list = field(default_factory=list)
    mode: str = ""  # "circuit_aware" or baseline mode name


class StaticLayoutSynthesizer:
    """
    Produces a ``LayoutEngine`` from a ``DAGCircuit``.

    Parameters can be overridden per-call or at construction time.
    """

    def __init__(
        self,
        placement_config: Optional[PlacementConfig] = None,
        num_lanes: Optional[int] = None,
    ):
        self.placement_config = placement_config or PlacementConfig()
        self.num_lanes = num_lanes

    # ------------------------------------------------------------------
    # Circuit-aware path
    # ------------------------------------------------------------------

    def synthesize(
        self, dag: DAGCircuit
    ) -> Tuple[LayoutEngine, SynthesisReport]:
        """
        Full circuit-aware synthesis: extract summary → choose template →
        optimised placement → emit LayoutEngine.
        """
        summary = extract_circuit_summary(dag)
        template = select_template(
            n_qubits=summary.num_qubits,
            max_parallelism=summary.parallelism_profile.get("max_pauli_per_layer", 0),
            num_lanes=self.num_lanes,
        )
        placement = circuit_aware_placement(summary, template, self.placement_config)
        engine = emit_layout(template, placement)

        report = SynthesisReport(
            template_name=template.name,
            grid_width=template.grid_width,
            grid_height=template.grid_height,
            num_data_sites=len(template.data_sites),
            num_magic_sites=len(template.magic_sites),
            assignment=dict(placement.assignment),
            placement_cost=placement.cost,
            cost_breakdown=dict(placement.cost_breakdown),
            improvement_history=list(placement.improvement_history),
            mode="circuit_aware",
        )
        return engine, report

    # ------------------------------------------------------------------
    # Baseline (non-circuit-aware) path
    # ------------------------------------------------------------------

    def synthesize_baseline(
        self,
        dag: DAGCircuit,
        mode: str = "row_major",
        seed: Optional[int] = None,
    ) -> Tuple[LayoutEngine, SynthesisReport]:
        """
        Baseline synthesis: same template, but qubits are placed without
        using circuit interaction data.
        """
        summary = extract_circuit_summary(dag)
        template = select_template(
            n_qubits=summary.num_qubits,
            max_parallelism=summary.parallelism_profile.get("max_pauli_per_layer", 0),
            num_lanes=self.num_lanes,
        )
        placement = baseline_placement(template, summary.num_qubits, mode=mode, seed=seed)
        engine = emit_layout(template, placement)

        report = SynthesisReport(
            template_name=template.name,
            grid_width=template.grid_width,
            grid_height=template.grid_height,
            num_data_sites=len(template.data_sites),
            num_magic_sites=len(template.magic_sites),
            assignment=dict(placement.assignment),
            placement_cost=placement.cost,
            cost_breakdown=dict(placement.cost_breakdown),
            improvement_history=list(placement.improvement_history),
            mode=mode,
        )
        return engine, report
