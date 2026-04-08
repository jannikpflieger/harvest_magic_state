"""
Circuit-aware static layout synthesis.

Translates DAG circuit structure into optimized lattice layouts
by analyzing qubit interaction patterns and placing high-interaction
qubits close together on the lattice.
"""

from .circuit_summary import CircuitSummary, extract_circuit_summary
from .templates import LayoutTemplate, bus_template, select_template
from .placement import (
    PlacementConfig,
    PlacementResult,
    circuit_aware_placement,
    baseline_placement,
)
from .emitter import emit_layout
from .synthesizer import StaticLayoutSynthesizer, SynthesisReport
