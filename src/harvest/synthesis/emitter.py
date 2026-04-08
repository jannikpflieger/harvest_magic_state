"""
Emit a LayoutEngine from a (LayoutTemplate, PlacementResult) pair.

The output is the *same type* as the existing preset layouts, so it plugs
directly into ``DAGProcessor(layout_engine=engine)`` with zero adapters.
"""

from __future__ import annotations

from typing import Tuple

from harvest.layout.engine import LayoutEngine, data_patch_1cell, magic_patch_1cell
from .templates import LayoutTemplate
from .placement import PlacementResult

Coord = Tuple[int, int]


def emit_layout(template: LayoutTemplate, placement: PlacementResult) -> LayoutEngine:
    """
    Build a fully populated ``LayoutEngine`` by placing data-qubit and
    magic-state patches according to *template* geometry and *placement*
    assignment.

    Data qubits are named ``q_{logical_index}`` (same convention used by
    ``nxm_ring_layout_single_qubits`` and ``DAGProcessor._get_qubit_terminals``).

    Magic patches follow the naming/side conventions from the ring presets.
    """
    eng = LayoutEngine(template.grid_width, template.grid_height)

    # --- Data patches ---
    for logical_qubit, site_index in placement.assignment.items():
        coord = template.data_sites[site_index]
        name = f"q_{logical_qubit}"
        eng.add_patch(data_patch_1cell(name, coord))

    # --- Magic patches (perimeter ring, same convention as presets.py) ---
    W, H = template.grid_width, template.grid_height

    for x, y in template.magic_sites:
        if y == 0:
            side = "S"
            name = f"mT{x}"
        elif y == H - 1:
            side = "N"
            name = f"mB{x}"
        elif x == 0:
            side = "E"
            name = f"mL{y}"
        elif x == W - 1:
            side = "W"
            name = f"mR{y}"
        else:
            # Interior magic site (shouldn't happen with current templates)
            side = "S"
            name = f"mI{x}_{y}"
        eng.add_patch(magic_patch_1cell(name, (x, y), side=side))

    return eng
