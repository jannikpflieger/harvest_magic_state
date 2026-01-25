# preset_layout_7x9_magic_ring.py
"""
Preset: 7x9 grid with
  - magic state patches on the boundary (skipping corners), ports facing inward
  - 4 vertical 2-tile / 2-qubit "paired patches" in the interior (alternating X/Z ports around perimeter)
    labeled as (0,1), (2,3), (4,5), (6,7)

This uses the layout engine + constructors from `lattice_double_patches.py`.

Typical usage:
--------------
from preset_layout_7x9_magic_ring import build_7x9_magic_ring_layout

eng = build_7x9_magic_ring_layout()
eng.visualize_layout("7x9 magic-ring + 4 paired 2Q patches")
"""

from __future__ import annotations
from typing import Tuple

from lattice_double_patches import LayoutEngine, magic_patch_1cell, paired_patches_2q_alternating, prune_degree0_nodes

Coord = Tuple[int, int]


def build_7x9_magic_ring_layout(
    *,
    start_with: str = "X",
    swap_xz: bool = False,
) -> LayoutEngine:
    """
    Build the 7x9 layout.

    Args:
        start_with: 'X' or 'Z' for the paired 2-qubit perimeter alternation
        swap_xz: if True, swap X<->Z after alternation for the paired patches

    Returns:
        LayoutEngine populated with patches.
    """
    W, H = 7, 9
    eng = LayoutEngine(W, H)

    # --- Magic patches on boundary, skipping corners ---
    # Top edge (y=0): face inward => port on 'S'
    for x in range(1, W - 1):
        eng.add_patch(magic_patch_1cell(f"mT{x}", (x, 0), side="S"))

    # Bottom edge (y=H-1): face inward => port on 'N'
    for x in range(1, W - 1):
        eng.add_patch(magic_patch_1cell(f"mB{x}", (x, H - 1), side="N"))

    # Left edge (x=0): face inward => port on 'E'
    for y in range(1, H - 1):
        eng.add_patch(magic_patch_1cell(f"mL{y}", (0, y), side="E"))

    # Right edge (x=W-1): face inward => port on 'W'
    for y in range(1, H - 1):
        eng.add_patch(magic_patch_1cell(f"mR{y}", (W - 1, y), side="W"))

    # --- 4 paired 2-qubit patches (vertical pairs) in a 2x2 arrangement ---
    placements = [
        ((2, 2), 0, 1),  # top-left
        ((4, 2), 2, 3),  # top-right
        ((2, 5), 4, 5),  # bottom-left
        ((4, 5), 6, 7),  # bottom-right
    ]

    for (ox, oy), a, b in placements:
        patches = paired_patches_2q_alternating(
            name_a=f"q{a}",
            name_b=f"q{b}",
            origin=(ox, oy),
            orientation="V",
            start_with=start_with,
            swap_xz=swap_xz,
        )
        for p in patches:
            eng.add_patch(p)

    return eng


if __name__ == "__main__":
    eng = build_7x9_magic_ring_layout()
    eng.visualize_layout("7x9 magic-ring + 4 paired 2Q patches")
    graph, ports_by_patch, pos, patch_used_by_port = eng.build_routing_graph()
    eng.visualize_graph(graph, pos, "Routing graph overlay")

    graph, pos, patch_used_by_port = prune_degree0_nodes(graph, pos, patch_used_by_port)
    eng.visualize_graph(graph, pos, "Routing graph overlay (pruned)")