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

from lattice_test.lattice_double_patches import LayoutEngine, magic_patch_1cell, paired_patches_2q_alternating, data_patch_1cell

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


def nxm_ring_layout_single_qubits(n: int, m: int, *, start_with: str = "X", swap_xz: bool = False) -> LayoutEngine:
    """
    Build an n x m grid layout with single qubits and a ring of magic states around it.

    Layout structure:
    - Interior: n x m grid of logical data qubits (single-cell patches)
    - Spacing: 1 patch between adjacent data qubits (in both X and Y)
    - Magic ring: magic state patches on the boundary, 1 patch spacing from the data qubits
    - Magic ports face inward (toward the center)

    Grid dimensions:
    - data qubits occupy: positions (2i, 2j) for i in [0..n-1], j in [0..m-1]
    - data grid footprint: n*(2) x m*(2) = (2n-1) x (2m-1) cells
    - with 1-patch spacing between qubits and magic states: (2n+1) x (2m+1) cells minimum
    - actual layout: (2n+3) x (2m+3) cells

    Minimum layout: n=1, m=1 => layout is 5x5 (data at (2,2), magic at boundary)
    """
    # Calculate grid dimensions
    # n x m data qubits with 1-patch spacing between them: (2n-1) x (2m-1) cells for data
    # + 1 patch on each side for spacing before magic: (2n+1) x (2m+1) cells
    # + 1 patch on each side for magic patches: (2n+3) x (2m+3) cells
    W = 2 * n + 3
    H = 2 * m + 3

    eng = LayoutEngine(W, H)

    # --- Place logical data qubits in n x m grid ---
    # Position (i, j) -> cell (2*i + 2, 2*j + 2)
    # This ensures 1-patch spacing between adjacent qubits
    for i in range(n):
        for j in range(m):
            x = 2 * i + 2
            y = 2 * j + 2
            qubit_name = f"q_{i}_{j}"
            eng.add_patch(data_patch_1cell(qubit_name, (x, y), swap_xz=swap_xz))

    # --- Place magic state patches around the perimeter ---
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

    return eng


def nxm_ring_layout_single_qubits_large_spacing(n: int, m: int, *, start_with: str = "X", swap_xz: bool = False) -> LayoutEngine:
    """
    Build an n x m grid layout with single qubits and a ring of magic states around it.
    
    This variant uses larger spacing:
    - 2-patch spacing between adjacent data qubits (in both X and Y)
    - 1-patch spacing between data qubits and magic state patches
    - Magic ring: magic state patches on the boundary, facing inward
    
    Layout structure:
    - Interior: n x m grid of logical data qubits (single-cell patches)
    - Spacing: 2 patches between adjacent data qubits (in both X and Y)
    - Magic ring: magic state patches on the boundary, 1 patch spacing from the data qubits
    - Magic ports face inward (toward the center)

    Grid dimensions:
    - data qubits occupy: positions (3i + 2, 3j + 2) for i in [0..n-1], j in [0..m-1]
    - data grid footprint: spacing of 3 between qubits
    - with 1-patch spacing between qubits and magic states
    - actual layout: (3n + 2) x (3m + 2) cells
    
    Example: n=5, m=5 => layout is 17x17
    """
    # Calculate grid dimensions
    # n x m data qubits with 2-patch spacing between them: (3n-1) x (3m-1) cells for data
    # + 1 patch on each side for spacing before magic: (3n+1) x (3m+1) cells
    # + 1 patch on each side for magic patches: (3n+2) x (3m+2) cells
    W = 3 * n + 2
    H = 3 * m + 2

    eng = LayoutEngine(W, H)

    # --- Place logical data qubits in n x m grid ---
    # Position (i, j) -> cell (3*i + 2, 3*j + 2)
    # This ensures 2-patch spacing between adjacent qubits
    for i in range(n):
        for j in range(m):
            x = 3 * i + 2
            y = 3 * j + 2
            qubit_name = f"q_{i}_{j}"
            eng.add_patch(data_patch_1cell(qubit_name, (x, y), swap_xz=swap_xz))

    # --- Place magic state patches around the perimeter ---
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

    return eng


if __name__ == "__main__":
    # Test with the existing 7x9 layout
    #eng = build_7x9_magic_ring_layout()
    #eng.visualize_layout("7x9 magic-ring + 4 paired 2Q patches")
    #graph, ports_by_patch, pos, patch_used_by_port = eng.build_routing_graph()
    #eng.visualize_graph(graph, pos, "Routing graph overlay")

    
    #eng_2x2 = nxm_ring_layout_single_qubits(10, 10)
    #eng_2x2.visualize_layout("2x2 single-qubit layout (7x7 grid)")

    eng_5x5_large = nxm_ring_layout_single_qubits_large_spacing(5, 5)
    eng_5x5_large.visualize_layout("5x5 single-qubit layout with large spacing (17x17 grid)")
    graph, ports_by_patch, pos, patch_used_by_port = eng_5x5_large.build_routing_graph()
    eng_5x5_large.visualize_graph(graph, pos, "Routing graph overlay for 5x5 large spacing")