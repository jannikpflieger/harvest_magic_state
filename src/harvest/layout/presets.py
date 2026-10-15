from __future__ import annotations
from typing import Optional, Tuple

from harvest.layout.engine import LayoutEngine, magic_patch_1cell, data_patch_1cell

Coord = Tuple[int, int]


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
    # Qubits are numbered sequentially: q_0, q_1, q_2, ..., q_(n*m-1)
    qubit_idx = 0
    for i in range(n):
        for j in range(m):
            x = 2 * i + 2
            y = 2 * j + 2
            qubit_name = f"q_{qubit_idx}"
            eng.add_patch(data_patch_1cell(qubit_name, (x, y), swap_xz=swap_xz))
            qubit_idx += 1

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
    # Qubits are numbered sequentially: q_0, q_1, q_2, ..., q_(n*m-1)
    qubit_idx = 0
    for i in range(n):
        for j in range(m):
            x = 3 * i + 2
            y = 3 * j + 2
            qubit_name = f"q_{qubit_idx}"
            eng.add_patch(data_patch_1cell(qubit_name, (x, y), swap_xz=swap_xz))
            qubit_idx += 1

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


def blocks_of_four_qubit_patches(
    n: int, m: int, *, start_with: str = "X", swap_xz: bool = False,
    magic_side: Optional[str] = None, magic_ring_count: Optional[int] = None,
) -> LayoutEngine:
    """
    Build an n x m grid layout with blocks of 4-qubit patches and a ring of magic states around it.
    
    Layout structure:
    - Interior: n x m blocks of 4-qubit patches (2x2 arrangement of single-cell patches)
    - Spacing: 1 patch between adjacent blocks
    - Magic ring: magic state patches on the boundary, 1 patch spacing from the blocks
    - Magic ports face inward (toward the center)
    - ``magic_side`` ("top", "bottom", "left", "right"): magic patches on that
      edge only instead of the full ring
    - ``magic_ring_count``: exactly that many magic patches, spread evenly over
      the four edges

    Grid dimensions:
    - Each block occupies a 3x3 area (2x2 for qubits + 1 for spacing)
    - n x m blocks => (3n-1) x (3m-1) cells for blocks
    - with 1-patch spacing between blocks and magic states: (3n+1) x (3m+1) cells
    - actual layout: (3n+3) x (3m+3) cells
    
    Example: n=2, m=2 => layout is 9x9
    """
    # Calculate grid dimensions
    # n x m blocks of 4 qubits with spacing: (3n-1) x (3m-1) cells for blocks
    # + 1 patch on each side for spacing before magic: (3n+1) x (3m+1) cells
    # + 1 patch on each side for magic patches: (3n+3) x (3m+3) cells
    W = 3 * n + 3
    H = 3 * m + 3

    eng = LayoutEngine(W, H)
    
    # --- Place blocks of 4 qubits in n x m grid ---
    # Position (i, j) -> block origin at cell (3*i + 2, 3*j + 2)
    # Qubits are numbered sequentially: q_0, q_1, q_2, ..., q_(n*m*4-1)
    qubit_idx = 0
    for i in range(n):
        for j in range(m):
            ox = 3 * i + 2
            oy = 3 * j + 2
            block_patches = [
                data_patch_1cell(f"q_{qubit_idx}", (ox, oy), swap_xz=swap_xz),
                data_patch_1cell(f"q_{qubit_idx + 1}", (ox + 1, oy), swap_xz=swap_xz),
                data_patch_1cell(f"q_{qubit_idx + 2}", (ox, oy + 1), swap_xz=swap_xz),
                data_patch_1cell(f"q_{qubit_idx + 3}", (ox + 1, oy + 1), swap_xz=swap_xz),
            ]
            for p in block_patches:
                eng.add_patch(p)
            qubit_idx += 4

    if magic_ring_count is not None:
        # Spread exactly ``magic_ring_count`` patches over the four edges;
        # the remainder goes to top, bottom, left, right in that order.
        base, extra = divmod(magic_ring_count, 4)
        for k, side in enumerate(("top", "bottom", "left", "right")):
            _add_magic_fixed_count(eng, W, H, base + (k < extra), side=side)
        return eng
    if magic_side is not None:
        _add_magic_one_side(eng, W, H, side=magic_side)
        return eng

    for x in range(1, W - 1):
        eng.add_patch(magic_patch_1cell(f"mT{x}", (x, 0), side="S"))
    
    for x in range(1, W - 1):
        eng.add_patch(magic_patch_1cell(f"mB{x}", (x, H - 1), side="N"))
    
    for y in range(1, H - 1):
        eng.add_patch(magic_patch_1cell(f"mL{y}", (0, y), side="E"))
    
    for y in range(1, H - 1):
        eng.add_patch(magic_patch_1cell(f"mR{y}", (W - 1, y), side="W"))
    
    return eng

_SIDE_CONFIG = {
    "top":    lambda W, H: ([(x, 0)     for x in range(1, W - 1)], "S", "mT"),
    "bottom": lambda W, H: ([(x, H - 1) for x in range(1, W - 1)], "N", "mB"),
    "left":   lambda W, H: ([(0, y)     for y in range(1, H - 1)], "E", "mL"),
    "right":  lambda W, H: ([(W - 1, y) for y in range(1, H - 1)], "W", "mR"),
}


def _add_magic_one_side(eng, W, H, side="top"):
    """Place magic patches along a single side of the grid (skipping corners)."""
    if side not in _SIDE_CONFIG:
        raise ValueError(f"side must be one of {list(_SIDE_CONFIG)}, got {side!r}")
    positions, port_side, prefix = _SIDE_CONFIG[side](W, H)
    for coord in positions:
        idx = coord[0] if side in ("top", "bottom") else coord[1]
        eng.add_patch(magic_patch_1cell(f"{prefix}{idx}", coord, side=port_side))


def _add_magic_fixed_count(eng, W, H, num_magic, side="top"):
    """Place exactly *num_magic* magic patches evenly along one side."""
    if side not in _SIDE_CONFIG:
        raise ValueError(f"side must be one of {list(_SIDE_CONFIG)}, got {side!r}")
    positions, port_side, prefix = _SIDE_CONFIG[side](W, H)
    available = len(positions)
    if num_magic > available:
        import warnings
        warnings.warn(
            f"Requested {num_magic} magic patches on {side} but only "
            f"{available} positions available; capping at {available}."
        )
        num_magic = available
    if num_magic <= 0:
        return
    # Pick evenly-spaced indices into the positions list
    if num_magic == 1:
        chosen = [available // 2]
    else:
        chosen = [round(i * (available - 1) / (num_magic - 1)) for i in range(num_magic)]
    for ci in chosen:
        coord = positions[ci]
        idx = coord[0] if side in ("top", "bottom") else coord[1]
        eng.add_patch(magic_patch_1cell(f"{prefix}{idx}", coord, side=port_side))


def nxm_one_side_magic_layout_single_qubits(
    n: int, m: int, *, side: str = "top", swap_xz: bool = False,
) -> LayoutEngine:
    """
    Same grid as ``nxm_ring_layout_single_qubits`` (single spacing, W=2n+3,
    H=2m+3) but magic patches placed **only on one side**.

    Parameters
    ----------
    n, m : int
        Number of data-qubit columns / rows.
    side : str
        Which edge to place magic patches on: "top", "bottom", "left", "right".
    swap_xz : bool
        Swap X/Z port types on data patches.
    """
    W = 2 * n + 3
    H = 2 * m + 3
    eng = LayoutEngine(W, H)

    qubit_idx = 0
    for i in range(n):
        for j in range(m):
            x = 2 * i + 2
            y = 2 * j + 2
            eng.add_patch(data_patch_1cell(f"q_{qubit_idx}", (x, y), swap_xz=swap_xz))
            qubit_idx += 1

    _add_magic_one_side(eng, W, H, side=side)
    return eng


def nxm_one_side_magic_layout_single_qubits_large_spacing(
    n: int, m: int, *, side: str = "top", swap_xz: bool = False,
) -> LayoutEngine:
    """
    Same grid as ``nxm_ring_layout_single_qubits_large_spacing`` (double
    spacing, W=3n+2, H=3m+2) but magic patches placed **only on one side**.
    """
    W = 3 * n + 2
    H = 3 * m + 2
    eng = LayoutEngine(W, H)

    qubit_idx = 0
    for i in range(n):
        for j in range(m):
            x = 3 * i + 2
            y = 3 * j + 2
            eng.add_patch(data_patch_1cell(f"q_{qubit_idx}", (x, y), swap_xz=swap_xz))
            qubit_idx += 1

    _add_magic_one_side(eng, W, H, side=side)
    return eng


def nxm_fixed_magic_count_layout_single_qubits(
    n: int, m: int, num_magic: int, *, side: str = "top", swap_xz: bool = False,
) -> LayoutEngine:
    """
    Same grid as ``nxm_ring_layout_single_qubits`` (single spacing, W=2n+3,
    H=2m+3) but with exactly *num_magic* magic patches evenly distributed
    along one side.

    Parameters
    ----------
    n, m : int
        Number of data-qubit columns / rows.
    num_magic : int
        Exact number of magic patches to place.
    side : str
        Which edge to place magic patches on: "top", "bottom", "left", "right".
    swap_xz : bool
        Swap X/Z port types on data patches.
    """
    W = 2 * n + 3
    H = 2 * m + 3
    eng = LayoutEngine(W, H)

    qubit_idx = 0
    for i in range(n):
        for j in range(m):
            x = 2 * i + 2
            y = 2 * j + 2
            eng.add_patch(data_patch_1cell(f"q_{qubit_idx}", (x, y), swap_xz=swap_xz))
            qubit_idx += 1

    _add_magic_fixed_count(eng, W, H, num_magic, side=side)
    return eng


def nxm_fixed_magic_count_layout_single_qubits_large_spacing(
    n: int, m: int, num_magic: int, *, side: str = "top", swap_xz: bool = False,
) -> LayoutEngine:

    W = 3 * n + 2
    H = 3 * m + 2
    eng = LayoutEngine(W, H)

    qubit_idx = 0
    for i in range(n):
        for j in range(m):
            x = 3 * i + 2
            y = 3 * j + 2
            eng.add_patch(data_patch_1cell(f"q_{qubit_idx}", (x, y), swap_xz=swap_xz))
            qubit_idx += 1

    _add_magic_fixed_count(eng, W, H, num_magic, side=side)
    return eng


def dense_comb_layout(num_qubits: int, num_magic: int = 2, *, swap_xz: bool = False) -> LayoutEngine:
    """Dense layout (~2.25 tiles per qubit) where every data patch keeps X and Z access.

    Columns repeat ``R D D`` and end with ``R`` (R = routing column spanning
    all rows); rows repeat bands of ``D R D`` (data row, routing row, data
    row).  Every single-cell data patch (X ports N/S, Z ports E/W) therefore
    touches the routing row on its N or S side and a routing column on its E
    or W side, so X-, Z- and Y-type products stay routable without patch
    rotations.  A 3x3 block holds four qubits and five routing cells, versus
    the ring presets' one-tile spacing around every patch plus a full magic
    ring.

    Magic patches sit in an extra top row directly above the routing columns
    (port facing S); the remaining cells of that row are blocked so they add
    no routing space.  ``num_magic`` ports are spread evenly over the
    ``units + 1`` routing columns; if more are requested, a bottom magic row is
    added the same way.

    Grid: ``W = 3*units + 1``, ``H = 3*bands + 1 (+1)``, chosen near-square with
    ``4 * units * bands >= num_qubits``.  Unused data slots stay free cells.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be positive")
    if num_magic < 1:
        raise ValueError("num_magic must be at least 1")
    best = None
    for bands in range(1, num_qubits + 1):
        units = -(-num_qubits // (4 * bands))
        W, H = 3 * units + 1, 3 * bands + 1
        key = (W * H, abs(W - H))
        if best is None or key < best[0]:
            best = (key, units, bands)
        if 4 * bands > num_qubits:
            break
    _, units, bands = best
    ports_per_row = units + 1
    bottom_row = num_magic > ports_per_row
    W = 3 * units + 1
    H = 3 * bands + 1 + (1 if bottom_row else 0)
    top = 1  # data/routing rows start below the magic row
    eng = LayoutEngine(W, H)

    qubit_idx = 0
    for band in range(bands):
        for y in (top + 3 * band, top + 3 * band + 2):  # the two data rows of the band
            for unit in range(units):
                for x in (3 * unit + 1, 3 * unit + 2):
                    if qubit_idx < num_qubits:
                        eng.add_patch(data_patch_1cell(f"q_{qubit_idx}", (x, y), swap_xz=swap_xz))
                        qubit_idx += 1

    def magic_row(y: int, count: int, side: str, prefix: str) -> None:
        columns = [3 * u for u in range(units + 1)]
        if count >= len(columns):
            chosen = columns
        elif count == 1:
            chosen = [columns[len(columns) // 2]]
        else:
            chosen = [columns[round(i * (len(columns) - 1) / (count - 1))] for i in range(count)]
        for x in chosen:
            eng.add_patch(magic_patch_1cell(f"{prefix}{x}", (x, y), side=side))
        eng.add_blocked({(x, y) for x in range(W) if x not in chosen})

    magic_row(0, min(num_magic, ports_per_row), "S", "mT")
    if bottom_row:
        magic_row(H - 1, num_magic - ports_per_row, "N", "mB")
    return eng
