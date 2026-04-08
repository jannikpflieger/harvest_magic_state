"""
Layout templates for circuit-aware synthesis.

A LayoutTemplate describes the geometry of a lattice layout *before* any
logical qubit has been assigned to a specific site.  It lists data sites,
ancilla/routing sites, magic sites, and precomputed pairwise distances
between data sites so the placement heuristic can evaluate assignment cost
in O(1) per pair.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

Coord = Tuple[int, int]


@dataclass
class LayoutTemplate:
    """Geometry of a lattice layout without qubit assignment."""

    name: str
    grid_width: int
    grid_height: int

    data_sites: List[Coord]
    """Coordinates where data qubits can be placed."""

    magic_sites: List[Coord]
    """Coordinates reserved for magic-state patches."""

    routing_lanes: int
    """Number of routing rows between each pair of adjacent data rows."""

    pairwise_distances: Dict[Tuple[int, int], float] = field(default_factory=dict)
    """Manhattan distance between data site indices (i < j)."""

    site_centrality: Dict[int, float] = field(default_factory=dict)
    """Centrality score for each data site index (higher = more central)."""


def _manhattan(a: Coord, b: Coord) -> float:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _compute_distances_and_centrality(sites: List[Coord]) -> Tuple[Dict[Tuple[int, int], float], Dict[int, float]]:
    """Precompute pairwise Manhattan distances and centrality for data sites."""
    n = len(sites)
    distances: Dict[Tuple[int, int], float] = {}
    total_dist: Dict[int, float] = {i: 0.0 for i in range(n)}

    for i in range(n):
        for j in range(i + 1, n):
            d = _manhattan(sites[i], sites[j])
            distances[(i, j)] = d
            total_dist[i] += d
            total_dist[j] += d

    # Centrality = inverse of average distance to all other sites
    centrality: Dict[int, float] = {}
    for i in range(n):
        avg_d = total_dist[i] / max(n - 1, 1)
        centrality[i] = 1.0 / max(avg_d, 0.01)

    return distances, centrality


def bus_template(n_qubits: int, num_lanes: int = 2) -> LayoutTemplate:
    """
    Rectangular bus template.

    Data qubits are arranged in rows with ``num_lanes`` routing rows between them.
    Magic-state patches form a ring around the perimeter (same pattern as
    ``nxm_ring_layout_single_qubits`` in presets.py).

    The number of columns is chosen to make the grid as close to square as
    possible.

    Args:
        n_qubits: Number of data-qubit sites to create (≥ 1).
        num_lanes: Routing rows between adjacent data rows (1, 2, or 3).

    Returns:
        LayoutTemplate with precomputed distances and centrality.
    """
    # Decide grid dimensions (data columns x data rows)
    cols = max(1, math.ceil(math.sqrt(n_qubits)))
    rows = max(1, math.ceil(n_qubits / cols))

    # Spacing between adjacent data qubits:
    #   horizontal: every other column (spacing = 2 in x) for num_lanes == 1
    #   vertical: (num_lanes + 1) rows apart
    x_spacing = 2  # matches nxm_ring_layout_single_qubits
    y_spacing = num_lanes + 1  # 1 lane → 2, 2 lanes → 3, 3 lanes → 4

    # Grid origin for data: leave 2 cells of margin for magic ring
    x_origin = 2
    y_origin = 2

    data_sites: List[Coord] = []
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            if idx >= n_qubits:
                break
            x = x_origin + c * x_spacing
            y = y_origin + r * y_spacing
            data_sites.append((x, y))

    # Grid dimensions: data footprint + 2-cell border for magic ring
    max_x = max(s[0] for s in data_sites) if data_sites else x_origin
    max_y = max(s[1] for s in data_sites) if data_sites else y_origin
    W = max_x + 3  # +1 for cell width + 2 for right margin
    H = max_y + 3

    # Magic-state patches on the perimeter (skip corners)
    magic_sites: List[Coord] = []
    for x in range(1, W - 1):
        magic_sites.append((x, 0))
        magic_sites.append((x, H - 1))
    for y in range(1, H - 1):
        magic_sites.append((0, y))
        magic_sites.append((W - 1, y))

    distances, centrality = _compute_distances_and_centrality(data_sites)

    return LayoutTemplate(
        name=f"bus_{num_lanes}lane_{n_qubits}q",
        grid_width=W,
        grid_height=H,
        data_sites=data_sites,
        magic_sites=magic_sites,
        routing_lanes=num_lanes,
        pairwise_distances=distances,
        site_centrality=centrality,
    )


def select_template(
    n_qubits: int,
    max_parallelism: int = 0,
    num_lanes: Optional[int] = None,
) -> LayoutTemplate:
    """
    Auto-select a bus template based on circuit properties.

    If *num_lanes* is given it is used directly.  Otherwise the number of
    routing lanes is chosen from ``max_parallelism`` (the widest DAG layer):

    * ≤ 5  → 1 lane (compact)
    * ≤ 10 → 2 lanes (standard)
    * > 10 → 3 lanes (spacious)
    """
    if num_lanes is None:
        if max_parallelism <= 5:
            num_lanes = 1
        elif max_parallelism <= 10:
            num_lanes = 2
        else:
            num_lanes = 3

    return bus_template(n_qubits, num_lanes=num_lanes)
