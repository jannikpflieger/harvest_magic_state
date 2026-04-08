"""
Placement heuristics: assign logical qubits to data sites on a LayoutTemplate.

Two strategies:
  - **circuit_aware_placement**: uses the interaction graph from CircuitSummary
    to minimise a weighted-distance cost function.
  - **baseline_placement**: simple row-major, column-major, or random assignment
    (no circuit information used).
"""

from __future__ import annotations

import random as _random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .circuit_summary import CircuitSummary
from .templates import LayoutTemplate


@dataclass
class PlacementConfig:
    """Knobs for the placement heuristic."""

    alpha: float = 1.0
    """Weight for interaction-distance cost."""

    beta: float = 0.5
    """Weight for Y-count accessibility penalty (Y qubits need both X+Z ports)."""

    max_swap_iterations: int = 100
    """Maximum pairwise-swap improvement passes."""

    seed: Optional[int] = None
    """Random seed for tie-breaking."""


@dataclass
class PlacementResult:
    """Output of a placement run."""

    assignment: Dict[int, int]
    """Logical qubit index → data-site index in the template."""

    cost: float
    """Final cost value."""

    cost_breakdown: Dict[str, float] = field(default_factory=dict)
    """Per-component cost (interaction, accessibility)."""

    improvement_history: List[float] = field(default_factory=list)
    """Cost after each swap iteration (for diagnostics)."""


# ---------------------------------------------------------------------------
# Cost evaluation
# ---------------------------------------------------------------------------

def _interaction_cost(
    assignment: Dict[int, int],
    interaction_graph: Dict[Tuple[int, int], float],
    pairwise_distances: Dict[Tuple[int, int], float],
) -> float:
    """Sum of weight * distance for every interacting qubit pair."""
    total = 0.0
    for (qi, qj), weight in interaction_graph.items():
        si = assignment.get(qi)
        sj = assignment.get(qj)
        if si is None or sj is None:
            continue
        key = (min(si, sj), max(si, sj))
        dist = pairwise_distances.get(key, 0.0)
        total += weight * dist
    return total


def _accessibility_penalty(
    assignment: Dict[int, int],
    pauli_profile: Dict,
    site_centrality: Dict[int, float],
) -> float:
    """Penalise Y-heavy qubits placed on low-centrality (peripheral) sites."""
    total = 0.0
    for qubit, site in assignment.items():
        y_count = pauli_profile.get(qubit, {}).get('Y', 0)
        if y_count > 0:
            centrality = site_centrality.get(site, 0.0)
            # Lower centrality → higher penalty
            total += y_count * (1.0 / max(centrality, 0.01))
    return total


def _total_cost(
    assignment: Dict[int, int],
    summary: CircuitSummary,
    template: LayoutTemplate,
    config: PlacementConfig,
) -> Tuple[float, Dict[str, float]]:
    ic = config.alpha * _interaction_cost(
        assignment, summary.interaction_graph, template.pairwise_distances
    )
    ap = config.beta * _accessibility_penalty(
        assignment, summary.pauli_profile, template.site_centrality
    )
    breakdown = {"interaction": ic, "accessibility": ap}
    return ic + ap, breakdown


# ---------------------------------------------------------------------------
# Circuit-aware placement
# ---------------------------------------------------------------------------

def circuit_aware_placement(
    summary: CircuitSummary,
    template: LayoutTemplate,
    config: Optional[PlacementConfig] = None,
) -> PlacementResult:
    """
    Greedy initial placement + local pairwise-swap improvement.

    1. Rank qubits by total interaction degree (most connected first).
    2. Rank data sites by centrality (most central first).
    3. Greedily assign highest-degree qubit to most central available site.
    4. Iteratively swap pairs that reduce total cost.
    """
    if config is None:
        config = PlacementConfig()
    rng = _random.Random(config.seed)

    qubits = summary.qubit_list
    n = len(qubits)
    n_sites = len(template.data_sites)
    if n > n_sites:
        raise ValueError(
            f"More qubits ({n}) than data sites ({n_sites}) in template."
        )

    # Interaction degree per qubit
    degree: Dict[int, float] = {q: 0.0 for q in qubits}
    for (qi, qj), w in summary.interaction_graph.items():
        if qi in degree:
            degree[qi] += w
        if qj in degree:
            degree[qj] += w

    qubits_sorted = sorted(qubits, key=lambda q: degree[q], reverse=True)
    sites_sorted = sorted(range(n_sites), key=lambda s: template.site_centrality.get(s, 0.0), reverse=True)

    # Greedy initial assignment
    assignment: Dict[int, int] = {}
    used_sites: set = set()
    for q, s in zip(qubits_sorted, sites_sorted):
        assignment[q] = s
        used_sites.add(s)

    cost, breakdown = _total_cost(assignment, summary, template, config)
    history: List[float] = [cost]

    # Local improvement: pairwise swap
    qubit_list = list(assignment.keys())
    for _iteration in range(config.max_swap_iterations):
        improved = False
        rng.shuffle(qubit_list)

        for i in range(len(qubit_list)):
            for j in range(i + 1, len(qubit_list)):
                qi, qj = qubit_list[i], qubit_list[j]
                si, sj = assignment[qi], assignment[qj]

                # Try swap
                assignment[qi], assignment[qj] = sj, si
                new_cost, new_bd = _total_cost(assignment, summary, template, config)

                if new_cost < cost:
                    cost = new_cost
                    breakdown = new_bd
                    improved = True
                else:
                    # Revert
                    assignment[qi], assignment[qj] = si, sj

        history.append(cost)
        if not improved:
            break

    return PlacementResult(
        assignment=assignment,
        cost=cost,
        cost_breakdown=breakdown,
        improvement_history=history,
    )


# ---------------------------------------------------------------------------
# Baseline (non-circuit-aware) placement
# ---------------------------------------------------------------------------

def baseline_placement(
    template: LayoutTemplate,
    n_qubits: int,
    mode: str = "row_major",
    seed: Optional[int] = None,
) -> PlacementResult:
    """
    Assign qubits to data sites without any circuit information.

    Modes:
      - ``"row_major"``: qubit k → site k (default, matches nxm_ring preset ordering)
      - ``"column_major"``: qubit k → site at column-first traversal order
      - ``"random"``: random permutation of sites
    """
    n_sites = len(template.data_sites)
    if n_qubits > n_sites:
        raise ValueError(
            f"More qubits ({n_qubits}) than data sites ({n_sites}) in template."
        )

    if mode == "row_major":
        assignment = {q: q for q in range(n_qubits)}
    elif mode == "column_major":
        # Sites are laid out row-major in the template (row r, col c → index r*cols + c).
        # Column-major maps qubit k to column-first traversal: go down columns first.
        import math
        cols = max(1, math.ceil(math.sqrt(n_sites)))
        rows = max(1, math.ceil(n_sites / cols))
        assignment = {}
        for q in range(n_qubits):
            col = q // rows
            row = q % rows
            site = row * cols + col
            # Fall back to identity if computed site is out of range
            assignment[q] = site if site < n_sites else q
    elif mode == "random":
        rng = _random.Random(seed)
        sites = list(range(n_sites))
        rng.shuffle(sites)
        assignment = {q: sites[q] for q in range(n_qubits)}
    else:
        raise ValueError(f"Unknown baseline mode: {mode!r}")

    return PlacementResult(assignment=assignment, cost=0.0)
