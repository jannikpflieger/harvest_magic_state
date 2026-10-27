"""Qubit placement on an existing layout (EVAL_PLAN E6).

``StaticLayoutSynthesizer`` places qubits on its own bus template.  To compare
placements on *identical* layouts (ring, blocks4, dense), this module treats
any preset's data patches as the sites of a ``LayoutTemplate``, runs the same
placement heuristics (``circuit_aware_placement`` / ``baseline_placement``)
and its cost ``C(pi)`` on them, and renames the data patches accordingly.  The
geometry -- and therefore the area -- is unchanged; only which logical qubit
sits on which site differs.
"""

from __future__ import annotations

import re
from itertools import count
from dataclasses import replace
from typing import Dict, Optional, Tuple

from harvest.layout.engine import LayoutEngine, Patch
from harvest.synthesis.circuit_summary import extract_circuit_summary
from harvest.synthesis.placement import (
    PlacementConfig,
    _total_cost,
    baseline_placement,
    circuit_aware_placement,
)
from harvest.synthesis.templates import LayoutTemplate, _compute_distances_and_centrality

PLACEMENTS = ("index", "random", "circuit_aware")
_DATA = re.compile(r"^q_(\d+)$")


def _sites(eng: LayoutEngine):
    """Data sites in index order: [(original name, centroid cell)]."""
    sites = []
    for name, patch in eng.patches.items():
        match = _DATA.match(name)
        if match:
            x, y = min(patch.cells)
            sites.append((int(match.group(1)), name, (x, y)))
    sites.sort()
    return [(name, cell) for _, name, cell in sites]


def layout_template(eng: LayoutEngine) -> LayoutTemplate:
    sites = _sites(eng)
    magic = [min(p.cells) for p in eng.patches.values()
             if any(port.port_type == "M" for port in p.ports)]
    distances, centrality = _compute_distances_and_centrality([cell for _, cell in sites])
    return LayoutTemplate(
        name="preset", grid_width=eng.W, grid_height=eng.H,
        data_sites=[cell for _, cell in sites], magic_sites=magic, routing_lanes=1,
        pairwise_distances=distances, site_centrality=centrality,
    )


def relabel(eng: LayoutEngine, assignment: Dict[int, int]) -> LayoutEngine:
    """Copy of ``eng`` where logical qubit ``q`` sits on data site ``assignment[q]``."""
    sites = _sites(eng)
    new_name = {sites[site][0]: f"q_{q}" for q, site in assignment.items()}
    spare = (i for i in count() if i not in assignment)
    for name, _ in sites:
        new_name.setdefault(name, f"q_{next(spare)}")
    out = LayoutEngine(eng.W, eng.H)
    for cell, value in eng.occ.items():
        if value == "BLOCKED":
            out.occ[cell] = "BLOCKED"
    for name, patch in eng.patches.items():
        renamed = new_name.get(name, name)
        ports = [replace(port, patch=renamed) for port in patch.ports]
        out.add_patch(Patch(name=renamed, kind=patch.kind, cells=set(patch.cells), ports=ports))
    return out


def place(
    eng: LayoutEngine, dag, mode: str, seed: Optional[int] = None,
    config: Optional[PlacementConfig] = None,
) -> Tuple[LayoutEngine, float, Dict[int, int]]:
    """Return ``(relabelled layout, placement cost C(pi), assignment)``."""
    if mode not in PLACEMENTS:
        raise ValueError(f"unknown placement {mode!r}; expected one of {PLACEMENTS}")
    config = config or PlacementConfig(alpha=1.0, beta=0.5, max_swap_iterations=100, seed=seed)
    summary = extract_circuit_summary(dag)
    template = layout_template(eng)
    if mode == "circuit_aware":
        result = circuit_aware_placement(summary, template, config)
    else:
        result = baseline_placement(
            template, summary.num_qubits, mode="row_major" if mode == "index" else "random",
            seed=seed,
        )
    cost, _ = _total_cost(result.assignment, summary, template, config)
    return relabel(eng, result.assignment), cost, dict(result.assignment)
