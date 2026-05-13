"""
Shared metric helpers for routing experiments.
"""

from collections import defaultdict


def compute_active_volume(results: list, mode: str) -> int:
    """Compute active volume from routing result dicts.

    Active volume = Σ_t |{ grid cells occupied at timestep t }|

    For packing / pathfinder mode: group results by time_step, take the
    union of (x, y) tuple nodes per step, then sum cardinalities.

    For sequential mode (steiner_tree): every result corresponds to its own
    step, so we simply count all tuple nodes across all results.

    Args:
        results: List of result dicts from any scheduler function.
        mode: Scheduler mode string ("steiner_tree", "steiner_packing",
              or "steiner_pathfinder").

    Returns:
        Total active volume (integer, in cell-steps).
    """
    if mode == "steiner_tree":
        total = 0
        for r in results:
            for node in r.get("steiner_nodes", set()):
                if isinstance(node, tuple):
                    total += 1
        return total

    # Packing / pathfinder: group by time_step and union cells per step.
    by_step: dict = defaultdict(set)
    for r in results:
        t = r.get("time_step", 0)
        for node in r.get("steiner_nodes", set()):
            if isinstance(node, tuple):
                by_step[t].add(node)
    return sum(len(cells) for cells in by_step.values())
