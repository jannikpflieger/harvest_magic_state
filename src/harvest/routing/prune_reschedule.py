"""HARVEST-3: prune rarely used tiles, then schedule again.

After a schedule, pruning keeps every routing cell and magic patch some route
used, however rarely.  HARVEST-3 repeats:

1. count, per retained routing cell and magic patch, how many routed products
   used it (:func:`usage_counts`);
2. build the layout of the pruned footprint without the least-used
   ``fraction`` of the routing cells and of the magic patches
   (:func:`reduced_layout`; register data patches are always kept);
3. schedule the circuit again on that layout.

A round is kept only if it completes and lowers the volume; otherwise the next
round starts again from the best layout with half the fraction.  The loop
stops after ``max_rounds`` reschedules or when nothing can be removed, and
returns the best schedule, so the result is never worse than the first one.  The
scheduler itself (HARVEST or HARVEST-2) is unchanged; only the layout it gets
shrinks.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class RescheduleConfig:
    # share of the retained routing cells / magic patches removed per round
    fraction: float = 0.1
    max_rounds: int = 4


def usage_counts(processor, results) -> Tuple[Counter, Counter]:
    """(routed products per routing cell, per magic patch) over the schedule."""
    cells: Counter = Counter()
    patches: Counter = Counter()
    patch_of = processor.patch_used_by_port
    for r in results:
        if not r.get("success", True):
            continue
        cells.update(n for n in r.get("steiner_nodes", ()) if isinstance(n, tuple))
        magic = r.get("magic_terminal")
        if magic is not None and magic in patch_of:
            patches[patch_of[magic]] += 1
    return cells, patches


def _connected(graph, removed, ports) -> bool:
    """Every port in ``ports`` reaches every other through routing cells not in ``removed``."""
    def cells_next_to(u):
        return [v for v, _ in graph[u] if isinstance(v, tuple) and v not in removed]

    ports = list(ports)
    if len(ports) < 2:
        return True
    seen = set(cells_next_to(ports[0]))
    stack = list(seen)
    while stack:
        for v in cells_next_to(stack.pop()):
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return all(any(c in seen for c in cells_next_to(p)) for p in ports[1:])


def least_used(processor, results, num_qubits: int, fraction: float):
    """The ``fraction`` least-used retained routing cells and magic patches (ties by position/name).

    A cell is only dropped if every port a routed product used still reaches
    every other such port without it (and without the cells dropped before),
    so the reduced layout can route every product the schedule routed.
    """
    cells, patches = usage_counts(processor, results)
    register = {f"q_{i}" for i in range(num_qubits)}
    kept_magic = sorted((patches[p], p) for p in processor.ports_by_patch if p not in register)
    drop_patches = {p for _, p in kept_magic[:math.floor(fraction * len(kept_magic))]}
    graph = processor.graph
    patch_of = processor.patch_used_by_port
    ports = {t for r in results if r.get("success", True) for t in r.get("all_terminals", ())
             if t in graph and patch_of.get(t) not in drop_patches}
    kept_cells = sorted((cells[c], c) for c in graph if isinstance(c, tuple))
    target = math.ceil(fraction * len(kept_cells))
    drop_cells: set = set()
    for _, c in kept_cells:
        if len(drop_cells) == target:
            break
        if _connected(graph, drop_cells | {c}, ports):
            drop_cells.add(c)
    return drop_cells, drop_patches


def reduced_layout(processor, num_qubits: int, drop_cells=(), drop_patches=()):
    """The layout of ``processor``'s pruned footprint, minus ``drop_cells`` / ``drop_patches``.

    As ``rq3_provisioning.right_sized_layout``: routing cells absent from the
    pruned routing graph and magic patches without a retained port become
    BLOCKED; every register qubit's data patch is kept.
    """
    from harvest.layout.engine import LayoutEngine

    eng = processor.eng
    kept_cells = {n for n in processor.graph if isinstance(n, tuple)} - set(drop_cells)
    kept_patches = set(processor.ports_by_patch) - set(drop_patches)
    out = LayoutEngine(eng.W, eng.H)
    for cell, value in eng.occ.items():
        if value == "BLOCKED":
            out.occ[cell] = "BLOCKED"
    for x in range(eng.W):
        for y in range(eng.H):
            if eng.is_free((x, y)) and (x, y) not in kept_cells:
                out.occ[(x, y)] = "BLOCKED"
    register = {f"q_{i}" for i in range(num_qubits)}
    for patch in eng.patches.values():
        magic = any(port.port_type == "M" for port in patch.ports)
        if (magic and patch.name not in kept_patches) or (not magic and patch.name not in register):
            for cell in patch.cells:
                out.occ[cell] = "BLOCKED"
            continue
        out.add_patch(patch)
    return out


def prune_reschedule(schedule: Callable[[Optional[Any]], Tuple[Dict[str, Any], Dict[str, Any]]],
                     num_qubits: int, config: Optional[RescheduleConfig] = None) -> Dict[str, Any]:
    """Run HARVEST-3 around ``schedule``.

    ``schedule(layout)`` runs the scheduler (``layout=None``: the original
    layout) and returns ``(row, sink)``: the result row (``status``,
    ``volume_tile_cycles``, ``T_cycles``, ``area_tiles``) and the adapter's
    sink with the pruned ``processor`` and the ``results``.  Returns the best
    round's row plus ``rounds`` (one dict per round, the first is round 0).
    """
    config = config or RescheduleConfig()
    row, sink = schedule(None)
    rounds: List[Dict[str, Any]] = [_round(0, row, 0, 0)]
    best, best_sink, best_k = row, sink, 0
    if row.get("status") != "ok":
        return {**row, "rounds": rounds, "best_round": 0}
    fraction = config.fraction
    for k in range(1, config.max_rounds + 1):
        drop_cells, drop_patches = least_used(best_sink["processor"], best_sink["results"],
                                              num_qubits, fraction)
        if not drop_cells and not drop_patches:
            break  # nothing left that can be removed
        layout = reduced_layout(best_sink["processor"], num_qubits, drop_cells, drop_patches)
        try:
            row, sink = schedule(layout)
        except Exception as exc:  # unroutable layout: counts as no improvement
            row, sink = {"status": "error", "error": str(exc)[:200]}, None
        rounds.append({**_round(k, row, len(drop_cells), len(drop_patches)), "fraction": fraction})
        if row.get("status") == "ok" and row["volume_tile_cycles"] < best["volume_tile_cycles"]:
            best, best_sink, best_k = row, sink, k
        else:
            fraction /= 2  # too much at once: retry from the best layout with half as much
    return {**best, "rounds": rounds, "best_round": best_k}


def _round(k: int, row: Dict[str, Any], cells: int, patches: int) -> Dict[str, Any]:
    return {"round": k, "status": row.get("status"), "T": row.get("T_cycles"),
            "area": row.get("area_tiles"), "volume": row.get("volume_tile_cycles"),
            "dropped_cells": cells, "dropped_patches": patches}
