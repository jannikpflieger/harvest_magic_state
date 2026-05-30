"""
Lattice pruner — post-scheduling pass that removes unused nodes from the
routing-graph data structures produced by :meth:`LayoutEngine.build_routing_graph`.

After the scheduler has produced a list of result dicts (one per operation), many
routing cells and patch-port nodes may never have appeared in any Steiner tree or
terminal set.  This module collects the set of *used* nodes from the result list
and returns leaner versions of all four routing-graph structures:

    graph, ports_by_patch, pos, patch_used_by_port

The :class:`LayoutEngine` itself (``eng.patches``, ``eng.occ``) is intentionally
left unchanged — only the derived routing-graph view is pruned.

Public API
----------
prune_lattice(results, graph, ports_by_patch, pos, patch_used_by_port)
    → (pruned_graph, pruned_ports_by_patch, pruned_pos, pruned_patch_used_by_port, stats)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Set, Tuple

logger = logging.getLogger("HarvestMagicState.LatticePruner")

# Type aliases matching engine.py conventions
_Graph = Dict[Any, List[Tuple[Any, int]]]
_PortsByPatch = Dict[str, Dict[str, List[str]]]
_Pos = Dict[Any, Tuple[float, float]]
_PatchUsedByPort = Dict[str, str]


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def prune_lattice(
    results: List[Dict],
    graph: _Graph,
    ports_by_patch: _PortsByPatch,
    pos: _Pos,
    patch_used_by_port: _PatchUsedByPort,
) -> Tuple[_Graph, _PortsByPatch, _Pos, _PatchUsedByPort, Dict]:
    """Remove graph nodes that were never referenced during scheduling.

    Scans every result dict in *results* and builds the set of nodes that were
    actually used:

    * ``r["steiner_nodes"]`` — routing cells and port nodes touched by the
      Steiner tree for each operation.
    * ``r["magic_terminal"]`` — the magic-state port selected for each
      operation.
    * ``r["qubit_terminals"]`` — the data-qubit port nodes for each operation.

    Every node in *graph* that is absent from this used set is removed.  Edges
    that reference a removed node are also dropped.  The companion structures
    (*ports_by_patch*, *pos*, *patch_used_by_port*) are updated consistently.

    Args:
        results:            List of per-operation result dicts returned by the
                            scheduler (``process_entire_dag`` / adaptive variant).
        graph:              Adjacency-list routing graph
                            ``{node: [(neighbour, weight), ...]}``.
        ports_by_patch:     Port mapping ``{patch_name: {port_type: [port_ids]}}``.
        pos:                Position dict ``{node: (x, y)}``.
        patch_used_by_port: Reverse mapping ``{port_id: patch_name}``.

    Returns:
        A 5-tuple ``(pruned_graph, pruned_ports_by_patch, pruned_pos,
        pruned_patch_used_by_port, stats)`` where *stats* is::

            {
                "nodes_before":        int,  # nodes in original graph
                "nodes_after":         int,  # nodes remaining after pruning
                "nodes_removed":       int,  # nodes removed
                "routing_cells_removed": int, # (x, y) tuple nodes removed
                "port_nodes_removed":  int,  # MAGIC string port nodes removed
                "port_nodes_removed_total": int,
                "magic_port_nodes_removed": int,
                "data_port_nodes_removed": int,
                "patches_before":      int,  # patches with ≥1 port before
                "patches_after":       int,  # patches with ≥1 port after
                "magic_patches_before": int,
                "magic_patches_after":  int,
                "magic_patches_removed": int,
                "data_patches_before": int,
                "data_patches_after":  int,
                "data_patches_removed": int,
            }
    """
    # ------------------------------------------------------------------
    # 1. Collect the set of nodes that were actually used.
    # ------------------------------------------------------------------
    used_nodes: Set = set()
    for r in results:
        # Routing cells and port nodes traversed by the Steiner tree
        for node in r.get("steiner_nodes", set()):
            used_nodes.add(node)

        # Explicitly selected magic terminal (a port-node string)
        magic_term = r.get("magic_terminal")
        if magic_term:
            used_nodes.add(magic_term)

        # Data-qubit port nodes
        for term in r.get("qubit_terminals", []):
            used_nodes.add(term)

    def _is_magic_patch(type_map: Dict[str, List[str]]) -> bool:
        # In this codebase magic patches expose M-ports.
        return "M" in type_map and len(type_map.get("M", [])) > 0

    magic_patches_before_set = {
        patch_name for patch_name, type_map in ports_by_patch.items() if _is_magic_patch(type_map)
    }
    data_patches_before_set = set(ports_by_patch) - magic_patches_before_set

    nodes_before = len(graph)
    routing_before = sum(1 for n in graph if isinstance(n, tuple))
    port_before = sum(1 for n in graph if isinstance(n, str))
    patches_before = len(ports_by_patch)

    logger.debug(
        "Pruner: graph has %d nodes (%d routing cells, %d port nodes) across %d patches before pruning.",
        nodes_before, routing_before, port_before, patches_before,
    )
    logger.debug("Pruner: %d nodes were used during scheduling.", len(used_nodes))

    # ------------------------------------------------------------------
    # 2. Rebuild the graph, keeping only used nodes and their edges.
    # ------------------------------------------------------------------
    pruned_graph: _Graph = {}
    for node, neighbours in graph.items():
        if node not in used_nodes:
            continue
        pruned_graph[node] = [
            (nbr, w) for (nbr, w) in neighbours if nbr in used_nodes
        ]

    # ------------------------------------------------------------------
    # 3. Filter the companion structures.
    # ------------------------------------------------------------------
    pruned_pos: _Pos = {n: v for n, v in pos.items() if n in used_nodes}

    pruned_patch_used_by_port: _PatchUsedByPort = {
        pid: pname
        for pid, pname in patch_used_by_port.items()
        if pid in used_nodes
    }

    # Rebuild ports_by_patch: drop port IDs not in used_nodes, then drop
    # empty port-type lists and empty patch entries.
    pruned_ports_by_patch: _PortsByPatch = {}
    for patch_name, type_map in ports_by_patch.items():
        new_type_map: Dict[str, List[str]] = {}
        for port_type, port_ids in type_map.items():
            kept = [pid for pid in port_ids if pid in used_nodes]
            if kept:
                new_type_map[port_type] = kept
        if new_type_map:
            pruned_ports_by_patch[patch_name] = new_type_map

    # ------------------------------------------------------------------
    # 4. Compute and log statistics.
    # ------------------------------------------------------------------
    magic_patches_after_set = {
        patch_name for patch_name, type_map in pruned_ports_by_patch.items() if _is_magic_patch(type_map)
    }
    data_patches_after_set = set(pruned_ports_by_patch) - magic_patches_after_set

    nodes_after = len(pruned_graph)
    routing_after = sum(1 for n in pruned_graph if isinstance(n, tuple))
    port_after = sum(1 for n in pruned_graph if isinstance(n, str))
    patches_after = len(pruned_ports_by_patch)

    removed_nodes = set(graph) - set(pruned_graph)
    magic_port_nodes_removed = 0
    data_port_nodes_removed = 0
    for node in removed_nodes:
        if not isinstance(node, str):
            continue
        patch_name = patch_used_by_port.get(node)
        if patch_name in magic_patches_before_set:
            magic_port_nodes_removed += 1
        elif patch_name in data_patches_before_set:
            data_port_nodes_removed += 1

    port_nodes_removed_total = magic_port_nodes_removed + data_port_nodes_removed

    stats = {
        "nodes_before":          nodes_before,
        "nodes_after":           nodes_after,
        "nodes_removed":         nodes_before - nodes_after,
        "routing_cells_removed": routing_before - routing_after,
        # Keep historical key name, but align semantics with user expectation:
        # only magic-state port removals count toward "port_nodes_removed".
        "port_nodes_removed":    magic_port_nodes_removed,
        "port_nodes_removed_total": port_nodes_removed_total,
        "magic_port_nodes_removed": magic_port_nodes_removed,
        "data_port_nodes_removed": data_port_nodes_removed,
        "patches_before":        patches_before,
        "patches_after":         patches_after,
        "magic_patches_before":  len(magic_patches_before_set),
        "magic_patches_after":   len(magic_patches_after_set),
        "magic_patches_removed": len(magic_patches_before_set - magic_patches_after_set),
        "data_patches_before":   len(data_patches_before_set),
        "data_patches_after":    len(data_patches_after_set),
        "data_patches_removed":  len(data_patches_before_set - data_patches_after_set),
    }

    logger.info(
        "Pruner: removed %d / %d nodes (%d routing cells, %d magic port nodes, %d data port nodes). "
        "Patches with active ports: %d → %d (magic: %d → %d).",
        stats["nodes_removed"], nodes_before,
        stats["routing_cells_removed"], stats["magic_port_nodes_removed"], stats["data_port_nodes_removed"],
        patches_before, patches_after,
        stats["magic_patches_before"], stats["magic_patches_after"],
    )

    return pruned_graph, pruned_ports_by_patch, pruned_pos, pruned_patch_used_by_port, stats
