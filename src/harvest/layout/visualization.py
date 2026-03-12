"""
Visualization helpers for the LayoutEngine.

All public functions take the engine instance as the first argument so that
LayoutEngine can delegate to them without carrying the plotting code itself.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple, Union

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

if TYPE_CHECKING:
    from harvest.layout.engine import LayoutEngine

Coord = Tuple[int, int]
Node = Union[Coord, str]


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _induced_subgraph(graph, pos, keep_nodes):
    """
    Return graph/pos restricted to *keep_nodes*.
    Keeps only edges (u->v) where both u and v are kept.
    """
    keep_nodes = set(keep_nodes)

    new_graph = {}
    for u in keep_nodes:
        if u in graph:
            new_graph[u] = [(v, w) for (v, w) in graph[u] if v in keep_nodes]

    new_pos = None
    if pos is not None:
        new_pos = {u: pos[u] for u in keep_nodes if u in pos}

    return new_graph, new_pos


def _draw_background(eng: "LayoutEngine", ax):
    """Draw background tiles (free / blocked / patch‐hatched)."""
    for x in range(eng.W):
        for y in range(eng.H):
            c = (x, y)
            status = eng.occ.get(c, "FREE")
            rect = plt.Rectangle((x, y), 1, 1, fill=True, edgecolor="k", linewidth=0.2)
            if status == "FREE":
                rect.set_facecolor((1, 1, 1, 1))
            elif status == "BLOCKED":
                rect.set_facecolor((0.9, 0.9, 0.9, 1))
            else:
                rect.set_facecolor((1, 1, 1, 1))
                rect.set_hatch("///")
            ax.add_patch(rect)


def _draw_patch_labels(eng: "LayoutEngine", ax, fontsize=9):
    for p in eng.patches.values():
        cx, cy = eng._patch_centroid(p.cells)
        ax.text(cx, cy, p.name, ha="center", va="center", fontsize=fontsize)


def _finish_axes(eng: "LayoutEngine", ax, title):
    ax.set_title(title)
    ax.set_xlim(0, eng.W)
    ax.set_ylim(0, eng.H)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks(range(0, eng.W + 1))
    ax.set_yticks(range(0, eng.H + 1))


MARKER_MAP = {"X": "o", "Z": "s", "M": "^"}
COLOR_MAP = {"X": "red", "Z": "blue", "M": "yellow"}


def _draw_port_nodes(ax, port_nodes, pos):
    for n in port_nodes:
        x, y = pos[n]
        t = n.split(":")[-1]
        ax.scatter([x], [y],
                   marker=MARKER_MAP.get(t, "o"),
                   s=85,
                   c=COLOR_MAP.get(t, "k"),
                   edgecolors="k",
                   linewidths=0.6)


# ------------------------------------------------------------------
# Public visualisation functions
# ------------------------------------------------------------------

def visualize_packing_solution(
    eng: "LayoutEngine",
    graph,
    pos,
    packing_results,
    *,
    title="Steiner Packing Solution",
    only_used=True,
):
    """Visualize multiple Steiner trees from packing with different colours."""
    # Collect all used nodes and edges
    all_used_nodes: set = set()
    all_used_edges: set = set()
    all_terminals: set = set()

    for result in packing_results:
        if result["success"]:
            all_used_nodes.update(result["sol_nodes"])
            all_used_edges.update(result["sol_edges"])
        all_terminals.update(result["terminal_set"])

    if only_used:
        graph, pos = _induced_subgraph(graph, pos, all_used_nodes | all_terminals)

    fig, ax = plt.subplots(figsize=(max(6, eng.W / 3), max(4, eng.H / 3)))

    _draw_background(eng, ax)
    _draw_patch_labels(eng, ax)

    # Draw graph edges faintly
    faint_segments = []
    seen: set = set()
    for u, nbrs in graph.items():
        for v, w in nbrs:
            a, b = (u, v) if str(u) <= str(v) else (v, u)
            key = (a, b, w)
            if key in seen:
                continue
            seen.add(key)
            if u in pos and v in pos and (a, b) not in all_used_edges:
                faint_segments.append([pos[u], pos[v]])
    if faint_segments:
        ax.add_collection(LineCollection(faint_segments, linewidths=0.4, colors=[(0, 0, 0, 0.15)]))

    # Colour scheme for different Steiner trees
    colors = ["red", "blue", "green", "orange", "purple", "brown", "pink", "gray", "olive", "cyan"]

    # Highlight solution edges with different colours per tree
    for i, result in enumerate(packing_results):
        if not result["success"]:
            continue
        color = colors[i % len(colors)]
        sol_segments = []
        for a, b in result["sol_edges"]:
            if a in pos and b in pos:
                sol_segments.append([pos[a], pos[b]])
        if sol_segments:
            ax.add_collection(
                LineCollection(sol_segments, linewidths=3.0, colors=[color], alpha=0.8, label=f"Tree {i+1}")
            )

    # Draw routing nodes
    routing_x, routing_y = [], []
    port_nodes = []
    for n in graph.keys():
        if isinstance(n, tuple) and n in pos:
            routing_x.append(pos[n][0])
            routing_y.append(pos[n][1])
        elif n in pos:
            port_nodes.append(n)
    ax.scatter(routing_x, routing_y, s=6)

    _draw_port_nodes(ax, port_nodes, pos)

    # Circle terminals with different styles per set
    for i, result in enumerate(packing_results):
        color = colors[i % len(colors)]
        for tnode in result["terminal_set"]:
            if tnode in pos:
                x, y = pos[tnode]
                ax.scatter([x], [y], s=200, facecolors="none", edgecolors=color, linewidths=2.5, alpha=0.8)

    # Add legend if multiple trees
    if len([r for r in packing_results if r["success"]]) > 1:
        ax.legend(loc="upper right")

    _finish_axes(eng, ax, title)
    plt.show()


def visualize_solution(
    eng: "LayoutEngine",
    graph,
    pos,
    sol_edges,
    *,
    title="Steiner solution",
    terminals=None,
    only_used=True,
):
    """Visualize the routing graph lightly and highlight *sol_edges* strongly."""
    used: set = set()
    for a, b in sol_edges:
        used.add(a)
        used.add(b)
    if terminals:
        used.update(terminals)

    if only_used:
        graph, pos = _induced_subgraph(graph, pos, used)

    fig, ax = plt.subplots(figsize=(max(6, eng.W / 3), max(4, eng.H / 3)))

    _draw_background(eng, ax)
    _draw_patch_labels(eng, ax)

    # Draw graph edges faintly
    faint_segments = []
    seen: set = set()
    for u, nbrs in graph.items():
        for v, w in nbrs:
            a, b = (u, v) if str(u) <= str(v) else (v, u)
            key = (a, b, w)
            if key in seen:
                continue
            seen.add(key)
            if u in pos and v in pos:
                faint_segments.append([pos[u], pos[v]])
    if faint_segments:
        ax.add_collection(LineCollection(faint_segments, linewidths=0.4, colors=[(0, 0, 0, 0.15)]))

    # Highlight solution edges
    sol_segments = []
    for a, b in sol_edges:
        if a in pos and b in pos:
            sol_segments.append([pos[a], pos[b]])
    if sol_segments:
        ax.add_collection(LineCollection(sol_segments, linewidths=3.0, colors=[(0, 0, 0, 0.9)]))

    # Draw nodes
    routing_x, routing_y = [], []
    port_nodes = []
    for n in graph.keys():
        if isinstance(n, tuple) and n in pos:
            routing_x.append(pos[n][0])
            routing_y.append(pos[n][1])
        elif n in pos:
            port_nodes.append(n)
    ax.scatter(routing_x, routing_y, s=6)

    _draw_port_nodes(ax, port_nodes, pos)

    # Circle terminals if provided
    if terminals:
        for tnode in terminals:
            if tnode in pos:
                x, y = pos[tnode]
                ax.scatter([x], [y], s=200, facecolors="none", edgecolors="k", linewidths=2.2)

    _finish_axes(eng, ax, title)
    plt.show()


def visualize_layout(eng: "LayoutEngine", title="Layout", show_ports=True):
    """Draw the layout grid with patches and (optionally) ports."""
    fig, ax = plt.subplots(figsize=(max(6, eng.W / 3), max(4, eng.H / 3)))

    for x in range(eng.W):
        for y in range(eng.H):
            c = (x, y)
            status = eng.occ.get(c, "FREE")
            rect = plt.Rectangle((x, y), 1, 1, fill=True, edgecolor="k", linewidth=0.25)
            if status == "FREE":
                rect.set_facecolor((1, 1, 1, 1))
            elif status == "BLOCKED":
                rect.set_facecolor((0.9, 0.9, 0.9, 1))
            else:
                rect.set_facecolor((0.75, 0.75, 0.75, 1))
            ax.add_patch(rect)

    for p in eng.patches.values():
        cx, cy = eng._patch_centroid(p.cells)
        ax.text(cx, cy, p.name, ha="center", va="center", fontsize=10)

        if show_ports:
            for port in p.ports:
                px, py = eng._port_position(port)
                ax.scatter([px], [py],
                           marker=MARKER_MAP.get(port.port_type, "o"),
                           s=70,
                           c=COLOR_MAP.get(port.port_type, "k"),
                           edgecolors="k",
                           linewidths=0.6)

    _finish_axes(eng, ax, title)
    # NOTE: intentionally not calling plt.show() here (consistent with original)


def visualize_graph(eng: "LayoutEngine", graph, pos, title="Routing graph overlay"):
    """Draw routing‐graph nodes/edges on top of the layout grid."""
    fig, ax = plt.subplots(figsize=(max(6, eng.W / 3), max(4, eng.H / 3)))

    for x in range(eng.W):
        for y in range(eng.H):
            c = (x, y)
            status = eng.occ.get(c, "FREE")
            rect = plt.Rectangle((x, y), 1, 1, fill=True, edgecolor="k", linewidth=0.2)
            if status == "FREE":
                rect.set_facecolor((1, 1, 1, 1))
            elif status == "BLOCKED":
                rect.set_facecolor((0.9, 0.9, 0.9, 1))
            else:
                rect.set_facecolor((0.75, 0.75, 0.75, 1))
            ax.add_patch(rect)

    _draw_patch_labels(eng, ax)

    # Unique edges
    segments = []
    seen: set = set()
    for u, nbrs in graph.items():
        for v, w in nbrs:
            a, b = (u, v) if str(u) <= str(v) else (v, u)
            key = (a, b, w)
            if key in seen:
                continue
            seen.add(key)
            if u in pos and v in pos:
                segments.append([pos[u], pos[v]])
    if segments:
        ax.add_collection(LineCollection(segments, linewidths=0.6))

    routing_x, routing_y = [], []
    port_nodes = []
    for n in graph.keys():
        if isinstance(n, tuple):
            routing_x.append(pos[n][0])
            routing_y.append(pos[n][1])
        else:
            port_nodes.append(n)
    ax.scatter(routing_x, routing_y, s=6)

    _draw_port_nodes(ax, port_nodes, pos)

    _finish_axes(eng, ax, title)
    plt.show()
