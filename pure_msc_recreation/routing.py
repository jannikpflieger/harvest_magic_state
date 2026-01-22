from __future__ import annotations
from collections import deque
from typing import Dict, List, Set, Tuple, Optional
from models import Coord, Layout

def neighbors(c: Coord, w: int, h: int) -> List[Coord]:
    x, y = c
    out = []
    if x > 0: out.append((x-1, y))
    if x+1 < w: out.append((x+1, y))
    if y > 0: out.append((x, y-1))
    if y+1 < h: out.append((x, y+1))
    return out

def bfs_shortest_path(layout: Layout, start: Coord, goal: Coord, allowed: Set[Coord]) -> Optional[List[Coord]]:
    # BFS on grid restricted to allowed coords
    if start == goal:
        return [start]
    q = deque([start])
    prev: Dict[Coord, Optional[Coord]] = {start: None}
    while q:
        cur = q.popleft()
        for nb in neighbors(cur, layout.w, layout.h):
            if nb not in allowed or nb in prev:
                continue
            prev[nb] = cur
            if nb == goal:
                # reconstruct
                path = [goal]
                p = cur
                while p is not None:
                    path.append(p)
                    p = prev[p]
                path.reverse()
                return path
            q.append(nb)
    return None

def approx_steiner_tree(layout: Layout, terminals: List[Coord], allowed: Set[Coord]) -> Optional[Set[Coord]]:
    # Greedy "connect closest terminal to current tree" heuristic
    if not terminals:
        return set()
    tree: Set[Coord] = {terminals[0]}
    remaining = terminals[1:]

    while remaining:
        best = None  # (path_len, term_idx, path_nodes)
        for i, t in enumerate(remaining):
            # connect t to *any* node already in tree: try closest by BFS to each tree node (cheap version)
            # For speed later: multi-source BFS from tree.
            for s in tree:
                p = bfs_shortest_path(layout, s, t, allowed)
                if p is None:
                    continue
                cand = (len(p), i, set(p))
                if best is None or cand[0] < best[0]:
                    best = cand
        if best is None:
            return None
        _, idx, path_nodes = best
        tree |= path_nodes
        remaining.pop(idx)

    return tree
