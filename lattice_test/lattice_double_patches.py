# lattice_engine_doublepatches.py
# Layout engine for lattice-surgery routing with:
#   - single-cell data patches (1 qubit) with 4 ports (X opposite, Z opposite)
#   - wide 1-qubit patches occupying 2 cells ("2-tile 1-qubit"), with 2 ports on long sides
#   - 2-qubit paired patches occupying 2 cells ("2-tile 2-qubit"), with alternating X/Z ports around perimeter
#   - magic patches with a single basis-agnostic "M" port
#
# Ports are edge-terminals:
#   - Each port is a *single node*.
#   - A port can connect to multiple routing cells if it spans multiple boundary cells.
#   - For "two ports on a long side" we model two separate port nodes, each anchored to one boundary cell.

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Union, Set
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

Coord = Tuple[int, int]
Node = Union[Coord, str]


# -------------------------
# Data structures
# -------------------------

@dataclass(frozen=True)
class EdgePort:
    patch: str
    port_type: str   # "X", "Z", "M"
    side: str        # "N","S","E","W"
    port_id: str     # unique within patch, e.g. "N0","N1","W","E","M_W"
    anchors: Tuple[Coord, ...]  # footprint cells that this port sits on along that side

    @property
    def node_id(self) -> str:
        # parseable: last field is the type
        return f"P:{self.patch}:{self.port_id}:{self.port_type}"


@dataclass
class Patch:
    name: str
    kind: str
    cells: Set[Coord]
    ports: List[EdgePort] = field(default_factory=list)


# -------------------------
# Layout Engine
# -------------------------

class LayoutEngine:
    def __init__(self, width: int, height: int):
        self.W = width
        self.H = height
        # "PATCH:<name>" or "BLOCKED"; missing => FREE
        self.occ: Dict[Coord, str] = {}
        self.patches: Dict[str, Patch] = {}

    # ---- utilities ----
    def in_bounds(self, c: Coord) -> bool:
        x, y = c
        return 0 <= x < self.W and 0 <= y < self.H

    def is_free(self, c: Coord) -> bool:
        return self.occ.get(c, "FREE") == "FREE"

    def side_to_delta(self, side: str) -> Coord:
        return {"N": (0, -1), "S": (0, 1), "W": (-1, 0), "E": (1, 0)}[side]

    # ---- placement ----
    def add_blocked(self, cells: Set[Coord]):
        for c in cells:
            if not self.in_bounds(c):
                raise ValueError(f"Blocked out of bounds: {c}")
            if c in self.occ and self.occ[c].startswith("PATCH:"):
                raise ValueError(f"Blocked overlaps patch at {c}")
            self.occ[c] = "BLOCKED"

    def add_patch(self, patch: Patch):
        if patch.name in self.patches:
            raise ValueError(f"Patch {patch.name} exists")

        for c in patch.cells:
            if not self.in_bounds(c):
                raise ValueError(f"Patch {patch.name} out of bounds cell: {c}")
            if c in self.occ:
                raise ValueError(f"Cell {c} already occupied by {self.occ[c]}")

        for c in patch.cells:
            self.occ[c] = f"PATCH:{patch.name}"
        self.patches[patch.name] = patch

    # ---- geometry helpers ----
    def _patch_centroid(self, footprint: Set[Coord]) -> Tuple[float, float]:
        xs = [x + 0.5 for x, _ in footprint]
        ys = [y + 0.5 for _, y in footprint]
        return sum(xs)/len(xs), sum(ys)/len(ys)

    def _port_position(self, port: EdgePort) -> Tuple[float, float]:
        xs = [x + 0.5 for x, _ in port.anchors]
        ys = [y + 0.5 for _, y in port.anchors]
        cx, cy = sum(xs)/len(xs), sum(ys)/len(ys)
        dx, dy = self.side_to_delta(port.side)
        return cx + 0.55*dx, cy + 0.55*dy

    def _attach_cells_for_port(self, footprint: Set[Coord], port: EdgePort) -> List[Coord]:
        dx, dy = self.side_to_delta(port.side)
        out = []
        for (x, y) in port.anchors:
            a = (x + dx, y + dy)
            # ignore anchors whose neighbor is inside footprint (internal boundary)
            if a not in footprint:
                out.append(a)
        return out

    # ---- graph builder ----
    def build_routing_graph(self):
        """
        Returns:
          graph: Dict[node, List[(neighbor, weight)]]
          ports_by_patch: patch -> port_type -> list of port node IDs
          pos: node -> (x,y) float coords
          patch_used_by_port: port_node -> patch_name
        """
        graph: Dict[Node, List[Tuple[Node, int]]] = {}
        pos: Dict[Node, Tuple[float, float]] = {}
        patch_used_by_port: Dict[str, str] = {}

        # routing nodes
        for x in range(self.W):
            for y in range(self.H):
                c = (x, y)
                if self.is_free(c):
                    graph[c] = []
                    pos[c] = (x + 0.5, y + 0.5)

        # routing edges (4-neighbor)
        for (x, y) in list(graph.keys()):
            for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                n = (x + dx, y + dy)
                if self.in_bounds(n) and self.is_free(n):
                    graph[(x, y)].append((n, 1))

        # ports
        ports_by_patch: Dict[str, Dict[str, List[str]]] = {}
        for p in self.patches.values():
            ports_by_patch[p.name] = {}
            for port in p.ports:
                pid = port.node_id
                patch_used_by_port[pid] = p.name
                ports_by_patch[p.name].setdefault(port.port_type, []).append(pid)
                graph.setdefault(pid, [])
                pos[pid] = self._port_position(port)

                for a in self._attach_cells_for_port(p.cells, port):
                    if self.in_bounds(a) and self.is_free(a):
                        graph[pid].append((a, 0))
                        graph[a].append((pid, 0))
        
        graph, ports_by_patch, pos, patch_used_by_port = self._prune_degree0_nodes(graph, ports_by_patch, pos, patch_used_by_port)

        return graph, ports_by_patch, pos, patch_used_by_port

    def _prune_degree0_nodes(self, graph, ports_by_patch, pos, patch_used_by_port):
        """
        Remove nodes u where graph[u] has no neighbors (degree 0).
        Also prunes ports_by_patch / pos / patch_used_by_port accordingly.
        """
        keep = {u for u, nbrs in graph.items() if len(nbrs) > 0}

        # prune graph + edges
        new_graph = {}
        for u, nbrs in graph.items():
            if u not in keep:
                continue
            new_graph[u] = [(v, w) for (v, w) in nbrs if v in keep]

        # prune pos
        if pos is not None:
            pos = {u: pos[u] for u in keep if u in pos}

        # prune patch_used_by_port
        if patch_used_by_port is not None:
            patch_used_by_port = {u: patch_used_by_port[u] for u in keep if u in patch_used_by_port}

        # prune ports_by_patch
        if ports_by_patch is not None:
            for p in list(ports_by_patch.keys()):
                for t in list(ports_by_patch[p].keys()):
                    ports_by_patch[p][t] = [pid for pid in ports_by_patch[p][t] if pid in keep]
                    if not ports_by_patch[p][t]:
                        del ports_by_patch[p][t]
                if not ports_by_patch[p]:
                    del ports_by_patch[p]

        return new_graph, ports_by_patch, pos, patch_used_by_port

    # ---- visualization ----
    def visualize_layout(self, title="Layout", show_ports=True):
        fig, ax = plt.subplots(figsize=(max(6, self.W/3), max(4, self.H/3)))

        for x in range(self.W):
            for y in range(self.H):
                c = (x, y)
                status = self.occ.get(c, "FREE")
                rect = plt.Rectangle((x, y), 1, 1, fill=True, edgecolor="k", linewidth=0.25)
                if status == "FREE":
                    rect.set_facecolor((1,1,1,1))
                elif status == "BLOCKED":
                    rect.set_facecolor((0.9,0.9,0.9,1))
                else:
                    rect.set_facecolor((0.75,0.75,0.75,1))
                ax.add_patch(rect)

        marker_map = {"X": "o", "Z": "s", "M": "^"}
        color_map  = {"X": "red", "Z": "blue", "M": "yellow"}

        for p in self.patches.values():
            cx, cy = self._patch_centroid(p.cells)
            ax.text(cx, cy, p.name, ha="center", va="center", fontsize=10)

            if show_ports:
                for port in p.ports:
                    px, py = self._port_position(port)
                    ax.scatter([px], [py],
                               marker=marker_map.get(port.port_type, "o"),
                               s=70,
                               c=color_map.get(port.port_type, "k"),
                               edgecolors="k",
                               linewidths=0.6)

        ax.set_title(title)
        ax.set_xlim(0, self.W)
        ax.set_ylim(0, self.H)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_xticks(range(0, self.W+1))
        ax.set_yticks(range(0, self.H+1))
        plt.show()

    def visualize_graph(self, graph, pos, title="Routing graph overlay"):
        fig, ax = plt.subplots(figsize=(max(6, self.W/3), max(4, self.H/3)))

        for x in range(self.W):
            for y in range(self.H):
                c = (x, y)
                status = self.occ.get(c, "FREE")
                rect = plt.Rectangle((x, y), 1, 1, fill=True, edgecolor="k", linewidth=0.2)
                if status == "FREE":
                    rect.set_facecolor((1,1,1,1))
                elif status == "BLOCKED":
                    rect.set_facecolor((0.9,0.9,0.9,1))
                else:
                    rect.set_facecolor((0.75,0.75,0.75,1))
                ax.add_patch(rect)

        for p in self.patches.values():
            cx, cy = self._patch_centroid(p.cells)
            ax.text(cx, cy, p.name, ha="center", va="center", fontsize=9)

        # unique edges
        segments = []
        seen = set()
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

        marker_map = {"X": "o", "Z": "s", "M": "^"}
        color_map  = {"X": "red", "Z": "blue", "M": "yellow"}
        for n in port_nodes:
            x, y = pos[n]
            t = n.split(":")[-1]  # port_type
            ax.scatter([x], [y],
                       marker=marker_map.get(t, "o"),
                       s=85,
                       c=color_map.get(t, "k"),
                       edgecolors="k",
                       linewidths=0.6)

        ax.set_title(title)
        ax.set_xlim(0, self.W)
        ax.set_ylim(0, self.H)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_xticks(range(0, self.W+1))
        ax.set_yticks(range(0, self.H+1))
        plt.show()


# -------------------------
# Patch constructors
# -------------------------

def _swap_type(t: str, swap_xz: bool) -> str:
    if not swap_xz:
        return t
    if t == "X":
        return "Z"
    if t == "Z":
        return "X"
    return t

def data_patch_1cell(name: str, cell: Coord, swap_xz: bool = False) -> Patch:
    """Single-cell data patch (1 qubit). Default: X on N/S, Z on E/W. swap_xz swaps X<->Z."""
    ports = [
        EdgePort(name, _swap_type("X", swap_xz), "N", "N", (cell,)),
        EdgePort(name, _swap_type("X", swap_xz), "S", "S", (cell,)),
        EdgePort(name, _swap_type("Z", swap_xz), "E", "E", (cell,)),
        EdgePort(name, _swap_type("Z", swap_xz), "W", "W", (cell,)),
    ]
    return Patch(name=name, kind="data_1cell", cells={cell}, ports=ports)

def magic_patch_1cell(name: str, cell: Coord, side: str = "W") -> Patch:
    """Magic patch with a single basis-agnostic M-port."""
    ports = [EdgePort(name, "M", side, f"M_{side}", (cell,))]
    return Patch(name=name, kind="magic", cells={cell}, ports=ports)

def wide_patch_1q_2cells(name: str, origin: Coord, orientation: str = "H", swap_xz: bool = False) -> Patch:
    """
    Type-1 double patch: 1 qubit occupying 2 tiles.

    Horizontal (H): cells (x,y) and (x+1,y)
      - short sides (W/E): 1 port each (default X)
      - long sides (N/S): 2 ports each (default Z), one per tile:
          top: N0 (left tile), N1 (right tile)
          bot: S0 (left tile), S1 (right tile)

    Vertical (V): cells (x,y) and (x,y+1)
      - short sides (N/S): 1 port each (default X)
      - long sides (W/E): 2 ports each (default Z), one per tile:
          left:  W0 (top tile), W1 (bottom tile)
          right: E0 (top tile), E1 (bottom tile)

    swap_xz swaps X<->Z.
    """
    x, y = origin
    ori = orientation.upper()

    if ori == "H":
        c0 = (x, y)       # left
        c1 = (x + 1, y)   # right
        cells = {c0, c1}
        ports = [
            # short sides (default X)
            EdgePort(name, _swap_type("X", swap_xz), "W", "W", (c0,)),
            EdgePort(name, _swap_type("X", swap_xz), "E", "E", (c1,)),
            # long top (default Z)
            EdgePort(name, _swap_type("Z", swap_xz), "N", "N0", (c0,)),
            EdgePort(name, _swap_type("Z", swap_xz), "N", "N1", (c1,)),
            # long bottom (default Z)
            EdgePort(name, _swap_type("Z", swap_xz), "S", "S0", (c0,)),
            EdgePort(name, _swap_type("Z", swap_xz), "S", "S1", (c1,)),
        ]

    elif ori == "V":
        c0 = (x, y)       # top
        c1 = (x, y + 1)   # bottom
        cells = {c0, c1}
        ports = [
            # short sides (default X)
            EdgePort(name, _swap_type("X", swap_xz), "N", "N", (c0,)),
            EdgePort(name, _swap_type("X", swap_xz), "S", "S", (c1,)),
            # long left (default Z)
            EdgePort(name, _swap_type("Z", swap_xz), "W", "W0", (c0,)),
            EdgePort(name, _swap_type("Z", swap_xz), "W", "W1", (c1,)),
            # long right (default Z)
            EdgePort(name, _swap_type("Z", swap_xz), "E", "E0", (c0,)),
            EdgePort(name, _swap_type("Z", swap_xz), "E", "E1", (c1,)),
        ]
    else:
        raise ValueError("orientation must be 'H' or 'V'")

    return Patch(name=name, kind="wide_1q_2cells", cells=cells, ports=ports)

def paired_patches_2q_alternating(
    name_a: str,
    name_b: str,
    origin: Coord,
    orientation: str = "H",
    start_with: str = "X",
    swap_xz: bool = False,
) -> List[Patch]:
    """
    Type-2 double patch: 2 tiles hosting 2 qubits (two patches), with alternating X/Z ports around the *outer perimeter*.

    Horizontal (H): A=left tile, B=right tile
      perimeter segments (6) order:
        1) A.N, 2) B.N, 3) B.E, 4) B.S, 5) A.S, 6) A.W
      internal edges omitted: A.E and B.W

    Vertical (V): A=top tile, B=bottom tile
      perimeter segments (6) order:
        1) A.N, 2) A.E, 3) B.E, 4) B.S, 5) B.W, 6) A.W
      internal edges omitted: A.S and B.N

    start_with chooses the first segment type ('X' or 'Z'), then alternates.
    swap_xz swaps X<->Z after alternation.
    """
    start = start_with.upper()
    if start not in ("X", "Z"):
        raise ValueError("start_with must be 'X' or 'Z'")
    other = "Z" if start == "X" else "X"
    seq = [start, other, start, other, start, other]
    seq = [_swap_type(t, swap_xz) for t in seq]

    x, y = origin
    ori = orientation.upper()

    if ori == "H":
        a_cell = (x, y)         # left
        b_cell = (x + 1, y)     # right
        segs = [
            (name_a, a_cell, "N", "N"),  # 1
            (name_b, b_cell, "N", "N"),  # 2
            (name_b, b_cell, "E", "E"),  # 3
            (name_b, b_cell, "S", "S"),  # 4
            (name_a, a_cell, "S", "S"),  # 5
            (name_a, a_cell, "W", "W"),  # 6
        ]
    elif ori == "V":
        a_cell = (x, y)         # top
        b_cell = (x, y + 1)     # bottom
        segs = [
            (name_a, a_cell, "N", "N"),  # 1
            (name_a, a_cell, "E", "E"),  # 2
            (name_b, b_cell, "E", "E"),  # 3
            (name_b, b_cell, "S", "S"),  # 4
            (name_b, b_cell, "W", "W"),  # 5
            (name_a, a_cell, "W", "W"),  # 6
        ]
    else:
        raise ValueError("orientation must be 'H' or 'V'")

    ports_a: List[EdgePort] = []
    ports_b: List[EdgePort] = []
    for t, (pname, cell, side, pid) in zip(seq, segs):
        ep = EdgePort(pname, t, side, pid, (cell,))
        if pname == name_a:
            ports_a.append(ep)
        else:
            ports_b.append(ep)

    # each qubit is its own patch (as you want: 1 patch == 1 qubit)
    return [
        Patch(name=name_a, kind="paired_2q", cells={a_cell}, ports=ports_a),
        Patch(name=name_b, kind="paired_2q", cells={b_cell}, ports=ports_b),
    ]


# -------------------------
# Demo / Visual check
# -------------------------

if __name__ == "__main__":
    eng = LayoutEngine(18, 10)

    # Single-cell data patches (one rotated)
    eng.add_patch(data_patch_1cell("d0", (2, 2), swap_xz=False))
    eng.add_patch(data_patch_1cell("d1", (2, 7), swap_xz=True))

    # Type-1: 1-qubit wide patches (2 tiles)
    eng.add_patch(wide_patch_1q_2cells("wH", origin=(6, 2), orientation="H", swap_xz=False))
    eng.add_patch(wide_patch_1q_2cells("wV", origin=(6, 6), orientation="V", swap_xz=False))

    # Type-2: 2-qubit paired patches with alternating perimeter ports
    for p in paired_patches_2q_alternating("pA", "pB", origin=(11, 2), orientation="H", start_with="X", swap_xz=False):
        eng.add_patch(p)
    for p in paired_patches_2q_alternating("qA", "qB", origin=(12, 7), orientation="V", start_with="Z", swap_xz=False):
        eng.add_patch(p)

    # Magic patch
    eng.add_patch(magic_patch_1cell("m0", (16, 1), side="W"))

    eng.visualize_layout("Double patch variants: wide 1Q (2 tiles) + paired 2Q alternating")

    graph, ports_by_patch, pos, patch_used_by_port = eng.build_routing_graph()
    eng.visualize_graph(graph, pos, "Routing graph overlay")

    print("ports_by_patch =", ports_by_patch)
