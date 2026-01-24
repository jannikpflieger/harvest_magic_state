# layout_engine.py
# Cell-based lattice layout engine for lattice-surgery routing:
# - data patches occupy 1 cell and expose 4 ports (X on N/S, Z on E/W)
# - magic patches occupy 1 cell and expose 1 port (M) on a chosen side
# - builds a routing graph over FREE cells + explicit port nodes
# - visualizes layout and graph overlay with matplotlib

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Union, Set
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

Coord = Tuple[int, int]
Node = Union[Coord, str]  # routing cell (x,y) or port node id "P:<patch>:<side>:<type>"


# -------------------------
# Data structures
# -------------------------

@dataclass(frozen=True)
class SidePort:
    patch: str
    port_type: str   # "X", "Z", "M"
    side: str        # "N","S","E","W"

    @property
    def node_id(self) -> str:
        return f"P:{self.patch}:{self.side}:{self.port_type}"


@dataclass
class CellPatch:
    name: str
    kind: str            # "data" or "magic"
    cell: Coord          # occupies exactly one grid cell
    ports: List[SidePort] = field(default_factory=list)


# -------------------------
# Layout engine
# -------------------------

class LayoutEngine:
    def __init__(self, width: int, height: int):
        self.W = width
        self.H = height
        # "PATCH:<name>" or "BLOCKED"; missing => FREE
        self.occ: Dict[Coord, str] = {}
        self.patches: Dict[str, CellPatch] = {}

    # --- utilities ---
    def in_bounds(self, c: Coord) -> bool:
        x, y = c
        return 0 <= x < self.W and 0 <= y < self.H

    def is_free(self, c: Coord) -> bool:
        return self.occ.get(c, "FREE") == "FREE"

    def side_to_delta(self, side: str) -> Coord:
        return {"N": (0, -1), "S": (0, 1), "W": (-1, 0), "E": (1, 0)}[side]

    def neighbor_cell_for_port(self, patch_cell: Coord, side: str) -> Coord:
        dx, dy = self.side_to_delta(side)
        return (patch_cell[0] + dx, patch_cell[1] + dy)

    # --- placement ---
    def add_blocked(self, cells: Set[Coord]):
        for c in cells:
            if not self.in_bounds(c):
                raise ValueError(f"Blocked out of bounds: {c}")
            if c in self.occ and self.occ[c].startswith("PATCH:"):
                raise ValueError(f"Blocked overlaps patch at {c}")
            self.occ[c] = "BLOCKED"

    def add_patch(self, patch: CellPatch):
        if patch.name in self.patches:
            raise ValueError(f"Patch {patch.name} exists")
        if not self.in_bounds(patch.cell):
            raise ValueError(f"Patch {patch.name} out of bounds: {patch.cell}")
        if patch.cell in self.occ:
            raise ValueError(f"Cell {patch.cell} occupied by {self.occ[patch.cell]}")
        self.occ[patch.cell] = f"PATCH:{patch.name}"
        self.patches[patch.name] = patch

    # --- graph builder ---
    def build_routing_graph(self):
        """
        Returns:
          graph: Dict[node, List[(neighbor, weight)]]
          ports_by_patch: Dict[patch][port_type] -> List[port_node_id]
          pos: Dict[node] -> (x,y) float coords for visualization
          patch_used_by_port: Dict[port_node_id] -> patch_name  (useful for packing constraints)
        """
        graph: Dict[Node, List[Tuple[Node, int]]] = {}
        pos: Dict[Node, Tuple[float, float]] = {}
        patch_used_by_port: Dict[str, str] = {}

        # routing nodes = all FREE cells
        for x in range(self.W):
            for y in range(self.H):
                c = (x, y)
                if self.is_free(c):
                    graph[c] = []
                    pos[c] = (x + 0.5, y + 0.5)

        # routing edges (4-neighbor)
        for (x, y) in list(graph.keys()):
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                n = (x + dx, y + dy)
                if self.in_bounds(n) and self.is_free(n):
                    graph[(x, y)].append((n, 1))

        # port nodes
        ports_by_patch: Dict[str, Dict[str, List[str]]] = {}
        for p in self.patches.values():
            ports_by_patch[p.name] = {}
            px, py = p.cell

            for port in p.ports:
                pid = port.node_id
                patch_used_by_port[pid] = p.name
                ports_by_patch[p.name].setdefault(port.port_type, []).append(pid)
                graph.setdefault(pid, [])

                # port position slightly outside patch cell
                dx, dy = self.side_to_delta(port.side)
                pos[pid] = (px + 0.5 + 0.55 * dx, py + 0.5 + 0.55 * dy)

                # connect port to the adjacent FREE routing cell
                attach = self.neighbor_cell_for_port(p.cell, port.side)
                if self.in_bounds(attach) and self.is_free(attach):
                    graph[pid].append((attach, 0))
                    graph[attach].append((pid, 0))

        return graph, ports_by_patch, pos, patch_used_by_port

    # -------------------------
    # Visualization
    # -------------------------

    def visualize_layout(self, title="Layout", show_ports=True):
        """Draw patches (hatched), blocked cells (gray), and ports (colored markers)."""
        fig, ax = plt.subplots(figsize=(max(6, self.W / 3), max(4, self.H / 3)))

        # grid tiles
        for x in range(self.W):
            for y in range(self.H):
                c = (x, y)
                status = self.occ.get(c, "FREE")
                rect = plt.Rectangle((x, y), 1, 1, fill=True, edgecolor="k", linewidth=0.25)
                if status == "FREE":
                    rect.set_facecolor((1, 1, 1, 1))
                elif status == "BLOCKED":
                    rect.set_facecolor((0.9, 0.9, 0.9, 1))
                else:  # patch
                    rect.set_facecolor((1, 1, 1, 1))
                    rect.set_hatch("///")
                ax.add_patch(rect)

        marker_map = {"X": "o", "Z": "s", "M": "^"}
        color_map = {"X": "red", "Z": "blue", "M": "yellow"}

        for p in self.patches.values():
            x, y = p.cell
            ax.text(x + 0.5, y + 0.5, p.name, ha="center", va="center", fontsize=10)

            if show_ports:
                for port in p.ports:
                    dx, dy = self.side_to_delta(port.side)
                    px = x + 0.5 + 0.55 * dx
                    py = y + 0.5 + 0.55 * dy

                    mk = marker_map[port.port_type]  # raises if unknown -> good
                    col = color_map[port.port_type]

                    ax.scatter([px], [py], marker=mk, s=70, c=col, edgecolors="k", linewidths=0.6)

        ax.set_title(title)
        ax.set_xlim(0, self.W)
        ax.set_ylim(0, self.H)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_xticks(range(0, self.W + 1))
        ax.set_yticks(range(0, self.H + 1))
        plt.show()

    def visualize_graph(self, graph, pos, title="Routing graph overlay"):
        """Overlay the routing graph (edges + nodes) on the layout."""
        fig, ax = plt.subplots(figsize=(max(6, self.W / 3), max(4, self.H / 3)))

        # background tiles
        for x in range(self.W):
            for y in range(self.H):
                c = (x, y)
                status = self.occ.get(c, "FREE")
                rect = plt.Rectangle((x, y), 1, 1, fill=True, edgecolor="k", linewidth=0.2)
                if status == "FREE":
                    rect.set_facecolor((1, 1, 1, 1))
                elif status == "BLOCKED":
                    rect.set_facecolor((0.9, 0.9, 0.9, 1))
                else:
                    rect.set_facecolor((1, 1, 1, 1))
                    rect.set_hatch("///")
                ax.add_patch(rect)

        for p in self.patches.values():
            x, y = p.cell
            ax.text(x + 0.5, y + 0.5, p.name, ha="center", va="center", fontsize=9)

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

        # nodes
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
        color_map = {"X": "red", "Z": "blue", "M": "yellow"}

        for n in port_nodes:
            x, y = pos[n]
            t = n.split(":")[-1]
            ax.scatter(
                [x], [y],
                marker=marker_map.get(t, "o"),
                s=85,
                c=color_map.get(t, "k"),
                edgecolors="k",
                linewidths=0.6,
            )

        ax.set_title(title)
        ax.set_xlim(0, self.W)
        ax.set_ylim(0, self.H)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_xticks(range(0, self.W + 1))
        ax.set_yticks(range(0, self.H + 1))
        plt.show()


# -------------------------
# Helper constructors
# -------------------------

def data_patch_4ports(name: str, cell: Coord) -> CellPatch:
    """Data patch: X ports on N/S, Z ports on E/W."""
    return CellPatch(
        name=name,
        kind="data",
        cell=cell,
        ports=[
            SidePort(name, "X", "N"),
            SidePort(name, "X", "S"),
            SidePort(name, "Z", "E"),
            SidePort(name, "Z", "W"),
        ],
    )


def magic_patch_1port(name: str, cell: Coord, side: str = "W") -> CellPatch:
    """Magic patch: single basis-agnostic magic port type 'M'."""
    return CellPatch(
        name=name,
        kind="magic",
        cell=cell,
        ports=[SidePort(name, "M", side)],
    )


# -------------------------
# Demo / quick test
# -------------------------

if __name__ == "__main__":
    eng = LayoutEngine(4, 5)

    eng.add_patch(data_patch_4ports("d0", (1, 1)))
    eng.add_patch(data_patch_4ports("d1", (1, 3)))

    eng.add_patch(magic_patch_1port("m0", (3, 1), side="W"))

    # Example obstacle band (remove if you want empty workspace)
    #eng.add_blocked({(x, 7) for x in range(8, 16)})

    #eng.visualize_layout("Layout with X/Z/M ports (red/blue/yellow)")
    graph, ports_by_patch, pos, patch_used_by_port = eng.build_routing_graph()
    eng.visualize_graph(graph, pos, "Routing graph overlay (free-space graph + port nodes)")

    print("ports_by_patch =", ports_by_patch)
