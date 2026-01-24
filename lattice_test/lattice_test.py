from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union, Set
import matplotlib.pyplot as plt

Coord = Tuple[int, int]
Node = Union[Coord, str]  # routing cell (x,y) or port node id "P:patch:side"

@dataclass(frozen=True)
class SidePort:
    patch: str
    port_type: str   # "X" or "Z" or "IN"/"OUT"
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

class LayoutEngine:
    def __init__(self, width: int, height: int):
        self.W = width
        self.H = height
        self.occ: Dict[Coord, str] = {}      # "PATCH:<name>" or "BLOCKED"; missing => FREE
        self.patches: Dict[str, CellPatch] = {}

    def in_bounds(self, c: Coord) -> bool:
        x, y = c
        return 0 <= x < self.W and 0 <= y < self.H

    def add_blocked(self, cells: Set[Coord]):
        for c in cells:
            if not self.in_bounds(c):
                raise ValueError(f"Blocked cell out of bounds: {c}")
            if c in self.occ and self.occ[c].startswith("PATCH:"):
                raise ValueError(f"Blocked overlaps patch at {c}")
            self.occ[c] = "BLOCKED"

    def add_patch(self, patch: CellPatch):
        if patch.name in self.patches:
            raise ValueError(f"Patch {patch.name} exists")
        if not self.in_bounds(patch.cell):
            raise ValueError(f"Patch {patch.name} out of bounds: {patch.cell}")
        if patch.cell in self.occ:
            raise ValueError(f"Cell {patch.cell} already occupied by {self.occ[patch.cell]}")
        self.occ[patch.cell] = f"PATCH:{patch.name}"
        self.patches[patch.name] = patch

    def is_free(self, c: Coord) -> bool:
        return self.occ.get(c, "FREE") == "FREE"

    def side_to_delta(self, side: str) -> Coord:
        return {"N": (0, -1), "S": (0, 1), "W": (-1, 0), "E": (1, 0)}[side]

    def neighbor_cell_for_port(self, patch_cell: Coord, side: str) -> Coord:
        dx, dy = self.side_to_delta(side)
        return (patch_cell[0] + dx, patch_cell[1] + dy)

    def build_routing_graph(self):
        """
        Nodes:
          - Free routing cells (x,y)
          - Port nodes "P:<patch>:<side>:<type>"
        Edges:
          - 4-neighbor between free routing cells (weight 1)
          - Port node <-> adjacent free routing cell (weight 0)
        """
        graph: Dict[Node, List[Tuple[Node, int]]] = {}

        # add routing nodes
        for x in range(self.W):
            for y in range(self.H):
                c = (x, y)
                if self.is_free(c):
                    graph[c] = []

        # connect routing nodes (4-neighbor)
        for (x, y) in list(graph.keys()):
            if not isinstance((x, y), tuple):
                continue
            for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                n = (x + dx, y + dy)
                if self.in_bounds(n) and self.is_free(n):
                    graph[(x, y)].append((n, 1))

        # add ports
        ports_by_patch: Dict[str, Dict[str, List[str]]] = {}
        for p in self.patches.values():
            ports_by_patch[p.name] = {}
            for port in p.ports:
                pid = port.node_id
                ports_by_patch[p.name].setdefault(port.port_type, []).append(pid)
                graph.setdefault(pid, [])

                attach = self.neighbor_cell_for_port(p.cell, port.side)
                if self.in_bounds(attach) and self.is_free(attach):
                    graph[pid].append((attach, 0))
                    graph[attach].append((pid, 0))
                # else: port exists but currently not attachable (blocked / out of bounds / patch)

        return graph, ports_by_patch

    # ---------- Visualization ----------
    def visualize(self, title="Layout", show_ports=True, show_grid=True):
        fig, ax = plt.subplots(figsize=(max(6, self.W/3), max(4, self.H/3)))

        # draw base grid
        for x in range(self.W):
            for y in range(self.H):
                c = (x, y)
                status = self.occ.get(c, "FREE")
                rect = plt.Rectangle(
                    (x, y), 1, 1,
                    fill=True,
                    edgecolor='k' if show_grid else 'none',
                    linewidth=0.3
                )
                if status == "FREE":
                    rect.set_facecolor((1,1,1,1))
                elif status == "BLOCKED":
                    rect.set_facecolor((0.9,0.9,0.9,1))
                elif status.startswith("PATCH:"):
                    rect.set_facecolor((1,1,1,1))
                    rect.set_hatch("///")
                ax.add_patch(rect)

        # label patches and draw ports as single markers per side
        marker_map = {"X": "o", "Z": "s", "M": "D"}
        color_map  = {"X": "red", "Z": "blue", "M": "yellow"} 
        for p in self.patches.values():
            x, y = p.cell
            ax.text(x + 0.5, y + 0.5, p.name, ha="center", va="center", fontsize=10)

            if show_ports:
                # place port markers slightly outside the patch cell boundary
                for port in p.ports:
                    dx, dy = self.side_to_delta(port.side)
                    # marker center: offset 0.5 to cell center, then push 0.55 toward that side
                    px = x + 0.5 + 0.55 * dx
                    py = y + 0.5 + 0.55 * dy
                    mk = marker_map[port.port_type]
                    col = color_map.get(port.port_type, "k")
                    ax.scatter([px], [py], marker=mk, s=70, c=col, edgecolors="k", linewidths=0.6)


        ax.set_title(title)
        ax.set_xlim(0, self.W)
        ax.set_ylim(0, self.H)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_xticks(range(0, self.W+1))
        ax.set_yticks(range(0, self.H+1))
        if not show_grid:
            ax.set_xticks([])
            ax.set_yticks([])
        plt.show()

# ---- helper constructors ----

def data_patch_4ports(name: str, cell: Coord) -> CellPatch:
    # X opposite: N,S ; Z opposite: E,W
    ports = [
        SidePort(name, "X", "N"),
        SidePort(name, "X", "S"),
        SidePort(name, "Z", "E"),
        SidePort(name, "Z", "W"),
    ]
    return CellPatch(name=name, kind="data", cell=cell, ports=ports)

def magic_patch_1port(name: str, cell: Coord, side: str = "W") -> CellPatch:
    ports = [SidePort(name, "M", side)]
    return CellPatch(name=name, kind="magic", cell=cell, ports=ports)

def magic_patch_default(name: str, cell: Coord) -> CellPatch:
    # Example: let magic expose one "IN" port on the west side by default (adjust as you like)
    ports = [SidePort(name, "M", "W")]
    return CellPatch(name=name, kind="magic", cell=cell, ports=ports)

# ---- demo ----
if __name__ == "__main__":
    eng = LayoutEngine(20, 12)
    eng.add_patch(data_patch_4ports("d0", (5, 4)))
    eng.add_patch(data_patch_4ports("d1", (5, 8)))
    eng.add_patch(magic_patch_default("m0", (15, 4)))
    #eng.add_blocked({(x, 6) for x in range(7, 14)})  # obstacle band
    eng.visualize("Cell patches with 4 side-ports")

    graph, ports = eng.build_routing_graph()
    # ports["d0"]["X"] -> list of 2 port node IDs (N,S); ports["d0"]["Z"] -> 2 (E,W)
