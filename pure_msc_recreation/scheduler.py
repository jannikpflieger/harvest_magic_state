from __future__ import annotations
from typing import Dict, List, Set, Tuple, Optional
import random
from models import Layout, PauliProduct, sample_cultivation_cycles, CellType
from routing import approx_steiner_tree

class Scheduler:
    def __init__(self, layout: Layout, lam: float, code_distance: int, seed: int = 0):
        self.layout = layout
        self.lam = lam
        self.code_distance = code_distance
        self.rng = random.Random(seed)
        self.time = 0
        self.done: Set[int] = set()

    def tick_cultivators(self, used_for_routing: Set[Tuple[int,int]], consumed_magic: Set[Tuple[int,int]], pure_magic: bool):
        # Update cultivation timers for all cultivators
        for coord, cult in self.layout.cultivators.items():
            if coord in consumed_magic:
                # consumed: restart next cycle
                cult.ready = False
                cult.remaining = sample_cultivation_cycles(self.lam, self.code_distance, self.rng)
            elif pure_magic and coord in used_for_routing:
                # Pure Magic: cancel cultivation when repurposed for routing, restart
                cult.ready = False
                cult.remaining = sample_cultivation_cycles(self.lam, self.code_distance, self.rng)
            else:
                cult.tick()

    def ready_products(self, products: Dict[int, PauliProduct]) -> List[PauliProduct]:
        out = []
        for p in products.values():
            if p.pid in self.done:
                continue
            if any(dep not in self.done for dep in p.deps):
                continue
            out.append(p)
        return out

    def ready_magic_cells(self) -> List[Tuple[int,int]]:
        return [c for c, cult in self.layout.cultivators.items() if cult.ready]

    def run(self, products: Dict[int, PauliProduct], pure_magic: bool) -> Tuple[int, int]:
        # returns (T_cycles, volume=N*T)
        N = self.layout.w * self.layout.h  # you can replace by “count of patches” if you prefer
        while len(self.done) < len(products):
            scheduled, used_nodes, consumed_magic = self.schedule_one_cycle(products, pure_magic)
            # mark scheduled complete
            for pid in scheduled:
                self.done.add(pid)
            # cultivation evolution
            self.tick_cultivators(used_for_routing=used_nodes, consumed_magic=consumed_magic, pure_magic=pure_magic)
            self.time += 1
        return self.time, N * self.time

    def schedule_one_cycle(self, products: Dict[int, PauliProduct], pure_magic: bool):
        avail_products = self.ready_products(products)

        # Available routing cells: all non-data cells that are not occupied
        allowed: Set[Tuple[int,int]] = set()
        for y in range(self.layout.h):
            for x in range(self.layout.w):
                cell = self.layout.grid[y][x]
                if cell.typ != CellType.DATA and not cell.occupied:
                    allowed.add((x,y))
        # Also allow terminals on data positions (so trees can attach)
        allowed |= set(self.layout.data_pos.values())

        scheduled: List[int] = []
        used_nodes: Set[Tuple[int,int]] = set()
        consumed_magic: Set[Tuple[int,int]] = set()

        # Greedy MINFIT packing
        while True:
            best = None  # (cost, pid, tree_nodes, magic_coord)
            ready_magic = self.ready_magic_cells()

            if not ready_magic:
                break

            for p in avail_products:
                # choose the best magic cell for this product (try a few)
                # terminals: all data coords involved + one magic coord
                data_terms = [self.layout.data_pos[t.qubit] for t in p.terms]

                # Try a small subset of magic cells to keep runtime sane
                for m in ready_magic[: min(10, len(ready_magic))]:
                    terminals = data_terms + [m]
                    tree = approx_steiner_tree(self.layout, terminals, allowed - used_nodes)
                    if tree is None:
                        continue
                    cost = len(tree)
                    cand = (cost, p.pid, tree, m)
                    if best is None or cand[0] < best[0]:
                        best = cand

            if best is None:
                break

            cost, pid, tree, m = best
            scheduled.append(pid)
            consumed_magic.add(m)
            used_nodes |= tree

            # remove scheduled product from avail list for this cycle
            avail_products = [p for p in avail_products if p.pid != pid]

        return scheduled, used_nodes, consumed_magic
