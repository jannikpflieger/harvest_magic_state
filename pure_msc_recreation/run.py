# run_demo.py
from models import Cell, CellType, Cultivator, Layout, sample_cultivation_cycles
from scheduler import Scheduler
from experiments import make_synthetic_products
import random

def build_bus_layout(w: int, h: int, n_data: int, lam: float, d: int, seed: int = 0) -> Layout:
    rng = random.Random(seed)

    # simple placement: data in the middle row(s), magic on boundary, bus everywhere else
    grid = [[Cell(CellType.BUS) for _ in range(w)] for _ in range(h)]
    data_pos = {}
    cultivators = {}

    # place data patches
    x0 = (w - n_data) // 2
    y0 = h // 2
    for q in range(n_data):
        x = x0 + q
        y = y0
        grid[y][x] = Cell(CellType.DATA)
        data_pos[q] = (x, y)

    # magic cultivators on top boundary (excluding data columns if they collide)
    for x in range(w):
        if grid[0][x].typ == CellType.DATA:
            continue
        grid[0][x] = Cell(CellType.MAGIC)
        cultivators[(x, 0)] = Cultivator(
            coord=(x, 0),
            remaining=sample_cultivation_cycles(lam, d, rng),
            ready=False
        )

    return Layout(w=w, h=h, grid=grid, data_pos=data_pos, cultivators=cultivators)

def build_pure_magic_layout(w: int, h: int, n_data: int, lam: float, d: int, seed: int = 0) -> Layout:
    rng = random.Random(seed)

    # ancilla everywhere, data in middle
    grid = [[Cell(CellType.ANCILLA) for _ in range(w)] for _ in range(h)]
    data_pos = {}
    cultivators = {}

    # place data
    x0 = (w - n_data) // 2
    y0 = h // 2
    for q in range(n_data):
        x = x0 + q
        y = y0
        grid[y][x] = Cell(CellType.DATA)
        data_pos[q] = (x, y)

    # every non-data cell is a cultivator by default (Pure Magic)
    for y in range(h):
        for x in range(w):
            if grid[y][x].typ != CellType.DATA:
                cultivators[(x, y)] = Cultivator(
                    coord=(x, y),
                    remaining=sample_cultivation_cycles(lam, d, rng),
                    ready=False
                )

    return Layout(w=w, h=h, grid=grid, data_pos=data_pos, cultivators=cultivators)

def main():
    # ---- knobs you can change ----
    w, h = 20, 10
    n_data = 10
    n_products = 300
    avg_weight = 3      # average Pauli product weight (number of qubits touched)
    lam = 1.0           # cultivation rate (bigger -> faster)
    d = 15              # code distance scaling
    seed = 0
    # -----------------------------

    products = make_synthetic_products(n_qubits=n_data, n_products=n_products, avg_weight=avg_weight, seed=seed)

    bus_layout = build_bus_layout(w, h, n_data, lam, d, seed=seed)
    pure_layout = build_pure_magic_layout(w, h, n_data, lam, d, seed=seed)

    bus_sched = Scheduler(bus_layout, lam=lam, code_distance=d, seed=seed)
    pure_sched = Scheduler(pure_layout, lam=lam, code_distance=d, seed=seed)

    T_bus, V_bus = bus_sched.run(products, pure_magic=False)

    # IMPORTANT: regenerate products because Scheduler mutates "done" state internally, but
    # products themselves are immutable; easiest is just reload same synthetic instance:
    products2 = make_synthetic_products(n_qubits=n_data, n_products=n_products, avg_weight=avg_weight, seed=seed)
    T_pure, V_pure = pure_sched.run(products2, pure_magic=True)

    print("=== Results ===")
    print(f"Bus:       T = {T_bus} cycles,  V = {V_bus}")
    print(f"PureMagic: T = {T_pure} cycles, V = {V_pure}")
    print(f"Speedup (T): {T_bus / T_pure:.3f}x")
    print(f"Volume ratio (V_bus / V_pure): {V_bus / V_pure:.3f}x")

if __name__ == "__main__":
    main()
