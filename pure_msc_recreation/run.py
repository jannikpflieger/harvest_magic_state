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

def visualize_msc_layout(layout: Layout, title: str = "MSC Layout") -> str:
    """
    Create a text-based visualization of the MSC recreation layout.
    
    Args:
        layout: The Layout object to visualize
        title: Title for the visualization
        
    Returns:
        A string containing the ASCII visualization
        
    Legend:
        D = Data qubit
        M = Magic cultivator
        A = Ancilla cultivator
        B = Bus cell
        X = Occupied cell
    """
    lines = [f"\n{title}"]
    lines.append("=" * len(title))
    lines.append("")
    
    # Create header with column numbers
    header = "   "
    for x in range(layout.w):
        header += f"{x % 10}"
    lines.append(header)
    lines.append("")
    
    # Create each row
    for y in range(layout.h):
        row = f"{y:2} "
        for x in range(layout.w):
            cell = layout.grid[y][x]
            
            if cell.typ == CellType.DATA:
                if cell.occupied:
                    row += "X"  # Occupied data
                else:
                    row += "D"  # Data qubit
            elif cell.typ == CellType.MAGIC:
                if cell.occupied:
                    row += "X"  # Occupied magic
                else:
                    # Show cultivation status if available
                    coord = (x, y)
                    if coord in layout.cultivators:
                        cultivator = layout.cultivators[coord]
                        if cultivator.ready:
                            row += "R"  # Ready magic
                        else:
                            row += "M"  # Cultivating magic
                    else:
                        row += "M"  # Magic cultivator
            elif cell.typ == CellType.ANCILLA:
                if cell.occupied:
                    row += "X"  # Occupied ancilla
                else:
                    # Show cultivation status if available
                    coord = (x, y)
                    if coord in layout.cultivators:
                        cultivator = layout.cultivators[coord]
                        if cultivator.ready:
                            row += "r"  # Ready ancilla (lowercase to distinguish from Magic)
                        else:
                            row += "A"  # Cultivating ancilla
                    else:
                        row += "A"  # Ancilla cultivator
            elif cell.typ == CellType.BUS:
                if cell.occupied:
                    row += "X"  # Occupied bus
                else:
                    row += "B"  # Bus cell
            else:
                row += "?"
        
        lines.append(row)
    
    # Add legend
    lines.append("")
    lines.append("Legend:")
    lines.append("  D = Data qubit")
    lines.append("  M = Magic cultivator (cultivating)")
    lines.append("  R = Magic cultivator (ready)")
    lines.append("  A = Ancilla cultivator (cultivating)")
    lines.append("  r = Ancilla cultivator (ready)")
    lines.append("  B = Bus cell")
    lines.append("  X = Occupied cell")
    
    # Add statistics
    data_count = 0
    magic_count = 0
    ancilla_count = 0
    bus_count = 0
    occupied_count = 0
    ready_cultivators = 0
    total_cultivators = len(layout.cultivators)
    
    for y in range(layout.h):
        for x in range(layout.w):
            cell = layout.grid[y][x]
            if cell.occupied:
                occupied_count += 1
            
            if cell.typ == CellType.DATA:
                data_count += 1
            elif cell.typ == CellType.MAGIC:
                magic_count += 1
            elif cell.typ == CellType.ANCILLA:
                ancilla_count += 1
            elif cell.typ == CellType.BUS:
                bus_count += 1
    
    # Count ready cultivators
    for cultivator in layout.cultivators.values():
        if cultivator.ready:
            ready_cultivators += 1
    
    lines.append("")
    lines.append("Statistics:")
    lines.append(f"  Total size: {layout.w} x {layout.h} = {layout.w * layout.h}")
    lines.append(f"  Data qubits: {data_count}")
    lines.append(f"  Magic cultivators: {magic_count}")
    lines.append(f"  Ancilla cultivators: {ancilla_count}")
    lines.append(f"  Bus cells: {bus_count}")
    lines.append(f"  Occupied cells: {occupied_count}")
    lines.append(f"  Total cultivators: {total_cultivators}")
    lines.append(f"  Ready cultivators: {ready_cultivators} / {total_cultivators}")
    
    return "\n".join(lines)


def test_visualizations():
    """
    Test function to demonstrate layout visualizations.
    Creates sample layouts and displays their text visualizations.
    """
    print("Testing Layout Visualizations")
    print("=" * 40)
    
    # Create a small bus layout
    bus_layout = build_bus_layout(w=12, h=6, n_data=5, lam=1.0, d=15, seed=42)
    print(visualize_msc_layout(bus_layout, "Bus Layout Example"))
    
    # Create a small pure magic layout
    pure_layout = build_pure_magic_layout(w=10, h=8, n_data=4, lam=1.0, d=15, seed=42)
    print(visualize_msc_layout(pure_layout, "Pure Magic Layout Example"))
    
    print("\n" + "=" * 40)
    print("Visualization test completed!")


def main():
    # ---- knobs you can change ----
    w, h = 5,4
    n_data = 3
    n_products = 1000
    avg_weight = 4      # average Pauli product weight (number of qubits touched)
    lam = 1.0           # cultivation rate (bigger -> faster)
    d = 15              # code distance scaling
    seed = 0
    # -----------------------------

    products = make_synthetic_products(n_qubits=n_data, n_products=n_products, avg_weight=avg_weight, seed=seed)

    bus_layout = build_bus_layout(w, h, n_data, lam, d, seed=seed)
    pure_layout = build_pure_magic_layout(w, h, n_data, lam, d, seed=seed)

    # Optional: Show layout visualizations (comment out for performance)
    print(visualize_msc_layout(bus_layout, "Bus Layout"))
    print(visualize_msc_layout(pure_layout, "Pure Magic Layout"))

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

def demo_visualization():
    """
    Simple demo function showing how to visualize different layouts.
    You can call this function to see examples of both layout types.
    """
    print("=== Layout Visualization Demo ===")
    
    # Create smaller layouts for demo
    print("\n1. Small Bus Layout (8x5 with 3 data qubits):")
    small_bus = build_bus_layout(w=8, h=5, n_data=3, lam=1.0, d=15, seed=123)
    print(visualize_msc_layout(small_bus, "Small Bus Layout"))
    
    print("\n2. Small Pure Magic Layout (7x6 with 3 data qubits):")
    small_pure = build_pure_magic_layout(w=7, h=6, n_data=3, lam=1.0, d=15, seed=123)
    print(visualize_msc_layout(small_pure, "Small Pure Magic Layout"))
    
    print("\n3. After some cultivation time simulation:")
    # Simulate some cultivation cycles to show different states
    for _ in range(3):
        for cultivator in small_pure.cultivators.values():
            cultivator.tick()
    
    print(visualize_msc_layout(small_pure, "After 3 Cultivation Cycles"))


if __name__ == "__main__":
    import sys
    
    # Check if user wants to run demo
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo_visualization()
    elif len(sys.argv) > 1 and sys.argv[1] == "test":
        test_visualizations()
    else:
        main()
