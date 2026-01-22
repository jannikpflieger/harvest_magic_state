"""
Text-based visualization functions for different layout types in the quantum computing lattice system.

This module provides ASCII-style text visualizations for both the lattice layout system
and the pure_msc_recreation layout system, highlighting different types of qubits.
"""

from typing import Dict, List, Optional, Tuple
import sys
import os

# Add the current directory to Python path to import from subdirectories
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from lattice.lattice_layout import Layout as LatticeLayout, AncillaState
from pure_msc_recreation.models import Layout as MSCLayout, CellType


def visualize_lattice_layout(layout: LatticeLayout, title: str = "Lattice Layout") -> str:
    """
    Create a text-based visualization of the lattice layout.
    
    Args:
        layout: The LatticeLayout object to visualize
        title: Title for the visualization
        
    Returns:
        A string containing the ASCII visualization
        
    Legend:
        D = Data qubit
        C = Ancilla (Cultivating)
        R = Ancilla (Ready Magic State)
        B = Ancilla (Routing Busy)
        . = Empty ancilla
    """
    lines = [f"\n{title}"]
    lines.append("=" * len(title))
    lines.append("")
    
    # Create header with column numbers
    header = "   "
    for x in range(layout.W):
        header += f"{x % 10}"
    lines.append(header)
    lines.append("")
    
    # Create each row
    for y in range(layout.H):
        row = f"{y:2} "
        for x in range(layout.W):
            coord = (x, y)
            
            if layout.is_data(coord):
                # Data qubit - show the qubit ID
                data_patch = layout.data_at[coord]
                row += "D"
            elif layout.is_ancilla(coord):
                # Ancilla qubit - show state
                if coord in layout.ancilla_state:
                    state = layout.ancilla_state[coord]
                    if state == AncillaState.CULTIVATING:
                        row += "C"
                    elif state == AncillaState.READY_MAGIC:
                        row += "R"
                    elif state == AncillaState.ROUTING_BUSY:
                        row += "B"
                    else:
                        row += "?"
                else:
                    row += "."
            else:
                row += " "
        
        lines.append(row)
    
    # Add legend
    lines.append("")
    lines.append("Legend:")
    lines.append("  D = Data qubit")
    lines.append("  C = Ancilla (Cultivating)")
    lines.append("  R = Ancilla (Ready Magic State)")
    lines.append("  B = Ancilla (Routing Busy)")
    lines.append("  . = Empty ancilla")
    
    # Add statistics
    data_count = len(layout.data_at)
    ancilla_count = 0
    cultivating_count = 0
    ready_count = 0
    busy_count = 0
    
    for y in range(layout.H):
        for x in range(layout.W):
            coord = (x, y)
            if layout.is_ancilla(coord):
                ancilla_count += 1
                if coord in layout.ancilla_state:
                    state = layout.ancilla_state[coord]
                    if state == AncillaState.CULTIVATING:
                        cultivating_count += 1
                    elif state == AncillaState.READY_MAGIC:
                        ready_count += 1
                    elif state == AncillaState.ROUTING_BUSY:
                        busy_count += 1
    
    lines.append("")
    lines.append("Statistics:")
    lines.append(f"  Total size: {layout.W} x {layout.H} = {layout.W * layout.H}")
    lines.append(f"  Data qubits: {data_count}")
    lines.append(f"  Ancilla qubits: {ancilla_count}")
    lines.append(f"    - Cultivating: {cultivating_count}")
    lines.append(f"    - Ready: {ready_count}")
    lines.append(f"    - Busy: {busy_count}")
    
    return "\n".join(lines)


def visualize_msc_layout(layout: MSCLayout, title: str = "MSC Layout") -> str:
    """
    Create a text-based visualization of the MSC recreation layout.
    
    Args:
        layout: The MSCLayout object to visualize
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
    Test function to demonstrate both visualization functions.
    Creates sample layouts and displays their text visualizations.
    """
    print("Testing Layout Visualizations")
    print("=" * 40)
    
    # Test MSC Layout visualization
    print("\n1. Testing MSC Layout Visualization:")
    print("-" * 35)
    
    try:
        from pure_msc_recreation.run import build_bus_layout, build_pure_magic_layout
        
        # Create a small bus layout
        bus_layout = build_bus_layout(w=12, h=6, n_data=5, lam=1.0, d=15, seed=42)
        print(visualize_msc_layout(bus_layout, "Bus Layout Example"))
        
        # Create a small pure magic layout
        pure_layout = build_pure_magic_layout(w=10, h=8, n_data=4, lam=1.0, d=15, seed=42)
        print(visualize_msc_layout(pure_layout, "Pure Magic Layout Example"))
        
    except ImportError as e:
        print(f"Could not test MSC layouts: {e}")
    
    # Test Lattice Layout visualization
    print("\n2. Testing Lattice Layout Visualization:")
    print("-" * 38)
    
    try:
        from lattice.lattice_layout import Layout, DataPatch, AncillaState
        
        # Create a simple lattice layout for demonstration
        width, height = 8, 6
        data_patches = {
            (2, 2): DataPatch(qid=0, pos=(2, 2)),
            (4, 2): DataPatch(qid=1, pos=(4, 2)),
            (6, 2): DataPatch(qid=2, pos=(6, 2)),
        }
        
        ancilla_states = {
            (1, 1): AncillaState.CULTIVATING,
            (3, 1): AncillaState.READY_MAGIC,
            (5, 1): AncillaState.ROUTING_BUSY,
            (1, 3): AncillaState.READY_MAGIC,
            (3, 3): AncillaState.CULTIVATING,
            (5, 3): AncillaState.READY_MAGIC,
        }
        
        cultivation_progress = {
            (1, 1): 2,
            (3, 3): 4,
        }
        
        lattice_layout = Layout(
            W=width,
            H=height,
            data_at=data_patches,
            ancilla_state=ancilla_states,
            cultivation_progress=cultivation_progress,
            cultivation_latency=5
        )
        
        print(visualize_lattice_layout(lattice_layout, "Sample Lattice Layout"))
        
    except ImportError as e:
        print(f"Could not test lattice layouts: {e}")
    
    print("\n" + "=" * 40)
    print("Visualization test completed!")


if __name__ == "__main__":
    test_visualizations()