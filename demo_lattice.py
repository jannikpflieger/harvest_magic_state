#!/usr/bin/env python3
"""
Simple demo script for quantum lattice layout and visualization.

This script demonstrates the basic functionality of creating and visualizing
quantum lattice layouts with data qubits (blue) and ancilla qubits (white).
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from lattice import (
    make_layout, 
    visualize_layout
)

def main():
    """Simple demo of lattice functionality."""
    print("🔬 Quantum Lattice Layout Demo")
    print("=" * 50)
    
    # Create a lattice with 6 data qubits
    print("Creating lattice with 6 data qubits...")
    layout = make_layout(n_qubits=6)
    
    # Create visualization
    print("Creating visualization...")
    try:
        visualize_layout(layout, 
                        title="Quantum Lattice Layout",
                        save_path="simple_layout.png")
        print("✅ Visualization saved as 'simple_layout.png'")
    except Exception as e:
        print(f"❌ Visualization error: {e}")
    
    print("\nDemo completed!")

if __name__ == "__main__":
    main()