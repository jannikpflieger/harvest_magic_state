#!/usr/bin/env python3
"""
Simple test script for quantum lattice layout functionality.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from lattice import (
    make_layout, 
    construct_lattice_layout, 
    create_default_lattice_layout,
    visualize_layout
)

def main():
    """Test basic lattice functionality."""
    print("QUANTUM LATTICE LAYOUT TESTS")
    print("=" * 60)
    
    # Test 1: Basic layout
    print("Test 1: Basic Layout (6 qubits)")
    layout1 = make_layout(n_qubits=6, W=12, H=8)
    print(f"Data qubits: {len(layout1.data_at)}")
    
    try:
        visualize_layout(layout1, "Basic Layout", save_path="test_basic.png")
        print("✅ Basic layout visualization saved")
    except Exception as e:
        print(f"❌ Visualization error: {e}")
    
    print("\n" + "-"*40 + "\n")
    
    # Test 2: Square lattice
    print("Test 2: Square Lattice (10x10)")
    layout2 = construct_lattice_layout('square', (10, 10))
    print(f"Data qubits: {len(layout2.data_at)}")
    
    try:
        visualize_layout(layout2, "Square Lattice", save_path="test_square.png")
        print("✅ Square lattice visualization saved")
    except Exception as e:
        print(f"❌ Visualization error: {e}")
    
    print("\n" + "-"*40 + "\n")
    
    # Test 3: Default layout
    print("Test 3: Default Layout (8 qubits)")
    layout3 = create_default_lattice_layout(8)
    print(f"Data qubits: {len(layout3.data_at)}")
    
    try:
        visualize_layout(layout3, "Default Layout", save_path="test_default.png")
        print("✅ Default layout visualization saved")
    except Exception as e:
        print(f"❌ Visualization error: {e}")
    
    print("\n" + "="*60)
    print("All tests completed!")
    print("Generated files: test_basic.png, test_square.png, test_default.png")

if __name__ == "__main__":
    main()