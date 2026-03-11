#!/usr/bin/env python3
"""
Random Circuit Generator

This script generates randomized benchmark circuits with specified qubit counts and depths.
The generated circuits are saved as QASM files in the benchmark_circuits/qasm/random_circuits directory.

Usage:
    python generate_random_circuits.py
    
The script generates circuits for 50, 60, 70, 80, 90, and 100 qubits with depth 100.
For each configuration, multiple circuit variations are generated with different random seeds.
"""

import os
import sys
from pathlib import Path
from circuit_to_pbc_dag import create_random_circuit
from qiskit.qasm2 import dump as qasm_dump

def generate_random_circuit_suite(qubit_counts=None, depth=100, circuits_per_size=5):
    """
    Generate a suite of random circuits with different qubit counts and depths.
    
    Args:
        qubit_counts (list): List of qubit counts to generate circuits for
        depth (int): Circuit depth for all generated circuits
        circuits_per_size (int): Number of circuit variations to generate per qubit count
    """
    if qubit_counts is None:
        qubit_counts = [50, 60, 70, 80, 90, 100]
    
    # Get the directory path relative to this script
    script_dir = Path(__file__).parent
    output_dir = script_dir / "benchmark_circuits" / "qasm" / "random_circuits"
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating random circuits...")
    print(f"Qubit counts: {qubit_counts}")
    print(f"Depth: {depth}")
    print(f"Circuits per size: {circuits_per_size}")
    print(f"Output directory: {output_dir}")
    print("-" * 50)
    
    total_circuits = len(qubit_counts) * circuits_per_size
    current_circuit = 0
    
    for num_qubits in qubit_counts:
        print(f"\nGenerating circuits for {num_qubits} qubits...")
        
        for circuit_idx in range(circuits_per_size):
            current_circuit += 1
            
            # Use circuit index as seed for reproducibility
            seed = circuit_idx + 1
            
            # Generate the random circuit
            try:
                circuit = create_random_circuit(num_qubits, depth, seed=seed)
                
                # Create filename
                filename = f"random_{num_qubits}q_depth{depth}_seed{seed}.qasm"
                filepath = output_dir / filename
                
                # Save the circuit as QASM
                qasm_dump(circuit, str(filepath))
                
                # Print progress and circuit info
                gates_count = circuit.count_ops()
                actual_depth = circuit.depth()
                
                print(f"  [{current_circuit:2d}/{total_circuits}] Generated {filename}")
                print(f"      Gates: {dict(gates_count)}")
                print(f"      Actual depth: {actual_depth}")
                
            except Exception as e:
                print(f"  Error generating circuit {circuit_idx+1} for {num_qubits} qubits: {e}")
                continue
    
    print("\n" + "=" * 50)
    print(f"✓ Successfully generated {current_circuit} random circuits")
    print(f"  Saved to: {output_dir}")
    
    # List generated files
    qasm_files = sorted(output_dir.glob("*.qasm"))
    print(f"\nGenerated files ({len(qasm_files)}):")
    for qasm_file in qasm_files:
        print(f"  - {qasm_file.name}")

def generate_single_circuit(num_qubits, depth, seed=None, output_filename=None):
    """
    Generate a single random circuit and save it as QASM.
    
    Args:
        num_qubits (int): Number of qubits
        depth (int): Circuit depth
        seed (int, optional): Random seed for reproducibility
        output_filename (str, optional): Custom output filename
    
    Returns:
        str: Path to the saved QASM file
    """
    script_dir = Path(__file__).parent
    output_dir = script_dir / "benchmark_circuits" / "qasm" / "random_circuits"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate the circuit
    circuit = create_random_circuit(num_qubits, depth, seed=seed)
    
    # Create filename
    if output_filename is None:
        seed_suffix = f"_seed{seed}" if seed is not None else ""
        output_filename = f"random_{num_qubits}q_depth{depth}{seed_suffix}.qasm"
    
    filepath = output_dir / output_filename
    
    # Save the circuit
    qasm_dump(circuit, str(filepath))
    
    print(f"Generated circuit: {output_filename}")
    print(f"  Qubits: {num_qubits}, Depth: {circuit.depth()}")
    print(f"  Gates: {dict(circuit.count_ops())}")
    print(f"  Saved to: {filepath}")
    
    return str(filepath)

def main():
    """Main function to generate the default suite of random circuits."""
    try:
        # Generate the standard suite: 50, 60, 70, 80, 90, 100 qubits with depth 100
        generate_random_circuit_suite(
            qubit_counts=[50, 60, 70, 80, 90, 100],
            depth=100,
            circuits_per_size=5
        )
    except Exception as e:
        print(f"Error during circuit generation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()