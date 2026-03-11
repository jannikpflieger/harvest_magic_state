"""
QASM file loading and discovery utilities.
"""

import os
import glob
from pathlib import Path
from typing import List

from qiskit import QuantumCircuit


def qasm_to_circuit(filename):
    """
    Load a quantum circuit from a QASM file.

    Args:
        filename: Path to the QASM file
    Returns:
        QuantumCircuit: The loaded quantum circuit
    """
    try:
        circuit = QuantumCircuit.from_qasm_file(filename)
        print(f"Successfully loaded circuit from {filename}")
        print(f"Gates: {dict(circuit.count_ops())}")
        print(f"Depth: {circuit.depth()}")
        print(f"Qubits: {circuit.num_qubits}")
        return circuit
    except Exception as e:
        print(f"Error loading QASM file: {e}")
        raise


def find_qasm_files(base_dir: str) -> List[str]:
    """
    Recursively find all .qasm files in the given directory.

    Args:
        base_dir: Base directory to search in

    Returns:
        List of paths to all .qasm files found
    """
    qasm_files = []
    base_path = Path(base_dir)

    if not base_path.exists():
        print(f"Directory {base_dir} does not exist!")
        return qasm_files

    # Use glob to find all .qasm files recursively
    pattern = os.path.join(base_dir, "**", "*.qasm")
    qasm_files = glob.glob(pattern, recursive=True)

    print(f"Found {len(qasm_files)} QASM files in {base_dir}")
    return sorted(qasm_files)
