"""
Circuit package for quantum circuit processing and conversion.

This package provides functionality for:
- Converting quantum circuits to PCB (Pauli-based Circuit Block) format
- Creating random quantum circuits
- Visualizing circuit DAGs
- Working with Clifford+T gate sets
"""

from .circuit_to_pbc_dag import (
    convert_to_PCB,
    create_random_circuit,
    create_dag,
    mqt_bench_pipeline
)

from .visualizer import visualize_dag

from .utils import CLIFFORD_T_GATE_SET

__all__ = [
    'convert_to_PCB',
    'convert_to_CliffordT', 
    'create_random_circuit',
    'create_dag',
    'mqt_bench_pipeline',
    'visualize_dag',
    'CLIFFORD_T_GATE_SET'
]