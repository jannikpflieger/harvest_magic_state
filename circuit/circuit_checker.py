#!/usr/bin/env python3
"""
Circuit Checker - Load QASM circuits from benchmark_circuits/qasm/ and convert to PBC format
"""

import os
import sys
from pathlib import Path
from circuit_to_pbc_dag import qasm_to_circuit, convert_to_PCB
from qiskit import transpile, QuantumCircuit
from qiskit.transpiler.passes import RemoveFinalMeasurements
from qiskit.transpiler import PassManager

from bqskit.ft import CliffordTModel
from bqskit import compile
from bqskit.ir.circuit import Circuit
from bqskit.ext import qiskit_to_bqskit, bqskit_to_qiskit
from bqskit.passes import ScanningGateRemovalPass



def is_identity_op(op) -> bool:
    g = op.gate
    # conservative: only drop if it's clearly identity
    return getattr(g, "name", "").lower() in {"i", "id", "identity", "identity1"}

def strip_identities_by_name(circ):
    new = Circuit(circ.num_qudits, circ.radixes)
    for op in circ:
        if is_identity_op(op):
            continue
        new.append(op)
    return new

def strip_qiskit_identities(qc: QuantumCircuit) -> QuantumCircuit:
    new = QuantumCircuit(*qc.qregs, *qc.cregs, name=qc.name)

    for inst, qargs, cargs in qc.data:
        name = inst.name.lower()

        # Common identity names: 'id', 'i' and your custom 'identity1'
        if name in {"id", "i"} or name.startswith("identity"):
            continue

        new.append(inst, qargs, cargs)

    return new



def main(qasm_file_path):
    """
    Load a QASM file and convert it to PBC format.
    
    Args:
        qasm_file_path (str): Path to the QASM file relative to benchmark_circuits/qasm/
                             or absolute path to the QASM file
    """
    # Get the directory of this script
    script_dir = Path(__file__).parent
    
    # Check if path is absolute or relative
    if os.path.isabs(qasm_file_path):
        full_path = qasm_file_path
    else:
        # Construct path relative to benchmark_circuits/qasm/
        full_path = script_dir / "benchmark_circuits" / "qasm" / qasm_file_path
    
    # Check if file exists
    if not os.path.exists(full_path):
        print(f"❌ Error: File not found: {full_path}")
        print(f"Available subdirs in benchmark_circuits/qasm/:")
        qasm_dir = script_dir / "benchmark_circuits" / "qasm"
        if qasm_dir.exists():
            for item in qasm_dir.iterdir():
                if item.is_dir():
                    print(f"  - {item.name}/")
        return
    
    print(f"🔍 Loading circuit from: {full_path}")
    print("=" * 60)
    
    try:
        # Load the circuit from QASM file
        circuit = qasm_to_circuit(str(full_path))

        # Remove measurements if any
        pm = PassManager(RemoveFinalMeasurements())
        circuit = pm.run(circuit)

        bqs_circuit = qiskit_to_bqskit(circuit)
        model = CliffordTModel(bqs_circuit.num_qudits)

        ft_circuit = compile(bqs_circuit, model, optimization_level=2)
        #print(f"📊 BQSKit compiled circuit gates: {dict(ft_circuit.count_ops())}")

        ft_circuit = strip_identities_by_name(ft_circuit)

        transpiled_circuit = bqskit_to_qiskit(ft_circuit)
        transpiled_circuit = strip_qiskit_identities(transpiled_circuit)
        
        
        
        print("\n" + "=" * 60)
        print("🔧 Transpiling to Clifford+T gate set...")
        print("=" * 60)
        
        # Transpile to Clifford+T gate set (gates supported by LitinskiTransformation)
        #clifford_t_basis = ["id", "x", "y", "z", "h", "s", "sdg", "sx", "sxdg", "cx", "cz", "cy", "swap", "iswap", "ecr", "dcx", "t", "tdg"] 
        #transpiled_circuit = transpile(circuit, basis_gates=clifford_t_basis, optimization_level=3, approximation_degree=1)
        #transpiled_circuit = transpile(circuit, optimization_level=3)

        print(f"📊 Transpiled circuit gates: {dict(transpiled_circuit.count_ops())}")
        print(f"📏 Transpiled circuit depth: {transpiled_circuit.depth()}")
        
        print("\n" + "=" * 60)
        print("🔄 Converting to PBC format...")
        print("=" * 60)
        
        # Convert to PBC format
        pbc_circuit = convert_to_PCB(transpiled_circuit, fix_clifford=False, verbose=True)
        
        print("\n" + "=" * 60)
        print("📋 PBC Circuit Summary:")
        print("=" * 60)
        print(f"🔢 Number of qubits: {pbc_circuit.num_qubits}")
        print(f"📏 Circuit depth: {pbc_circuit.depth()}")
        print(f"🎯 Gate counts: {dict(pbc_circuit.count_ops())}")
        
        # Print the circuit
        print("\n" + "=" * 60)
        print("📖 PBC Circuit:")
        print("=" * 60)
        #print(pbc_circuit)
        
    except Exception as e:
        print(f"❌ Error processing circuit: {e}")
        import traceback
        traceback.print_exc()

def print_usage():
    """Print usage information"""
    print("Usage: python circuit_checker.py <path_to_qasm_file>")
    print("\nExamples:")
    print("  python circuit_checker.py qaoa/qaoa_barabasi_albert_N10_3reps.qasm")
    print("  python circuit_checker.py clifford/clifford_20_12345.qasm")
    print("  python circuit_checker.py qft/qft_10.qasm")
    print("\nOr provide absolute path:")
    print("  python circuit_checker.py /path/to/your/circuit.qasm")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print_usage()
        sys.exit(1)
    
    qasm_file_path = sys.argv[1]
    main(qasm_file_path)
