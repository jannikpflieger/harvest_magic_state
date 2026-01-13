from qiskit import QuantumCircuit
import utils
from qiskit.transpiler.passes import LitinskiTransformation
from qiskit.transpiler import PassManager
import numpy as np

def convert_to_PCB(circuit, fix_clifford=True):
    """
    Convert a quantum circuit to PCB (Pauli-based Circuit Block) format using LitinskiTransformation.
    
    This transformation moves Clifford gates to the end of the circuit and converts RZ-rotations 
    to product pauli rotations (implemented as PauliEvolutionGate gates), and changes Z-measurements 
    to product pauli measurements (implemented using PauliProductMeasurement instructions).
    
    The pass supports Clifford gates: ["id", "x", "y", "z", "h", "s", "sdg", "sx", "sxdg", 
    "cx", "cz", "cy", "swap", "iswap", "ecr", "dcx"] and RZ-rotations: ["t", "tdg", "rz"]
    
    Args:
        circuit: The input quantum circuit containing Clifford gates, RZ-rotations, and Z-measurements
        fix_clifford (bool): If False, omits final Clifford gates from the output circuit
        
    Returns:
        Transformed quantum circuit in PCB format
        
    Raises:
        TranspilerError: if the circuit contains gates not supported by the pass
    """
    # Create the LitinskiTransformation pass
    litinski_pass = LitinskiTransformation(fix_clifford=fix_clifford)
    
    # Create a pass manager and add the LitinskiTransformation
    pass_manager = PassManager([litinski_pass])
    
    # Apply the transformation
    pcb_circuit = pass_manager.run(circuit)
    
    return pcb_circuit

def convert_to_CliffordT(circuit):
    pass

def create_random_circuit(num_qubits, depth, gate_set=None):
    pass