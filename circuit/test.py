from qiskit import QuantumCircuit
import numpy as np
import traceback
from pre_processing import convert_to_PCB

def test_convert_to_PCB():
    """Test the convert_to_PCB function with a simple circuit."""

    # Create a test circuit with supported gates (without measurements)
    qc = QuantumCircuit(3)
    
    # Add some Clifford gates
    qc.h(0)  # Hadamard
    qc.s(1)  # S gate
    qc.x(2)  # Pauli X
    
    # Add some two-qubit Clifford gates
    qc.cx(0, 1)  # CNOT
    qc.cz(1, 2)  # Controlled-Z
    
    # Add some RZ-rotations
    qc.t(0)  # T gate
    qc.tdg(1)  # T dagger
    qc.rz(np.pi/4, 2)  # RZ rotation
    
    # Add more Clifford gates
    qc.sx(0)  # sqrt(X)
    qc.y(1)  # Pauli Y
    
    # Note: Regular measurements are not supported by LitinskiTransformation
    # The pass expects Z-measurements to be handled differently
    
    print("Original circuit:")
    print(qc)
    print(f"Original circuit depth: {qc.depth()}")
    print(f"Original circuit gates: {qc.count_ops()}")
    print("\n" + "="*50 + "\n")
    
    # Test the PCB conversion
    try:
        pcb_circuit = convert_to_PCB(qc)
        print("PCB converted circuit:")
        print(pcb_circuit)
        print(f"PCB circuit depth: {pcb_circuit.depth()}")
        print(f"PCB circuit gates: {pcb_circuit.count_ops()}")
        print("\nPCB conversion successful!")

        print("\nTest completed successfully!")
        
    except Exception as e:
        print(f"Error during PCB conversion: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    test_convert_to_PCB()