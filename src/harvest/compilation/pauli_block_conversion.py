from qiskit import QuantumCircuit
import numpy as np
from qiskit.converters import circuit_to_dag
from qiskit.transpiler.passes import LitinskiTransformation
from qiskit.transpiler import PassManager
import random


def convert_to_PCB(circuit, fix_clifford=False, verbose=True):
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
        verbose (bool): If True, prints information about the conversion

    Returns:
        Transformed quantum circuit in PCB format

    Raises:
        TranspilerError: if the circuit contains gates not supported by the pass
    """
    if verbose:
        print(f"\n=== PBC Conversion ===")
        print(f"Input circuit gates: {dict(circuit.count_ops())}")

        # Check what gates can be converted
        clifford_gates = {"id", "x", "y", "z", "h", "s", "sdg", "sx", "sxdg", "cx", "cz", "cy", "swap", "iswap", "ecr", "dcx"}
        rz_gates = {"t", "tdg", "rz"}
        supported_gates = clifford_gates | rz_gates

        circuit_gates = set(circuit.count_ops().keys())
        convertible_gates = circuit_gates & rz_gates
        unsupported_gates = circuit_gates - supported_gates

        if convertible_gates:
            print(f"Gates to be converted to PauliEvolution: {convertible_gates}")
        else:
            print("⚠ No RZ-rotation gates found - circuit is already Clifford-only")

        if unsupported_gates:
            print(f"⚠ Unsupported gates that may cause errors: {unsupported_gates}")

    # Create the LitinskiTransformation pass
    litinski_pass = LitinskiTransformation(fix_clifford=fix_clifford)

    pass_manager = PassManager([litinski_pass])

    # Apply the transformation
    pcb_circuit = pass_manager.run(circuit)

    if verbose:
        print(f"Output circuit gates: {dict(pcb_circuit.count_ops())}")

        # Check if transformation actually occurred
        input_ops = circuit.count_ops()
        output_ops = pcb_circuit.count_ops()

        if 'PauliEvolution' in output_ops:
            print("✓ PBC conversion successful - RZ rotations converted to PauliEvolution gates")
        elif input_ops == output_ops:
            print("ℹ No transformation applied - circuit was already in suitable form")
        else:
            print("✓ Circuit structure modified during conversion")

        print(f"=== End PBC Conversion ===\n")

    return pcb_circuit


def create_random_circuit(num_qubits, depth, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    qc = QuantumCircuit(num_qubits)

    single_qubit_gates = [
        lambda qc, q: qc.h(q),
        lambda qc, q: qc.s(q), 
        lambda qc, q: qc.x(q),      
        lambda qc, q: qc.y(q),      
        lambda qc, q: qc.z(q),     
        lambda qc, q: qc.t(q),      
        lambda qc, q: qc.tdg(q),    
        lambda qc, q: qc.sx(q),     
        lambda qc, q: qc.sxdg(q),   
        lambda qc, q: qc.rz(np.random.uniform(0, 2*np.pi), q), 
    ]

    two_qubit_gates = [
        lambda qc, q1, q2: qc.cx(q1, q2),    
        lambda qc, q1, q2: qc.cz(q1, q2), 
        lambda qc, q1, q2: qc.cy(q1, q2), 
    ]

    for layer in range(depth):
        # Decide what gates to place in this layer
        available_qubits = list(range(num_qubits))
        random.shuffle(available_qubits)

        used_qubits = set()

        # Try to place gates without conflicts
        attempts = 0
        while available_qubits and attempts < num_qubits * 2:
            attempts += 1

            # Decide between single and two-qubit gate
            if len(available_qubits) >= 2 and random.random() < 0.3:  # 30% chance for 2-qubit gate
                # Two-qubit gate
                q1 = available_qubits[0]
                q2 = available_qubits[1]

                if q1 not in used_qubits and q2 not in used_qubits:
                    gate = random.choice(two_qubit_gates)
                    gate(qc, q1, q2)
                    used_qubits.add(q1)
                    used_qubits.add(q2)
                    available_qubits.remove(q1)
                    available_qubits.remove(q2)
            else:
                # Single-qubit gate
                if available_qubits:
                    q = available_qubits[0]
                    if q not in used_qubits:
                        gate = random.choice(single_qubit_gates)
                        gate(qc, q)
                        used_qubits.add(q)
                        available_qubits.remove(q)

    return qc


def create_dag(circuit):
    dag = circuit_to_dag(circuit)
    return dag
