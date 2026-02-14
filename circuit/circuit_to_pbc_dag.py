from qiskit import QuantumCircuit, transpile
import numpy as np
from qiskit.dagcircuit import DAGCircuit
from qiskit.converters import circuit_to_dag
from qiskit.transpiler.passes import RemoveFinalMeasurements, LitinskiTransformation
from qiskit.transpiler import PassManager
import random

import mqt.bench as mqtbench
from mqt.bench import get_benchmark, BenchmarkLevel, targets


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
    
    # Create a pass manager and add the LitinskiTransformation
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

def convert_to_CliffordT(circuit):
    pass

def create_random_circuit(num_qubits, depth, seed=None):
    """
    Create a random quantum circuit with specified width and depth.
    
    Args:
        num_qubits (int): Number of qubits in the circuit
        depth (int): Target depth of the circuit
        seed (int, optional): Random seed for reproducibility
        
    Returns:
        QuantumCircuit: Random circuit with the specified parameters
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    qc = QuantumCircuit(num_qubits)
    
    # Available single-qubit gates (Clifford + T gates + rotations)
    single_qubit_gates = [
        lambda qc, q: qc.h(q),      # Hadamard
        lambda qc, q: qc.s(q),      # S gate
        lambda qc, q: qc.x(q),      # Pauli X
        lambda qc, q: qc.y(q),      # Pauli Y
        lambda qc, q: qc.z(q),      # Pauli Z
        lambda qc, q: qc.t(q),      # T gate
        lambda qc, q: qc.tdg(q),    # T dagger
        lambda qc, q: qc.sx(q),     # sqrt(X)
        lambda qc, q: qc.sxdg(q),   # sqrt(X) dagger
        lambda qc, q: qc.rz(np.random.uniform(0, 2*np.pi), q),  # Random RZ rotation
    ]
    
    # Available two-qubit gates
    two_qubit_gates = [
        lambda qc, q1, q2: qc.cx(q1, q2),    # CNOT
        lambda qc, q1, q2: qc.cz(q1, q2),    # Controlled-Z
        lambda qc, q1, q2: qc.cy(q1, q2),    # Controlled-Y
    ]
    
    # Generate circuit layer by layer to achieve target depth
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
    """
    Convert circuit to DAG.
    
    Args:
        circuit: Qiskit QuantumCircuit
        
    Returns:
        DAGCircuit: DAG representation of the circuit
    """
    # Convert circuit to DAG
    dag = circuit_to_dag(circuit)
    return dag

def mqt_bench_pipeline(num_qubits, algorithm="ae", level=BenchmarkLevel.NATIVEGATES):
    """
    Get a benchmark circuit from MQT Bench that contains non-Clifford gates suitable for PBC conversion.
    
    Args:
        num_qubits: Number of qubits for the benchmark circuit
        algorithm: Algorithm to use ("ae", "qft", "portfoliovqe", "qaoa", etc.)
        level: Benchmark level (NATIVEGATES includes T gates, ALGORITHMIC might not)
        
    Returns:
        QuantumCircuit: Benchmark circuit without measurements, suitable for PBC conversion
    """
    # List of algorithms that typically contain non-Clifford gates
    non_clifford_algorithms = ["ae", "qft", "portfoliovqe", "qaoa", "qgan", "qnn"]
    
    # Use a default algorithm that contains T gates if the requested one might not
    if algorithm not in non_clifford_algorithms:
        print(f"Warning: Algorithm '{algorithm}' might only contain Clifford gates.")
        print(f"Consider using one of: {non_clifford_algorithms}")
    
    try:
        # Get benchmark with Clifford+T gate set to ensure non-Clifford gates
        qc = get_benchmark(
            algorithm, 
            level, 
            circuit_size=num_qubits, 
            target=targets.get_target_for_gateset("clifford+t", num_qubits),
            opt_level=3  # Lower optimization to preserve T gates
        )
    except Exception as e:
        print(f"Failed to get {algorithm} benchmark, falling back to 'ae'")
        qc = get_benchmark(
            "ae", 
            level, 
            circuit_size=num_qubits, 
            target=targets.get_target_for_gateset("clifford+t", num_qubits),
            opt_level=3
        )
    
    # Remove final measurements using transpile with specific pass
    pm = PassManager(RemoveFinalMeasurements())
    qc_without_measurements = pm.run(qc)

    print("MQT Bench Circuit:")
    print(f"Gates: {dict(qc_without_measurements.count_ops())}")
    print(f"Depth: {qc_without_measurements.depth()}")
    print(f"Qubits: {qc_without_measurements.num_qubits}")
    
    # Check if circuit contains non-Clifford gates
    clifford_gates = {"id", "x", "y", "z", "h", "s", "sdg", "sx", "sxdg", "cx", "cz", "cy", "swap", "iswap", "ecr", "dcx"}
    circuit_gates = set(qc_without_measurements.count_ops().keys())
    non_clifford_gates = circuit_gates - clifford_gates
    
    if non_clifford_gates:
        print(f"✓ Non-Clifford gates found: {non_clifford_gates} - suitable for PBC conversion")
    else:
        print("⚠ Warning: Circuit contains only Clifford gates - PBC conversion will have no effect")
        print("   Consider using a different algorithm or adding T gates manually")
    
    return qc_without_measurements