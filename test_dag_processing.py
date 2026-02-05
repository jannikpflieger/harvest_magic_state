"""
Simple test to demonstrate DAG processing with Steiner algorithm integration.
"""

from circuit.circuit_to_pbc_dag import create_random_circuit, create_dag, convert_to_PCB
from circuit.visualizer import visualize_dag
from dag_steiner_processor import DAGProcessor, process_dag_with_steiner


def test_simple_dag_processing():
    """Test the DAG processing with a simple circuit."""
    
    print("="*50)
    print("SIMPLE DAG PROCESSING TEST")
    print("="*50)
    
    # Create a simple test circuit
    qc = create_random_circuit(num_qubits=4, depth=10, seed=42)
    print(f"Created circuit with {qc.num_qubits} qubits and depth {qc.depth()}")
    print(f"Original gates: {qc.count_ops()}")
    
    # Convert to PCB format
    pcb_circuit = convert_to_PCB(qc)
    print(pcb_circuit)
    print(f"PCB gates: {pcb_circuit.count_ops()}")
    
    # Create DAG
    dag = create_dag(pcb_circuit)
    visualize_dag(dag, "Simple PCB Circuit DAG")
    print(f"DAG has {len(list(dag.op_nodes()))} operation nodes")
    
    # Process with Steiner algorithm
    processor, results = process_dag_with_steiner(
        dag, layout_rows=4, layout_cols=1, visualize_steps=True
    )
    
    # Print detailed results
    print(f"\nSuccessfully processed {len(results)} nodes:")
    for i, result in enumerate(results):
        print(f"  {i+1}. {result['gate_name']} on qubits {result['qubits']} - "
              f"{len(result['steiner_edges'])} edges in solution")
    
    return processor, results


def test_step_by_step_visualization():
    """Test with step-by-step visualization."""
    
    print("="*50)
    print("STEP-BY-STEP VISUALIZATION TEST")
    print("="*50)
    
    # Create a very simple circuit for clear visualization
    from qiskit import QuantumCircuit
    
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.t(1)
    
    print("Created simple circuit: H-CNOT-T")
    print(qc)
    
    # Convert and process
    pcb_circuit = convert_to_PCB(qc)
    print(pcb_circuit)
    dag = create_dag(pcb_circuit)
    
    processor, results = process_dag_with_steiner(
        dag, layout_rows=2, layout_cols=2, visualize_steps=True
    )
    
    return processor, results


if __name__ == "__main__":
    # Run simple test first
    processor1, results1 = test_simple_dag_processing()
    
    print("\n" + "="*70 + "\n")
    
    # Run visualization test
    #processor2, results2 = test_step_by_step_visualization()
    
    print("\nAll tests completed!")