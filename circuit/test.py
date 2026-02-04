import traceback
from circuit_to_pbc_dag import convert_to_PCB, create_dag, create_random_circuit, mqt_bench_pipeline
from visualizer import visualize_dag


def test_convert_to_PCB(qc):
    """Test the convert_to_PCB function with a random circuit."""
    
    print("Generated random circuit:")
    print(qc)
    print(f"Circuit depth: {qc.depth()}")
    print(f"Circuit gates: {qc.count_ops()}")
    print("\n" + "="*50 + "\n")
    
    # Test the PCB conversion
    try:
        pcb_circuit = convert_to_PCB(qc, False)
        print("PCB converted circuit:")
        print(pcb_circuit)
        print(f"PCB circuit depth: {pcb_circuit.depth()}")
        print(f"PCB circuit gates: {pcb_circuit.count_ops()}")
        print("\nPCB conversion successful!")

        # Create and visualize the original circuit DAG
        print("\n" + "="*50)
        print("VISUALIZING ORIGINAL CIRCUIT DAG")
        print("="*50)
        original_dag = create_dag(qc)
        visualize_dag(original_dag, "Original Circuit DAG")
        
        # Create and visualize the PCB circuit DAG
        print("\n" + "="*50)
        print("VISUALIZING PCB CIRCUIT DAG")
        print("="*50)
        pcb_dag = create_dag(pcb_circuit)
        visualize_dag(pcb_dag, "PBC Format Circuit DAG")

        print("\nTest completed successfully!")
    
        return pcb_dag
        
    except Exception as e:
        print(f"Error during PCB conversion: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    # You can customize the circuit parameters here
    num_qubits = 20  # Number of qubits
    depth = 10       # Target depth
    seed = 42       # For reproducible results
    
    qc = create_random_circuit(num_qubits, depth, seed)
    #qc = mqt_bench_pipeline(num_qubits)
    test_convert_to_PCB(qc)