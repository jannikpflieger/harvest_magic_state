"""
Test script to compare Steiner tree vs Steiner packing DAG processing modes.
"""

import sys
sys.path.append('/home/jannik/Documents/TUM_Semester_5/MasterThesis/code/harvest_magic_state')
sys.path.append('/home/jannik/Documents/TUM_Semester_5/MasterThesis/code/harvest_magic_state/lattice_test')

from dag_steiner_processor import process_dag_with_steiner
from circuit.circuit_to_pbc_dag import create_random_circuit, convert_to_PCB, create_dag
from circuit.visualizer import visualize_dag
import logging
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('HarvestMagicState.DAGProcessor')

def test_both_processing_modes():
    """Compare sequential Steiner tree vs parallel Steiner packing processing."""
    
    print("=" * 70)
    print("DAG Processing Mode Comparison Test")
    print("=" * 70)
    
    # Create a test DAG with some parallelizable operations
    print("\nCreating test DAG with Pauli evolution operations...")
    
    # Create a random circuit and convert to PCB format
    circuit = create_random_circuit(num_qubits=9, depth=5)
    pcb_circuit = convert_to_PCB(circuit)
    print(pcb_circuit)
    dag = create_dag(pcb_circuit)
    
    num_nodes = len(list(dag.op_nodes()))
    print(f"Test DAG has {num_nodes} operation nodes")
    
    # Visualize the DAG structure
    visualize_dag(dag, "Test DAG Structure")
    
    
    processor1, results1 = process_dag_with_steiner(
        dag, 
        layout_rows=3, layout_cols=3,
        visualize_steps=False,  # Show step-by-step visualization
        mode="steiner_tree"
    )
    
    print(processor1.get_summary("steiner_tree"))
    
    # Test Mode 2: Parallel Steiner Packing Processing  
    print("\n" + "="*50)
    print("MODE 2: Parallel Steiner Packing Processing")
    print("="*50)

    visualize_dag(dag, "DAG Before Packing Processing")
    
    processor2, results2 = process_dag_with_steiner(
        dag,
        layout_rows=3, layout_cols=3, 
        visualize_steps=True,  # Show step-by-step visualization
        mode="steiner_packing"
    )
    
    print("\nPacking Processing Results:")
    print(processor2.get_summary("steiner_packing"))
    
    return processor1, results1, processor2, results2


if __name__ == "__main__":
    try:
        # Test both processing modes
        test_both_processing_modes()
        
        print("\n✓ All tests completed successfully!")
        
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()