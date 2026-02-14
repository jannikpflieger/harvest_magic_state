import traceback
import numpy as np
from circuit_to_pbc_dag import convert_to_PCB, create_dag, create_random_circuit, mqt_bench_pipeline
from visualizer import visualize_dag


def analyze_dag_layers(dag, title="Circuit DAG Analysis"):
    """
    Analyze a DAG and count operations per layer, focusing on PauliEvolution gates.
    
    Args:
        dag: Qiskit DAGCircuit object
        title: Title for the analysis output
    
    Returns:
        dict: Dictionary with layer analysis data
    """
    print(f"\n=== {title} ===")
    print(f"Total operations: {len(dag.op_nodes())}")
    print(f"Circuit depth: {dag.depth()}")
    print(f"Number of qubits: {dag.num_qubits()}")
    print(f"Gate counts: {dag.count_ops()}")
    
    # Get layers from the DAG
    layers = list(dag.layers())
    print(f"Number of layers: {len(layers)}")
    
    layer_data = {
        'layer_numbers': [],
        'total_gates': [],
        'pauli_evolutions': [],
        'pauli_evolution_weights': [],
        'clifford_gates': [],
        'other_gates': []
    }
    
    all_pauli_weights = []
    
    # Analyze each layer
    for layer_idx, layer in enumerate(layers):
        layer_ops = list(layer['graph'].op_nodes())
        
        pauli_evolution_count = 0
        layer_pauli_weights = []
        clifford_count = 0
        other_count = 0
        
        for op_node in layer_ops:
            op_name = op_node.op.name
            if op_name == 'PauliEvolution':
                pauli_evolution_count += 1
                
                # Get weight (number of qubits the PauliEvolution acts on)
                weight = len(op_node.qargs) if op_node.qargs else 0
                layer_pauli_weights.append(weight)
                all_pauli_weights.append(weight)
                
            elif op_name in {'h', 'x', 'y', 'z', 's', 'sdg', 'sx', 'sxdg', 'cx', 'cz', 'cy', 'swap'}:
                clifford_count += 1
            else:
                other_count += 1
        
        layer_data['layer_numbers'].append(layer_idx)
        layer_data['total_gates'].append(len(layer_ops))
        layer_data['pauli_evolutions'].append(pauli_evolution_count)
        layer_data['pauli_evolution_weights'].append(layer_pauli_weights)
        layer_data['clifford_gates'].append(clifford_count)
        layer_data['other_gates'].append(other_count)
        
        if pauli_evolution_count > 0 and layer_idx < 10:
            weights_str = f", weights: {layer_pauli_weights}" if layer_pauli_weights else ""
            print(f"Layer {layer_idx}: {len(layer_ops)} total gates ({pauli_evolution_count} PauliEvolution{weights_str}, {clifford_count} Clifford, {other_count} other)")
    
    # Summary statistics
    total_pauli_evolutions = sum(layer_data['pauli_evolutions'])
    max_pauli_per_layer = max(layer_data['pauli_evolutions']) if layer_data['pauli_evolutions'] else 0
    layers_with_pauli = sum(1 for count in layer_data['pauli_evolutions'] if count > 0)
    
    if layers_with_pauli > 10:
        print(f"... and {layers_with_pauli - 10} more layers with PauliEvolution gates")
    
    print(f"\nSummary:")
    print(f"Total PauliEvolution gates: {total_pauli_evolutions}")
    print(f"Max PauliEvolution gates per layer: {max_pauli_per_layer}")
    print(f"Layers containing PauliEvolution gates: {layers_with_pauli}/{len(layers)}")
    
    if layers_with_pauli > 0:
        avg_pauli_per_layer = total_pauli_evolutions / layers_with_pauli
        print(f"Average PauliEvolution gates per active layer: {avg_pauli_per_layer:.2f}")
    
    # Weight statistics
    if all_pauli_weights:
        print(f"\nPauliEvolution Weight Statistics:")
        print(f"Weight range: {min(all_pauli_weights)} - {max(all_pauli_weights)}")
        print(f"Average weight: {np.mean(all_pauli_weights):.2f}")
        print(f"Weight distribution: {dict(zip(*np.unique(all_pauli_weights, return_counts=True)))}")
    
    return layer_data



def test_convert_to_PCB(qc, visualize=False, analyze_layers=True):
    """Test the convert_to_PCB function with a random circuit."""
    
    print("Generated random circuit:")
    print(f"Circuit depth: {qc.depth()}")
    print(f"Circuit gates: {qc.count_ops()}")
    print("\n" + "="*50 + "\n")
    
    # Test the PCB conversion
    try:
        pcb_circuit = convert_to_PCB(qc, False)
        print("PCB converted circuit:")
        print(f"PCB circuit depth: {pcb_circuit.depth()}")
        print(f"PCB circuit gates: {pcb_circuit.count_ops()}")
        print("\nPCB conversion successful!")

        # Create and analyze the original circuit DAG
        original_dag = create_dag(qc)
        if visualize:
            visualize_dag(original_dag, "Original Circuit DAG")
        
        if analyze_layers:
            original_layer_data = analyze_dag_layers(original_dag, "Original Circuit DAG Analysis")
        
        # Create and analyze the PCB circuit DAG
        pcb_dag = create_dag(pcb_circuit)
        if visualize:
            visualize_dag(pcb_dag, "PBC Format Circuit DAG")
            
        if analyze_layers:
            pcb_layer_data = analyze_dag_layers(pcb_dag, "PBC Circuit DAG Analysis")

        print("\nTest completed successfully!")
    
        return pcb_dag
        
    except Exception as e:
        print(f"Error during PCB conversion: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    # You can customize the circuit parameters here
    num_qubits = 3  # Number of qubits
    depth = 10       # Target depth
    seed = 42       # For reproducible results
    
    qc = create_random_circuit(num_qubits, depth, seed)
    #qc = mqt_bench_pipeline(num_qubits, algorithm="qaoa")
    test_convert_to_PCB(qc, visualize=True, analyze_layers=True)