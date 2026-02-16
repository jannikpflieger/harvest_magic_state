import matplotlib
matplotlib.use('TkAgg')  # Use interactive backend for displaying the plot
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path

# Import necessary functions from existing modules
from circuit_to_pbc_dag import qasm_to_circuit, convert_to_PCB, create_dag
from visualizer import visualize_dag
from qiskit import transpile


def load_and_visualize_qft_dag():
    """
    Load the QFT N010 circuit and visualize its DAG structure.
    """
    # Path to the QFT N010 QASM file
    qasm_file = "benchmark_circuits/qasm/qft/qft_N010.qasm"
    
    print("=" * 60)
    print("QFT N010 Circuit PBC DAG Visualization")
    print("=" * 60)
    
    try:
        # Load the QASM circuit
        print(f"Loading circuit from: {qasm_file}")
        circuit = qasm_to_circuit(qasm_file)
        
        # Transpile to supported gate set for PBC conversion
        print("\nTranspiling to PBC-compatible gates...")
        # Transpile to basis gates supported by LitinskiTransformation
        basis_gates = ['id', 'x', 'y', 'z', 'h', 's', 'sdg', 'sx', 'sxdg', 'cx', 'cz', 'cy', 
                      'swap', 'iswap', 'ecr', 'dcx', 't', 'tdg', 'rz']
        transpiled_circuit = transpile(circuit, basis_gates=basis_gates, optimization_level=0)
        print(f"Transpiled circuit gates: {dict(transpiled_circuit.count_ops())}")
        
        # Convert circuit to PBC format
        print("\nConverting circuit to PBC format...")
        pbc_circuit = convert_to_PCB(transpiled_circuit, fix_clifford=False, verbose=True)
        
        # Create DAG from the PBC circuit
        print("\nConverting PBC circuit to DAG...")
        dag = create_dag(pbc_circuit)
        
        print(f"DAG created successfully!")
        print(f"DAG nodes: {len(list(dag.op_nodes()))}")
        print(f"DAG depth: {dag.depth()}")
        print(f"DAG qubits: {dag.num_qubits()}")
        
        # Visualize the PBC DAG
        print("\nGenerating PBC DAG visualization...")
        visualize_dag(dag, title=f"QFT N010 PBC DAG ({dag.num_qubits()} qubits, {dag.depth()} depth)")
        
        print("\n" + "=" * 60)
        print("PBC DAG visualization displayed!")
        print("Close the plot window to continue...")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you're running this from the circuit directory")
        print("and that the QFT file exists.")


def save_qft_dag_plot():
    """
    Load the QFT N010 circuit and save its DAG visualization to a file.
    """
    # Path to the QFT N010 QASM file
    qasm_file = "benchmark_circuits/qasm/qft/qft_N010.qasm"
    output_file = "../plots/qft_N010_pbc_dag.png"
    
    print("=" * 60)
    print("QFT N010 Circuit PBC DAG Visualization (Save to File)")
    print("=" * 60)
    
    try:
        # Load the QASM circuit
        print(f"Loading circuit from: {qasm_file}")
        circuit = qasm_to_circuit(qasm_file)
        
        # Transpile to supported gate set for PBC conversion
        print("\nTranspiling to PBC-compatible gates...")
        # Transpile to basis gates supported by LitinskiTransformation
        basis_gates = ['id', 'x', 'y', 'z', 'h', 's', 'sdg', 'sx', 'sxdg', 'cx', 'cz', 'cy', 
                      'swap', 'iswap', 'ecr', 'dcx', 't', 'tdg', 'rz']
        transpiled_circuit = transpile(circuit, basis_gates=basis_gates, optimization_level=0)
        print(f"Transpiled circuit gates: {dict(transpiled_circuit.count_ops())}")
        
        # Convert circuit to PBC format
        print("\nConverting circuit to PBC format...")
        pbc_circuit = convert_to_PCB(transpiled_circuit, fix_clifford=False, verbose=True)
        
        # Create DAG from the PBC circuit
        print("\nConverting PBC circuit to DAG...")
        dag = create_dag(pbc_circuit)
        
        print(f"DAG created successfully!")
        print(f"DAG nodes: {len(list(dag.op_nodes()))}")
        print(f"DAG depth: {dag.depth()}")
        print(f"DAG qubits: {dag.num_qubits()}")
        
        # Create plots directory if it doesn't exist
        Path("../plots").mkdir(parents=True, exist_ok=True)
        
        # Change matplotlib backend for saving
        matplotlib.use('Agg')
        
        # Visualize and save the PBC DAG
        print(f"\nGenerating and saving PBC DAG visualization to: {output_file}")
        
        # Get all nodes and create a simple graph
        all_nodes = list(dag.op_nodes())
        if not all_nodes:
            print("No operation nodes found!")
            return
        
        # Create NetworkX graph
        G = nx.DiGraph()
        labels = {}
        
        # Add all nodes with simple labels
        for i, node in enumerate(all_nodes):
            G.add_node(i)
            gate_name = node.op.name
            qubits = [dag.find_bit(q).index for q in node.qargs] if node.qargs else []
            labels[i] = f"{gate_name}\nq{qubits}"
        
        # Add edges based on the actual DAG structure
        edge_count = 0
        
        # Create mapping from DAG nodes to our indices
        node_to_idx = {node: i for i, node in enumerate(all_nodes)}
        
        # Use DAG's edges() method to get the actual edges
        for source, target, edge_data in dag.edges(all_nodes):
            # Only add edges between operation nodes
            if (source in node_to_idx and target in node_to_idx):
                src_idx = node_to_idx[source]
                tgt_idx = node_to_idx[target]
                G.add_edge(src_idx, tgt_idx)
                edge_count += 1
        
        # Create chronological layout based on topological layers
        plt.figure(figsize=(20, 12))
        
        if G.number_of_nodes() > 0:
            pos = {}
            
            # Calculate layers based on longest path from start
            layers = {}
            for node_idx in range(len(all_nodes)):
                # Find the maximum layer of all predecessors + 1
                max_pred_layer = -1
                for pred_idx in G.predecessors(node_idx):
                    if pred_idx in layers:
                        max_pred_layer = max(max_pred_layer, layers[pred_idx])
                layers[node_idx] = max_pred_layer + 1
            
            # Group nodes by layer
            layer_nodes = {}
            for node_idx, layer in layers.items():
                if layer not in layer_nodes:
                    layer_nodes[layer] = []
                layer_nodes[layer].append(node_idx)
            
            # Position nodes: x = layer (time), y = qubit lane position
            qubit_y_positions = {}
            num_qubits = dag.num_qubits()
            
            # Assign y coordinates to qubits (lanes)
            for i in range(num_qubits):
                qubit_y_positions[i] = (num_qubits - 1 - i) * 2.0
            
            for layer, nodes_in_layer in layer_nodes.items():
                for node_idx in nodes_in_layer:
                    node = all_nodes[node_idx]
                    
                    # Calculate y position based on which qubits this operation acts on
                    if node.qargs:
                        qubit_indices = [dag.find_bit(q).index for q in node.qargs]
                        # Position at average of target qubits (middle for multi-qubit gates)
                        avg_y = sum(qubit_y_positions[q_idx] for q_idx in qubit_indices) / len(qubit_indices)
                    else:
                        # Fallback for operations without qubits
                        avg_y = 0.0
                    
                    pos[node_idx] = (layer * 3.0, avg_y)
            
            # Draw horizontal qubit lanes for reference
            for qubit_idx, y_pos in qubit_y_positions.items():
                plt.axhline(y=y_pos, color='lightgray', linestyle='--', alpha=0.3, linewidth=1)
                plt.text(-1.0, y_pos, f'q[{qubit_idx}]', ha='right', va='center', 
                        fontsize=10, color='gray')
            
            # Draw edges with high visibility
            nx.draw_networkx_edges(G, pos, 
                                  edge_color='red', 
                                  arrows=True, 
                                  arrowsize=20,
                                  arrowstyle='->', 
                                  width=1.5,
                                  alpha=0.7,
                                  connectionstyle='arc3,rad=0.1')
            
            # Draw nodes
            nx.draw_networkx_nodes(G, pos,
                                  node_color='lightblue',
                                  node_size=2000,
                                  alpha=0.9,
                                  edgecolors='black',
                                  linewidths=1.5)
            
            # Draw labels
            nx.draw_networkx_labels(G, pos, labels,
                                   font_size=8,
                                   font_weight='bold',
                                   font_color='black')
            
            # Add layer labels at the bottom
            max_layer = max(layers.values()) if layers else 0
            for layer in range(max_layer + 1):
                plt.text(layer * 3.0, -3, f"T{layer}", 
                        ha='center', va='top', fontsize=10, 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        plt.title(f"QFT N010 PBC DAG - {dag.num_qubits()} Qubits, {dag.depth()} Depth, {len(all_nodes)} Gates", 
                 fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.margins(0.1)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"PBC DAG visualization saved to: {output_file}")
        print("\n" + "=" * 60)
        print("PBC DAG saved successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you're running this from the circuit directory")
        print("and that the QFT file exists.")


if __name__ == "__main__":
    print("QFT N010 PBC DAG Visualizer")
    print("-" * 40)
    print("1. Display interactive PBC DAG visualization")
    print("2. Save PBC DAG visualization to file")
    print("3. Both (display and save)")
    
    choice = input("\nSelect option (1/2/3): ").strip()
    
    if choice == "1":
        load_and_visualize_qft_dag()
    elif choice == "2":
        save_qft_dag_plot()
    elif choice == "3":
        load_and_visualize_qft_dag()
        save_qft_dag_plot()
    else:
        print("Invalid choice. Running interactive PBC visualization by default.")
        load_and_visualize_qft_dag()