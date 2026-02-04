import matplotlib.pyplot as plt
import networkx as nx


def visualize_dag(dag, title="Circuit DAG"):
    """
    Visualize a DAG.
    
    Args:
        dag: DAGCircuit to visualize
        title: Title for the visualization
        
    Returns:
        DAGCircuit: The input DAG (for method chaining)
    """
    
    print(f"\n{title}:")
    print(f"Operations: {dag.count_ops()}, Depth: {dag.depth()}, Qubits: {dag.num_qubits()}")
    
    # Get all nodes and create a simple graph
    all_nodes = list(dag.op_nodes())
    if not all_nodes:
        print("No operation nodes found!")
        return dag
    
    #print(f"Found {len(all_nodes)} operation nodes")
    
    # Create NetworkX graph
    G = nx.DiGraph()
    labels = {}
    
    # Add all nodes with simple labels
    for i, node in enumerate(all_nodes):
        G.add_node(i)
        gate_name = node.op.name
        qubits = [dag.find_bit(q).index for q in node.qargs] if node.qargs else []
        labels[i] = f"{gate_name}\nq{qubits}"
        #print(f"Node {i}: {gate_name} on qubits {qubits}")
    
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
            #print(f"Edge: {src_idx} -> {tgt_idx}")
    
    #print(f"Added {edge_count} edges")
    
    # Visualize with chronological layout (left to right)
    plt.figure(figsize=(14, 10))
    
    if G.number_of_nodes() > 0:
        # Create chronological layout based on topological layers
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
        qubit_y_positions = {}  # Map qubit index to y coordinate
        num_qubits = dag.num_qubits()
        
        # Assign y coordinates to qubits (lanes)
        for i in range(num_qubits):
            qubit_y_positions[i] = (num_qubits - 1 - i) * 2.0  # Reverse order, space them out
        
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
            plt.text(-0.5, y_pos, f'q[{qubit_idx}]', ha='right', va='center', 
                    fontsize=10, color='gray')
        
        # Draw edges with high visibility
        nx.draw_networkx_edges(G, pos, 
                              edge_color='red', 
                              arrows=True, 
                              arrowsize=25,
                              arrowstyle='->', 
                              width=2.5,
                              alpha=0.9,
                              connectionstyle='arc3,rad=0.1')
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos,
                              node_color='lightblue',
                              node_size=3000,
                              alpha=0.9,
                              edgecolors='black',
                              linewidths=2)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, labels,
                               font_size=10,
                               font_weight='bold',
                               font_color='black')
        
        # Add layer labels at the bottom
        max_layer = max(layers.values()) if layers else 0
        for layer in range(max_layer + 1):
            plt.text(layer * 3.0, -4, f"Layer {layer}", 
                    ha='center', va='top', fontsize=12, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.margins(0.15)
    plt.tight_layout()
    plt.show()
    
    return dag