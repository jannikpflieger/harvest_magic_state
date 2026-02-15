import matplotlib.pyplot as plt
import networkx as nx
import json
import os
import glob
import numpy as np
from pathlib import Path


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


def calculate_moving_window_stats(data, window_size=100):
    """
    Calculate moving window statistics for the data.
    
    Args:
        data: Array-like data
        window_size: Size of the moving window
        
    Returns:
        (moving_max, moving_avg): Arrays of moving window maximum and average
    """
    data = np.array(data)
    n = len(data)
    
    if n == 0:
        return np.array([]), np.array([])
    
    # If we have fewer data points than window size, use the available data
    effective_window = min(window_size, n)
    
    moving_max = []
    moving_avg = []
    
    for i in range(n):
        # Define window bounds
        start_idx = max(0, i - effective_window + 1)
        end_idx = i + 1
        
        window_data = data[start_idx:end_idx]
        moving_max.append(np.max(window_data))
        moving_avg.append(np.mean(window_data))
    
    return np.array(moving_max), np.array(moving_avg)


def load_single_circuit_data(json_file):
    """
    Load circuit analysis data from a single JSON file.
    
    Args:
        json_file: Path to circuit analysis JSON file
        
    Returns:
        Dictionary with layer-wise metrics
    """
    try:
        with open(json_file, 'r') as f:
            content = json.load(f)
            circuit_data = content.get('circuit', {})
            
            if circuit_data.get('analysis_status') != 'SUCCESS':
                print(f"Circuit analysis was not successful: {circuit_data.get('error', 'Unknown error')}")
                return None
            
            # Extract layer-wise data
            pauli_evolutions_per_layer = circuit_data.get('pauli_evolutions_per_layer', [])
            pauli_evolution_sizes_per_layer = circuit_data.get('pauli_evolution_sizes_per_layer', [])
            
            # Calculate layer-wise metrics
            layer_data = {
                'circuit_name': circuit_data.get('circuit_name', ''),
                'num_qubits': circuit_data.get('num_qubits', 0),
                'num_layers': len(pauli_evolutions_per_layer),
                'pauli_counts_per_layer': pauli_evolutions_per_layer,
                'max_pauli_size_per_layer': [],
                'avg_pauli_size_per_layer': []
            }
            
            # Calculate max and average size per layer
            for layer_sizes in pauli_evolution_sizes_per_layer:
                if layer_sizes:  # If there are Pauli evolutions in this layer
                    layer_data['max_pauli_size_per_layer'].append(max(layer_sizes))
                    layer_data['avg_pauli_size_per_layer'].append(sum(layer_sizes) / len(layer_sizes))
                else:  # No Pauli evolutions in this layer
                    layer_data['max_pauli_size_per_layer'].append(0)
                    layer_data['avg_pauli_size_per_layer'].append(0)
            
            return layer_data
            
    except Exception as e:
        print(f"Error loading {json_file}: {e}")
        return None


def visualize_circuit_information(json_file, window_size=500, save_file=None):
    """
    Visualize single circuit analysis with 4 metrics over layers using moving windows.
    
    Args:
        json_file: Path to circuit analysis JSON file
        window_size: Size of moving window for calculations (default: 100)
        save_file: Optional file path to save the plot
    """
    # Load single circuit data
    data = load_single_circuit_data(json_file)
    
    if data is None:
        print("Failed to load circuit data!")
        return
    
    print(f"Loaded circuit: {data['circuit_name']}")
    print(f"Qubits: {data['num_qubits']}, Layers: {data['num_layers']}")
    
    # Calculate moving window statistics for each metric
    layers = range(data['num_layers'])
    
    # 1. Maximum size of Pauli evolutions (moving window maximum)
    max_size_max, max_size_avg = calculate_moving_window_stats(data['max_pauli_size_per_layer'], window_size)
    
    # 2. Average size of Pauli evolutions (moving window average)
    avg_size_max, avg_size_avg = calculate_moving_window_stats(data['avg_pauli_size_per_layer'], window_size)
    
    # 3. Maximum number of Pauli evolutions (moving window maximum)
    max_count_max, max_count_avg = calculate_moving_window_stats(data['pauli_counts_per_layer'], window_size)
    
    # 4. Average number of Pauli evolutions (moving window average) 
    avg_count_max, avg_count_avg = calculate_moving_window_stats(data['pauli_counts_per_layer'], window_size)
    
    # Create single plot with 4 lines
    plt.figure(figsize=(15, 8))
    
    # Plot the 4 metrics
    plt.plot(layers, max_size_max, 'b-', linewidth=2, label='Max Pauli Size (Moving Max)', alpha=0.8)
    plt.plot(layers, avg_size_avg, 'g-', linewidth=2, label='Avg Pauli Size (Moving Avg)', alpha=0.8)
    plt.plot(layers, max_count_max, 'r-', linewidth=2, label='Max Pauli Count (Moving Max)', alpha=0.8)
    plt.plot(layers, avg_count_avg, 'm-', linewidth=2, label='Avg Pauli Count (Moving Avg)', alpha=0.8)
    
    plt.title(f'Pauli Evolution Metrics - {data["circuit_name"]} ({data["num_qubits"]} qubits)\n'
              f'Moving Window Size: {window_size}', fontsize=14, fontweight='bold')
    plt.xlabel('Layer Index', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot if file path provided
    if save_file:
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_file}")
    
    plt.show()
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print("CIRCUIT LAYER ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"Circuit: {data['circuit_name']} ({data['num_qubits']} qubits)")
    print(f"Total Layers: {data['num_layers']}")
    print(f"Max Pauli Size across all layers: {max(data['max_pauli_size_per_layer'])}")
    print(f"Avg Pauli Size range: {min(data['avg_pauli_size_per_layer']):.3f} - {max(data['avg_pauli_size_per_layer']):.3f}")
    print(f"Max Pauli Count in any layer: {max(data['pauli_counts_per_layer'])}")
    print(f"Avg Pauli Count per layer: {sum(data['pauli_counts_per_layer'])/len(data['pauli_counts_per_layer']):.2f}")
    print(f"{'='*60}")