from qiskit import QuantumCircuit
import numpy as np
import traceback
from pre_processing import convert_to_PCB
from qiskit.dagcircuit import DAGCircuit
from qiskit.converters import circuit_to_dag
import matplotlib.pyplot as plt
import networkx as nx
import random

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

def visualize_dag(circuit, title="Circuit DAG"):
    """
    Convert circuit to DAG and visualize it.
    
    Args:
        circuit: Qiskit QuantumCircuit
        title: Title for the visualization
    """
    # Convert circuit to DAG
    dag = circuit_to_dag(circuit)
    
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

def test_convert_to_PCB(num_qubits=3, depth=100, seed=42):
    """Test the convert_to_PCB function with a random circuit."""

    # Create a random test circuit
    qc = create_random_circuit(num_qubits, depth, seed)
    
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

        # Visualize the original circuit DAG
        print("\n" + "="*50)
        print("VISUALIZING ORIGINAL CIRCUIT DAG")
        print("="*50)
        original_dag = visualize_dag(qc, "Original Circuit DAG")
        
        # Visualize the PCB circuit DAG
        print("\n" + "="*50)
        print("VISUALIZING PCB CIRCUIT DAG")
        print("="*50)
        pcb_dag = visualize_dag(pcb_circuit, "PBC Format Circuit DAG")

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
    
    test_convert_to_PCB(num_qubits, depth, seed)