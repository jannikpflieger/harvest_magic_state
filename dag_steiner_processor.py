import logging
from lattice_test.lattice_double_patches import LayoutEngine
from lattice_test.attached_preset_layout import build_7x9_magic_ring_layout, nxm_ring_layout_single_qubits
import random
from collections import defaultdict

# Get logger for this module
logger = logging.getLogger('HarvestMagicState.DAGProcessor')
detailed_logger = logging.getLogger('HarvestMagicState.Detailed')


class DAGProcessor:
    """
    Processes DAG nodes sequentially, integrating with Steiner algorithm for magic state routing.
    """
    
    def __init__(self, layout_rows=4, layout_cols=4):
        """
        Initialize the DAG processor with a lattice layout.
        
        Args:
            layout_rows (int): Number of rows in the lattice layout
            layout_cols (int): Number of columns in the lattice layout
        """
        # Store layout dimensions
        self.layout_rows = layout_rows
        self.layout_cols = layout_cols
        
        # Setup the layout engine
        self.eng = nxm_ring_layout_single_qubits(layout_rows, layout_cols)
        self.graph, self.ports_by_patch, self.pos, self.patch_used_by_port = self.eng.build_routing_graph()
        
        # Track magic state terminals and their usage
        self.magic_terminals = self._get_magic_terminals()
        self.used_magic_terminals = set()
        
        # Store results for each processed node
        self.processing_results = []
        
    def _get_magic_terminals(self):
        """Get all available magic state terminals from the layout."""
        magic_terminals = []
        for patch_name, patch_ports in self.ports_by_patch.items():
            if 'M' in patch_ports:  # Magic state ports
                magic_terminals.extend(patch_ports['M'])
        return magic_terminals
    
    def _choose_magic_terminal(self):
        """
        Choose an available magic state terminal.
        
        Returns:
            A magic state terminal that hasn't been used yet, or None if all are used.
        """
        available_terminals = [t for t in self.magic_terminals if t not in self.used_magic_terminals]
        if not available_terminals:
            logger.warning("No more magic terminals available!")
            return None
        
        # Simple strategy: choose the first available
        # You can implement more sophisticated selection logic here
        chosen = available_terminals[0]
        self.used_magic_terminals.add(chosen)
        detailed_logger.info(f"Selected magic terminal: {chosen}")
        return chosen
    
    def _get_qubit_terminals(self, dag, node):
        """
        Get the data qubit terminals for a given DAG node.
        
        Args:
            dag: The DAG circuit
            node: The DAG node to process
            
        Returns:
            List of data qubit terminals
        """
        terminals = []
        
        if node.qargs:
            qubit_indices = [dag.find_bit(q).index for q in node.qargs]
            detailed_logger.info(f"Node {node.op.name} acts on qubits: {qubit_indices}")
            
            # For Pauli evolution gates, we need terminals for each qubit the gate acts on
            for qubit_index in qubit_indices:
                # Map qubit index to layout patch
                # Simple mapping: q_0 -> q_0_0, q_1 -> q_0_1, q_2 -> q_1_0, etc.
                # This assumes a row-major layout where qubits are arranged in rows
                row = qubit_index // self.layout_cols
                col = qubit_index % self.layout_cols
                patch_name = f"q_{row}_{col}"
                
                detailed_logger.info(f"Mapping qubit {qubit_index} to patch {patch_name}")
                
                if patch_name in self.ports_by_patch:
                    # Get the appropriate port type based on the Pauli operation
                    port_type = self._get_port_type_for_pauli_gate(node, qubit_index)
                    
                    if port_type in self.ports_by_patch[patch_name] and self.ports_by_patch[patch_name][port_type]:
                        # Use the first available port of this type
                        terminal = self.ports_by_patch[patch_name][port_type][0]
                        terminals.append(terminal)
                        detailed_logger.info(f"Added terminal {terminal} for qubit {qubit_index} (port type: {port_type})")
                    else:
                        logger.warning(f"No {port_type} ports available for patch {patch_name}")
                else:
                    logger.warning(f"Patch {patch_name} not found in layout")
        
        return terminals
    
    def _get_port_type_for_pauli_gate(self, node, qubit_index):
        """
        Determine the appropriate port type for a Pauli evolution gate acting on a specific qubit.
        
        Args:
            node: The DAG node (should be a PauliEvolution gate)
            qubit_index: The index of the qubit we're getting the port for
            
        Returns:
            str: Port type ('X', 'Z', or 'Y')
        """
        # For PauliEvolution gates, check the operator if available
        if hasattr(node.op, 'operator') and node.op.operator is not None:
            operator = node.op.operator
            detailed_logger.info(f"Operator: {operator}")
            
            # Handle SparseObservable format
            if hasattr(operator, 'terms'):
                try:
                    terms = list(operator.terms())
                    detailed_logger.info(f"Terms: {terms}")
                    
                    # Each term is (bit_pattern, coefficient)
                    for bit_pattern, coeff in terms:
                        # bit_pattern represents the Pauli operators
                        # We need to find which Pauli acts on our qubit
                        detailed_logger.info(f"Bit pattern: {bit_pattern}, Coefficient: {coeff}")
                        
                        # For SparseObservable, the bit pattern might encode X/Z operations
                        # This is a simplified approach - might need adjustment
                        if 'Z' in str(operator):
                            return 'Z'
                        elif 'X' in str(operator):
                            return 'X' 
                        elif 'Y' in str(operator):
                            return 'Y'
                        else:
                            return 'Z'  # Default
                except Exception as e:
                    detailed_logger.warning(f"Could not parse SparseObservable terms: {e}")
            
            # Try string representation as fallback
            op_str = str(operator)
            if 'Z' in op_str:
                return 'Z'
            elif 'X' in op_str:
                return 'X'
            elif 'Y' in op_str:
                return 'Y'
        
        # Check for legacy pauli attribute
        elif hasattr(node.op, 'pauli') and node.op.pauli is not None:
            pauli_str = str(node.op.pauli)
            detailed_logger.info(f"Pauli string: {pauli_str}")
            
            # Find which Pauli operator acts on this qubit
            if qubit_index < len(pauli_str):
                # Try both forward and reverse indexing
                pauli_op_reverse = pauli_str[-(qubit_index+1)] if qubit_index < len(pauli_str) else 'I'
                
                detailed_logger.info(f"Qubit {qubit_index}: Pauli operator = {pauli_op_reverse}")
                
                if pauli_op_reverse == 'X':
                    return 'X'
                elif pauli_op_reverse == 'Z':
                    return 'Z'
                elif pauli_op_reverse == 'Y':
                    return 'Y'
                else:  # 'I' or unknown
                    return 'Z'  # Default for identity or unknown
        
        # Fallback for other gate types
        detailed_logger.info(f"Using fallback port type detection for gate: {node.op.name}")
        return self._get_port_type_for_gate(node.op.name)
    
    def _get_port_type_for_gate(self, gate_name):
        """
        Determine the appropriate port type based on gate name.
        
        Args:
            gate_name (str): Name of the quantum gate
            
        Returns:
            str: Port type ('X', 'Z', or 'Y')
        """
        # Simple mapping - you can make this more sophisticated
        if gate_name in ['x', 'cx', 'sx', 'sxdg']:
            return 'X'
        elif gate_name in ['z', 'cz', 's', 'sdg', 'rz', 't', 'tdg']:
            return 'Z'
        else:
            return 'X'  # Default to X
    
    def process_dag_node(self, dag, node):
        """
        Process a single DAG node: get qubit info, choose magic terminal, run Steiner.
        
        Args:
            dag: The DAG circuit
            node: The DAG node to process
            
        Returns:
            dict: Processing results including terminals and Steiner solution
        """
        # Get qubit terminals for this node
        qubit_terminals = self._get_qubit_terminals(dag, node)
        
        # Choose a magic state terminal
        magic_terminal = self._choose_magic_terminal()
        
        if magic_terminal is None:
            logger.warning(f"Skipping node {node.op.name} - no magic terminals available")
            return None
        
        # Combine all terminals
        all_terminals = [magic_terminal] + qubit_terminals
        
        if len(all_terminals) < 2:
            logger.warning(f"Not enough terminals for node {node.op.name}")
            return None
        
        # Run Steiner algorithm
        try:
            sol_nodes, sol_edges = self.eng.steiner_tree(self.graph, all_terminals)
            
            result = {
                'node': node,
                'gate_name': node.op.name,
                'qubits': [dag.find_bit(q).index for q in node.qargs] if node.qargs else [],
                'magic_terminal': magic_terminal,
                'qubit_terminals': qubit_terminals,
                'all_terminals': all_terminals,
                'steiner_nodes': sol_nodes,
                'steiner_edges': sol_edges
            }
            
            detailed_logger.info(f"Processed node: {node.op.name} on qubits {result['qubits']} with {len(sol_edges)} Steiner edges")
            return result
            
        except Exception as e:
            logger.error(f"Error processing node {node.op.name}: {e}")
            return None
    
    def process_entire_dag(self, dag, visualize_each_step=False):
        """
        Process the entire DAG, node by node, in topological order.
        
        Args:
            dag: The DAG circuit to process
            visualize_each_step (bool): Whether to visualize each Steiner solution
            
        Returns:
            list: Results from processing all nodes
        """
        num_nodes = len(list(dag.op_nodes()))
        logger.info(f"Starting to process DAG with {num_nodes} operation nodes")
        
        # Track which nodes have been processed
        processed_nodes = set()
        results = []
        step = 0
        
        all_op_nodes = list(dag.op_nodes())
        
        while len(processed_nodes) < len(all_op_nodes):
            # Get nodes that have no unprocessed predecessors (ready to execute)
            ready_nodes = []
            for node in all_op_nodes:
                if node in processed_nodes:
                    continue
                    
                # Check if all operation predecessors have been processed
                predecessors = list(dag.predecessors(node))
                op_predecessors = [p for p in predecessors if hasattr(p, 'op')]
                unprocessed_predecessors = [p for p in op_predecessors if p not in processed_nodes]
                
                if len(unprocessed_predecessors) == 0:
                    ready_nodes.append(node)
            
            if not ready_nodes:
                logger.warning("No ready nodes found, but not all nodes processed. Breaking to avoid infinite loop.")
                break
            
            # Process the first ready node
            node = ready_nodes[0]
            result = self.process_dag_node(dag, node)
            
            if result is not None:
                results.append(result)
                
                # Visualize this step if requested
                if visualize_each_step:
                    title = f"Step {step}: {result['gate_name']} on qubits {result['qubits']}"
                    self.eng.visualize_solution(
                        self.graph, self.pos, result['steiner_edges'],
                        terminals=result['all_terminals'], title=title
                    )
            
            # Mark the node as processed
            processed_nodes.add(node)
            step += 1
        
        logger.info(f"Finished processing DAG. Processed {len(results)} nodes successfully.")
        return results
    
    def visualize_layout(self):
        """Visualize the initial routing graph layout."""
        self.eng.visualize_graph(self.graph, self.pos, "Routing graph overlay")
    
    def get_summary(self):
        """Get a summary of processing results."""
        if not self.processing_results:
            return "No nodes processed yet."
        
        gate_counts = defaultdict(int)
        for result in self.processing_results:
            gate_counts[result['gate_name']] += 1
        
        summary = f"Processed {len(self.processing_results)} nodes:\n"
        for gate, count in gate_counts.items():
            summary += f"  {gate}: {count}\n"
        summary += f"Used magic terminals: {len(self.used_magic_terminals)}/{len(self.magic_terminals)}"
        
        return summary


def process_dag_with_steiner(dag, layout_rows=4, layout_cols=4, visualize_steps=False):
    """
    Convenience function to process a DAG with Steiner algorithm integration.
    
    Args:
        dag: The DAG circuit to process
        layout_rows (int): Number of rows in lattice layout
        layout_cols (int): Number of columns in lattice layout  
        visualize_steps (bool): Whether to visualize each processing step
        
    Returns:
        tuple: (DAGProcessor instance, processing results)
    """
    processor = DAGProcessor(layout_rows, layout_cols)
    
    # Visualize the initial layout
    detailed_logger.info("Visualizing initial routing graph")
    processor.visualize_layout()
    
    # Process the entire DAG
    results = processor.process_entire_dag(dag, visualize_each_step=visualize_steps)
    
    # Store results in processor for later access
    processor.processing_results = results
    
    # Log summary
    summary = processor.get_summary()
    logger.info("Processing Summary:")
    for line in summary.split('\n'):
        if line.strip():
            detailed_logger.info(f"  {line}")
    
    return processor, results