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
            for qubit in node.qargs:
                qubit_index = dag.find_bit(qubit).index
                # Map qubit index to layout patch
                # Assuming logical qubits map to q_x_y patches
                patch_name = f"q_{qubit_index//4}_{qubit_index%4}"  # Example mapping
                
                if patch_name in self.ports_by_patch:
                    # Choose appropriate port type based on gate type
                    port_type = self._get_port_type_for_gate(node.op.name)
                    if port_type in self.ports_by_patch[patch_name]:
                        terminals.append(self.ports_by_patch[patch_name][port_type][0])
        
        return terminals
    
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