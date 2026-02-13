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
        
        # Debug: log position information
        detailed_logger.info(f"Position dictionary contains {len(self.pos)} entries:")
        for patch_name, pos in list(self.pos.items())[:10]:  # Show first 10 entries
            detailed_logger.info(f"  {patch_name}: {pos}")
        if len(self.pos) > 10:
            detailed_logger.info(f"  ... and {len(self.pos) - 10} more entries")
        
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
    
    def _calculate_distance(self, terminal1, terminal2):
        """Calculate Euclidean distance between two terminals."""
        pos1 = self.pos.get(terminal1)
        pos2 = self.pos.get(terminal2)
        if pos1 is None or pos2 is None:
            return float('inf')  # Can't route if position unknown
        return ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5
    
    def _choose_optimal_magic_terminal(self, qubit_terminals, available_magic_terminals=None):
        """Choose the magic terminal closest to the data qubit terminals.
        
        Args:
            qubit_terminals: List of data qubit terminal ports
            available_magic_terminals: List of available magic terminals. If None, uses all unused terminals.
            
        Returns:
            Best magic terminal or None if none available
        """
        if available_magic_terminals is None:
            available_magic_terminals = [t for t in self.magic_terminals if t not in self.used_magic_terminals]
            
        if not available_magic_terminals or not qubit_terminals:
            return None
        
        # Calculate average distance from each magic terminal to all qubit terminals
        best_magic = None
        min_avg_distance = float('inf')
        
        for magic_terminal in available_magic_terminals:
            total_distance = 0
            for qubit_terminal in qubit_terminals:
                total_distance += self._calculate_distance(magic_terminal, qubit_terminal)
            
            avg_distance = total_distance / len(qubit_terminals)
            
            if avg_distance < min_avg_distance:
                min_avg_distance = avg_distance
                best_magic = magic_terminal
        
        detailed_logger.info(f"Selected optimal magic terminal: {best_magic} (avg distance: {min_avg_distance:.2f})")
        return best_magic
    
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
                    port_type = self._get_port_type_for_pauli_gate(node, qubit_index, qubit_indices)
                    
                    if port_type == 'Y':
                        # Y operation requires both X and Z ports from the same qubit
                        detailed_logger.info(f"Y operation detected for qubit {qubit_index} - adding both X and Z ports")
                        
                        # Get the best oriented ports for both X and Z
                        magic_terminal = self._choose_magic_terminal()  # Get current magic state position
                        best_x_port = self._get_best_oriented_port(patch_name, 'X', magic_terminal)
                        best_z_port = self._get_best_oriented_port(patch_name, 'Z', magic_terminal)
                        
                        if best_x_port:
                            terminals.append(best_x_port)
                            detailed_logger.info(f"Added oriented X terminal {best_x_port} for Y operation on qubit {qubit_index}")
                        else:
                            logger.warning(f"No X ports available for Y operation on patch {patch_name}")
                        
                        if best_z_port:
                            terminals.append(best_z_port)
                            detailed_logger.info(f"Added oriented Z terminal {best_z_port} for Y operation on qubit {qubit_index}")
                        else:
                            logger.warning(f"No Z ports available for Y operation on patch {patch_name}")
                    
                    else:
                        # Single port type (X or Z) - choose best oriented port
                        magic_terminal = self._choose_magic_terminal()  # Get current magic state position
                        best_port = self._get_best_oriented_port(patch_name, port_type, magic_terminal)
                        
                        if best_port:
                            terminals.append(best_port)
                            detailed_logger.info(f"Added oriented terminal {best_port} for qubit {qubit_index} (port type: {port_type})")
                        else:
                            logger.warning(f"No {port_type} ports available for patch {patch_name}")
                else:
                    logger.warning(f"Patch {patch_name} not found in layout")
        
        return terminals
    
    def _get_best_oriented_port(self, data_patch_name, port_type, magic_terminal):
        """
        Get the best oriented port on a data patch based on magic state position.
        
        Args:
            data_patch_name (str): Name of the data patch (e.g., 'q_0_1')
            port_type (str): Type of port needed ('X', 'Z', 'Y')
            magic_terminal: The magic state terminal being used
            
        Returns:
            str: The best oriented terminal, or None if not available
        """
        if data_patch_name not in self.ports_by_patch:
            return None
            
        if port_type not in self.ports_by_patch[data_patch_name]:
            return None
            
        available_ports = self.ports_by_patch[data_patch_name][port_type]
        if not available_ports:
            return None
        
        # If only one port available, return it
        if len(available_ports) == 1:
            return available_ports[0]
        
        # Get positions of magic state and data patch
        magic_patch_name = self._extract_patch_name_from_terminal(magic_terminal)
        
        # Find positions by looking up any port from these patches
        magic_pos = self._get_patch_position(magic_terminal)
        data_pos = self._get_patch_position_from_name(data_patch_name)
        
        if magic_pos is None or data_pos is None:
            # Fallback: return first available port
            detailed_logger.warning(f"Could not determine positions for magic terminal {magic_terminal} or data patch {data_patch_name}")
            return available_ports[0]
        
        # Calculate relative direction from data patch to magic patch
        dx = magic_pos[0] - data_pos[0]  # x-difference (negative = magic is left, positive = magic is right)
        dy = magic_pos[1] - data_pos[1]  # y-difference (negative = magic is below, positive = magic is above)
        
        detailed_logger.info(f"Magic at {magic_pos}, Data at {data_pos}, Delta: ({dx}, {dy})")
        
        # Determine preferred port directions based on magic state quadrant
        preferred_directions = []
        
        if dx < 0 and dy > 0:  # Magic is top-left
            preferred_directions = ['N', 'W']  # North or West ports
        elif dx > 0 and dy > 0:  # Magic is top-right  
            preferred_directions = ['N', 'E']  # North or East ports
        elif dx < 0 and dy < 0:  # Magic is bottom-left
            preferred_directions = ['S', 'W']  # South or West ports
        elif dx > 0 and dy < 0:  # Magic is bottom-right
            preferred_directions = ['S', 'E']  # South or East ports
        elif dx == 0 and dy > 0:  # Magic is directly above
            preferred_directions = ['N']
        elif dx == 0 and dy < 0:  # Magic is directly below
            preferred_directions = ['S'] 
        elif dx < 0 and dy == 0:  # Magic is directly left
            preferred_directions = ['W']
        elif dx > 0 and dy == 0:  # Magic is directly right
            preferred_directions = ['E']
        else:  # Same position (shouldn't happen)
            preferred_directions = ['N', 'S', 'E', 'W']
        
        detailed_logger.info(f"Preferred port directions: {preferred_directions}")
        
        # Look for ports in preferred directions
        for direction in preferred_directions:
            for port in available_ports:
                if self._port_has_direction(port, direction):
                    detailed_logger.info(f"Selected {direction} oriented port: {port}")
                    return port
        
        # Fallback: return first available port
        detailed_logger.info(f"No preferred direction found, using first available: {available_ports[0]}")
        return available_ports[0]
    
    def _extract_patch_name_from_terminal(self, terminal):
        """Extract patch name from terminal string like 'P:mT1:M_S:M'."""
        if isinstance(terminal, str) and ':' in terminal:
            parts = terminal.split(':')
            if len(parts) >= 2:
                return parts[1]  # Extract patch name
        return str(terminal)  # Fallback
    
    def _port_has_direction(self, port_terminal, direction):
        """
        Check if a port terminal has a specific direction (N/S/E/W).
        
        Args:
            port_terminal (str): Terminal string like 'P:q_0_0:N:X'
            direction (str): Direction to check ('N', 'S', 'E', 'W')
            
        Returns:
            bool: True if port has the specified direction
        """
        if isinstance(port_terminal, str) and ':' in port_terminal:
            parts = port_terminal.split(':')
            if len(parts) >= 3:
                port_direction = parts[2]  # Extract direction part
                return direction in port_direction
        return False
    
    def _get_patch_position(self, terminal):
        """
        Get the position of a patch from a terminal.
        
        Args:
            terminal: Terminal string (e.g., 'P:mT1:M_S:M')
            
        Returns:
            Position tuple (x, y) or None if not found
        """
        if terminal in self.pos:
            return self.pos[terminal]
        return None
    
    def _get_patch_position_from_name(self, patch_name):
        """
        Get the position of a data patch by finding any port from that patch.
        
        Args:
            patch_name: Name of the patch (e.g., 'q_0_0')
            
        Returns:
            Position tuple (x, y) or None if not found
        """
        if patch_name in self.ports_by_patch:
            # Get any port from this patch and look up its position
            for port_type, ports in self.ports_by_patch[patch_name].items():
                if ports:
                    first_port = ports[0]
                    if first_port in self.pos:
                        detailed_logger.info(f"Found position for patch {patch_name} via port {first_port}: {self.pos[first_port]}")
                        return self.pos[first_port]
        
        detailed_logger.warning(f"Could not find position for patch {patch_name}")
        return None
    
    def _get_port_type_for_pauli_gate(self, node, qubit_index, all_qubits_in_operation):
        """
        Determine the appropriate port type for a Pauli evolution gate acting on a specific qubit.
        
        Args:
            node: The DAG node (should be a PauliEvolution gate)
            qubit_index: The index of the qubit we're getting the port for
            all_qubits_in_operation: List of all qubit indices in this operation
            
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
                    
                    # For multi-qubit operations, we need to parse the bit pattern
                    # to determine which Pauli acts on which qubit
                    for bit_pattern, coeff in terms:
                        detailed_logger.info(f"Bit pattern: {bit_pattern}, Coefficient: {coeff}")
                        
                        # Try to extract individual Pauli operations from the pattern
                        # This is a simplified approach and might need refinement
                        # For now, we'll analyze the string representation
                        break
                        
                except Exception as e:
                    detailed_logger.warning(f"Could not parse SparseObservable terms: {e}")
            
            # Parse the string representation to extract per-qubit Pauli operations
            op_str = str(operator)
            detailed_logger.info(f"Operator string representation: {op_str}")
            
            # For multi-qubit operators, extract the Pauli string
            # Example: "Z_1 Y_0" where indices refer to positions in the Pauli string, not circuit qubits
            if '_' in op_str:
                # Get the position of this qubit in the operation
                try:
                    qubit_position = all_qubits_in_operation.index(qubit_index)
                    detailed_logger.info(f"Qubit {qubit_index} is at position {qubit_position} in operation")
                except ValueError:
                    detailed_logger.warning(f"Qubit {qubit_index} not found in operation qubits {all_qubits_in_operation}")
                    return 'Z'  # fallback
                
                # Look for patterns like "X_N", "Y_N", "Z_N" where N is the position in Pauli string
                import re
                pattern = rf'([XYZ])_{qubit_position}(?:\D|$)'
                match = re.search(pattern, op_str)
                if match:
                    pauli_op = match.group(1)
                    detailed_logger.info(f"Found {pauli_op} operation for qubit {qubit_index} at position {qubit_position}")
                    return pauli_op
                else:
                    detailed_logger.warning(f"No pattern match for position {qubit_position} in '{op_str}'")
            else:
                # Try to extract from a Pauli string format without underscores
                # This is more complex for SparseObservable, so we'll use fallback
                detailed_logger.info(f"No underscore pattern found in operator string")
                
                # Fallback: analyze the overall string for single operations
                if 'Y' in op_str:
                    return 'Y'
                elif 'Z' in op_str:
                    return 'Z'
                elif 'X' in op_str:
                    return 'X'
        
        # Check for legacy pauli attribute
        elif hasattr(node.op, 'pauli') and node.op.pauli is not None:
            pauli_str = str(node.op.pauli)
            detailed_logger.info(f"Pauli string: {pauli_str}")
            
            # Find which Pauli operator acts on this qubit
            if qubit_index < len(pauli_str):
                # Use reverse indexing as it's common in Qiskit (rightmost = qubit 0)
                pauli_op_reverse = pauli_str[-(qubit_index+1)] if qubit_index < len(pauli_str) else 'I'
                
                detailed_logger.info(f"Qubit {qubit_index}: Pauli operator = {pauli_op_reverse}")
                
                if pauli_op_reverse in ['X', 'Y', 'Z']:
                    return pauli_op_reverse
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
        # First get potential qubit terminals to find the optimal magic terminal
        available_magic = [t for t in self.magic_terminals if t not in self.used_magic_terminals]
        if not available_magic:
            logger.warning(f"Skipping node {node.op.name} - no magic terminals available")
            return None
        
        # Get potential qubit terminals using a dummy magic terminal
        dummy_magic = available_magic[0]
        potential_qubit_terminals = self._get_qubit_terminals_with_magic_direction(dag, node, dummy_magic)
        
        if len(potential_qubit_terminals) == 0:
            logger.warning(f"No qubit terminals found for node {node.op.name}")
            return None
        
        # Choose the magic terminal closest to the qubit terminals
        magic_terminal = self._choose_optimal_magic_terminal(potential_qubit_terminals, available_magic)
        if magic_terminal is None:
            logger.warning(f"Skipping node {node.op.name} - no suitable magic terminals available")
            return None
        
        # Now properly mark it as used (since _choose_optimal_magic_terminal doesn't modify used set)
        self.used_magic_terminals.add(magic_terminal)
        
        # Get qubit terminals for this node with the optimal magic terminal
        qubit_terminals = self._get_qubit_terminals_with_magic_direction(dag, node, magic_terminal)
        
        if len(qubit_terminals) == 0:
            logger.warning(f"No qubit terminals found for node {node.op.name}")
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
    
    def _get_qubit_terminals_with_magic_direction(self, dag, node, magic_terminal):
        """
        Get the data qubit terminals for a given DAG node with directional selection based on magic state position.
        
        Args:
            dag: The DAG circuit
            node: The DAG node to process
            magic_terminal: The chosen magic state terminal
            
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
                    port_type = self._get_port_type_for_pauli_gate(node, qubit_index, qubit_indices)
                    
                    if port_type == 'Y':
                        # Y operation requires both X and Z ports from the same qubit
                        detailed_logger.info(f"Y operation detected for qubit {qubit_index} - adding both X and Z ports")
                        
                        # Get the best oriented ports for both X and Z
                        best_x_port = self._get_best_oriented_port(patch_name, 'X', magic_terminal)
                        best_z_port = self._get_best_oriented_port(patch_name, 'Z', magic_terminal)
                        
                        if best_x_port:
                            terminals.append(best_x_port)
                            detailed_logger.info(f"Added oriented X terminal {best_x_port} for Y operation on qubit {qubit_index}")
                        else:
                            logger.warning(f"No X ports available for Y operation on patch {patch_name}")
                        
                        if best_z_port:
                            terminals.append(best_z_port)
                            detailed_logger.info(f"Added oriented Z terminal {best_z_port} for Y operation on qubit {qubit_index}")
                        else:
                            logger.warning(f"No Z ports available for Y operation on patch {patch_name}")
                    
                    else:
                        # Single port type (X or Z) - choose best oriented port
                        best_port = self._get_best_oriented_port(patch_name, port_type, magic_terminal)
                        
                        if best_port:
                            terminals.append(best_port)
                            detailed_logger.info(f"Added oriented terminal {best_port} for qubit {qubit_index} (port type: {port_type})")
                        else:
                            logger.warning(f"No {port_type} ports available for patch {patch_name}")
                else:
                    logger.warning(f"Patch {patch_name} not found in layout")
        
        return terminals
    
    def process_entire_dag(self, dag, visualize_each_step=False, mode="steiner_tree"):
        """
        Process the entire DAG, node by node, in topological order.
        
        Args:
            dag: The DAG circuit to process
            visualize_each_step (bool): Whether to visualize each Steiner solution
            mode (str): Processing mode - "steiner_tree" or "steiner_packing"
            
        Returns:
            list: Results from processing all nodes
        """
        if mode == "steiner_tree":
            return self._process_dag_sequential(dag, visualize_each_step)
        elif mode == "steiner_packing":
            return self._process_dag_with_packing(dag, visualize_each_step)
        else:
            raise ValueError(f"Unknown processing mode: {mode}")
    
    def _process_dag_sequential(self, dag, visualize_each_step=False):
        """
        Original sequential processing using individual Steiner trees.
        """
        num_nodes = len(list(dag.op_nodes()))
        logger.info(f"Starting sequential DAG processing with {num_nodes} operation nodes")
        
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
        
        logger.info(f"Finished sequential processing. Processed {len(results)} nodes successfully.")
        return results
    
    def _process_dag_with_packing(self, dag, visualize_each_step=False):
        """
        New parallel processing using Steiner forest packing.
        Maximizes nodes processed per time step while respecting dependencies.
        """
        num_nodes = len(list(dag.op_nodes()))
        logger.info(f"Starting DAG processing with Steiner packing - {num_nodes} operation nodes")
        
        processed_nodes = set()
        all_results = []
        time_step = 0
        all_op_nodes = list(dag.op_nodes())
        
        while len(processed_nodes) < len(all_op_nodes):
            logger.info(f"=== Time Step {time_step} ===")
            
            ready_nodes = self._get_ready_nodes(dag, all_op_nodes, processed_nodes)
            
            if not ready_nodes:
                logger.warning("No ready nodes found, breaking to avoid infinite loop")
                break
            
            logger.info(f"Found {len(ready_nodes)} ready nodes: {[n.op.name for n in ready_nodes]}")
            
            # Use the full graph for each time step - routing cells can be reused
            working_graph = self.graph.copy()
            
            # Process this time step
            time_step_results = self._process_time_step_with_packing(
                dag, ready_nodes, time_step, working_graph, visualize_each_step
            )
            
            # Update tracking
            successful_nodes = []
            for result in time_step_results:
                if result['success']:
                    processed_nodes.add(result['node'])
                    all_results.append(result)
                    successful_nodes.append(result['node'])
                    # Note: Only magic terminals are permanently used, routing cells can be reused
            
            successful_count = len(successful_nodes)
            failed_count = len(time_step_results) - successful_count
            
            logger.info(f"Time step {time_step}: {successful_count} successful, {failed_count} failed")
            
            # If no progress, we're stuck
            if successful_count == 0:
                logger.warning(f"No progress in time step {time_step}, remaining nodes cannot be routed")
                break
                
            time_step += 1
            
            # Safety check
            if time_step > num_nodes:
                logger.error("Too many time steps - stopping")
                break
        
        logger.info(f"Finished packing processing in {time_step} time steps. Processed {len(all_results)}/{num_nodes} nodes successfully.")
        return all_results
    
    def _get_ready_nodes(self, dag, all_op_nodes, processed_nodes):
        """Get all nodes ready for execution."""
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
        
        return ready_nodes
    
    def _process_time_step_with_packing(self, dag, ready_nodes, time_step, working_graph, visualize_each_step):
        """Process multiple nodes in a single time step using Steiner packing."""
        if not ready_nodes:
            return []
        
        terminal_sets = []
        node_to_terminals = {}
        temp_used_magic = self.used_magic_terminals.copy()
        
        for node in ready_nodes:
            # Get available magic terminals for this time step
            available_magic = [t for t in self.magic_terminals if t not in temp_used_magic]
            if not available_magic:
                logger.warning(f"No magic terminals available for node {node.op.name}")
                continue
            
            # First get potential qubit terminals to find the optimal magic terminal
            # Try with a dummy magic terminal to get qubit positions
            dummy_magic = available_magic[0]
            potential_qubit_terminals = self._get_qubit_terminals_with_magic_direction(dag, node, dummy_magic)
            if not potential_qubit_terminals:
                continue
            
            # Choose the magic terminal closest to the qubit terminals
            magic_terminal = self._choose_optimal_magic_terminal(potential_qubit_terminals, available_magic)
            if not magic_terminal:
                logger.warning(f"No suitable magic terminal found for node {node.op.name}")
                continue
                
            temp_used_magic.add(magic_terminal)
            
            # Get qubit terminals with the chosen optimal magic terminal
            qubit_terminals = self._get_qubit_terminals_with_magic_direction(dag, node, magic_terminal)
            if not qubit_terminals:
                temp_used_magic.remove(magic_terminal)
                continue
                
            # Create terminal set
            terminals = [magic_terminal] + qubit_terminals
            terminal_sets.append(terminals)
            node_to_terminals[node] = {
                'magic_terminal': magic_terminal,
                'qubit_terminals': qubit_terminals,
                'all_terminals': terminals
            }
        
        if not terminal_sets:
            logger.warning(f"No valid terminal sets for time step {time_step}")
            return []
        
        # Run Steiner packing on working graph
        logger.info(f"Running Steiner packing on {len(terminal_sets)} terminal sets")
        packing_results, _ = self.eng.steiner_packing(
            working_graph, terminal_sets, greedy_order="min_size"
        )
        
        # Process results
        time_step_results = []
        nodes_with_terminals = [node for node in ready_nodes if node in node_to_terminals]
        
        for i, (node, packing_result) in enumerate(zip(nodes_with_terminals, packing_results)):
            node_terminals = node_to_terminals[node]
            
            result = {
                'node': node,
                'gate_name': node.op.name,
                'qubits': [dag.find_bit(q).index for q in node.qargs] if node.qargs else [],
                'time_step': time_step,
                'success': packing_result['success'],
                'magic_terminal': node_terminals['magic_terminal'],
                'qubit_terminals': node_terminals['qubit_terminals'],
                'all_terminals': node_terminals['all_terminals'],
                'steiner_nodes': packing_result['sol_nodes'] if packing_result['success'] else set(),
                'steiner_edges': packing_result['sol_edges'] if packing_result['success'] else set()
            }
            
            if packing_result['success']:
                # Permanently allocate magic terminal
                self.used_magic_terminals.add(node_terminals['magic_terminal'])
                logger.info(f"Successfully routed {node.op.name} in time step {time_step}")
            else:
                logger.info(f"Failed to route {node.op.name} in time step {time_step}")
                if 'error' in packing_result:
                    result['error'] = packing_result['error']
            
            time_step_results.append(result)
        
        # Visualize if requested
        if visualize_each_step and time_step_results:
            successful_results = [r for r in time_step_results if r['success']]
            if successful_results:
                self.eng.visualize_packing_solution(
                    working_graph, self.pos,
                    [{'terminal_set': r['all_terminals'], 
                      'sol_nodes': r['steiner_nodes'],
                      'sol_edges': r['steiner_edges'],
                      'success': r['success']} for r in successful_results],
                    title=f"Time Step {time_step}: {len(successful_results)} nodes routed"
                )
        
        return time_step_results
    
    def visualize_layout(self):
        """Visualize the initial routing graph layout."""
        self.eng.visualize_graph(self.graph, self.pos, "Routing graph overlay")
    
    def get_summary(self, mode="steiner_tree"):
        """Get a summary of processing results."""
        if not self.processing_results:
            return "No nodes processed yet."
        
        gate_counts = defaultdict(int)
        successful_results = []
        failed_results = []
        
        for result in self.processing_results:
            gate_counts[result['gate_name']] += 1
            if result.get('success', True):  # Default True for backward compatibility
                successful_results.append(result)
            else:
                failed_results.append(result)
        
        summary = f"Processing mode: {mode}\n"
        summary += f"Processed {len(successful_results)} nodes successfully, {len(failed_results)} failed:\n"
        
        for gate, count in gate_counts.items():
            summary += f"  {gate}: {count}\n"
        
        summary += f"Used magic terminals: {len(self.used_magic_terminals)}/{len(self.magic_terminals)}\n"
        
        if mode == "steiner_packing" and self.processing_results:
            # Add time step analysis for packing mode
            time_steps = set()
            for result in successful_results:
                if 'time_step' in result:
                    time_steps.add(result['time_step'])
            
            if time_steps:
                summary += f"Completed in {len(time_steps)} time steps\n"
                
                # Show nodes per time step
                for ts in sorted(time_steps):
                    ts_nodes = [r for r in successful_results if r.get('time_step') == ts]
                    summary += f"  Time step {ts}: {len(ts_nodes)} nodes\n"
        
        return summary


def process_dag_with_steiner(dag, layout_rows=4, layout_cols=4, visualize_steps=False, mode="steiner_tree"):
    """
    Convenience function to process a DAG with Steiner algorithm integration.
    
    Args:
        dag: The DAG circuit to process
        layout_rows (int): Number of rows in lattice layout
        layout_cols (int): Number of columns in lattice layout  
        visualize_steps (bool): Whether to visualize each processing step
        mode (str): Processing mode - "steiner_tree" for sequential, "steiner_packing" for parallel
        
    Returns:
        tuple: (DAGProcessor instance, processing results)
    """
    processor = DAGProcessor(layout_rows, layout_cols)
    
    # Visualize the initial layout
    detailed_logger.info("Visualizing initial routing graph")
    processor.visualize_layout()
    
    # Process the entire DAG with specified mode
    logger.info(f"Processing DAG with mode: {mode}")
    results = processor.process_entire_dag(dag, visualize_each_step=visualize_steps, mode=mode)
    
    # Store results in processor for later access
    processor.processing_results = results
    
    # Log summary
    summary = processor.get_summary(mode)
    logger.info("Processing Summary:")
    for line in summary.split('\n'):
        if line.strip():
            detailed_logger.info(f"  {line}")
    
    return processor, results