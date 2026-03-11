"""
DAGProcessor — thin orchestration layer for routing DAG nodes
through the lattice layout using Steiner-tree algorithms.
"""

import re
import logging
from collections import defaultdict

from harvest.layout.presets import nxm_ring_layout_single_qubits
from .magic_terminal_selection import (
    get_magic_terminals,
    choose_optimal_magic_terminal,
)
from .scheduler import (
    process_dag_sequential,
    process_dag_with_packing,
    process_dag_with_pathfinder,
)

logger = logging.getLogger('HarvestMagicState.DAGProcessor')
detailed_logger = logging.getLogger('HarvestMagicState.Detailed')


class DAGProcessor:
    """
    Processes DAG nodes sequentially, integrating with Steiner algorithm for magic state routing.
    """

    def __init__(self, layout_rows=4, layout_cols=4):
        self.layout_rows = layout_rows
        self.layout_cols = layout_cols

        # Setup the layout engine
        self.eng = nxm_ring_layout_single_qubits(layout_rows, layout_cols)
        self.graph, self.ports_by_patch, self.pos, self.patch_used_by_port = self.eng.build_routing_graph()

        detailed_logger.info(f"Position dictionary contains {len(self.pos)} entries:")
        for patch_name, pos in list(self.pos.items())[:10]:
            detailed_logger.info(f"  {patch_name}: {pos}")
        if len(self.pos) > 10:
            detailed_logger.info(f"  ... and {len(self.pos) - 10} more entries")

        # Track magic state terminals and their usage
        self.magic_terminals = get_magic_terminals(self.ports_by_patch)
        self.used_magic_terminals = set()

        # Store results for each processed node
        self.processing_results = []

    # ------------------------------------------------------------------
    # Port-type helpers
    # ------------------------------------------------------------------

    def _get_port_type_for_pauli_gate(self, node, qubit_index, all_qubits_in_operation):
        """
        Determine the appropriate port type for a Pauli evolution gate acting on a specific qubit.
        """
        if hasattr(node.op, 'operator') and node.op.operator is not None:
            operator = node.op.operator
            detailed_logger.info(f"Operator: {operator}")

            if hasattr(operator, 'terms'):
                try:
                    terms = list(operator.terms())
                    detailed_logger.info(f"Terms: {terms}")
                    for bit_pattern, coeff in terms:
                        detailed_logger.info(f"Bit pattern: {bit_pattern}, Coefficient: {coeff}")
                        break
                except Exception as e:
                    detailed_logger.warning(f"Could not parse SparseObservable terms: {e}")

            op_str = str(operator)
            detailed_logger.info(f"Operator string representation: {op_str}")

            if '_' in op_str:
                try:
                    qubit_position = all_qubits_in_operation.index(qubit_index)
                    detailed_logger.info(f"Qubit {qubit_index} is at position {qubit_position} in operation")
                except ValueError:
                    detailed_logger.warning(f"Qubit {qubit_index} not found in operation qubits {all_qubits_in_operation}")
                    return 'Z'

                pattern = rf'([XYZ])_{qubit_position}(?:\D|$)'
                match = re.search(pattern, op_str)
                if match:
                    pauli_op = match.group(1)
                    detailed_logger.info(f"Found {pauli_op} operation for qubit {qubit_index} at position {qubit_position}")
                    return pauli_op
                else:
                    detailed_logger.warning(f"No pattern match for position {qubit_position} in '{op_str}'")
            else:
                detailed_logger.info(f"No underscore pattern found in operator string")
                if 'Y' in op_str:
                    return 'Y'
                elif 'Z' in op_str:
                    return 'Z'
                elif 'X' in op_str:
                    return 'X'

        elif hasattr(node.op, 'pauli') and node.op.pauli is not None:
            pauli_str = str(node.op.pauli)
            detailed_logger.info(f"Pauli string: {pauli_str}")

            if qubit_index < len(pauli_str):
                pauli_op_reverse = pauli_str[-(qubit_index + 1)] if qubit_index < len(pauli_str) else 'I'
                detailed_logger.info(f"Qubit {qubit_index}: Pauli operator = {pauli_op_reverse}")
                if pauli_op_reverse in ['X', 'Y', 'Z']:
                    return pauli_op_reverse
                else:
                    return 'Z'

        detailed_logger.info(f"Using fallback port type detection for gate: {node.op.name}")
        return self._get_port_type_for_gate(node.op.name)

    @staticmethod
    def _get_port_type_for_gate(gate_name):
        if gate_name in ['x', 'cx', 'sx', 'sxdg']:
            return 'X'
        elif gate_name in ['z', 'cz', 's', 'sdg', 'rz', 't', 'tdg']:
            return 'Z'
        return 'X'

    # ------------------------------------------------------------------
    # Terminal / orientation helpers
    # ------------------------------------------------------------------

    def _get_best_oriented_port(self, data_patch_name, port_type, magic_terminal):
        """Get the best oriented port on a data patch based on magic state position."""
        if data_patch_name not in self.ports_by_patch:
            return None
        if port_type not in self.ports_by_patch[data_patch_name]:
            return None

        available_ports = self.ports_by_patch[data_patch_name][port_type]
        if not available_ports:
            return None
        if len(available_ports) == 1:
            return available_ports[0]

        magic_pos = self._get_patch_position(magic_terminal)
        data_pos = self._get_patch_position_from_name(data_patch_name)

        if magic_pos is None or data_pos is None:
            detailed_logger.warning(
                f"Could not determine positions for magic terminal {magic_terminal} "
                f"or data patch {data_patch_name}"
            )
            return available_ports[0]

        dx = magic_pos[0] - data_pos[0]
        dy = magic_pos[1] - data_pos[1]

        detailed_logger.info(f"Magic at {magic_pos}, Data at {data_pos}, Delta: ({dx}, {dy})")

        if dx < 0 and dy > 0:
            preferred_directions = ['N', 'W']
        elif dx > 0 and dy > 0:
            preferred_directions = ['N', 'E']
        elif dx < 0 and dy < 0:
            preferred_directions = ['S', 'W']
        elif dx > 0 and dy < 0:
            preferred_directions = ['S', 'E']
        elif dx == 0 and dy > 0:
            preferred_directions = ['N']
        elif dx == 0 and dy < 0:
            preferred_directions = ['S']
        elif dx < 0 and dy == 0:
            preferred_directions = ['W']
        elif dx > 0 and dy == 0:
            preferred_directions = ['E']
        else:
            preferred_directions = ['N', 'S', 'E', 'W']

        detailed_logger.info(f"Preferred port directions: {preferred_directions}")

        for direction in preferred_directions:
            for port in available_ports:
                if self._port_has_direction(port, direction):
                    detailed_logger.info(f"Selected {direction} oriented port: {port}")
                    return port

        detailed_logger.info(f"No preferred direction found, using first available: {available_ports[0]}")
        return available_ports[0]

    @staticmethod
    def _extract_patch_name_from_terminal(terminal):
        if isinstance(terminal, str) and ':' in terminal:
            parts = terminal.split(':')
            if len(parts) >= 2:
                return parts[1]
        return str(terminal)

    @staticmethod
    def _port_has_direction(port_terminal, direction):
        if isinstance(port_terminal, str) and ':' in port_terminal:
            parts = port_terminal.split(':')
            if len(parts) >= 3:
                return direction in parts[2]
        return False

    def _get_patch_position(self, terminal):
        return self.pos.get(terminal)

    def _get_patch_position_from_name(self, patch_name):
        if patch_name in self.ports_by_patch:
            for port_type, ports in self.ports_by_patch[patch_name].items():
                if ports:
                    first_port = ports[0]
                    if first_port in self.pos:
                        detailed_logger.info(f"Found position for patch {patch_name} via port {first_port}: {self.pos[first_port]}")
                        return self.pos[first_port]
        detailed_logger.warning(f"Could not find position for patch {patch_name}")
        return None

    # ------------------------------------------------------------------
    # Qubit terminal resolution
    # ------------------------------------------------------------------

    def _get_qubit_terminals(self, dag, node):
        """Get the data qubit terminals for a given DAG node."""
        terminals = []

        if node.qargs:
            qubit_indices = [dag.find_bit(q).index for q in node.qargs]
            detailed_logger.info(f"Node {node.op.name} acts on qubits: {qubit_indices}")

            for qubit_index in qubit_indices:
                row = qubit_index // self.layout_cols
                col = qubit_index % self.layout_cols
                patch_name = f"q_{row}_{col}"

                detailed_logger.info(f"Mapping qubit {qubit_index} to patch {patch_name}")

                if patch_name in self.ports_by_patch:
                    port_type = self._get_port_type_for_pauli_gate(node, qubit_index, qubit_indices)

                    if port_type == 'Y':
                        detailed_logger.info(f"Y operation detected for qubit {qubit_index} - adding both X and Z ports")
                        from .magic_terminal_selection import choose_magic_terminal
                        magic_terminal = choose_magic_terminal(self.magic_terminals, self.used_magic_terminals)

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
                        from .magic_terminal_selection import choose_magic_terminal
                        magic_terminal = choose_magic_terminal(self.magic_terminals, self.used_magic_terminals)
                        best_port = self._get_best_oriented_port(patch_name, port_type, magic_terminal)

                        if best_port:
                            terminals.append(best_port)
                            detailed_logger.info(f"Added oriented terminal {best_port} for qubit {qubit_index} (port type: {port_type})")
                        else:
                            logger.warning(f"No {port_type} ports available for patch {patch_name}")
                else:
                    logger.warning(f"Patch {patch_name} not found in layout")

        return terminals

    def _get_qubit_terminals_with_magic_direction(self, dag, node, magic_terminal):
        """Get data qubit terminals with directional selection based on magic state position."""
        terminals = []

        if node.qargs:
            qubit_indices = [dag.find_bit(q).index for q in node.qargs]
            detailed_logger.info(f"Node {node.op.name} acts on qubits: {qubit_indices}")

            for qubit_index in qubit_indices:
                row = qubit_index // self.layout_cols
                col = qubit_index % self.layout_cols
                patch_name = f"q_{row}_{col}"

                detailed_logger.info(f"Mapping qubit {qubit_index} to patch {patch_name}")

                if patch_name in self.ports_by_patch:
                    port_type = self._get_port_type_for_pauli_gate(node, qubit_index, qubit_indices)

                    if port_type == 'Y':
                        detailed_logger.info(f"Y operation detected for qubit {qubit_index} - adding both X and Z ports")

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
                        best_port = self._get_best_oriented_port(patch_name, port_type, magic_terminal)

                        if best_port:
                            terminals.append(best_port)
                            detailed_logger.info(f"Added oriented terminal {best_port} for qubit {qubit_index} (port type: {port_type})")
                        else:
                            logger.warning(f"No {port_type} ports available for patch {patch_name}")
                else:
                    logger.warning(f"Patch {patch_name} not found in layout")

        return terminals

    # ------------------------------------------------------------------
    # Node processing
    # ------------------------------------------------------------------

    def process_dag_node(self, dag, node):
        """Process a single DAG node: get qubit info, choose magic terminal, run Steiner."""
        available_magic = [t for t in self.magic_terminals if t not in self.used_magic_terminals]
        if not available_magic:
            logger.warning(f"Skipping node {node.op.name} - no magic terminals available")
            return None

        dummy_magic = available_magic[0]
        potential_qubit_terminals = self._get_qubit_terminals_with_magic_direction(dag, node, dummy_magic)

        if len(potential_qubit_terminals) == 0:
            logger.warning(f"No qubit terminals found for node {node.op.name}")
            return None

        magic_terminal = choose_optimal_magic_terminal(
            self.pos, self.magic_terminals, self.used_magic_terminals,
            potential_qubit_terminals, available_magic
        )
        if magic_terminal is None:
            logger.warning(f"Skipping node {node.op.name} - no suitable magic terminals available")
            return None

        self.used_magic_terminals.add(magic_terminal)

        qubit_terminals = self._get_qubit_terminals_with_magic_direction(dag, node, magic_terminal)

        if len(qubit_terminals) == 0:
            logger.warning(f"No qubit terminals found for node {node.op.name}")
            return None

        all_terminals = [magic_terminal] + qubit_terminals

        if len(all_terminals) < 2:
            logger.warning(f"Not enough terminals for node {node.op.name}")
            return None

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

            detailed_logger.info(
                f"Processed node: {node.op.name} on qubits {result['qubits']} "
                f"with {len(sol_edges)} Steiner edges"
            )
            return result

        except Exception as e:
            logger.error(f"Error processing node {node.op.name}: {e}")
            return None

    # ------------------------------------------------------------------
    # DAG-level processing
    # ------------------------------------------------------------------

    def process_entire_dag(self, dag, visualize_each_step=False, mode="steiner_tree"):
        """
        Process the entire DAG, node by node, in topological order.

        Args:
            dag: The DAG circuit to process
            visualize_each_step: Whether to visualize each Steiner solution
            mode: "steiner_tree", "steiner_packing", or "steiner_pathfinder"

        Returns:
            list: Results from processing all nodes
        """
        if mode == "steiner_tree":
            return process_dag_sequential(self, dag, visualize_each_step)
        elif mode == "steiner_packing":
            return process_dag_with_packing(self, dag, visualize_each_step)
        elif mode == "steiner_pathfinder":
            return process_dag_with_pathfinder(self, dag, visualize_each_step)
        else:
            raise ValueError(f"Unknown processing mode: {mode}")

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

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
            if result.get('success', True):
                successful_results.append(result)
            else:
                failed_results.append(result)

        summary = f"Processing mode: {mode}\n"
        summary += f"Processed {len(successful_results)} nodes successfully, {len(failed_results)} failed:\n"

        for gate, count in gate_counts.items():
            summary += f"  {gate}: {count}\n"

        summary += f"Used magic terminals: {len(self.used_magic_terminals)}/{len(self.magic_terminals)}\n"

        if mode in ["steiner_packing", "steiner_pathfinder"] and self.processing_results:
            time_steps = set()
            for result in successful_results:
                if 'time_step' in result:
                    time_steps.add(result['time_step'])

            if time_steps:
                summary += f"Completed in {len(time_steps)} time steps\n"
                for ts in sorted(time_steps):
                    ts_nodes = [r for r in successful_results if r.get('time_step') == ts]
                    summary += f"  Time step {ts}: {len(ts_nodes)} nodes\n"

        return summary


def process_dag_with_steiner(dag, layout_rows=4, layout_cols=4, visualize_steps=False, mode="steiner_tree"):
    """
    Convenience function to process a DAG with Steiner algorithm integration.

    Args:
        dag: The DAG circuit to process
        layout_rows: Number of rows in lattice layout
        layout_cols: Number of columns in lattice layout
        visualize_steps: Whether to visualize each processing step
        mode: "steiner_tree", "steiner_packing", or "steiner_pathfinder"

    Returns:
        tuple: (DAGProcessor instance, processing results)
    """
    processor = DAGProcessor(layout_rows, layout_cols)

    detailed_logger.info("Visualizing initial routing graph")
    processor.visualize_layout()

    logger.info(f"Processing DAG with mode: {mode}")
    results = processor.process_entire_dag(dag, visualize_each_step=visualize_steps, mode=mode)

    processor.processing_results = results

    summary = processor.get_summary(mode)
    logger.info("Processing Summary:")
    for line in summary.split('\n'):
        if line.strip():
            detailed_logger.info(f"  {line}")

    return processor, results
