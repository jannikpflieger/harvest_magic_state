"""
DAG traversal and time-step scheduling strategies for routing.
"""

import logging

logger = logging.getLogger('HarvestMagicState.DAGProcessor')
detailed_logger = logging.getLogger('HarvestMagicState.Detailed')


def get_ready_nodes(dag, all_op_nodes, processed_nodes):
    """Get all nodes ready for execution (no unprocessed predecessors)."""
    ready_nodes = []
    for node in all_op_nodes:
        if node in processed_nodes:
            continue

        predecessors = list(dag.predecessors(node))
        op_predecessors = [p for p in predecessors if hasattr(p, 'op')]
        unprocessed_predecessors = [p for p in op_predecessors if p not in processed_nodes]

        if len(unprocessed_predecessors) == 0:
            ready_nodes.append(node)

    return ready_nodes


def process_dag_sequential(processor, dag, visualize_each_step=False):
    """
    Original sequential processing using individual Steiner trees.

    Args:
        processor: DAGProcessor instance
        dag: The DAG circuit to process
        visualize_each_step: Whether to visualize each step

    Returns:
        list: Results from processing all nodes
    """
    num_nodes = len(list(dag.op_nodes()))
    logger.info(f"Starting sequential DAG processing with {num_nodes} operation nodes")

    processed_nodes = set()
    results = []
    step = 0

    all_op_nodes = list(dag.op_nodes())

    while len(processed_nodes) < len(all_op_nodes):
        ready_nodes = get_ready_nodes(dag, all_op_nodes, processed_nodes)

        if not ready_nodes:
            logger.warning("No ready nodes found, but not all nodes processed. Breaking to avoid infinite loop.")
            break

        # Reset magic terminal tracking for each step
        processor.used_magic_terminals = set()

        node = ready_nodes[0]
        result = processor.process_dag_node(dag, node)

        if result is not None:
            results.append(result)

            if visualize_each_step:
                title = f"Step {step}: {result['gate_name']} on qubits {result['qubits']}"
                processor.eng.visualize_solution(
                    processor.graph, processor.pos, result['steiner_edges'],
                    terminals=result['all_terminals'], title=title
                )

        processed_nodes.add(node)
        step += 1

    logger.info(f"Finished sequential processing. Processed {len(results)} nodes successfully.")
    return results


def process_dag_with_packing(processor, dag, visualize_each_step=False):
    """
    Parallel processing using Steiner forest packing.
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

        processor.used_magic_terminals = set()

        ready_nodes = get_ready_nodes(dag, all_op_nodes, processed_nodes)

        if not ready_nodes:
            logger.warning("No ready nodes found, breaking to avoid infinite loop")
            break

        logger.info(f"Found {len(ready_nodes)} ready nodes: {[n.op.name for n in ready_nodes]}")

        working_graph = processor.graph.copy()

        time_step_results = _process_time_step_with_packing(
            processor, dag, ready_nodes, time_step, working_graph, visualize_each_step
        )

        successful_nodes = []
        for result in time_step_results:
            if result['success']:
                processed_nodes.add(result['node'])
                all_results.append(result)
                successful_nodes.append(result['node'])

        successful_count = len(successful_nodes)
        failed_count = len(time_step_results) - successful_count

        logger.info(f"Time step {time_step}: {successful_count} successful, {failed_count} failed")

        if successful_count == 0:
            logger.warning(f"No progress in time step {time_step}, remaining nodes cannot be routed")
            break

        time_step += 1

        if time_step > num_nodes:
            logger.error("Too many time steps - stopping")
            break

    logger.info(f"Finished packing processing in {time_step} time steps. Processed {len(all_results)}/{num_nodes} nodes successfully.")
    return all_results


def process_dag_with_pathfinder(processor, dag, visualize_each_step=False):
    """
    Process DAG using pathfinder-based approach with iterative improvements.
    """
    num_nodes = len(list(dag.op_nodes()))
    logger.info(f"Starting DAG processing with Steiner pathfinder - {num_nodes} operation nodes")

    processed_nodes = set()
    all_results = []
    time_step = 0
    all_op_nodes = list(dag.op_nodes())

    while len(processed_nodes) < len(all_op_nodes):
        logger.info(f"=== Time Step {time_step} ===")

        processor.used_magic_terminals = set()

        ready_nodes = get_ready_nodes(dag, all_op_nodes, processed_nodes)

        if not ready_nodes:
            logger.warning("No ready nodes found, breaking to avoid infinite loop")
            break

        logger.info(f"Found {len(ready_nodes)} ready nodes: {[n.op.name for n in ready_nodes]}")

        working_graph = processor.graph.copy()

        time_step_results = _process_time_step_with_pathfinder(
            processor, dag, ready_nodes, time_step, working_graph, visualize_each_step
        )

        successful_nodes = []
        for result in time_step_results:
            if result['success']:
                processed_nodes.add(result['node'])
                all_results.append(result)
                successful_nodes.append(result['node'])

        successful_count = len(successful_nodes)
        failed_count = len(time_step_results) - successful_count

        logger.info(f"Time step {time_step}: {successful_count} successful, {failed_count} failed")

        if successful_count == 0:
            logger.warning(f"No progress in time step {time_step}, remaining nodes cannot be routed")
            break

        time_step += 1

        if time_step > num_nodes:
            logger.error("Too many time steps - stopping")
            break

    logger.info(f"Finished pathfinder processing in {time_step} time steps. Processed {len(all_results)}/{num_nodes} nodes successfully.")
    return all_results


def _prepare_terminal_sets(processor, dag, ready_nodes):
    """Prepare terminal sets for packing/pathfinder algorithms.

    Returns:
        (terminal_sets, node_to_terminals) or ([], {}) if no valid sets.
    """
    terminal_sets = []
    node_to_terminals = {}
    temp_used_magic = processor.used_magic_terminals.copy()

    for node in ready_nodes:
        available_magic = [t for t in processor.magic_terminals if t not in temp_used_magic]
        if not available_magic:
            logger.warning(f"No magic terminals available for node {node.op.name}")
            continue

        dummy_magic = available_magic[0]
        potential_qubit_terminals = processor._get_qubit_terminals_with_magic_direction(dag, node, dummy_magic)
        if not potential_qubit_terminals:
            continue

        from .magic_terminal_selection import choose_optimal_magic_terminal
        magic_terminal = choose_optimal_magic_terminal(
            processor.pos, processor.magic_terminals, temp_used_magic,
            potential_qubit_terminals, available_magic
        )
        if not magic_terminal:
            logger.warning(f"No suitable magic terminal found for node {node.op.name}")
            continue

        temp_used_magic.add(magic_terminal)

        qubit_terminals = processor._get_qubit_terminals_with_magic_direction(dag, node, magic_terminal)
        if not qubit_terminals:
            temp_used_magic.remove(magic_terminal)
            continue

        terminals = [magic_terminal] + qubit_terminals
        terminal_sets.append(terminals)
        node_to_terminals[node] = {
            'magic_terminal': magic_terminal,
            'qubit_terminals': qubit_terminals,
            'all_terminals': terminals
        }

    return terminal_sets, node_to_terminals


def _collect_time_step_results(processor, ready_nodes, node_to_terminals, packing_results, time_step):
    """Collect results from a packing/pathfinder run into result dicts."""
    time_step_results = []
    nodes_with_terminals = [node for node in ready_nodes if node in node_to_terminals]

    for i, (node, packing_result) in enumerate(zip(nodes_with_terminals, packing_results)):
        from qiskit.dagcircuit import DAGOpNode
        node_terminals = node_to_terminals[node]

        result = {
            'node': node,
            'gate_name': node.op.name,
            'qubits': [],
            'time_step': time_step,
            'success': packing_result['success'],
            'magic_terminal': node_terminals['magic_terminal'],
            'qubit_terminals': node_terminals['qubit_terminals'],
            'all_terminals': node_terminals['all_terminals'],
            'steiner_nodes': packing_result['sol_nodes'] if packing_result['success'] else set(),
            'steiner_edges': packing_result['sol_edges'] if packing_result['success'] else set()
        }

        if packing_result['success']:
            processor.used_magic_terminals.add(node_terminals['magic_terminal'])
            logger.info(f"Successfully routed {node.op.name} in time step {time_step}")
        else:
            logger.info(f"Failed to route {node.op.name} in time step {time_step}")
            if 'error' in packing_result:
                result['error'] = packing_result['error']

        time_step_results.append(result)

    return time_step_results


def _process_time_step_with_packing(processor, dag, ready_nodes, time_step, working_graph, visualize_each_step):
    """Process multiple nodes in a single time step using Steiner packing."""
    if not ready_nodes:
        return []

    terminal_sets, node_to_terminals = _prepare_terminal_sets(processor, dag, ready_nodes)

    if not terminal_sets:
        logger.warning(f"No valid terminal sets for time step {time_step}")
        return []

    logger.info(f"Running Steiner packing on {len(terminal_sets)} terminal sets")
    packing_results, _ = processor.eng.steiner_packing(
        working_graph, terminal_sets, greedy_order="min_size"
    )

    time_step_results = _collect_time_step_results(
        processor, ready_nodes, node_to_terminals, packing_results, time_step
    )

    if visualize_each_step and time_step_results:
        successful_results = [r for r in time_step_results if r['success']]
        if successful_results:
            processor.eng.visualize_packing_solution(
                working_graph, processor.pos,
                [{'terminal_set': r['all_terminals'],
                  'sol_nodes': r['steiner_nodes'],
                  'sol_edges': r['steiner_edges'],
                  'success': r['success']} for r in successful_results],
                title=f"Time Step {time_step}: {len(successful_results)} nodes routed"
            )

    return time_step_results


def _process_time_step_with_pathfinder(processor, dag, ready_nodes, time_step, working_graph, visualize_each_step):
    """Process multiple nodes in a single time step using Steiner pathfinder."""
    if not ready_nodes:
        return []

    terminal_sets, node_to_terminals = _prepare_terminal_sets(processor, dag, ready_nodes)

    if len(terminal_sets) == 0:
        logger.warning(f"No valid terminal sets for time step {time_step}")
        return []

    logger.info(f"Running Steiner pathfinder on {len(terminal_sets)} terminal sets")
    packing_results, _ = processor.eng.steiner_packing_pathfinder(
        working_graph,
        terminal_sets,
        max_iters=15,
        alpha=6.0,
        beta=1.5,
        greedy_order="min_size",
    )

    time_step_results = _collect_time_step_results(
        processor, ready_nodes, node_to_terminals, packing_results, time_step
    )

    if visualize_each_step and time_step_results:
        successful_results = [r for r in time_step_results if r['success']]
        if successful_results:
            processor.eng.visualize_packing_solution(
                working_graph, processor.pos,
                [{'terminal_set': r['all_terminals'],
                  'sol_nodes': r['steiner_nodes'],
                  'sol_edges': r['steiner_edges'],
                  'success': r['success']} for r in successful_results],
                title=f"Time Step {time_step} (Pathfinder): {len(successful_results)} nodes routed"
            )

    return time_step_results
