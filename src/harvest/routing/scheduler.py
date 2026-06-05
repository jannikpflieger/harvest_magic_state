"""
DAG traversal and time-step scheduling strategies for routing.
"""

import logging

from harvest.compilation.utils import node_needs_magic_state

logger = logging.getLogger('HarvestMagicState.DAGProcessor')
detailed_logger = logging.getLogger('HarvestMagicState.Detailed')

# Maximum consecutive idle cycles (waiting for magic states) before giving up.
_MAX_IDLE_CYCLES = 500


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

    When a ``magic_source`` is attached to *processor*, nodes that require
    a magic state will stall until one becomes available.  The source is
    ticked once per scheduler step.

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
    idle_streak = 0

    all_op_nodes = list(dag.op_nodes())
    factory = processor.magic_source  # may be None

    while len(processed_nodes) < len(all_op_nodes):
        # Advance the factory clock at the start of every step.
        if factory:
            factory.tick()

        ready_nodes = get_ready_nodes(dag, all_op_nodes, processed_nodes)

        if not ready_nodes:
            logger.warning("No ready nodes found, but not all nodes processed. Breaking to avoid infinite loop.")
            break

        # Reset magic terminal tracking for each step
        processor.used_magic_terminals = set()

        node = ready_nodes[0]

        # --- magic-state availability gate (per-terminal) ---
        ready_terminals = None
        if factory and not factory.unlimited and node_needs_magic_state(node):
            ready_terminals = factory.get_ready_terminals()
            if not ready_terminals:
                # No terminal is ready — stall for this cycle.
                factory.record_wait_cycle()
                idle_streak += 1
                logger.info(
                    f"Step {step}: waiting for magic state "
                    f"(idle streak {idle_streak}, ready={factory.num_ready})"
                )
                if idle_streak > _MAX_IDLE_CYCLES:
                    logger.error("Maximum idle cycles exceeded while waiting for magic states — aborting.")
                    break
                step += 1
                continue
        elif factory and not factory.unlimited:
            ready_terminals = factory.get_ready_terminals()

        idle_streak = 0  # reset on progress

        result = processor.process_dag_node(dag, node, ready_terminals=ready_terminals)

        if result is not None:
            # Consume from the chosen terminal AFTER successful routing
            if factory and not factory.unlimited and node_needs_magic_state(node):
                factory.consume(result['magic_terminal'])

            result['magic_wait_cycles'] = 0  # sequential; wait captured via idle steps
            results.append(result)

        processed_nodes.add(node)
        step += 1

    logger.info(f"Finished sequential processing in {step} total steps. "
                f"Processed {len(results)}/{num_nodes} nodes successfully.")
    processor._scheduling_metadata = {
        "total_elapsed_steps": step,
        "num_nodes_completed": len(results),
        "num_nodes_total": num_nodes,
        "completed": len(processed_nodes) == len(all_op_nodes),
    }
    return results


def process_dag_with_packing(processor, dag, visualize_each_step=False):
    """
    Parallel processing using Steiner forest packing.
    Maximizes nodes processed per time step while respecting dependencies
    and magic-state availability.
    """
    num_nodes = len(list(dag.op_nodes()))
    logger.info(f"Starting DAG processing with Steiner packing - {num_nodes} operation nodes")

    processed_nodes = set()
    all_results = []
    time_step = 0
    idle_streak = 0
    all_op_nodes = list(dag.op_nodes())
    factory = processor.magic_source

    while len(processed_nodes) < len(all_op_nodes):
        logger.info(f"=== Time Step {time_step} ===")

        if factory:
            factory.tick()

        processor.used_magic_terminals = set()

        ready_nodes = get_ready_nodes(dag, all_op_nodes, processed_nodes)

        if not ready_nodes:
            logger.warning("No ready nodes found, breaking to avoid infinite loop")
            break

        # --- Check per-terminal availability ---
        ready_terminals = factory.get_ready_terminals() if factory and not factory.unlimited else None

        # Count how many magic-needing nodes we can admit
        executable_nodes = _filter_by_magic_availability(ready_nodes, factory, ready_terminals)

        if not executable_nodes:
            if factory:
                factory.record_wait_cycle()
            idle_streak += 1
            logger.info(
                f"Time step {time_step}: idle (waiting for magic states, "
                f"streak={idle_streak}, ready={factory.num_ready if factory else 'N/A'})"
            )
            if idle_streak > _MAX_IDLE_CYCLES:
                logger.error("Maximum idle cycles exceeded — aborting.")
                break
            time_step += 1
            continue

        idle_streak = 0

        logger.info(f"Found {len(executable_nodes)} executable nodes (of {len(ready_nodes)} ready): "
                     f"{[n.op.name for n in executable_nodes]}")

        working_graph = processor.graph.copy()

        time_step_results = _process_time_step_with_packing(
            processor, dag, executable_nodes, time_step, working_graph, visualize_each_step,
            ready_terminals=ready_terminals,
        )

        successful_nodes = []
        for result in time_step_results:
            if result['success']:
                processed_nodes.add(result['node'])
                all_results.append(result)
                successful_nodes.append(result['node'])
                # Consume from the terminal AFTER successful routing
                if factory and not factory.unlimited and node_needs_magic_state(result['node']):
                    factory.consume(result['magic_terminal'])

        successful_count = len(successful_nodes)
        failed_count = len(time_step_results) - successful_count

        logger.info(f"Time step {time_step}: {successful_count} successful, {failed_count} failed")

        if successful_count == 0:
            logger.warning(f"No progress in time step {time_step}, remaining nodes cannot be routed")
            break

        time_step += 1

        max_steps = num_nodes + _MAX_IDLE_CYCLES
        if factory and not factory.unlimited:
            prep = getattr(factory, 'preparation_cycles', 50)
            max_steps += num_nodes * prep
        if time_step > max_steps:
            logger.error("Too many time steps - stopping")
            break

    logger.info(f"Finished packing processing in {time_step} time steps. "
                f"Processed {len(all_results)}/{num_nodes} nodes successfully.")
    processor._scheduling_metadata = {
        "total_elapsed_steps": time_step,
        "num_nodes_completed": len(all_results),
        "num_nodes_total": num_nodes,
        "completed": len(processed_nodes) == len(all_op_nodes),
    }
    return all_results


def process_dag_with_pathfinder(processor, dag, visualize_each_step=False):
    """
    Process DAG using pathfinder-based approach with iterative improvements.
    Respects magic-state availability when a factory is attached.
    """
    num_nodes = len(list(dag.op_nodes()))
    logger.info(f"Starting DAG processing with Steiner pathfinder - {num_nodes} operation nodes")

    processed_nodes = set()
    all_results = []
    time_step = 0
    idle_streak = 0
    all_op_nodes = list(dag.op_nodes())
    factory = processor.magic_source

    while len(processed_nodes) < len(all_op_nodes):
        logger.info(f"=== Time Step {time_step} ===")

        if factory:
            factory.tick()

        processor.used_magic_terminals = set()

        ready_nodes = get_ready_nodes(dag, all_op_nodes, processed_nodes)

        if not ready_nodes:
            logger.warning("No ready nodes found, breaking to avoid infinite loop")
            break

        ready_terminals = factory.get_ready_terminals() if factory and not factory.unlimited else None

        executable_nodes = _filter_by_magic_availability(ready_nodes, factory, ready_terminals)

        if not executable_nodes:
            if factory:
                factory.record_wait_cycle()
            idle_streak += 1
            logger.info(
                f"Time step {time_step}: idle (waiting for magic states, "
                f"streak={idle_streak}, ready={factory.num_ready if factory else 'N/A'})"
            )
            if idle_streak > _MAX_IDLE_CYCLES:
                logger.error("Maximum idle cycles exceeded — aborting.")
                break
            time_step += 1
            continue

        idle_streak = 0

        logger.info(f"Found {len(executable_nodes)} executable nodes (of {len(ready_nodes)} ready): "
                     f"{[n.op.name for n in executable_nodes]}")

        working_graph = processor.graph.copy()

        time_step_results = _process_time_step_with_pathfinder(
            processor, dag, executable_nodes, time_step, working_graph, visualize_each_step,
            ready_terminals=ready_terminals,
        )

        successful_nodes = []
        for result in time_step_results:
            if result['success']:
                processed_nodes.add(result['node'])
                all_results.append(result)
                successful_nodes.append(result['node'])
                if factory and not factory.unlimited and node_needs_magic_state(result['node']):
                    factory.consume(result['magic_terminal'])

        successful_count = len(successful_nodes)
        failed_count = len(time_step_results) - successful_count

        logger.info(f"Time step {time_step}: {successful_count} successful, {failed_count} failed")

        if successful_count == 0:
            logger.warning(f"No progress in time step {time_step}, remaining nodes cannot be routed")
            break

        time_step += 1

        max_steps = num_nodes + _MAX_IDLE_CYCLES
        if factory and not factory.unlimited:
            prep = getattr(factory, 'preparation_cycles', 50)
            max_steps += num_nodes * prep
        if time_step > max_steps:
            logger.error("Too many time steps - stopping")
            break

    logger.info(f"Finished pathfinder processing in {time_step} time steps. "
                f"Processed {len(all_results)}/{num_nodes} nodes successfully.")
    processor._scheduling_metadata = {
        "total_elapsed_steps": time_step,
        "num_nodes_completed": len(all_results),
        "num_nodes_total": num_nodes,
        "completed": len(processed_nodes) == len(all_op_nodes),
    }
    return all_results


def _filter_by_magic_availability(ready_nodes, factory, ready_terminals=None):
    """Return the subset of *ready_nodes* that can execute this cycle.

    Clifford-only nodes always pass.  Magic-needing nodes are admitted
    only while there are still ready terminals available (one per node).
    No magic states are consumed here — consumption happens after routing.
    """
    if factory is None or factory.unlimited:
        return list(ready_nodes)

    # Count how many ready terminals we can hand out this step.
    remaining_ready = len(ready_terminals) if ready_terminals is not None else 0

    executable = []
    for node in ready_nodes:
        if node_needs_magic_state(node):
            if remaining_ready > 0:
                executable.append(node)
                remaining_ready -= 1
            # else: deferred — not enough ready terminals
        else:
            executable.append(node)
    return executable


def _prepare_terminal_sets(processor, dag, ready_nodes, ready_terminals=None):
    """Prepare terminal sets for packing/pathfinder algorithms.

    Args:
        processor: DAGProcessor instance
        dag: The DAG circuit
        ready_nodes: Nodes to prepare terminals for
        ready_terminals: If given, restrict magic terminal choice to this set.

    Returns:
        (terminal_sets, node_to_terminals) or ([], {}) if no valid sets.
    """
    terminal_sets = []
    node_to_terminals = {}
    temp_used_magic = processor.used_magic_terminals.copy()

    # Unlimited mode (no magic source or unlimited source) should not enforce
    # one-time terminal usage within a time step. Otherwise dense layers can
    # spuriously report "No magic terminals available" even though magic is
    # configured as always available.
    factory = getattr(processor, "magic_source", None)
    enforce_unique_magic = not (factory is None or getattr(factory, "unlimited", False))

    for node in ready_nodes:
        if enforce_unique_magic:
            available_magic = [t for t in processor.magic_terminals if t not in temp_used_magic]
            if ready_terminals is not None:
                ready_set = set(ready_terminals)
                available_magic = [t for t in available_magic if t in ready_set]
        else:
            available_magic = list(processor.magic_terminals)

        if not available_magic:
            # This can happen only in constrained mode.
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

        if enforce_unique_magic:
            temp_used_magic.add(magic_terminal)

        qubit_terminals = processor._get_qubit_terminals_with_magic_direction(dag, node, magic_terminal)
        if not qubit_terminals:
            if enforce_unique_magic and magic_terminal in temp_used_magic:
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
            'magic_wait_cycles': 0,
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


def _process_time_step_with_packing(processor, dag, ready_nodes, time_step, working_graph, visualize_each_step, ready_terminals=None):
    """Process multiple nodes in a single time step using Steiner packing."""
    if not ready_nodes:
        return []

    terminal_sets, node_to_terminals = _prepare_terminal_sets(processor, dag, ready_nodes, ready_terminals=ready_terminals)

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


def _process_time_step_with_pathfinder(processor, dag, ready_nodes, time_step, working_graph, visualize_each_step, ready_terminals=None):
    """Process multiple nodes in a single time step using Steiner pathfinder."""
    if not ready_nodes:
        return []

    terminal_sets, node_to_terminals = _prepare_terminal_sets(processor, dag, ready_nodes, ready_terminals=ready_terminals)

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

    return time_step_results


def _process_time_step_with_harvest(
    processor, dag, ready_nodes, time_step, working_graph,
    visualize_each_step, ready_terminals=None
):
    """Process one time step using the HARVEST negotiated-congestion router.

    Mirrors ``_process_time_step_with_pathfinder`` but delegates to
    ``LayoutEngine.steiner_packing_harvest`` and attaches the algorithm's
    internal statistics to every result dict under the ``"harvest_stats"``
    key.
    """
    if not ready_nodes:
        return []

    terminal_sets, node_to_terminals = _prepare_terminal_sets(
        processor, dag, ready_nodes, ready_terminals=ready_terminals
    )

    if not terminal_sets:
        logger.warning(f"No valid terminal sets for time step {time_step}")
        return []

    logger.info(f"Running HARVEST on {len(terminal_sets)} terminal sets")
    packing_results, _, harvest_stats = processor.eng.steiner_packing_harvest(
        working_graph,
        terminal_sets,
        max_iters=4,
        alpha=2.0,
        beta=1.5,
        stagnation_limit=2,
        greedy_order="min_size",
    )
    logger.info(
        f"HARVEST step {time_step}: iters={harvest_stats['iterations']}, "
        f"conflicted_reroutes={harvest_stats['conflicted_reroutes']}, "
        f"dropped={harvest_stats['dropped']}, "
        f"stopped_reason={harvest_stats['stopped_reason']}"
    )

    time_step_results = _collect_time_step_results(
        processor, ready_nodes, node_to_terminals, packing_results, time_step
    )

    # Attach HARVEST-specific metadata to every result in this time step.
    for r in time_step_results:
        r['harvest_stats'] = harvest_stats
        r['algorithm'] = 'harvest'


    return time_step_results


def process_dag_with_harvest(processor, dag, visualize_each_step=False):
    """Process DAG using HARVEST — a Pathfinder-derived negotiated-congestion
    scheduler with two runtime-focused improvements:

    1. **Selective rerouting**: only nets that participate in congestion are
       ripped up and rerouted each iteration, leaving conflict-free nets
       untouched.

    2. **Early stopping**: the congestion loop terminates as soon as the
       conflict score reaches zero or fails to improve for
       ``stagnation_limit`` consecutive iterations.

    The outer loop structure (time-step management, magic-state gating,
    dependency tracking) is identical to ``process_dag_with_pathfinder``
    so that both algorithms are directly comparable in benchmarks.

    Args:
        processor: DAGProcessor instance.
        dag: The DAGCircuit to process.
        visualize_each_step: Whether to visualise each time step.

    Returns:
        list: Result dicts for every successfully routed node.  Each dict
        contains the standard fields (``node``, ``gate_name``, ``qubits``,
        ``time_step``, ``success``, ``magic_wait_cycles``,
        ``magic_terminal``, ``qubit_terminals``, ``all_terminals``,
        ``steiner_nodes``, ``steiner_edges``) plus HARVEST-only fields
        ``"harvest_stats"`` and ``"algorithm": "harvest"``.
    """
    num_nodes = len(list(dag.op_nodes()))
    logger.info(
        f"Starting DAG processing with HARVEST - {num_nodes} operation nodes"
    )

    processed_nodes = set()
    all_results = []
    time_step = 0
    idle_streak = 0
    all_op_nodes = list(dag.op_nodes())
    factory = processor.magic_source

    while len(processed_nodes) < len(all_op_nodes):
        logger.info(f"=== Time Step {time_step} ===")

        if factory:
            factory.tick()

        processor.used_magic_terminals = set()

        ready_nodes = get_ready_nodes(dag, all_op_nodes, processed_nodes)

        if not ready_nodes:
            logger.warning("No ready nodes found, breaking to avoid infinite loop")
            break

        ready_terminals = (
            factory.get_ready_terminals()
            if factory and not factory.unlimited
            else None
        )

        executable_nodes = _filter_by_magic_availability(
            ready_nodes, factory, ready_terminals
        )

        if not executable_nodes:
            if factory:
                factory.record_wait_cycle()
            idle_streak += 1
            logger.info(
                f"Time step {time_step}: idle (waiting for magic states, "
                f"streak={idle_streak}, "
                f"ready={factory.num_ready if factory else 'N/A'})"
            )
            if idle_streak > _MAX_IDLE_CYCLES:
                logger.error("Maximum idle cycles exceeded — aborting.")
                break
            time_step += 1
            continue

        idle_streak = 0

        logger.info(
            f"Found {len(executable_nodes)} executable nodes "
            f"(of {len(ready_nodes)} ready): "
            f"{[n.op.name for n in executable_nodes]}"
        )

        working_graph = processor.graph.copy()

        time_step_results = _process_time_step_with_harvest(
            processor, dag, executable_nodes, time_step, working_graph,
            visualize_each_step, ready_terminals=ready_terminals,
        )

        successful_nodes = []
        for result in time_step_results:
            if result['success']:
                processed_nodes.add(result['node'])
                all_results.append(result)
                successful_nodes.append(result['node'])
                if factory and not factory.unlimited and node_needs_magic_state(result['node']):
                    factory.consume(result['magic_terminal'])

        successful_count = len(successful_nodes)
        failed_count = len(time_step_results) - successful_count

        logger.info(
            f"Time step {time_step}: {successful_count} successful, "
            f"{failed_count} failed"
        )

        if successful_count == 0:
            logger.warning(
                f"No progress in time step {time_step}, "
                "remaining nodes cannot be routed"
            )
            break

        time_step += 1

        max_steps = num_nodes + _MAX_IDLE_CYCLES
        if factory and not factory.unlimited:
            prep = getattr(factory, 'preparation_cycles', 50)
            max_steps += num_nodes * prep
        if time_step > max_steps:
            logger.error("Too many time steps - stopping")
            break

    logger.info(
        f"Finished HARVEST processing in {time_step} time steps. "
        f"Processed {len(all_results)}/{num_nodes} nodes successfully."
    )
    processor._scheduling_metadata = {
        "total_elapsed_steps": time_step,
        "num_nodes_completed": len(all_results),
        "num_nodes_total": num_nodes,
        "completed": len(processed_nodes) == len(all_op_nodes),
    }
    return all_results