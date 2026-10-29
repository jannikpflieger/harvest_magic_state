"""
DAG traversal and time-step scheduling strategies for routing.
"""

import logging

from harvest import faults
from harvest.compilation.utils import node_needs_magic_state
from .config import HarvestConfig

logger = logging.getLogger('HarvestMagicState.DAGProcessor')
detailed_logger = logging.getLogger('HarvestMagicState.Detailed')

# Maximum consecutive idle cycles (waiting for magic states) before giving up.
_MAX_IDLE_CYCLES = 500

# HARVEST scheduling policies are fields of ``HarvestConfig`` (routing/config.py).


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


class ReadyTracker:
    """Incremental :func:`get_ready_nodes`: same nodes, same (DAG) order.

    ``get_ready_nodes`` rescans every operation and its predecessors at every
    time step, which is quadratic in the circuit size.  The tracker keeps the
    number of unprocessed predecessors per node and updates it when a node is
    marked done, so a step costs only the size of the ready set.
    """

    def __init__(self, dag, all_op_nodes):
        self._nodes = list(all_op_nodes)
        index = {node._node_id: i for i, node in enumerate(self._nodes)}
        self._successors = [[] for _ in self._nodes]
        self._missing = [0] * len(self._nodes)
        for i, node in enumerate(self._nodes):
            preds = {index[p._node_id] for p in dag.predecessors(node) if hasattr(p, 'op')}
            self._missing[i] = len(preds)
            for p in preds:
                self._successors[p].append(i)
        self._index = index
        self._ready = {i for i, m in enumerate(self._missing) if m == 0}
        self._done = [False] * len(self._nodes)
        self.num_done = 0

    def ready(self):
        return [self._nodes[i] for i in sorted(self._ready)]

    def heights(self):
        """{node_id: longest chain of operations from the node to a sink (>= 1)}."""
        missing = list(self._missing)
        order = [i for i, m in enumerate(missing) if m == 0]
        for i in order:  # Kahn's algorithm; ``order`` grows while iterating
            for j in self._successors[i]:
                missing[j] -= 1
                if missing[j] == 0:
                    order.append(j)
        height = [1] * len(self._nodes)
        for i in reversed(order):
            for j in self._successors[i]:
                if height[j] + 1 > height[i]:
                    height[i] = height[j] + 1
        return {self._nodes[i]._node_id: height[i] for i in range(len(self._nodes))}

    def blocked(self):
        """Not-yet-done nodes with at least one unprocessed predecessor (DAG order)."""
        return [self._nodes[i] for i, m in enumerate(self._missing)
                if m > 0 and not self._done[i]]

    def mark_done(self, node):
        i = self._index[node._node_id]
        if self._done[i]:
            return
        self._done[i] = True
        self.num_done += 1
        self._ready.discard(i)
        for j in self._successors[i]:
            self._missing[j] -= 1
            if self._missing[j] == 0 and not self._done[j]:
                self._ready.add(j)


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
            result['time_step'] = step
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

        if successful_count == 0 and stalled:
            # Every routed product stalled on an empty terminal: wait a cycle.
            if factory:
                factory.record_wait_cycle()
            idle_streak += 1
            if idle_streak > _MAX_IDLE_CYCLES:
                logger.error("Maximum idle cycles exceeded — aborting.")
                break
            time_step += 1
            continue

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

        if successful_count == 0 and stalled:
            # Every routed product stalled on an empty terminal: wait a cycle.
            if factory:
                factory.record_wait_cycle()
            idle_streak += 1
            if idle_streak > _MAX_IDLE_CYCLES:
                logger.error("Maximum idle cycles exceeded — aborting.")
                break
            time_step += 1
            continue

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


def _magic_distances(processor):
    """Routing distance from every magic terminal to every node (layout is static)."""
    cache = getattr(processor, '_magic_distance_cache', None)
    if cache is None:
        cache = {t: processor.eng._dijkstra(processor.graph, t)[0] for t in processor.magic_terminals}
        processor._magic_distance_cache = cache
    return cache


def _port_specs(processor, dag, node):
    """[(patch, [port types])] of a product, as ``_get_qubit_terminals_with_magic_direction``
    derives them; cached per node because they do not depend on the magic terminal."""
    cache = processor.__dict__.setdefault('_port_spec_cache', {})
    specs = cache.get(node._node_id)
    if specs is None:
        specs = []
        qubit_indices = [dag.find_bit(q).index for q in node.qargs]
        for qubit_index in qubit_indices:
            patch_name = f"q_{qubit_index}"
            if patch_name not in processor.ports_by_patch:
                continue
            port_type = processor._get_port_type_for_pauli_gate(node, qubit_index, qubit_indices)
            if port_type == 'Y' and faults.active("drop_y_port"):
                faults.note("drop_y_port")
                specs.append((patch_name, ['X']))
                continue
            specs.append((patch_name, ['X', 'Z'] if port_type == 'Y' else [port_type]))
        cache[node._node_id] = specs
    return specs


def _oriented_terminals(processor, specs, magic_terminal):
    """Same ports as ``_get_qubit_terminals_with_magic_direction`` for these specs."""
    terminals = []
    for patch_name, port_types in specs:
        for port_type in port_types:
            port = processor._get_best_oriented_port(patch_name, port_type, magic_terminal)
            if port:
                terminals.append(port)
    return terminals


def _assign_terminals_matching(processor, dag, ready_nodes, available_magic):
    """Optimal product -> magic-terminal assignment for one time step.

    The first ``len(available_magic)`` products (in priority order) that have
    data ports receive a terminal; the assignment minimizes the total routing
    distance (sum over the product's oriented data ports of the shortest-path
    length from the magic terminal), unlike the greedy nearest choice, which
    lets an early product take the terminal a later one needs most.
    Returns ``[(node, magic_terminal, qubit_terminals)]`` in ready order.
    """
    from scipy.optimize import linear_sum_assignment

    if not available_magic:
        return []
    dist = _magic_distances(processor)
    unreachable = 10 ** 6
    rows = []  # (node, {terminal: qubit_terminals}, [cost per terminal])
    for node in ready_nodes:
        if len(rows) == len(available_magic):
            break
        options, costs = {}, []
        specs = _port_specs(processor, dag, node)
        for t in available_magic:
            qts = _oriented_terminals(processor, specs, t)
            options[t] = qts
            costs.append(sum(dist[t].get(p, unreachable) for p in qts) if qts else 10 * unreachable)
        if not any(options.values()):
            continue
        rows.append((node, options, costs))
    if not rows:
        return []
    assigned_rows, assigned_cols = linear_sum_assignment([costs for _, _, costs in rows])
    chosen = {}
    for r, c in zip(assigned_rows, assigned_cols):
        node, options, _ = rows[r]
        t = available_magic[c]
        if options[t]:
            chosen[r] = (node, t, options[t])
    return [chosen[r] for r in sorted(chosen)]


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

    # A magic patch can serve at most one product per time step, also when
    # states are always ready (no source / unlimited source): routes must be
    # node-disjoint, so two products sharing a magic terminal cannot both
    # succeed.  Handing the same terminal to several products made all but
    # one fail and retry next step (~80% of failed routings on dense
    # layouts).  Products beyond the number of terminals simply wait for the
    # next step, exactly as in Silva's EAF baseline.
    ready_set = set(ready_terminals) if ready_terminals is not None else None

    if getattr(processor, '_terminal_policy', 'nearest') == 'matching':
        available_magic = [t for t in processor.magic_terminals if t not in temp_used_magic]
        if ready_set is not None:
            available_magic = [t for t in available_magic if t in ready_set]
        for node, magic_terminal, qubit_terminals in _assign_terminals_matching(
            processor, dag, ready_nodes, available_magic
        ):
            terminals = [magic_terminal] + qubit_terminals
            terminal_sets.append(terminals)
            node_to_terminals[node] = {
                'magic_terminal': magic_terminal,
                'qubit_terminals': qubit_terminals,
                'all_terminals': terminals,
            }
        return terminal_sets, node_to_terminals

    for node in ready_nodes:
        available_magic = [t for t in processor.magic_terminals if t not in temp_used_magic]
        if ready_set is not None:
            available_magic = [t for t in available_magic if t in ready_set]

        if not available_magic:
            logger.debug(f"No free magic terminal left this step for node {node.op.name}")
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
            if magic_terminal in temp_used_magic:
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


def _inject_reuse_terminal(processor, results):
    """Fault ``reuse_terminal``: commit two products of one step on the same terminal."""
    routed = [r for r in results if r['success'] and r.get('magic_terminal') is not None]
    if len(routed) < 2 or not faults.fire_once(processor, "reuse_terminal"):
        return
    shared = routed[0]['magic_terminal']
    routed[1]['magic_terminal'] = shared
    routed[1]['all_terminals'] = [shared] + list(routed[1]['qubit_terminals'])


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


def _free_components(graph, blocked):
    """{node: component id} of ``graph`` without ``blocked`` (graph is symmetric)."""
    label = {}
    for root in graph:
        if root in label or root in blocked:
            continue
        label[root] = root
        stack = [root]
        while stack:
            u = stack.pop()
            for v, _ in graph.get(u, ()):
                if v not in label and v not in blocked:
                    label[v] = root
                    stack.append(v)
    return label


def _retry_failed_with_other_magic(
    processor, dag, graph, results, ready_terminals=None, ready_nodes=(), time_step=None
):
    """Re-route failed products through another free magic terminal (in place).

    Magic terminals are assigned before routing; a product whose assigned
    terminal cannot be reached around the routes committed this step was
    simply deferred, even when another magic patch was free and reachable.
    Silva's EAF attaches the nearest *reachable* free site instead.  This pass
    gives every failed product that chance: free terminals are tried nearest
    first, the data ports are re-oriented toward each candidate, and the
    product is routed on the graph without the routing cells already used in
    this step.

    Ready products that never received a terminal (more products than free
    terminals at assignment time) are tried as well, after the failed ones:
    a terminal left free by an unroutable product must not stay unused while
    another ready product could take it.
    """
    attempted = {r["node"] for r in results}
    for node in ready_nodes:
        if node not in attempted:
            results.append({
                "node": node, "gate_name": node.op.name, "qubits": [],
                "time_step": time_step, "success": False, "magic_wait_cycles": 0,
                "magic_terminal": None, "qubit_terminals": [], "all_terminals": [],
                "steiner_nodes": set(), "steiner_edges": set(),
                "error": "no magic terminal assigned", "_unassigned": True,
            })
    failed = [r for r in results if not r["success"]]
    if not failed:
        return
    committed = set()
    used_magic = set()
    for r in results:
        if r["success"]:
            committed |= {n for n in r["steiner_nodes"] if isinstance(n, tuple)}
            used_magic.add(r["magic_terminal"])
    allowed = set(ready_terminals) if ready_terminals is not None else None
    # The routing graph is symmetric, so a Steiner tree exists iff all its
    # terminals share a component of the free graph: candidates that fail this
    # test are skipped without running the (much costlier) Steiner search.
    components = None
    for r in failed:
        node = r["node"]
        candidates = [
            t for t in processor.magic_terminals
            if t not in used_magic and t != r["magic_terminal"]
            and (allowed is None or t in allowed)
        ]
        if not candidates:
            continue
        anchor = [processor.pos.get(t) for t in r["qubit_terminals"] if t in processor.pos]
        if not anchor:  # never assigned: anchor at the product's data patches
            qubits = [dag.find_bit(q).index for q in node.qargs]
            anchor = [
                processor.pos[t]
                for q in qubits
                for t in processor.ports_by_patch.get(f"q_{q}", [])
                if t in processor.pos
            ]

        def distance(t):
            p = processor.pos.get(t)
            if p is None or not anchor:
                return 0.0
            return min(abs(p[0] - a[0]) + abs(p[1] - a[1]) for a in anchor)

        for magic in sorted(candidates, key=distance):
            qubit_terminals = processor._get_qubit_terminals_with_magic_direction(dag, node, magic)
            if not qubit_terminals:
                continue
            terminals = [magic] + qubit_terminals
            if components is None:
                components = _free_components(graph, committed)
            label = components.get(magic)
            if label is None or any(components.get(t) != label for t in qubit_terminals):
                continue
            try:
                sol_nodes, sol_edges = processor.eng.steiner_tree(
                    graph, terminals, blocked=committed
                )
            except (ValueError, KeyError):
                continue
            r.update(
                success=True,
                magic_terminal=magic,
                qubit_terminals=qubit_terminals,
                all_terminals=terminals,
                steiner_nodes=sol_nodes,
                steiner_edges=sol_edges,
                magic_retry=True,
            )
            r.pop("error", None)
            r.pop("_unassigned", None)
            committed |= {n for n in sol_nodes if isinstance(n, tuple)}
            components = None
            used_magic.add(magic)
            processor.used_magic_terminals.add(magic)
            break
    results[:] = [r for r in results if not r.get("_unassigned")]


def _process_time_step_with_harvest(
    processor, dag, ready_nodes, time_step, working_graph,
    visualize_each_step, ready_terminals=None, config=None
):
    """Process one time step using the HARVEST negotiated-congestion router.

    Mirrors ``_process_time_step_with_pathfinder`` but delegates to
    ``LayoutEngine.steiner_packing_harvest`` and attaches the algorithm's
    internal statistics to every result dict under the ``"harvest_stats"``
    key.  With ``config.negotiate`` false the step is a plain greedy Steiner
    packing (``LayoutEngine.steiner_packing``) instead.
    """
    config = config or HarvestConfig()
    if not ready_nodes:
        return []

    terminal_sets, node_to_terminals = _prepare_terminal_sets(
        processor, dag, ready_nodes, ready_terminals=ready_terminals
    )

    if not terminal_sets:
        logger.warning(f"No valid terminal sets for time step {time_step}")
        return []

    priorities = None
    heights = getattr(processor, '_node_heights', None)
    if heights is not None and getattr(processor, '_router_priority', False):
        nodes_with_terminals = [n for n in ready_nodes if n in node_to_terminals]
        priorities = [heights[n._node_id] for n in nodes_with_terminals]

    logger.info(f"Running HARVEST on {len(terminal_sets)} terminal sets")
    if config.negotiate:
        packing_results, _, harvest_stats = processor.eng.steiner_packing_harvest(
            working_graph,
            terminal_sets,
            max_iters=config.max_iters,
            alpha=config.alpha,
            beta=config.beta,
            stagnation_limit=config.stagnation_limit,
            greedy_order="min_size",
            greedy_fill=config.greedy_fill,
            priorities=priorities,
        )
    else:
        packing_results, _ = processor.eng.steiner_packing(
            working_graph, terminal_sets, greedy_order="min_size"
        )
        harvest_stats = {
            "iterations": 0, "initial_routes": len(terminal_sets), "reroutes": 0,
            "conflicted_reroutes": 0, "dropped": 0, "filled": 0,
            "final_conflict_score": 0, "stopped_reason": "no_negotiation",
        }
    logger.info(
        f"HARVEST step {time_step}: iters={harvest_stats['iterations']}, "
        f"conflicted_reroutes={harvest_stats['conflicted_reroutes']}, "
        f"dropped={harvest_stats['dropped']}, "
        f"stopped_reason={harvest_stats['stopped_reason']}"
    )

    time_step_results = _collect_time_step_results(
        processor, ready_nodes, node_to_terminals, packing_results, time_step
    )

    if config.magic_retry:
        _retry_failed_with_other_magic(
            processor, dag, working_graph, time_step_results, ready_terminals,
            ready_nodes=ready_nodes, time_step=time_step,
        )
    _inject_reuse_terminal(processor, time_step_results)

    # Attach HARVEST-specific metadata to every result in this time step.
    for r in time_step_results:
        r['harvest_stats'] = harvest_stats
        r['algorithm'] = 'harvest'


    return time_step_results


def _inject_skip_dep(processor, tracker, ready_nodes):
    """Fault ``skip_dep``: one product whose predecessors are pending becomes ready."""
    if ready_nodes is None:
        return
    if processor.__dict__.get("_fault_skip_dep_done") or not faults.active("skip_dep"):
        return
    # Injected every step until a premature product is actually routed (then
    # noted as activated); a deferred one later runs legitimately.
    pending = tracker.blocked()
    if pending:
        processor._fault_skip_dep_node = pending[len(pending) // 2]
        ready_nodes.append(processor._fault_skip_dep_node)


def process_dag_with_harvest(processor, dag, visualize_each_step=False, config=None):
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
        config: :class:`HarvestConfig`; ``None`` means the defaults.

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
    tracker = ReadyTracker(dag, all_op_nodes)
    config = config or HarvestConfig()
    processor._terminal_policy = config.terminals
    processor._router_priority = config.router_priority
    heights = tracker.heights() if config.priority == "critical_path" else None
    processor._node_heights = heights
    # One entry per time step (evaluation instrumentation; never read back).
    step_log = processor._step_log = []

    while len(processed_nodes) < len(all_op_nodes):
        logger.info(f"=== Time Step {time_step} ===")

        if factory:
            factory.tick()

        processor.used_magic_terminals = set()

        ready_nodes = tracker.ready()
        if faults.active("skip_dep"):
            _inject_skip_dep(processor, tracker, ready_nodes)

        if not ready_nodes:
            logger.warning("No ready nodes found, breaking to avoid infinite loop")
            break
        if heights is not None:
            # Stable: equal heights keep DAG order.
            ready_nodes.sort(key=lambda n: -heights[n._node_id])

        ready_terminals = (
            factory.get_ready_terminals()
            if factory and not factory.unlimited
            else None
        )

        admission = config.admission or not factory or factory.unlimited
        if admission:
            executable_nodes = _filter_by_magic_availability(
                ready_nodes, factory, ready_terminals
            )
        else:
            executable_nodes = list(ready_nodes)

        if not executable_nodes:
            if factory:
                factory.record_wait_cycle()
            step_log.append({"ready": len(ready_nodes), "executable": 0, "routed": 0})
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
            visualize_each_step, ready_terminals=ready_terminals if admission else None,
            config=config,
        )
        stalled = 0
        if not admission:
            # Without admission control a product routed to a terminal that
            # holds no state cannot execute: its route is discarded (stall).
            ready_set = set(ready_terminals or ())
            for result in time_step_results:
                if (result['success'] and node_needs_magic_state(result['node'])
                        and result['magic_terminal'] not in ready_set):
                    result['success'] = False
                    result['stalled'] = True
                    stalled += 1

        early_consume = faults.active("early_consume")
        if early_consume and factory and not factory.unlimited:
            # Fault: every assigned terminal is consumed, routed or not.
            for result in time_step_results:
                if result.get('magic_terminal') is not None and node_needs_magic_state(result['node']):
                    if factory.consume(result['magic_terminal']) and not result['success']:
                        faults.note("early_consume")
        skipped = processor.__dict__.get("_fault_skip_dep_node")
        if skipped is not None:
            # Only a premature product routed in its injection step violates
            # anything; if it is deferred it later runs legitimately.
            if any(r['success'] and r['node'] is skipped for r in time_step_results):
                faults.note("skip_dep")
                processor._fault_skip_dep_done = True
            processor._fault_skip_dep_node = None

        successful_nodes = []
        for result in time_step_results:
            if result['success']:
                processed_nodes.add(result['node'])
                tracker.mark_done(result['node'])
                all_results.append(result)
                successful_nodes.append(result['node'])
                if (factory and not factory.unlimited and not early_consume
                        and node_needs_magic_state(result['node'])):
                    factory.consume(result['magic_terminal'])

        successful_count = len(successful_nodes)
        failed_count = len(time_step_results) - successful_count
        step_stats = time_step_results[0]['harvest_stats'] if time_step_results else {}
        step_log.append({
            "ready": len(ready_nodes),
            "executable": len(executable_nodes),
            "routed": successful_count,
            "iterations": step_stats.get("iterations", 0),
            "reroutes": step_stats.get("reroutes", 0),
            "dropped": step_stats.get("dropped", 0),
            "filled": step_stats.get("filled", 0),
            "magic_retries": sum(1 for r in time_step_results if r.get("magic_retry")),
            "stalled": stalled,
        })

        logger.info(
            f"Time step {time_step}: {successful_count} successful, "
            f"{failed_count} failed"
        )

        if successful_count == 0 and stalled:
            # Every routed product stalled on an empty terminal: wait a cycle.
            if factory:
                factory.record_wait_cycle()
            idle_streak += 1
            if idle_streak > _MAX_IDLE_CYCLES:
                logger.error("Maximum idle cycles exceeded — aborting.")
                break
            time_step += 1
            continue

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