"""
Policies for choosing magic state terminals based on proximity and availability.
"""

import logging

logger = logging.getLogger('HarvestMagicState.DAGProcessor')
detailed_logger = logging.getLogger('HarvestMagicState.Detailed')


def get_magic_terminals(ports_by_patch):
    """Get all available magic state terminals from the layout."""
    magic_terminals = []
    for patch_name, patch_ports in ports_by_patch.items():
        if 'M' in patch_ports:
            magic_terminals.extend(patch_ports['M'])
    return magic_terminals


def choose_magic_terminal(magic_terminals, used_magic_terminals):
    """
    Choose an available magic state terminal (first-available strategy).

    Returns:
        A magic state terminal that hasn't been used yet, or None if all are used.
    """
    available_terminals = [t for t in magic_terminals if t not in used_magic_terminals]
    if not available_terminals:
        logger.warning("No more magic terminals available!")
        return None

    chosen = available_terminals[0]
    detailed_logger.info(f"Selected magic terminal: {chosen}")
    return chosen


def calculate_distance(pos, terminal1, terminal2):
    """Calculate Euclidean distance between two terminals."""
    pos1 = pos.get(terminal1)
    pos2 = pos.get(terminal2)
    if pos1 is None or pos2 is None:
        return float('inf')
    return ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5


def choose_optimal_magic_terminal(pos, magic_terminals, used_magic_terminals, qubit_terminals, available_magic_terminals=None):
    """Choose the magic terminal closest to the data qubit terminals.

    Args:
        pos: Position dictionary mapping terminals to (x, y) coords
        magic_terminals: All magic terminals in the layout
        used_magic_terminals: Set of already-used magic terminals
        qubit_terminals: List of data qubit terminal ports
        available_magic_terminals: List of available magic terminals. If None, uses all unused terminals.

    Returns:
        Best magic terminal or None if none available
    """
    if available_magic_terminals is None:
        available_magic_terminals = [t for t in magic_terminals if t not in used_magic_terminals]

    if not available_magic_terminals or not qubit_terminals:
        return None

    best_magic = None
    min_avg_distance = float('inf')

    for magic_terminal in available_magic_terminals:
        total_distance = 0
        for qubit_terminal in qubit_terminals:
            total_distance += calculate_distance(pos, magic_terminal, qubit_terminal)

        avg_distance = total_distance / len(qubit_terminals)

        if avg_distance < min_avg_distance:
            min_avg_distance = avg_distance
            best_magic = magic_terminal

    detailed_logger.info(f"Selected optimal magic terminal: {best_magic} (avg distance: {min_avg_distance:.2f})")
    return best_magic
