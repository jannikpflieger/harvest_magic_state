import re

CLIFFORD_T_GATE_SET = {"id", "x", "y", "z", "h", "s", "sdg", "sx", "sxdg", "cx", "cz", "cy", "swap","iswap", "ecr", "dcx", "t", "tdg", "rz"}


def get_pauli_type_for_qubit(node, qubit_index, all_qubits_in_operation):
    """
    Determine the Pauli type (X, Y, or Z) for a specific qubit in a PauliEvolution gate.

    Standalone utility extracted from DAGProcessor._get_port_type_for_pauli_gate
    so it can be reused in circuit summary extraction without depending on a processor instance.

    Args:
        node: A DAGOpNode from a Qiskit DAGCircuit
        qubit_index: The global qubit index to query
        all_qubits_in_operation: List of all qubit indices involved in this operation

    Returns:
        str: 'X', 'Y', or 'Z'
    """
    if hasattr(node.op, 'operator') and node.op.operator is not None:
        op_str = str(node.op.operator)

        if '_' in op_str:
            try:
                qubit_position = all_qubits_in_operation.index(qubit_index)
            except ValueError:
                return 'Z'

            pattern = rf'([XYZ])_{qubit_position}(?:\D|$)'
            match = re.search(pattern, op_str)
            if match:
                return match.group(1)
        else:
            if 'Y' in op_str:
                return 'Y'
            elif 'Z' in op_str:
                return 'Z'
            elif 'X' in op_str:
                return 'X'

    elif hasattr(node.op, 'pauli') and node.op.pauli is not None:
        pauli_str = str(node.op.pauli)
        if qubit_index < len(pauli_str):
            pauli_op = pauli_str[-(qubit_index + 1)]
            if pauli_op in ('X', 'Y', 'Z'):
                return pauli_op

    return 'Z'


def node_needs_magic_state(node) -> bool:
    """Return True if the DAG operation node requires a magic state.

    After PCB conversion every non-Clifford rotation has been absorbed into
    a ``PauliEvolution`` gate, so checking the gate name is sufficient.
    """
    return getattr(node, 'op', None) is not None and node.op.name == 'PauliEvolution'