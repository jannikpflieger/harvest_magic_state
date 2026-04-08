"""
Extract a CircuitSummary from a Qiskit DAGCircuit.

The summary captures qubit interaction structure (weighted adjacency),
per-qubit Pauli profiles, and layer-level parallelism metrics — everything
the placement heuristic needs to make circuit-aware decisions.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Optional, Tuple

from qiskit.dagcircuit import DAGCircuit

from harvest.compilation.circuit_analysis import analyze_dag_layers
from harvest.compilation.utils import get_pauli_type_for_qubit


@dataclass
class CircuitSummary:
    """Compact description of DAG-level circuit structure."""

    qubit_list: List[int]
    """Sorted list of logical qubit indices that appear in the DAG."""

    interaction_graph: Dict[Tuple[int, int], float]
    """Weighted adjacency: (q_i, q_j) -> co-occurrence weight (i < j)."""

    pauli_profile: Dict[int, Counter]
    """Per-qubit Pauli type histogram: qubit -> Counter({'X': n, 'Y': m, 'Z': k})."""

    parallelism_profile: Dict
    """Output of analyze_dag_layers() — layer widths, max/median Pauli counts, etc."""

    num_qubits: int = 0
    """Total number of logical qubits."""

    total_pauli_evolutions: int = 0
    """Total PauliEvolution gates in the DAG."""

    magic_demand: Optional[Dict[int, int]] = field(default=None)
    """Per-qubit magic/T-state demand. Not available at DAG stage; placeholder."""


def extract_circuit_summary(dag: DAGCircuit) -> CircuitSummary:
    """
    Walk every PauliEvolution op-node in *dag* and build a CircuitSummary.

    For each PauliEvolution node:
      - extract the qubit indices it acts on
      - for every qubit, determine the Pauli type (X/Y/Z)
      - for every qubit *pair* in the node, increment the interaction weight

    The interaction weight between two qubits counts how many PauliEvolution
    gates they participate in together; this is the primary signal that the
    placement heuristic uses to decide which qubits should be close on the
    lattice.
    """
    interaction: Dict[Tuple[int, int], float] = {}
    pauli_profile: Dict[int, Counter] = {}
    qubit_set: set[int] = set()
    total_evolutions = 0

    for node in dag.op_nodes():
        if 'PauliEvolution' not in node.op.__class__.__name__:
            continue

        total_evolutions += 1
        qubit_indices = [dag.find_bit(q).index for q in node.qargs]
        qubit_set.update(qubit_indices)

        # Per-qubit Pauli type
        for qi in qubit_indices:
            ptype = get_pauli_type_for_qubit(node, qi, qubit_indices)
            if qi not in pauli_profile:
                pauli_profile[qi] = Counter()
            pauli_profile[qi][ptype] += 1

        # Interaction edges (undirected, canonical order)
        for qi, qj in combinations(qubit_indices, 2):
            key = (min(qi, qj), max(qi, qj))
            interaction[key] = interaction.get(key, 0.0) + 1.0

    qubit_list = sorted(qubit_set)

    # Reuse the existing layer analysis
    parallelism = analyze_dag_layers(dag)

    return CircuitSummary(
        qubit_list=qubit_list,
        interaction_graph=interaction,
        pauli_profile=pauli_profile,
        parallelism_profile=parallelism,
        num_qubits=len(qubit_list),
        total_pauli_evolutions=total_evolutions,
        magic_demand=None,
    )
