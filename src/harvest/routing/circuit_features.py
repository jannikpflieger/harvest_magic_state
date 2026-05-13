"""
Circuit-structure feature extraction for adaptive scheduler selection.

Computes a small set of interpretable features from a Qiskit DAGCircuit
(after PCB conversion) that are used by the adaptive scheduler heuristic.
"""

import statistics
from dataclasses import dataclass

from qiskit.dagcircuit import DAGCircuit

from harvest.compilation.circuit_analysis import analyze_dag_layers


@dataclass
class CircuitFeatures:
    """Lightweight summary of circuit structure for scheduler selection.

    Attributes:
        num_qubits:         Number of logical qubits in the DAG.
        num_pauli_products: Total number of PauliEvolution operations.
        avg_weight:         Mean qubit-weight of PauliEvolution operations.
        max_weight:         Maximum qubit-weight of any PauliEvolution operation.
        t_count_ratio:      ``num_pauli_products / num_qubits`` (magic-state
                            demand per logical qubit).
        avg_layer_width:    Mean number of PauliEvolution ops per non-empty
                            dependency layer — a proxy for circuit parallelism.
        dependency_depth:   Number of dependency layers that contain at least
                            one PauliEvolution operation.
    """

    num_qubits: int
    num_pauli_products: int
    avg_weight: float
    max_weight: int
    t_count_ratio: float
    avg_layer_width: float
    dependency_depth: int

    def __str__(self) -> str:
        return (
            f"CircuitFeatures("
            f"qubits={self.num_qubits}, "
            f"products={self.num_pauli_products}, "
            f"avg_weight={self.avg_weight:.2f}, "
            f"max_weight={self.max_weight}, "
            f"t_ratio={self.t_count_ratio:.2f}, "
            f"avg_layer_width={self.avg_layer_width:.2f}, "
            f"dep_depth={self.dependency_depth})"
        )


def compute_circuit_features(dag: DAGCircuit) -> CircuitFeatures:
    """Extract scheduling-relevant features from a Qiskit DAGCircuit.

    Reuses :func:`harvest.compilation.circuit_analysis.analyze_dag_layers`
    so that no logic is duplicated.

    Args:
        dag: A DAGCircuit produced after PCB conversion (containing
             PauliEvolution gates for non-Clifford rotations).

    Returns:
        :class:`CircuitFeatures` populated from the DAG structure.
    """
    layer_stats = analyze_dag_layers(dag)

    num_qubits = dag.num_qubits()
    num_pauli_products = layer_stats["total_pauli_evolutions"]
    avg_weight = layer_stats["avg_pauli_evolution_size"]
    max_weight = layer_stats["max_pauli_evolution_size"]

    t_count_ratio = num_pauli_products / max(num_qubits, 1)

    # avg_layer_width: mean ops-per-layer, considering only non-empty layers
    per_layer = layer_stats["pauli_evolutions_per_layer"]
    non_empty = [c for c in per_layer if c > 0]
    avg_layer_width = statistics.mean(non_empty) if non_empty else 0.0
    dependency_depth = len(non_empty)

    return CircuitFeatures(
        num_qubits=num_qubits,
        num_pauli_products=num_pauli_products,
        avg_weight=float(avg_weight),
        max_weight=int(max_weight),
        t_count_ratio=float(t_count_ratio),
        avg_layer_width=float(avg_layer_width),
        dependency_depth=dependency_depth,
    )
