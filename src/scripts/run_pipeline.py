#!/usr/bin/env python3
"""
Run the Harvest Magic State pipeline on a quantum circuit.
"""

import logging
from datetime import datetime

from harvest.compilation.pauli_block_conversion import convert_to_PCB, create_dag, create_random_circuit
from harvest.routing import process_dag_with_steiner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/harvest_magic_state_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('HarvestMagicState')

file_handler = logging.FileHandler(f'logs/harvest_magic_state_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

detailed_logger = logging.getLogger('HarvestMagicState.Detailed')
detailed_logger.addHandler(file_handler)
detailed_logger.setLevel(logging.INFO)
detailed_logger.propagate = False


def start_pipeline(circuit, lattice_layout=None, is_PCB=False, is_CliffordT=False,
                  layout_rows=4, layout_cols=4, visualize_steps=False, mode="steiner_packing"):
    """
    Complete pipeline: convert circuit -> create DAG -> process with Steiner algorithm.

    Returns:
        tuple: (DAGProcessor instance, processing results, final DAG)
    """
    logger.info("Starting Harvest Magic State Pipeline")

    detailed_logger.info("Step 1: Circuit preprocessing")
    if not is_CliffordT:
        detailed_logger.info("Converting to Clifford+T (placeholder)")

    if not is_PCB:
        detailed_logger.info("Converting to PCB format")
        circuit = convert_to_PCB(circuit)
        detailed_logger.info(f"PCB circuit depth: {circuit.depth()}")
        detailed_logger.info(f"PCB circuit gates: {circuit.count_ops()}")

    detailed_logger.info("Step 2: Creating DAG")
    dag = create_dag(circuit)
    detailed_logger.info(f"DAG operations: {dag.count_ops()}")
    detailed_logger.info(f"DAG depth: {dag.depth()}")
    detailed_logger.info(f"DAG qubits: {dag.num_qubits()}")

    detailed_logger.info(f"Step 3: Processing DAG with Steiner algorithm (mode: {mode})")
    detailed_logger.info(f"Using {layout_rows}x{layout_cols} lattice layout")
    processor, results = process_dag_with_steiner(
        dag, layout_rows=layout_rows, layout_cols=layout_cols,
        visualize_steps=visualize_steps, mode=mode
    )

    logger.info("Pipeline completed successfully")

    return processor, results, dag


if __name__ == "__main__":
    print("Creating test quantum circuit...")
    qc = create_random_circuit(num_qubits=20, depth=10, seed=42)

    print(f"\nOriginal Circuit:")
    print(f"  Qubits: {qc.num_qubits}")
    print(f"  Depth: {qc.depth()}")
    print(f"  Gates: {qc.count_ops()}")
    print(f"\nCircuit structure:")
    print(qc)

    logger.info(f"Created circuit with {qc.num_qubits} qubits, depth {qc.depth()}")
    detailed_logger.info(f"Circuit gates breakdown: {qc.count_ops()}")

    print("\nRunning pipeline with Steiner forest (parallel processing)...")
    processor, results, dag = start_pipeline(
        qc,
        layout_rows=5,
        layout_cols=5,
        visualize_steps=True,
        mode="steiner_packing"
    )

    print(f"\nProcessing completed! Processed {len(results)} DAG nodes")
    print(f"Check the log file for detailed execution information.")
