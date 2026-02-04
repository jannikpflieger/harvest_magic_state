import logging
from datetime import datetime
from circuit.circuit_to_pbc_dag import convert_to_PCB, convert_to_CliffordT, create_dag, create_random_circuit
from dag_steiner_processor import process_dag_with_steiner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'harvest_magic_state_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()  # Keep some output to console
    ]
)
logger = logging.getLogger('HarvestMagicState')

# Reduce console output by setting specific loggers to file only
file_handler = logging.FileHandler(f'harvest_magic_state_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

# Create a logger that only logs to file for detailed information
detailed_logger = logging.getLogger('HarvestMagicState.Detailed')
detailed_logger.addHandler(file_handler)
detailed_logger.setLevel(logging.INFO)
detailed_logger.propagate = False


def start_pipeline(circuit, lattice_layout=None, is_PCB=False, is_CliffordT=False, 
                  layout_rows=4, layout_cols=4, visualize_steps=False):
    """
    Complete pipeline: convert circuit -> create DAG -> process with Steiner algorithm.
    
    Args:
        circuit: Input quantum circuit
        lattice_layout: Custom lattice layout (not implemented yet)
        is_PCB (bool): Whether input is already in PCB format
        is_CliffordT (bool): Whether input is already in Clifford+T format
        layout_rows (int): Number of rows in lattice layout
        layout_cols (int): Number of columns in lattice layout
        visualize_steps (bool): Whether to visualize each processing step
        
    Returns:
        tuple: (DAGProcessor instance, processing results, final DAG)
    """
    logger.info("Starting Harvest Magic State Pipeline")
    
    # Step 1: Convert to appropriate format
    detailed_logger.info("Step 1: Circuit preprocessing")
    if not is_CliffordT:
        detailed_logger.info("Converting to Clifford+T (placeholder)")
        # circuit = convert_to_CliffordT(circuit)  # Currently a stub
    
    if not is_PCB:
        detailed_logger.info("Converting to PCB format")
        circuit = convert_to_PCB(circuit)
        detailed_logger.info(f"PCB circuit depth: {circuit.depth()}")
        detailed_logger.info(f"PCB circuit gates: {circuit.count_ops()}")

    # Step 2: Create DAG
    detailed_logger.info("Step 2: Creating DAG")
    dag = create_dag(circuit)
    detailed_logger.info(f"DAG operations: {dag.count_ops()}")
    detailed_logger.info(f"DAG depth: {dag.depth()}")
    detailed_logger.info(f"DAG qubits: {dag.num_qubits()}")

    # Step 3: Process DAG with Steiner algorithm
    detailed_logger.info(f"Step 3: Processing DAG with Steiner algorithm")
    detailed_logger.info(f"Using {layout_rows}x{layout_cols} lattice layout")
    processor, results = process_dag_with_steiner(
        dag, layout_rows=layout_rows, layout_cols=layout_cols, 
        visualize_steps=visualize_steps
    )

    logger.info("Pipeline completed successfully")
    
    return processor, results, dag


if __name__ == "__main__":
    # Create a test circuit
    print("Creating test quantum circuit...")
    qc = create_random_circuit(num_qubits=5, depth=20, seed=42)
    
    # Print circuit information to console
    print(f"\nOriginal Circuit:")
    print(f"  Qubits: {qc.num_qubits}")
    print(f"  Depth: {qc.depth()}")
    print(f"  Gates: {qc.count_ops()}")
    print(f"\nCircuit structure:")
    print(qc)
    
    # Log detailed information
    logger.info(f"Created circuit with {qc.num_qubits} qubits, depth {qc.depth()}")
    detailed_logger.info(f"Circuit gates breakdown: {qc.count_ops()}")
    
    # Run the complete pipeline
    processor, results, dag = start_pipeline(
        qc, 
        layout_rows=4, 
        layout_cols=4, 
        visualize_steps=False  # Set to True for step-by-step visualization
    )
    
    print(f"\nProcessing completed! Processed {len(results)} DAG nodes")
    print(f"Check the log file for detailed execution information.")
