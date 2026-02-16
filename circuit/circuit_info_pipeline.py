import os
import glob
import json
import statistics
from pathlib import Path
from typing import List, Tuple, Dict
import time
from circuit_to_pbc_dag import qasm_to_circuit, convert_to_PCB, create_dag
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import RemoveFinalMeasurements, RemoveBarriers
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary as sel
from qiskit import QuantumCircuit

from bqskit.ft import CliffordTModel
from bqskit import compile
from bqskit.ir.circuit import Circuit
from bqskit.ext import qiskit_to_bqskit, bqskit_to_qiskit


def is_identity_op(op) -> bool:
    g = op.gate
    # conservative: only drop if it's clearly identity
    return getattr(g, "name", "").lower() in {"i", "id", "identity", "identity1"}

def strip_identities_by_name(circ):
    new = Circuit(circ.num_qudits, circ.radixes)
    for op in circ:
        if is_identity_op(op):
            continue
        new.append(op)
    return new

def strip_qiskit_identities(qc: QuantumCircuit) -> QuantumCircuit:
    new = QuantumCircuit(*qc.qregs, *qc.cregs, name=qc.name)

    for inst, qargs, cargs in qc.data:
        name = inst.name.lower()

        # Common identity names: 'id', 'i' and your custom 'identity1'
        if name in {"id", "i"} or name.startswith("identity"):
            continue

        new.append(inst, qargs, cargs)

    return new

def find_qasm_files(base_dir: str) -> List[str]:
    """
    Recursively find all .qasm files in the given directory.
    
    Args:
        base_dir: Base directory to search in
        
    Returns:
        List of paths to all .qasm files found
    """
    qasm_files = []
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"Directory {base_dir} does not exist!")
        return qasm_files
    
    # Use glob to find all .qasm files recursively
    pattern = os.path.join(base_dir, "**", "*.qasm")
    qasm_files = glob.glob(pattern, recursive=True)
    
    print(f"Found {len(qasm_files)} QASM files in {base_dir}")
    return sorted(qasm_files)

def convert_rx_ry_to_rz(circuit):
    """
    Convert rx and ry gates to rz gates using basis translation.
    rx(θ) -> h, rz(θ), h
    ry(θ) -> ry(π/2), rz(θ), ry(-π/2) -> can be simplified to rz equivalents
    
    Args:
        circuit: QuantumCircuit with rx/ry gates
        
    Returns:
        QuantumCircuit with rx/ry converted to supported gates
    """
    from qiskit.transpiler.passes import UnrollCustomDefinitions, BasisTranslator
    from qiskit.circuit.library import RZGate, HGate, RXGate, RYGate
    
    # Define target basis (gates we want to keep)
    target_basis = ['id', 'x', 'y', 'z', 'h', 's', 'sdg', 'sx', 'sxdg', 
                   'cx', 'cz', 'cy', 'swap', 'iswap', 'ecr', 'dcx', 't', 'tdg', 'rz']
    
    # Create pass manager for basis translation
    pm = PassManager()
    pm.append(UnrollCustomDefinitions(sel, target_basis))
    pm.append(BasisTranslator(sel, target_basis))
    
    # Apply the transformation
    converted_circuit = pm.run(circuit)
    
    return converted_circuit

def has_unsupported_gates(circuit) -> Tuple[bool, List[str]]:
    """
    Check if circuit has gates that are unsupported by the PCB conversion.
    
    Args:
        circuit: QuantumCircuit to check
        
    Returns:
        (has_unsupported, unsupported_gates_list)
    """
    # Define supported gates for PCB conversion (including rx, ry for conversion)
    clifford_gates = {"id", "x", "y", "z", "h", "s", "sdg", "sx", "sxdg", 
                     "cx", "cz", "cy", "swap", "iswap", "ecr", "dcx"}
    rz_gates = {"t", "tdg", "rz"}
    convertible_gates = {"rx", "ry"}  # Gates we can convert to supported ones
    supported_gates = clifford_gates | rz_gates | convertible_gates
    
    gate_counts = circuit.count_ops()
    unsupported_gates = []
    
    for gate in gate_counts.keys():
        if gate not in supported_gates:
            unsupported_gates.append(gate)
    
    return len(unsupported_gates) > 0, unsupported_gates

def count_non_clifford_gates(circuit) -> int:
    """
    Count the number of non-Clifford gates in a circuit.
    
    Args:
        circuit: QuantumCircuit
        
    Returns:
        Number of non-Clifford gates
    """
    # Define Clifford gates
    clifford_gates = {"id", "x", "y", "z", "h", "s", "sdg", "sx", "sxdg", 
                     "cx", "cz", "cy", "swap", "iswap", "ecr", "dcx", "measure"}
    
    # Count non-Clifford gates
    gate_counts = circuit.count_ops()
    non_clifford_count = 0
    
    for gate, count in gate_counts.items():
        if gate not in clifford_gates:
            non_clifford_count += count
    
    return non_clifford_count

def analyze_dag_layers(dag: DAGCircuit) -> Dict:
    """
    Analyze DAG layers and count Pauli evolutions per layer, including their sizes.
    
    Args:
        dag: DAGCircuit to analyze
        
    Returns:
        Dictionary with layer analysis information
    """
    layers = list(dag.layers())
    num_layers = len(layers)
    
    pauli_evolutions_per_layer = []
    pauli_evolution_sizes_per_layer = []
    all_pauli_sizes = []
    
    for layer in layers:
        pauli_count = 0
        layer_sizes = []
        for node in layer['graph'].op_nodes():
            if 'PauliEvolution' in str(node.op.__class__.__name__):
                pauli_count += 1
                # Get the size (number of qubits) of the PauliEvolution
                size = len(node.qargs) if node.qargs else 0
                layer_sizes.append(size)
                all_pauli_sizes.append(size)
        
        pauli_evolutions_per_layer.append(pauli_count)
        pauli_evolution_sizes_per_layer.append(layer_sizes)
    
    # Calculate size statistics
    max_pauli_per_layer = max(pauli_evolutions_per_layer) if pauli_evolutions_per_layer else 0
    median_pauli_per_layer = statistics.median(pauli_evolutions_per_layer) if pauli_evolutions_per_layer else 0
    
    # Calculate size-related statistics
    max_pauli_evolution_size = max(all_pauli_sizes) if all_pauli_sizes else 0
    avg_pauli_evolution_size = statistics.mean(all_pauli_sizes) if all_pauli_sizes else 0
    
    # Calculate average sizes per layer (only for layers with Pauli evolutions)
    avg_pauli_sizes_per_layer = []
    for sizes in pauli_evolution_sizes_per_layer:
        if sizes:  # Only if there are Pauli evolutions in this layer
            avg_pauli_sizes_per_layer.append(statistics.mean(sizes))
        else:
            avg_pauli_sizes_per_layer.append(0)
    
    return {
        'num_layers': num_layers,
        'pauli_evolutions_per_layer': pauli_evolutions_per_layer,
        'max_pauli_per_layer': max_pauli_per_layer,
        'median_pauli_per_layer': median_pauli_per_layer,
        'total_pauli_evolutions': sum(pauli_evolutions_per_layer),
        'pauli_evolution_sizes_per_layer': pauli_evolution_sizes_per_layer,
        'avg_pauli_sizes_per_layer': avg_pauli_sizes_per_layer,
        'max_pauli_evolution_size': max_pauli_evolution_size,
        'avg_pauli_evolution_size': avg_pauli_evolution_size
    }

def save_individual_circuit_json(circuit_data: Dict, output_dir: str) -> str:
    """
    Save individual circuit analysis to JSON file.
    
    Args:
        circuit_data: Dictionary with circuit analysis
        output_dir: Directory to save JSON files
        
    Returns:
        Path to saved JSON file
    """
    circuit_id = circuit_data['circuit_id']
    json_filename = f"{circuit_id}.json"
    json_path = os.path.join(output_dir, json_filename)
    
    # Add individual file metadata
    json_data = {
        'metadata': {
            'analysis_timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'circuit_file': circuit_data['file_path']
        },
        'circuit': circuit_data
    }
    
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2, sort_keys=True)
    
    return json_path

def analyze_single_circuit(qasm_file: str, output_dir: str = None) -> Dict:
    """
    Perform comprehensive analysis of a single QASM circuit.
    
    Args:
        qasm_file: Path to QASM file
        output_dir: Directory to save individual JSON files (optional)
        
    Returns:
        Dictionary with all circuit information
    """
    try:
        # Basic circuit loading and analysis
        start_time = time.time()
        circuit = qasm_to_circuit(qasm_file)
        load_time = time.time() - start_time
        
        # Extract circuit identifier from filename
        circuit_name = os.path.basename(qasm_file).replace('.qasm', '')
        circuit_id = os.path.relpath(qasm_file).replace('/', '_').replace('.qasm', '')
        
        # Check if circuit needs rx/ry conversion
        gate_counts = circuit.count_ops()
        needs_conversion = 'rx' in gate_counts or 'ry' in gate_counts

        # Remove measurements if any
        pm = PassManager(RemoveFinalMeasurements())
        circuit = pm.run(circuit)

        #bqs_circuit = qiskit_to_bqskit(circuit)
        #model = CliffordTModel(bqs_circuit.num_qudits)

        #ft_circuit = compile(bqs_circuit, model, optimization_level=2)
        #print(f"📊 BQSKit compiled circuit gates: {dict(ft_circuit.count_ops())}")

        #ft_circuit = strip_identities_by_name(ft_circuit)

        #transpiled_circuit = bqskit_to_qiskit(ft_circuit)
        #circuit = strip_qiskit_identities(transpiled_circuit)
        
        if needs_conversion:
            print(f"Converting rx/ry gates to rz equivalents...")
            conversion_start = time.time()
            circuit = convert_rx_ry_to_rz(circuit)
            conversion_time = time.time() - conversion_start
            print(f"Conversion completed in {conversion_time:.3f}s")
            print(f"New gate counts: {dict(circuit.count_ops())}")
        else:
            conversion_time = 0.0
        
        # Basic circuit info
        basic_info = {
            'circuit_name': circuit_name,
            'circuit_id': circuit_id,
            'file_path': os.path.relpath(qasm_file),
            'num_qubits': circuit.num_qubits,
            'depth': circuit.depth(),
            'total_gates': sum(circuit.count_ops().values()),
            'gate_counts': dict(circuit.count_ops()),
            'non_clifford_gates': count_non_clifford_gates(circuit),
            'load_time_seconds': load_time,
            'rx_ry_conversion_needed': needs_conversion,
            'rx_ry_conversion_time_seconds': conversion_time
        }
        
        # PCB conversion analysis
        try:
            pcb_start_time = time.time()
            pcb_circuit = convert_to_PCB(circuit, verbose=False)
            pcb_time = time.time() - pcb_start_time
            
            pcb_info = {
                'pcb_conversion_successful': True,
                'pcb_depth': pcb_circuit.depth(),
                'pcb_total_gates': sum(pcb_circuit.count_ops().values()),
                'pcb_gate_counts': dict(pcb_circuit.count_ops()),
                'pcb_conversion_time_seconds': pcb_time
            }
            
            # DAG analysis
            try:
                dag_start_time = time.time()
                dag = create_dag(pcb_circuit)
                dag_time = time.time() - dag_start_time
                
                dag_analysis = analyze_dag_layers(dag)
                dag_info = {
                    'dag_analysis_successful': True,
                    'dag_creation_time_seconds': dag_time,
                    **dag_analysis
                }
            except Exception as dag_error:
                dag_info = {
                    'dag_analysis_successful': False,
                    'dag_error': str(dag_error),
                    'num_layers': None,
                    'pauli_evolutions_per_layer': None,
                    'max_pauli_per_layer': None,
                    'median_pauli_per_layer': None,
                    'total_pauli_evolutions': None,
                    'pauli_evolution_sizes_per_layer': None,
                    'avg_pauli_sizes_per_layer': None,
                    'max_pauli_evolution_size': None,
                    'avg_pauli_evolution_size': None
                }
                
        except Exception as pcb_error:
            pcb_info = {
                'pcb_conversion_successful': False,
                'pcb_error': str(pcb_error),
                'pcb_depth': None,
                'pcb_total_gates': None,
                'pcb_gate_counts': None
            }
            dag_info = {
                'dag_analysis_successful': False,
                'dag_error': 'PCB conversion failed',
                'num_layers': None,
                'pauli_evolutions_per_layer': None,
                'max_pauli_per_layer': None,
                'median_pauli_per_layer': None,
                'total_pauli_evolutions': None,
                'pauli_evolution_sizes_per_layer': None,
                'avg_pauli_sizes_per_layer': None,
                'max_pauli_evolution_size': None,
                'avg_pauli_evolution_size': None
            }
        
        # Combine all information
        result = {
            'analysis_status': 'SUCCESS',
            'error': None,
            **basic_info,
            **pcb_info,
            **dag_info
        }
        
        # Save individual JSON file if output directory is provided
        if output_dir:
            try:
                json_path = save_individual_circuit_json(result, output_dir)
                result['json_file_saved'] = json_path
            except Exception as e:
                print(f"Warning: Failed to save JSON for {circuit_name}: {e}")
                result['json_file_saved'] = None
        
        return result
        
    except Exception as e:
        # Extract what we can even if analysis fails
        circuit_name = os.path.basename(qasm_file).replace('.qasm', '')
        circuit_id = os.path.relpath(qasm_file).replace('/', '_').replace('.qasm', '')
        
        return {
            'analysis_status': 'FAILED',
            'error': str(e),
            'circuit_name': circuit_name,
            'circuit_id': circuit_id,
            'file_path': os.path.relpath(qasm_file),
            'num_qubits': None,
            'depth': None,
            'total_gates': None,
            'gate_counts': None,
            'non_clifford_gates': None,
            'pcb_conversion_successful': False,
            'dag_analysis_successful': False
        }

def pre_prep_circuit(circuit):
    # Remove measurements and barriers before analysis
    pm = PassManager()
    pm.append(RemoveFinalMeasurements())
    pm.append(RemoveBarriers())  # Remove barriers to avoid issues with gate checking
    # Remove barrier gates as they're not needed for gate checking
    circuit = pm.run(circuit)

    return circuit

def filter_supported_circuits(qasm_files: List[str]) -> Tuple[List[str], Dict[str, List[str]]]:
    """
    Filter circuits to only include those with supported gates for PCB conversion.
    
    Args:
        qasm_files: List of QASM file paths
        
    Returns:
        (supported_circuits, skipped_circuits_info)
    """
    supported_circuits = []
    skipped_circuits = {}
    
    print(f"\n{'='*80}")
    print(f"Filtering circuits for supported gates...")
    print(f"{'='*80}")
    
    for qasm_file in qasm_files:
        relative_path = os.path.relpath(qasm_file)
        
        try:
            # Load circuit quickly to check gates
            circuit = qasm_to_circuit(qasm_file)
            
            # Preprocess circuit: remove measurements and barriers before checking supported gates
            circuit = pre_prep_circuit(circuit)

            has_unsupported, unsupported_gates = has_unsupported_gates(circuit)
            
            if has_unsupported:
                skipped_circuits[relative_path] = unsupported_gates
                print(f"SKIP - {relative_path}: unsupported gates {unsupported_gates}")
            else:
                supported_circuits.append(qasm_file)
                print(f"✓ OK - {relative_path} ({circuit.num_qubits} qubits)")
                
        except Exception as e:
            skipped_circuits[relative_path] = [f"Loading error: {e}"]
            print(f"SKIP - {relative_path}: loading error")
    
    print(f"\nFiltering complete: {len(supported_circuits)} supported, {len(skipped_circuits)} skipped")
    return supported_circuits, skipped_circuits

def test_qasm_conversion(qasm_files: List[str], output_dir: str = None) -> Dict[str, Dict]:
    """
    Test converting each QASM file and perform comprehensive analysis.
    
    Args:
        qasm_files: List of paths to QASM files
        
    Returns:
        Dictionary with results for each file
    """
    results = {}
    successful = 0
    failed = 0
    
    print(f"\n{'='*80}")
    print(f"Performing comprehensive analysis of {len(qasm_files)} QASM files")
    print(f"{'='*80}")
    
    for i, qasm_file in enumerate(qasm_files, 1):
        relative_path = os.path.relpath(qasm_file)
        print(f"\n[{i}/{len(qasm_files)}] Analyzing: {relative_path}")
        
        result = analyze_single_circuit(qasm_file, output_dir)
        results[qasm_file] = result
        
        if result['analysis_status'] == 'SUCCESS':
            successful += 1
            pcb_status = "✓" if result.get('pcb_conversion_successful') else "✗"
            dag_status = "✓" if result.get('dag_analysis_successful') else "✗"
            print(f"✓ SUCCESS - {result['num_qubits']} qubits, depth {result['depth']}, "
                  f"{result['non_clifford_gates']} non-Clifford gates")
            print(f"  PCB: {pcb_status}, DAG: {dag_status}")
            if result.get('dag_analysis_successful'):
                print(f"  Layers: {result['num_layers']}, Max Pauli/layer: {result['max_pauli_per_layer']}")
        else:
            failed += 1
            print(f"✗ FAILED - Error: {result['error']}")
    
    print(f"\n{'='*80}")
    print(f"SUMMARY: {successful} successful, {failed} failed analyses")
    print(f"{'='*80}")
    
    return results

def analyze_results(results: Dict[str, Dict]) -> None:
    """
    Analyze and print detailed statistics about the analysis results.
    
    Args:
        results: Dictionary with analysis results
    """
    successful_results = [r for r in results.values() if r['analysis_status'] == 'SUCCESS']
    failed_results = [r for r in results.values() if r['analysis_status'] == 'FAILED']
    
    print(f"\n{'='*80}")
    print("DETAILED ANALYSIS REPORT")
    print(f"{'='*80}")
    
    if successful_results:
        print(f"\nSuccessful analyses: {len(successful_results)}")
        
        # Basic circuit statistics
        qubit_counts = [r['num_qubits'] for r in successful_results]
        depths = [r['depth'] for r in successful_results]
        total_gates = [r['total_gates'] for r in successful_results]
        non_clifford_gates = [r['non_clifford_gates'] for r in successful_results]
        
        print(f"  Qubit range: {min(qubit_counts)} - {max(qubit_counts)} (avg: {sum(qubit_counts)/len(qubit_counts):.1f})")
        print(f"  Depth range: {min(depths)} - {max(depths)} (avg: {sum(depths)/len(depths):.1f})")
        print(f"  Gate count range: {min(total_gates)} - {max(total_gates)} (avg: {sum(total_gates)/len(total_gates):.1f})")
        print(f"  Non-Clifford gates range: {min(non_clifford_gates)} - {max(non_clifford_gates)} (avg: {sum(non_clifford_gates)/len(non_clifford_gates):.1f})")
        
        # PCB conversion statistics
        pcb_successful = [r for r in successful_results if r.get('pcb_conversion_successful', False)]
        print(f"\n  PCB Conversions: {len(pcb_successful)}/{len(successful_results)} successful")
        
        if pcb_successful:
            pcb_depths = [r['pcb_depth'] for r in pcb_successful]
            pcb_gates = [r['pcb_total_gates'] for r in pcb_successful]
            print(f"    PCB depth range: {min(pcb_depths)} - {max(pcb_depths)} (avg: {sum(pcb_depths)/len(pcb_depths):.1f})")
            print(f"    PCB gate count range: {min(pcb_gates)} - {max(pcb_gates)} (avg: {sum(pcb_gates)/len(pcb_gates):.1f})")
        
        # DAG analysis statistics
        dag_successful = [r for r in successful_results if r.get('dag_analysis_successful', False)]
        print(f"\n  DAG Analyses: {len(dag_successful)}/{len(successful_results)} successful")
        
        if dag_successful:
            layer_counts = [r['num_layers'] for r in dag_successful if r['num_layers'] is not None]
            max_pauli_counts = [r['max_pauli_per_layer'] for r in dag_successful if r['max_pauli_per_layer'] is not None]
            total_pauli_counts = [r['total_pauli_evolutions'] for r in dag_successful if r['total_pauli_evolutions'] is not None]
            
            if layer_counts:
                print(f"    Layer count range: {min(layer_counts)} - {max(layer_counts)} (avg: {sum(layer_counts)/len(layer_counts):.1f})")
            if max_pauli_counts:
                print(f"    Max Pauli/layer range: {min(max_pauli_counts)} - {max(max_pauli_counts)} (avg: {sum(max_pauli_counts)/len(max_pauli_counts):.1f})")
            if total_pauli_counts:
                print(f"    Total Pauli evolutions range: {min(total_pauli_counts)} - {max(total_pauli_counts)} (avg: {sum(total_pauli_counts)/len(total_pauli_counts):.1f})")
        
        # Most common gates across all circuits
        all_gates = {}
        for r in successful_results:
            if r['gate_counts']:
                for gate, count in r['gate_counts'].items():
                    all_gates[gate] = all_gates.get(gate, 0) + count
        
        if all_gates:
            print(f"\n  Most common gates: {sorted(all_gates.items(), key=lambda x: x[1], reverse=True)[:10]}")
    
    if failed_results:
        print(f"\nFailed analyses: {len(failed_results)}")
        
        # Group errors by type
        error_types = {}
        for file_path, r in results.items():
            if r['analysis_status'] == 'FAILED':
                error = r['error']
                error_types[error] = error_types.get(error, []) + [os.path.relpath(file_path)]
        
        print("  Error types:")
        for error, files in error_types.items():
            print(f"    {error}: {len(files)} files")
            if len(files) <= 3:  # Show files if few
                for file in files:
                    print(f"      - {file}")
            elif len(files) <= 10:  # Show first few if not too many
                for file in files[:3]:
                    print(f"      - {file}")
                print(f"      ... and {len(files)-3} more")

def run_qasm_pipeline(benchmark_dir: str = None, max_files: int = None, 
                      subdirs: List[str] = None, output_json: str = None, 
                      skip_unsupported: bool = True) -> Dict[str, Dict]:
    """
    Main pipeline function to perform comprehensive analysis of all QASM files.
    
    Args:
        benchmark_dir: Directory containing QASM files. If None, uses default.
        max_files: Maximum number of files to test (for debugging/testing). If None, tests all.
        subdirs: List of subdirectories to test (e.g., ['feynman', 'bigint']). If None, tests all.
        output_json: Path to save JSON results. If None, uses default naming.
        skip_unsupported: If True, skip circuits with unsupported gates for PCB conversion.
        
    Returns:
        Dictionary with analysis results
    """
    if benchmark_dir is None:
        # Default to the benchmark circuits directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        benchmark_dir = os.path.join(current_dir, "benchmark_circuits", "qasm")
    
    print(f"Starting comprehensive QASM analysis pipeline...")
    print(f"Searching for QASM files in: {benchmark_dir}")
    
    # Find all QASM files
    if subdirs:
        qasm_files = []
        for subdir in subdirs:
            subdir_path = os.path.join(benchmark_dir, subdir)
            if os.path.exists(subdir_path):
                subdir_files = find_qasm_files(subdir_path)
                qasm_files.extend(subdir_files)
                print(f"Found {len(subdir_files)} files in {subdir}")
            else:
                print(f"Warning: Subdirectory {subdir} not found in {benchmark_dir}")
    else:
        qasm_files = find_qasm_files(benchmark_dir)
    
    if not qasm_files:
        print("No QASM files found!")
        return {}
    
    # Filter circuits for supported gates
    skipped_circuits = {}
    if skip_unsupported:
        qasm_files, skipped_circuits = filter_supported_circuits(qasm_files)
        if not qasm_files:
            print("No circuits with supported gates found!")
            return {}
    
    # Limit files if max_files is specified
    if max_files and len(qasm_files) > max_files:
        original_count = len(qasm_files)
        qasm_files = qasm_files[:max_files]
        print(f"Limiting to first {max_files} files out of {original_count} total")
    
    # Create output directory for individual JSON files
    if benchmark_dir:
        results_dir = os.path.join(os.path.dirname(benchmark_dir), "circuit_analysis_results")
    else:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        results_dir = os.path.join(current_dir, "benchmark_circuits", "circuit_analysis_results")
    
    os.makedirs(results_dir, exist_ok=True)
    print(f"\nIndividual circuit JSON files will be saved to: {results_dir}")
    
    # Perform comprehensive analysis
    results = test_qasm_conversion(qasm_files, results_dir)
    
    # Save consolidated JSON only if not using subdirs (when subdirs is used, individual JSONs are preferred)
    if not subdirs:
        # Generate JSON output filename if not provided
        if output_json is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_json = f"qasm_analysis_all_{timestamp}.json"
        
        # Save results to JSON
        try:
            # Convert results to a more JSON-friendly format
            json_results = {
                'metadata': {
                    'analysis_timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                    'benchmark_directory': benchmark_dir,
                    'total_files_found': len(results),
                    'max_files_limit': max_files,
                    'skip_unsupported_enabled': skip_unsupported,
                    'skipped_circuits_count': len(skipped_circuits),
                    'successful_analyses': len([r for r in results.values() if r['analysis_status'] == 'SUCCESS']),
                    'failed_analyses': len([r for r in results.values() if r['analysis_status'] == 'FAILED']),
                    'individual_json_directory': results_dir
                },
                'skipped_circuits': skipped_circuits,
                'circuits': {}
            }
            
            # Convert file paths to relative paths for cleaner JSON
            for file_path, result in results.items():
                relative_path = os.path.relpath(file_path)
                json_results['circuits'][relative_path] = result
            
            with open(output_json, 'w') as f:
                json.dump(json_results, f, indent=2, sort_keys=True)
            
            print(f"\nConsolidated results saved to: {output_json}")
            
        except Exception as e:
            print(f"Warning: Failed to save JSON results: {e}")
    else:
        print(f"\nWhen using subdirs, individual JSON files are saved to: {results_dir}")
        print(f"No consolidated JSON file created (individual JSONs preferred for subdir analysis)")
    
    # Analyze and display results
    analyze_results(results)
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive QASM circuit analysis pipeline")
    parser.add_argument("--max-files", type=int, help="Maximum number of files to analyze")
    parser.add_argument("--subdirs", nargs="+", help="Specific subdirectories to test (e.g., feynman bigint)")
    parser.add_argument("--benchmark-dir", help="Path to benchmark directory")
    parser.add_argument("--output-json", help="Path to save JSON results (default: auto-generated)")
    parser.add_argument("--include-unsupported", action="store_true", help="Include circuits with unsupported gates (default: skip them)")
    
    args = parser.parse_args()
    
    # Run the pipeline
    results = run_qasm_pipeline(
        benchmark_dir=args.benchmark_dir,
        max_files=args.max_files,
        subdirs=args.subdirs,
        output_json=args.output_json,
        skip_unsupported=not args.include_unsupported
    )