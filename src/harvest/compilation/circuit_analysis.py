"""
Circuit analysis, gate filtering, DAG layer metrics, and batch reporting.
"""

import os
import json
import statistics
from pathlib import Path
from typing import List, Tuple, Dict
import time

from .qasm_loader import qasm_to_circuit
from .pauli_block_conversion import convert_to_PCB, create_dag

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
        if name in {"id", "i"} or name.startswith("identity"):
            continue
        new.append(inst, qargs, cargs)
    return new


def convert_rx_ry_to_rz(circuit):
    """
    Convert rx and ry gates to rz gates using basis translation.
    """
    from qiskit.transpiler.passes import UnrollCustomDefinitions, BasisTranslator

    target_basis = ['id', 'x', 'y', 'z', 'h', 's', 'sdg', 'sx', 'sxdg',
                   'cx', 'cz', 'cy', 'swap', 'iswap', 'ecr', 'dcx', 't', 'tdg', 'rz']

    pm = PassManager()
    pm.append(UnrollCustomDefinitions(sel, target_basis))
    pm.append(BasisTranslator(sel, target_basis))

    converted_circuit = pm.run(circuit)
    return converted_circuit


def has_unsupported_gates(circuit) -> Tuple[bool, List[str]]:
    """
    Check if circuit has gates that are unsupported by the PCB conversion.

    Returns:
        (has_unsupported, unsupported_gates_list)
    """
    clifford_gates = {"id", "x", "y", "z", "h", "s", "sdg", "sx", "sxdg",
                     "cx", "cz", "cy", "swap", "iswap", "ecr", "dcx"}
    rz_gates = {"t", "tdg", "rz"}
    convertible_gates = {"rx", "ry"}
    supported_gates = clifford_gates | rz_gates | convertible_gates

    gate_counts = circuit.count_ops()
    unsupported_gates = []

    for gate in gate_counts.keys():
        if gate not in supported_gates:
            unsupported_gates.append(gate)

    return len(unsupported_gates) > 0, unsupported_gates


def count_non_clifford_gates(circuit) -> int:
    """Count the number of non-Clifford gates in a circuit."""
    clifford_gates = {"id", "x", "y", "z", "h", "s", "sdg", "sx", "sxdg",
                     "cx", "cz", "cy", "swap", "iswap", "ecr", "dcx", "measure"}

    gate_counts = circuit.count_ops()
    non_clifford_count = 0

    for gate, count in gate_counts.items():
        if gate not in clifford_gates:
            non_clifford_count += count

    return non_clifford_count


def analyze_dag_layers(dag: DAGCircuit) -> Dict:
    """
    Analyze DAG layers and count Pauli evolutions per layer, including their sizes.
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
                size = len(node.qargs) if node.qargs else 0
                layer_sizes.append(size)
                all_pauli_sizes.append(size)

        pauli_evolutions_per_layer.append(pauli_count)
        pauli_evolution_sizes_per_layer.append(layer_sizes)

    max_pauli_per_layer = max(pauli_evolutions_per_layer) if pauli_evolutions_per_layer else 0
    median_pauli_per_layer = statistics.median(pauli_evolutions_per_layer) if pauli_evolutions_per_layer else 0

    max_pauli_evolution_size = max(all_pauli_sizes) if all_pauli_sizes else 0
    avg_pauli_evolution_size = statistics.mean(all_pauli_sizes) if all_pauli_sizes else 0

    avg_pauli_sizes_per_layer = []
    for sizes in pauli_evolution_sizes_per_layer:
        if sizes:
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
    """Save individual circuit analysis to JSON file."""
    circuit_id = circuit_data['circuit_id']
    json_filename = f"{circuit_id}.json"
    json_path = os.path.join(output_dir, json_filename)

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
    """
    try:
        start_time = time.time()
        circuit = qasm_to_circuit(qasm_file)
        load_time = time.time() - start_time

        circuit_name = os.path.basename(qasm_file).replace('.qasm', '')
        circuit_id = os.path.relpath(qasm_file).replace('/', '_').replace('.qasm', '')

        gate_counts = circuit.count_ops()
        needs_conversion = 'rx' in gate_counts or 'ry' in gate_counts

        pm = PassManager(RemoveFinalMeasurements())
        circuit = pm.run(circuit)

        if needs_conversion:
            print(f"Converting rx/ry gates to rz equivalents...")
            conversion_start = time.time()
            circuit = convert_rx_ry_to_rz(circuit)
            conversion_time = time.time() - conversion_start
            print(f"Conversion completed in {conversion_time:.3f}s")
            print(f"New gate counts: {dict(circuit.count_ops())}")
        else:
            conversion_time = 0.0

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

        result = {
            'analysis_status': 'SUCCESS',
            'error': None,
            **basic_info,
            **pcb_info,
            **dag_info
        }

        if output_dir:
            try:
                json_path = save_individual_circuit_json(result, output_dir)
                result['json_file_saved'] = json_path
            except Exception as e:
                print(f"Warning: Failed to save JSON for {circuit_name}: {e}")
                result['json_file_saved'] = None

        return result

    except Exception as e:
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
    """Remove measurements and barriers before analysis."""
    pm = PassManager()
    pm.append(RemoveFinalMeasurements())
    pm.append(RemoveBarriers())
    circuit = pm.run(circuit)
    return circuit


def filter_supported_circuits(qasm_files: List[str]) -> Tuple[List[str], Dict[str, List[str]]]:
    """Filter circuits to only include those with supported gates for PCB conversion."""
    supported_circuits = []
    skipped_circuits = {}

    print(f"\n{'='*80}")
    print(f"Filtering circuits for supported gates...")
    print(f"{'='*80}")

    for qasm_file in qasm_files:
        relative_path = os.path.relpath(qasm_file)

        try:
            circuit = qasm_to_circuit(qasm_file)
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
    """Test converting each QASM file and perform comprehensive analysis."""
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
    """Analyze and print detailed statistics about the analysis results."""
    successful_results = [r for r in results.values() if r['analysis_status'] == 'SUCCESS']
    failed_results = [r for r in results.values() if r['analysis_status'] == 'FAILED']

    print(f"\n{'='*80}")
    print("DETAILED ANALYSIS REPORT")
    print(f"{'='*80}")

    if successful_results:
        print(f"\nSuccessful analyses: {len(successful_results)}")

        qubit_counts = [r['num_qubits'] for r in successful_results]
        depths = [r['depth'] for r in successful_results]
        total_gates = [r['total_gates'] for r in successful_results]
        non_clifford_gates = [r['non_clifford_gates'] for r in successful_results]

        print(f"  Qubit range: {min(qubit_counts)} - {max(qubit_counts)} (avg: {sum(qubit_counts)/len(qubit_counts):.1f})")
        print(f"  Depth range: {min(depths)} - {max(depths)} (avg: {sum(depths)/len(depths):.1f})")
        print(f"  Gate count range: {min(total_gates)} - {max(total_gates)} (avg: {sum(total_gates)/len(total_gates):.1f})")
        print(f"  Non-Clifford gates range: {min(non_clifford_gates)} - {max(non_clifford_gates)} (avg: {sum(non_clifford_gates)/len(non_clifford_gates):.1f})")

        pcb_successful = [r for r in successful_results if r.get('pcb_conversion_successful', False)]
        print(f"\n  PCB Conversions: {len(pcb_successful)}/{len(successful_results)} successful")

        if pcb_successful:
            pcb_depths = [r['pcb_depth'] for r in pcb_successful]
            pcb_gates = [r['pcb_total_gates'] for r in pcb_successful]
            print(f"    PCB depth range: {min(pcb_depths)} - {max(pcb_depths)} (avg: {sum(pcb_depths)/len(pcb_depths):.1f})")
            print(f"    PCB gate count range: {min(pcb_gates)} - {max(pcb_gates)} (avg: {sum(pcb_gates)/len(pcb_gates):.1f})")

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

        all_gates = {}
        for r in successful_results:
            if r['gate_counts']:
                for gate, count in r['gate_counts'].items():
                    all_gates[gate] = all_gates.get(gate, 0) + count

        if all_gates:
            print(f"\n  Most common gates: {sorted(all_gates.items(), key=lambda x: x[1], reverse=True)[:10]}")

    if failed_results:
        print(f"\nFailed analyses: {len(failed_results)}")

        error_types = {}
        for file_path, r in results.items():
            if r['analysis_status'] == 'FAILED':
                error = r['error']
                error_types[error] = error_types.get(error, []) + [os.path.relpath(file_path)]

        print("  Error types:")
        for error, files in error_types.items():
            print(f"    {error}: {len(files)} files")
            if len(files) <= 3:
                for file in files:
                    print(f"      - {file}")
            elif len(files) <= 10:
                for file in files[:3]:
                    print(f"      - {file}")
                print(f"      ... and {len(files)-3} more")


def run_qasm_pipeline(benchmark_dir: str = None, max_files: int = None,
                      subdirs: List[str] = None, output_json: str = None,
                      skip_unsupported: bool = True) -> Dict[str, Dict]:
    """
    Main pipeline function to perform comprehensive analysis of all QASM files.
    """
    from .qasm_loader import find_qasm_files

    if benchmark_dir is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        benchmark_dir = os.path.join(current_dir, "..", "..", "..", "benchmark_circuits", "qasm")

    print(f"Starting comprehensive QASM analysis pipeline...")
    print(f"Searching for QASM files in: {benchmark_dir}")

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

    skipped_circuits = {}
    if skip_unsupported:
        qasm_files, skipped_circuits = filter_supported_circuits(qasm_files)
        if not qasm_files:
            print("No circuits with supported gates found!")
            return {}

    if max_files and len(qasm_files) > max_files:
        original_count = len(qasm_files)
        qasm_files = qasm_files[:max_files]
        print(f"Limiting to first {max_files} files out of {original_count} total")

    if benchmark_dir:
        results_dir = os.path.join(os.path.dirname(benchmark_dir), "circuit_analysis_results")
    else:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        results_dir = os.path.join(current_dir, "..", "..", "..", "circuit", "benchmark_circuits", "circuit_analysis_results")

    os.makedirs(results_dir, exist_ok=True)
    print(f"\nIndividual circuit JSON files will be saved to: {results_dir}")

    results = test_qasm_conversion(qasm_files, results_dir)

    if not subdirs:
        if output_json is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_json = f"qasm_analysis_all_{timestamp}.json"

        try:
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

    analyze_results(results)

    return results
