"""
Comprehensive routing pipeline for testing and comparing different routing algorithms
with extensive performance metrics collection.

This pipeline:
1. Starts with a layout configuration
2. Generates or loads a circuit
3. Tests all available routing algorithms (steiner_tree, steiner_packing, steiner_pathfinder)
4. Collects extensive performance metrics
5. Integrates with circuit_info_pipeline data
6. Saves comprehensive results to JSON files
"""

import os
import json
import time
import statistics
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
from datetime import datetime
import logging

# Import existing modules
from dag_steiner_processor import DAGProcessor, process_dag_with_steiner
from circuit.circuit_to_pbc_dag import create_random_circuit, convert_to_PCB, create_dag, qasm_to_circuit
from circuit.circuit_info_pipeline import analyze_single_circuit
from circuit.visualizer import visualize_dag

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('ComprehensiveRoutingPipeline')


class PerformanceMetricsCollector:
    """Collects comprehensive performance metrics for routing algorithms."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics for a new test run."""
        # Basic processing metrics
        self.total_operations = 0
        self.successful_operations = 0
        self.failed_operations = 0
        
        # Wirelength metrics  
        self.total_wirelength = 0
        self.wirelengths_per_net = []
        
        # Runtime metrics
        self.total_runtime_ms = 0
        self.num_steiner_calls = 0
        
        # Magic terminal usage
        self.magic_terminals_used = 0
        self.total_magic_terminals = 0
        
        # Time step information
        self.num_time_steps = 0
        self.operations_per_timestep = []
        
    def add_operation_result(self, success: bool, wirelength: int = 0):
        """Add result from a single operation."""
        self.total_operations += 1
        if success:
            self.successful_operations += 1
            if wirelength > 0:
                self.wirelengths_per_net.append(wirelength)
                self.total_wirelength += wirelength
        else:
            self.failed_operations += 1
    
    def set_magic_terminal_info(self, used: int, total: int):
        """Set magic terminal usage information."""
        self.magic_terminals_used = used
        self.total_magic_terminals = total
    
    def set_timestep_info(self, num_steps: int, ops_per_step: List[int]):
        """Set time step processing information."""
        self.num_time_steps = num_steps
        self.operations_per_timestep = ops_per_step.copy()
    
    def calculate_final_metrics(self) -> Dict:
        """Calculate final aggregated metrics - only meaningful ones."""
        metrics = {}
        
        # Basic success/failure metrics
        metrics['total_operations'] = self.total_operations
        metrics['successful_operations'] = self.successful_operations
        metrics['failed_operations'] = self.failed_operations
        metrics['success_rate'] = self.successful_operations / max(self.total_operations, 1)
        
        # Wirelength metrics (only if we have actual data)
        metrics['total_wirelength'] = self.total_wirelength
        if self.wirelengths_per_net:
            metrics['avg_wirelength_per_net'] = statistics.mean(self.wirelengths_per_net)
            metrics['median_wirelength_per_net'] = statistics.median(self.wirelengths_per_net)
            metrics['max_wirelength'] = max(self.wirelengths_per_net)
            metrics['min_wirelength'] = min(self.wirelengths_per_net)
        else:
            metrics.update({
                'avg_wirelength_per_net': 0,
                'median_wirelength_per_net': 0,
                'max_wirelength': 0,
                'min_wirelength': 0
            })
        
        # Runtime metrics
        metrics['total_runtime_ms'] = self.total_runtime_ms
        metrics['num_steiner_calls'] = self.num_steiner_calls
        
        # Magic terminal usage
        metrics['magic_terminals_used'] = self.magic_terminals_used
        metrics['total_magic_terminals'] = self.total_magic_terminals
        if self.total_magic_terminals > 0:
            metrics['magic_terminal_utilization'] = self.magic_terminals_used / self.total_magic_terminals
        else:
            metrics['magic_terminal_utilization'] = 0
        
        # Time step information
        metrics['num_time_steps'] = self.num_time_steps
        if self.operations_per_timestep:
            metrics['avg_operations_per_timestep'] = statistics.mean(self.operations_per_timestep)
            metrics['max_operations_per_timestep'] = max(self.operations_per_timestep)
        else:
            metrics.update({
                'avg_operations_per_timestep': 0,
                'max_operations_per_timestep': 0
            })
        
        return metrics
    

class ComprehensiveRoutingPipeline:
    """Main pipeline class for comprehensive routing algorithm testing."""
    
    def __init__(self, output_dir: str = "routing_results"):
        """
        Initialize the comprehensive routing pipeline.
        
        Args:
            output_dir: Directory to save results JSON files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.algorithms = ["steiner_tree", "steiner_packing", "steiner_pathfinder"]
        
    def run_single_test(self, 
                       circuit_source: str,
                       layout_rows: int = 3,
                       layout_cols: int = 3,
                       circuit_depth: int = 100,
                       test_name: str = None,
                       visualize: bool = False) -> Dict:
        """
        Run a comprehensive test on a single circuit with all routing algorithms.
        
        Args:
            circuit_source: Either path to QASM file or "random" for random circuit generation
            layout_rows: Number of rows in layout
            layout_cols: Number of columns in layout
            circuit_depth: Depth of randomly generated circuit (only used when circuit_source="random")
            test_name: Optional name for this test
            visualize: Whether to visualize steps
            
        Returns:
            Dictionary with comprehensive results
        """
        start_time = time.time()
        
        # Generate test metadata
        test_id = f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if test_name:
            test_id = f"{test_name}_{test_id}"
        
        logger.info(f"Starting comprehensive routing test: {test_id}")
        
        # Load or generate circuit with comprehensive analysis
        if circuit_source == "random":
            circuit = create_random_circuit(num_qubits=layout_rows * layout_cols, depth=circuit_depth)
            # For random circuits, provide basic info
            circuit_info = {
                'source': 'random_generated',
                'generator_params': {
                    'num_qubits': layout_rows * layout_cols,
                    'depth': circuit_depth
                }
            }
        else:
            # Load from QASM file and get comprehensive circuit analysis
            circuit = qasm_to_circuit(circuit_source)
            logger.info(f"Running comprehensive circuit analysis for: {circuit_source}")
            circuit_info = analyze_single_circuit(circuit_source)
            circuit_info['source'] = 'qasm_file'
            circuit_info['file_path'] = circuit_source
            
            # Log analysis results
            if circuit_info.get('analysis_status') == 'SUCCESS':
                logger.info(f"Circuit analysis successful: {circuit_info.get('num_qubits')} qubits, {circuit_info.get('depth')} depth")
                logger.info(f"PCB conversion: {'✓' if circuit_info.get('pcb_conversion_successful') else '✗'}")
                logger.info(f"DAG analysis: {'✓' if circuit_info.get('dag_analysis_successful') else '✗'}")
            else:
                logger.warning(f"Circuit analysis failed: {circuit_info.get('error')}")
        
        # Convert to PCB and create DAG
        try:
            pcb_circuit = convert_to_PCB(circuit)
            dag = create_dag(pcb_circuit)
            
            conversion_success = True
            num_dag_nodes = len(list(dag.op_nodes()))
            logger.info(f"Successfully created DAG with {num_dag_nodes} operations")
            
        except Exception as e:
            logger.error(f"Failed to convert circuit to DAG: {e}")
            return {
                'test_id': test_id,
                'status': 'FAILED',
                'error': f"DAG conversion failed: {str(e)}",
                'circuit_info': circuit_info
            }
        
        # Test configuration
        test_config = {
            'test_id': test_id,
            'circuit_source': circuit_source,
            'layout_rows': layout_rows,
            'layout_cols': layout_cols,
            'circuit_depth': circuit_depth if circuit_source == "random" else None,
            'algorithms_tested': self.algorithms,
            'visualize_enabled': visualize,
            'num_dag_operations': num_dag_nodes
        }
        
        # Test all algorithms
        algorithm_results = {}
        
        for algorithm in self.algorithms:
            logger.info(f"Testing algorithm: {algorithm}")
            
            try:
                alg_start_time = time.time()
                
                # Run the algorithm
                processor, results = process_dag_with_steiner(
                    dag=dag,
                    layout_rows=layout_rows,
                    layout_cols=layout_cols,
                    visualize_steps=visualize,
                    mode=algorithm
                )
                
                alg_runtime = (time.time() - alg_start_time) * 1000  # Convert to ms
                
                # Extract metrics and routing schedule
                algorithm_metrics = self._extract_metrics_from_processor(processor, results, alg_runtime)
                routing_schedule = self._extract_routing_schedule(processor, algorithm)
                
                algorithm_results[algorithm] = {
                    'status': 'SUCCESS',
                    'runtime_ms': alg_runtime,
                    'processor_summary': processor.get_summary(algorithm) if hasattr(processor, 'get_summary') else {},
                    'metrics': algorithm_metrics,
                    'routing_schedule': routing_schedule
                }
                
                logger.info(f"Algorithm {algorithm} completed successfully in {alg_runtime:.2f}ms")
                
            except Exception as e:
                logger.error(f"Algorithm {algorithm} failed: {e}")
                algorithm_results[algorithm] = {
                    'status': 'FAILED',
                    'error': str(e),
                    'runtime_ms': 0,
                    'metrics': {}
                }
        
        total_runtime = (time.time() - start_time) * 1000
        
        # Compile comprehensive results
        comprehensive_results = {
            'test_metadata': {
                'test_id': test_id,
                'timestamp': datetime.now().isoformat(),
                'total_runtime_ms': total_runtime,
                **test_config
            },
            'circuit_info': circuit_info,
            'layout_info': {
                'rows': layout_rows,
                'cols': layout_cols,
                'total_patches': layout_rows * layout_cols
            },
            'algorithm_results': algorithm_results,
            'comparative_analysis': self._generate_comparative_analysis(algorithm_results)
        }
        
        # Save results 
        self._save_results(comprehensive_results)
        
        logger.info(f"Test {test_id} completed in {total_runtime:.2f}ms")
        return comprehensive_results
    
    def _extract_metrics_from_processor(self, processor: DAGProcessor, results: Any, runtime_ms: float) -> Dict:
        """
        Extract meaningful metrics from the processor results.
        Only collect metrics that we can actually measure accurately.
        """
        metrics_collector = PerformanceMetricsCollector()
        metrics_collector.total_runtime_ms = runtime_ms
        
        # Extract real data from processor
        if hasattr(processor, 'processing_results') and processor.processing_results:
            for result in processor.processing_results:
                # Success detection: look for explicit success field, default based on whether we have edges
                success = result.get('success', len(result.get('steiner_edges', [])) > 0)
                
                # Calculate wirelength from the number of edges in the Steiner tree
                wirelength = len(result.get('steiner_edges', []))
                
                metrics_collector.add_operation_result(success, wirelength)
            
            # Count actual Steiner calls made
            metrics_collector.num_steiner_calls = len(processor.processing_results)
        
        # Extract magic terminal usage if available
        if hasattr(processor, 'magic_terminals') and hasattr(processor, 'used_magic_terminals'):
            total_magic = len(processor.magic_terminals)
            used_magic = len(processor.used_magic_terminals)
            metrics_collector.set_magic_terminal_info(used_magic, total_magic)
        
        # Extract real time step information from processor results
        if hasattr(processor, 'processing_results') and processor.processing_results:
            # Collect time step data from actual results
            time_step_data = {}
            max_time_step = 0
            
            for result in processor.processing_results:
                time_step = result.get('time_step', None)
                if time_step is not None:
                    if time_step not in time_step_data:
                        time_step_data[time_step] = 0
                    time_step_data[time_step] += 1
                    max_time_step = max(max_time_step, time_step)
                else:
                    # For steiner_tree mode, each operation is sequential
                    # Assign sequential time steps
                    if not time_step_data:
                        current_step = 0
                    else:
                        current_step = max(time_step_data.keys()) + 1
                    time_step_data[current_step] = 1
                    max_time_step = current_step
            
            if time_step_data:
                num_time_steps = max_time_step + 1
                ops_per_step = []
                for step in range(num_time_steps):
                    ops_per_step.append(time_step_data.get(step, 0))
                
                metrics_collector.set_timestep_info(num_time_steps, ops_per_step)
        
        return metrics_collector.calculate_final_metrics()
    
    def _extract_routing_schedule(self, processor: DAGProcessor, algorithm: str) -> Dict:
        """Extract detailed routing schedule from processor results."""
        schedule = {
            'algorithm': algorithm,
            'time_steps': {},
            'operation_details': [],
            'routing_paths': []
        }
        
        if not hasattr(processor, 'processing_results') or not processor.processing_results:
            return schedule
            
        # Organize by time steps
        for result in processor.processing_results:
            op_time_step = result.get('time_step', None)
            
            # Create operation detail record
            operation_detail = {
                'gate_name': result.get('gate_name', 'unknown'),
                'qubits': result.get('qubits', []),
                'success': result.get('success', len(result.get('steiner_edges', [])) > 0),
                'wirelength': len(result.get('steiner_edges', [])),
                'magic_terminal': result.get('magic_terminal', None),
                'qubit_terminals': result.get('qubit_terminals', []),
                'all_terminals': result.get('all_terminals', []),
                'steiner_nodes': list(result.get('steiner_nodes', [])),
                'steiner_edges': list(result.get('steiner_edges', [])),
                'time_step': op_time_step
            }
            schedule['operation_details'].append(operation_detail)
            
            # Add to time step summary
            if op_time_step is not None:
                if op_time_step not in schedule['time_steps']:
                    schedule['time_steps'][op_time_step] = {
                        'operations': [],
                        'total_wirelength': 0,
                        'successful_operations': 0,
                        'failed_operations': 0
                    }
                
                schedule['time_steps'][op_time_step]['operations'].append({
                    'gate_name': operation_detail['gate_name'],
                    'qubits': operation_detail['qubits'],
                    'wirelength': operation_detail['wirelength'],
                    'success': operation_detail['success']
                })
                
                schedule['time_steps'][op_time_step]['total_wirelength'] += operation_detail['wirelength']
                if operation_detail['success']:
                    schedule['time_steps'][op_time_step]['successful_operations'] += 1
                else:
                    schedule['time_steps'][op_time_step]['failed_operations'] += 1
            
            # Add routing path information if successful
            if operation_detail['success'] and operation_detail['steiner_edges']:
                schedule['routing_paths'].append({
                    'gate_name': operation_detail['gate_name'],
                    'qubits': operation_detail['qubits'],
                    'time_step': op_time_step,
                    'path_edges': operation_detail['steiner_edges'],
                    'path_nodes': operation_detail['steiner_nodes'],
                    'terminals_used': operation_detail['all_terminals']
                })
        
        return schedule
    
    def _generate_comparative_analysis(self, algorithm_results: Dict) -> Dict:
        """Generate comparative analysis between algorithms."""
        analysis = {
            'successful_algorithms': [],
            'failed_algorithms': [],
            'performance_comparison': {},
            'metrics_comparison': {}
        }
        
        successful_results = {}
        
        for alg_name, alg_result in algorithm_results.items():
            if alg_result['status'] == 'SUCCESS':
                analysis['successful_algorithms'].append(alg_name)
                successful_results[alg_name] = alg_result
            else:
                analysis['failed_algorithms'].append(alg_name)
        
        # Compare performance metrics between successful algorithms
        if len(successful_results) > 1:
            metrics_to_compare = [
                'success_rate',
                'total_wirelength',
                'avg_wirelength_per_net',
                'total_runtime_ms',
                'magic_terminal_utilization',
                'num_time_steps'
            ]
            
            for metric in metrics_to_compare:
                metric_values = {}
                for alg_name, alg_result in successful_results.items():
                    value = alg_result.get('metrics', {}).get(metric, 0)
                    metric_values[alg_name] = value
                
                if metric_values:
                    analysis['metrics_comparison'][metric] = {
                        'values': metric_values,
                        'best_algorithm': max(metric_values.keys(), key=lambda k: metric_values[k]) if 'runtime' not in metric else min(metric_values.keys(), key=lambda k: metric_values[k]),
                        'worst_algorithm': min(metric_values.keys(), key=lambda k: metric_values[k]) if 'runtime' not in metric else max(metric_values.keys(), key=lambda k: metric_values[k])
                    }
        
        return analysis
    
    def _save_results(self, results: Dict):
        """Save results to JSON file."""
        test_id = results['test_metadata']['test_id']
        filename = f"{test_id}_comprehensive_results.json"
        filepath = self.output_dir / filename
        
        try:
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, sort_keys=True)
            logger.info(f"Results saved to: {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save results to {filepath}: {e}")
    
    def run_batch_tests(self, 
                       test_configs: List[Dict],
                       batch_name: str = "batch_test") -> Dict:
        """
        Run a batch of tests with different configurations.
        
        Args:
            test_configs: List of test configuration dictionaries
            batch_name: Name for this batch of tests
            
        Returns:
            Dictionary with batch results
        """
        batch_start_time = time.time()
        batch_id = f"{batch_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Starting batch test: {batch_id} with {len(test_configs)} configurations")
        
        batch_results = {
            'batch_metadata': {
                'batch_id': batch_id,
                'batch_name': batch_name,
                'timestamp': datetime.now().isoformat(),
                'num_tests': len(test_configs)
            },
            'individual_results': [],
            'batch_summary': {}
        }
        
        successful_tests = 0
        failed_tests = 0
        
        for i, config in enumerate(test_configs):
            logger.info(f"Running batch test {i+1}/{len(test_configs)}")
            
            try:
                test_config = {
                    'test_name': f"{batch_name}_test_{i+1}",
                    **config
                }
                
                result = self.run_single_test(**test_config)
                batch_results['individual_results'].append(result)
                
                if result.get('test_metadata', {}).get('status') != 'FAILED':
                    successful_tests += 1
                else:
                    failed_tests += 1
                    
            except Exception as e:
                logger.error(f"Batch test {i+1} failed: {e}")
                failed_tests += 1
                batch_results['individual_results'].append({
                    'test_id': f"{batch_name}_test_{i+1}_FAILED",
                    'status': 'FAILED',
                    'error': str(e),
                    'config': config
                })
        
        batch_runtime = (time.time() - batch_start_time) * 1000
        
        batch_results['batch_summary'] = {
            'total_runtime_ms': batch_runtime,
            'successful_tests': successful_tests,
            'failed_tests': failed_tests,
            'success_rate': successful_tests / len(test_configs) if test_configs else 0
        }
        
        # Save batch results
        batch_filename = f"{batch_id}_batch_results.json"
        batch_filepath = self.output_dir / batch_filename
        
        try:
            with open(batch_filepath, 'w') as f:
                json.dump(batch_results, f, indent=2, sort_keys=True)
            logger.info(f"Batch results saved to: {batch_filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save batch results: {e}")
        
        logger.info(f"Batch test {batch_id} completed: {successful_tests}/{len(test_configs)} successful")
        return batch_results
    
    def run_systematic_depth_experiment(self,
                                      layout_rows: int = 10,
                                      layout_cols: int = 10,
                                      depth_start: int = 50,
                                      depth_end: int = 200,
                                      depth_step: int = 25,
                                      runs_per_depth: int = 10,
                                      experiment_name: str = "depth_sweep") -> Dict:
        """
        Run a systematic experiment varying circuit depth with multiple runs per configuration.
        
        Args:
            layout_rows: Number of rows in layout (default: 10)
            layout_cols: Number of columns in layout (default: 10)
            depth_start: Starting circuit depth (default: 50)
            depth_end: Ending circuit depth (inclusive, default: 200)
            depth_step: Step size for depth variation (default: 25)
            runs_per_depth: Number of runs per depth configuration (default: 10)
            experiment_name: Name for this experiment series
            
        Returns:
            Dictionary with comprehensive experimental results
        """
        experiment_start_time = time.time()
        experiment_id = f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Generate depth range
        depths = list(range(depth_start, depth_end + 1, depth_step))
        total_experiments = len(depths) * runs_per_depth
        
        logger.info(f"Starting systematic depth experiment: {experiment_id}")
        logger.info(f"Layout: {layout_rows}x{layout_cols} ({layout_rows * layout_cols} qubits)")
        logger.info(f"Depth range: {depth_start} to {depth_end} (step: {depth_step}) -> {len(depths)} depths")
        logger.info(f"Runs per depth: {runs_per_depth}")
        logger.info(f"Total experiments: {total_experiments}")
        
        # Generate all test configurations
        test_configs = []
        for depth in depths:
            for run_num in range(1, runs_per_depth + 1):
                config = {
                    'circuit_source': 'random',
                    'layout_rows': layout_rows,
                    'layout_cols': layout_cols,
                    'circuit_depth': depth,
                    'test_name': f"{experiment_name}_depth{depth}_run{run_num}",
                    'visualize': False  # Disable visualization for batch experiments
                }
                test_configs.append(config)
        
        # Run the batch experiment
        logger.info(f"Executing {len(test_configs)} experiments...")
        batch_results = self.run_batch_tests(test_configs, batch_name=experiment_id)
        
        # Organize results by depth for analysis
        results_by_depth = {}
        for depth in depths:
            results_by_depth[depth] = {
                'depth': depth,
                'results': [],
                'summary': {
                    'successful_runs': 0,
                    'failed_runs': 0,
                    'success_rate': 0.0,
                    'avg_runtime_ms': 0.0,
                    'algorithm_performance': {}
                }
            }
        
        # Process and categorize results
        for result in batch_results['individual_results']:
            # Extract depth from test metadata
            test_metadata = result.get('test_metadata', {})
            depth = test_metadata.get('circuit_depth')
            
            if depth is not None and depth in results_by_depth:
                results_by_depth[depth]['results'].append(result)
                
                # Update summary statistics
                summary = results_by_depth[depth]['summary']
                if result.get('test_metadata', {}).get('status') != 'FAILED':
                    summary['successful_runs'] += 1
                    
                    # Aggregate algorithm performance
                    for alg_name, alg_result in result.get('algorithm_results', {}).items():
                        if alg_name not in summary['algorithm_performance']:
                            summary['algorithm_performance'][alg_name] = {
                                'successful_runs': 0,
                                'total_wirelength': [],
                                'runtime_ms': [],
                                'success_rates': []
                            }
                        
                        alg_perf = summary['algorithm_performance'][alg_name]
                        if alg_result.get('status') == 'SUCCESS':
                            alg_perf['successful_runs'] += 1
                            alg_perf['runtime_ms'].append(alg_result.get('runtime_ms', 0))
                            
                            metrics = alg_result.get('metrics', {})
                            if metrics.get('total_wirelength', 0) > 0:
                                alg_perf['total_wirelength'].append(metrics['total_wirelength'])
                            if 'success_rate' in metrics:
                                alg_perf['success_rates'].append(metrics['success_rate'])
                else:
                    summary['failed_runs'] += 1
        
        # Calculate final statistics for each depth
        for depth, depth_data in results_by_depth.items():
            summary = depth_data['summary']
            total_runs = summary['successful_runs'] + summary['failed_runs']
            
            if total_runs > 0:
                summary['success_rate'] = summary['successful_runs'] / total_runs
            
            # Calculate algorithm averages
            for alg_name, alg_perf in summary['algorithm_performance'].items():
                if alg_perf['runtime_ms']:
                    alg_perf['avg_runtime_ms'] = statistics.mean(alg_perf['runtime_ms'])
                    alg_perf['std_runtime_ms'] = statistics.stdev(alg_perf['runtime_ms']) if len(alg_perf['runtime_ms']) > 1 else 0
                
                if alg_perf['total_wirelength']:
                    alg_perf['avg_wirelength'] = statistics.mean(alg_perf['total_wirelength'])
                    alg_perf['std_wirelength'] = statistics.stdev(alg_perf['total_wirelength']) if len(alg_perf['total_wirelength']) > 1 else 0
                
                if alg_perf['success_rates']:
                    alg_perf['avg_success_rate'] = statistics.mean(alg_perf['success_rates'])
                    alg_perf['std_success_rate'] = statistics.stdev(alg_perf['success_rates']) if len(alg_perf['success_rates']) > 1 else 0
        
        experiment_runtime = (time.time() - experiment_start_time) * 1000
        
        # Compile comprehensive experimental results
        experimental_results = {
            'experiment_metadata': {
                'experiment_id': experiment_id,
                'experiment_name': experiment_name,
                'timestamp': datetime.now().isoformat(),
                'total_runtime_ms': experiment_runtime,
                'layout_rows': layout_rows,
                'layout_cols': layout_cols,
                'total_qubits': layout_rows * layout_cols,
                'depth_range': {
                    'start': depth_start,
                    'end': depth_end,
                    'step': depth_step,
                    'depths_tested': depths
                },
                'runs_per_depth': runs_per_depth,
                'total_experiments': total_experiments
            },
            'batch_results': batch_results,
            'results_by_depth': results_by_depth,
            'experiment_summary': self._generate_experiment_summary(results_by_depth, depths)
        }
        
        # Save experimental results
        self._save_experimental_results(experimental_results)
        
        logger.info(f"Systematic experiment {experiment_id} completed in {experiment_runtime/1000:.2f}s")
        logger.info(f"Total experiments: {total_experiments}, Overall success rate: {batch_results['batch_summary']['success_rate']:.2%}")
        
        return experimental_results
    
    def _generate_experiment_summary(self, results_by_depth: Dict, depths: List[int]) -> Dict:
        """Generate high-level summary of experimental results."""
        summary = {
            'depth_performance': {},
            'algorithm_comparison': {},
            'trends': {}
        }
        
        # Summarize performance by depth
        for depth in depths:
            if depth in results_by_depth:
                depth_data = results_by_depth[depth]
                summary['depth_performance'][depth] = {
                    'success_rate': depth_data['summary']['success_rate'],
                    'successful_runs': depth_data['summary']['successful_runs'],
                    'failed_runs': depth_data['summary']['failed_runs']
                }
        
        # Compare algorithms across all depths
        algorithm_names = set()
        for depth_data in results_by_depth.values():
            algorithm_names.update(depth_data['summary']['algorithm_performance'].keys())
        
        for alg_name in algorithm_names:
            alg_summary = {
                'total_successful_runs': 0,
                'avg_runtime_across_depths': [],
                'avg_wirelength_across_depths': [],
                'avg_success_rate_across_depths': []
            }
            
            for depth_data in results_by_depth.values():
                if alg_name in depth_data['summary']['algorithm_performance']:
                    alg_perf = depth_data['summary']['algorithm_performance'][alg_name]
                    alg_summary['total_successful_runs'] += alg_perf['successful_runs']
                    
                    if 'avg_runtime_ms' in alg_perf:
                        alg_summary['avg_runtime_across_depths'].append(alg_perf['avg_runtime_ms'])
                    if 'avg_wirelength' in alg_perf:
                        alg_summary['avg_wirelength_across_depths'].append(alg_perf['avg_wirelength'])
                    if 'avg_success_rate' in alg_perf:
                        alg_summary['avg_success_rate_across_depths'].append(alg_perf['avg_success_rate'])
            
            # Calculate overall averages
            if alg_summary['avg_runtime_across_depths']:
                alg_summary['overall_avg_runtime_ms'] = statistics.mean(alg_summary['avg_runtime_across_depths'])
            if alg_summary['avg_wirelength_across_depths']:
                alg_summary['overall_avg_wirelength'] = statistics.mean(alg_summary['avg_wirelength_across_depths'])
            if alg_summary['avg_success_rate_across_depths']:
                alg_summary['overall_avg_success_rate'] = statistics.mean(alg_summary['avg_success_rate_across_depths'])
            
            summary['algorithm_comparison'][alg_name] = alg_summary
        
        return summary
    
    def _save_experimental_results(self, results: Dict):
        """Save experimental results to JSON file."""
        experiment_id = results['experiment_metadata']['experiment_id']
        filename = f"{experiment_id}_experimental_results.json"
        filepath = self.output_dir / filename
        
        try:
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, sort_keys=True)
            logger.info(f"Experimental results saved to: {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save experimental results to {filepath}: {e}")


def main():
    """Main function demonstrating pipeline usage."""
    # Create pipeline instance
    pipeline = ComprehensiveRoutingPipeline("routing_experiment_results")
    
    # Option 1: Run systematic depth experiment (recommended for comprehensive analysis)
    print("🔬 Starting systematic depth experiment...")
    experimental_results = pipeline.run_systematic_depth_experiment(
        layout_rows=10,
        layout_cols=10,
        depth_start=50,
        depth_end=200,
        depth_step=25,
        runs_per_depth=10,
        experiment_name="comprehensive_depth_analysis"
    )
    
    print(f"✓ Systematic experiment completed!")
    print(f"   Total experiments: {experimental_results['experiment_metadata']['total_experiments']}")
    print(f"   Success rate: {experimental_results['batch_results']['batch_summary']['success_rate']:.2%}")
    print(f"   Runtime: {experimental_results['experiment_metadata']['total_runtime_ms']/1000:.1f}s")
    
    # Option 2: Quick batch test example (for comparison)
    print("\n🚀 Running quick batch test example...")
    batch_configs = [
        {'circuit_source': 'random', 'layout_rows': 5, 'layout_cols': 5, 'circuit_depth': 50},
        {'circuit_source': 'random', 'layout_rows': 5, 'layout_cols': 5, 'circuit_depth': 100},
        {'circuit_source': 'random', 'layout_rows': 5, 'layout_cols': 5, 'circuit_depth': 150},
    ]
    
    batch_result = pipeline.run_batch_tests(
        batch_configs,
        batch_name="quick_comparison"
    )
    
    print(f"✓ Batch test completed: {batch_result['batch_summary']['successful_tests']}/{batch_result['batch_metadata']['num_tests']} successful")


def run_depth_sweep_experiment():
    """
    Convenience function to run the systematic depth experiment.
    This is the main experimental function you'll want to use.
    """
    pipeline = ComprehensiveRoutingPipeline("routing_experiment_results")
    
    # Run systematic depth experiment: 10x10 layout, depth 50-200 (step 25), 10 runs per depth
    print("🔬 Starting comprehensive depth sweep experiment...")
    print("   Layout: 10x10 (100 qubits)")
    print("   Depths: 50, 75, 100, 125, 150, 175, 200")  
    print("   Runs per depth: 10")
    print("   Total experiments: 70")
    print("   This may take several minutes...")
    
    results = pipeline.run_systematic_depth_experiment(
        layout_rows=10,
        layout_cols=10,
        depth_start=50,
        depth_end=200,
        depth_step=25,
        runs_per_depth=10,
        experiment_name="depth_sweep_100qubits"
    )
    
    print("\n🎉 Experiment completed!")
    print(f"   Results saved to: routing_experiment_results/")
    print(f"   Check the *_experimental_results.json file for detailed analysis")
    
    return results


if __name__ == "__main__":
    # Run the systematic experiment
    run_depth_sweep_experiment()
    
    # Or run the main function with both examples
    # main()