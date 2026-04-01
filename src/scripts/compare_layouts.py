#!/usr/bin/env python3
"""
Compare different lattice layouts across various scheduling strategies.

This script runs a single circuit with multiple layout configurations:
1. Single spacing (nxm_ring_layout_single_qubits)
2. Large spacing (nxm_ring_layout_single_qubits_large_spacing)
3. Blocks of four (blocks_of_four_qubit_patches)

And tests them with all three schedulers:
1. steiner_tree (sequential)
2. steiner_packing (parallel with packing)
3. steiner_pathfinder (pathfinder-based)
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit

from harvest.compilation.pauli_block_conversion import convert_to_PCB, create_dag, create_random_circuit
from harvest.routing.processor import DAGProcessor
from harvest.layout.presets import (
    nxm_ring_layout_single_qubits,
    nxm_ring_layout_single_qubits_large_spacing,
    blocks_of_four_qubit_patches
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/layout_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('LayoutComparison')


def load_circuit(circuit_path):
    """Load a quantum circuit from a QASM file."""
    logger.info(f"Loading circuit from {circuit_path}")
    qc = QuantumCircuit.from_qasm_file(circuit_path)
    logger.info(f"Loaded circuit: {qc.num_qubits} qubits, depth {qc.depth()}")
    return qc


def run_experiment(circuit, layout_engine, scheduler_mode, layout_name, scheduler_name):
    """
    Run a single experiment: circuit + layout + scheduler.
    
    Returns:
        dict: Results including timesteps, success status, and other metrics
    """
    logger.info(f"Running: {layout_name} + {scheduler_name}")
    
    try:
        # Convert to PCB format
        pcb_circuit = convert_to_PCB(circuit)
        dag = create_dag(pcb_circuit)
        
        # Create processor with custom layout
        processor = DAGProcessor(layout_engine=layout_engine)
        
        # Process the DAG
        results = processor.process_entire_dag(dag, visualize_each_step=False, mode=scheduler_mode)
        
        # Extract metrics
        time_steps = set()
        for result in results:
            if 'time_step' in result:
                time_steps.add(result['time_step'])
        
        num_timesteps = len(time_steps) if time_steps else len(results)
        
        metrics = {
            'layout_name': layout_name,
            'scheduler_name': scheduler_name,
            'scheduler_mode': scheduler_mode,
            'num_timesteps': num_timesteps,
            'num_nodes_processed': len(results),
            'success': True,
            'magic_terminals_used': len(processor.used_magic_terminals),
            'magic_terminals_total': len(processor.magic_terminals)
        }
        
        logger.info(f"  ✓ Completed: {num_timesteps} timesteps, {len(results)} nodes")
        return metrics
        
    except Exception as e:
        logger.error(f"  ✗ Failed: {str(e)}")
        return {
            'layout_name': layout_name,
            'scheduler_name': scheduler_name,
            'scheduler_mode': scheduler_mode,
            'num_timesteps': None,
            'success': False,
            'error': str(e)
        }


def create_bar_plot(results, output_path):
    """
    Create a bar plot comparing layouts across schedulers.
    
    X-axis: Schedulers
    For each scheduler: 3 bars representing the 3 layouts
    Y-axis: Number of timesteps
    """
    # Organize data by scheduler and layout
    schedulers = ['No scheduling', 'Greedy', 'Advanced']
    layouts = ['Single Spacing', 'Large Spacing', 'Blocks of 4']
    
    data = {scheduler: {layout: None for layout in layouts} for scheduler in schedulers}
    
    for result in results:
        if result['success'] and result['layout_name'] in layouts:
            data[result['scheduler_name']][result['layout_name']] = result['num_timesteps']
    
    # Prepare data for plotting
    x = np.arange(len(schedulers))
    width = 0.25  # Width for 3 bars per group
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = ['#3498db', '#e74c3c', '#2ecc71']  # Blue, Red, Green
    
    for i, layout in enumerate(layouts):
        values = [data[scheduler][layout] for scheduler in schedulers]
        # Replace None with 0 for plotting (failed experiments)
        values = [v if v is not None else 0 for v in values]
        offset = width * (i - 1)  # Center bars: -1, 0, 1 for 3 bars
        bars = ax.bar(x + offset, values, width, label=layout, color=colors[i], alpha=0.8)
        
        # Add value labels on bars
        for bar, original_val in zip(bars, [data[scheduler][layout] for scheduler in schedulers]):
            height = bar.get_height()
            if height > 0 and original_val is not None:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}',
                       ha='center', va='bottom', fontsize=10)
            elif original_val is None:
                # Mark failed experiments
                ax.text(bar.get_x() + bar.get_width()/2., 0.5,
                       'Failed',
                       ha='center', va='bottom', fontsize=8, rotation=90)
    
    ax.set_xlabel('Scheduler Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Timesteps', fontsize=12, fontweight='bold')
    ax.set_title('Layout Comparison Across Schedulers\n 100-Qubit QAOA', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(schedulers)
    ax.legend(title='Layout Type', fontsize=10, title_fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved plot to {output_path}")
    plt.close()


def main():
    """Main execution function."""
    # Create a random 100-qubit circuit for testing
    # Using random circuit ensures gates are compatible with PCB conversion
    logger.info("Creating random 100-qubit circuit for layout comparison")
    circuit = create_random_circuit(num_qubits=100, depth=20, seed=42)
    logger.info(f"Created circuit: {circuit.num_qubits} qubits, depth {circuit.depth()}")
    logger.info(f"Circuit gates: {circuit.count_ops()}")
    
    # Determine layout size based on circuit qubits
    # For 100 qubits, we need at least a 10x10 grid
    layout_rows = 10
    layout_cols = 10
    
    # Define layouts to test
    layouts = [
        {
            'name': 'Single Spacing',
            'engine': nxm_ring_layout_single_qubits(layout_rows, layout_cols),
            'description': 'Standard single spacing between qubits'
        },
        {
            'name': 'Large Spacing',
            'engine': nxm_ring_layout_single_qubits_large_spacing(layout_rows, layout_cols),
            'description': 'Large spacing (2 patches between qubits)'
        },
        {
            'name': 'Blocks of 4',
            'engine': blocks_of_four_qubit_patches(layout_rows // 2, layout_cols // 2),
            'description': 'Qubits arranged in 2x2 blocks'
        }
    ]
    
    # Define schedulers to test
    schedulers = [
        {'name': 'No scheduling', 'mode': 'steiner_tree'},
        {'name': 'Greedy', 'mode': 'steiner_packing'},
        {'name': 'Advanced', 'mode': 'steiner_pathfinder'}
    ]
    
    # Run all experiments
    logger.info(f"Starting layout comparison experiments")
    logger.info(f"Layouts: {len(layouts)}")
    logger.info(f"Schedulers: {len(schedulers)}")
    logger.info(f"Total experiments: {len(layouts) * len(schedulers)}\n")
    
    results = []
    
    for layout in layouts:
        logger.info(f"\n--- Testing Layout: {layout['name']} ---")
        logger.info(f"    {layout['description']}")
        
        for scheduler in schedulers:
            result = run_experiment(
                circuit,
                layout['engine'],
                scheduler['mode'],
                layout['name'],
                scheduler['name']
            )
            results.append(result)
    
    # Save results to JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = f"routing_experiment_results/layout_comparison_{timestamp}.json"
    os.makedirs("routing_experiment_results", exist_ok=True)
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nSaved results to {results_path}")
    
    # Create visualization
    plot_path = f"plots/layout_comparison_{timestamp}.png"
    os.makedirs("plots", exist_ok=True)
    create_bar_plot(results, plot_path)
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("="*60)
    
    for result in results:
        if result['success']:
            logger.info(f"{result['layout_name']:20s} + {result['scheduler_name']:12s}: "
                       f"{result['num_timesteps']:4d} timesteps")
        else:
            logger.info(f"{result['layout_name']:20s} + {result['scheduler_name']:12s}: FAILED")
    
    logger.info("="*60)


if __name__ == "__main__":
    main()
