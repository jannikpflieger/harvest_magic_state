#!/usr/bin/env python3
"""
Algorithm Comparison Plot: Sequential vs Greedy Packing Scheduling

Compares 5 different quantum algorithms using two scheduling methods:
- Sequential (Steiner Tree)
- Greedy Packing (Steiner Packing)

Focus: Timesteps only (no wirelength or magic wait cycles)
Magic source: Unlimited (no preparation overhead)
Layout: Single space with magic states all around
"""

import json
import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def extract_timesteps_from_qaoa(file_path):
    """
    Extract sequential and greedy packing timesteps from QAOA JSON file.
    
    Expected structure: results array with scheduler_label and num_timesteps
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    results = data.get('results', [])
    
    sequential_ts = None
    greedy_ts = None
    
    for result in results:
        layout_label = result.get('layout_label', '')
        scheduler_label = result.get('scheduler_label', '')
        scheduler_mode = result.get('scheduler_mode', '')
        
        # Look for Circuit-Aware with Sequential scheduling
        if (layout_label == 'Circuit-Aware' and 
            scheduler_label == 'Sequential' and 
            scheduler_mode == 'steiner_tree'):
            sequential_ts = result.get('num_timesteps')
        
        # Look for Circuit-Aware with Greedy Packing
        if (layout_label == 'Circuit-Aware' and 
            scheduler_label == 'Greedy Packing' and 
            scheduler_mode == 'steiner_packing'):
            greedy_ts = result.get('num_timesteps')
    
    return sequential_ts, greedy_ts


def extract_timesteps_from_benchmark(file_path):
    """
    Extract sequential and greedy packing timesteps from benchmark JSON file.
    
    Expected structure: factory_vs_cultivation_results array with source_label, 
    scheduler_label, and num_timesteps
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    results = data.get('factory_vs_cultivation_results', [])
    
    sequential_ts = None
    greedy_ts = None
    
    for result in results:
        source_label = result.get('source_label', '')
        scheduler_label = result.get('scheduler_label', '')
        scheduler_mode = result.get('scheduler_mode', '')
        
        # Look for Unlimited source with Sequential scheduling
        if (source_label == 'Unlimited' and 
            scheduler_label == 'Sequential' and 
            scheduler_mode == 'steiner_tree'):
            sequential_ts = result.get('num_timesteps')
        
        # Look for Unlimited source with Greedy Packing
        if (source_label == 'Unlimited' and 
            scheduler_label == 'Greedy Packing' and 
            scheduler_mode == 'steiner_packing'):
            greedy_ts = result.get('num_timesteps')
    
    return sequential_ts, greedy_ts


def load_all_algorithm_data(base_path):
    """
    Load timestep data for all 5 algorithms.
    
    Returns:
        dict: {algorithm_name: {'sequential': int, 'greedy': int}}
    """
    data = {}
    
    # QAOA 100q
    qaoa_file = base_path / 'routing_experiment_results' / 'circuit_aware_qaoa_100q_20260409_112553.json'
    seq, greedy = extract_timesteps_from_qaoa(str(qaoa_file))
    data['QAOA 100q'] = {'sequential': seq, 'greedy': greedy}
    
    # Benchmark comparisons (all use factory_vs_cultivation_results format)
    benchmark_dir = base_path / 'routing_experiment_results' / 'benchmark_comparison'
    
    # Ising n66
    ising_file = benchmark_dir / 'ising_n66.json'
    seq, greedy = extract_timesteps_from_benchmark(str(ising_file))
    data['Ising n66'] = {'sequential': seq, 'greedy': greedy}
    
    # KNN n67
    knn_file = benchmark_dir / 'knn_n67.json'
    seq, greedy = extract_timesteps_from_benchmark(str(knn_file))
    data['KNN n67'] = {'sequential': seq, 'greedy': greedy}
    
    # QFT n033
    qft_file = benchmark_dir / 'qft_N033.json'
    seq, greedy = extract_timesteps_from_benchmark(str(qft_file))
    data['QFT n033'] = {'sequential': seq, 'greedy': greedy}
    
    # DNN n51
    dnn_file = benchmark_dir / 'dnn_n51.json'
    seq, greedy = extract_timesteps_from_benchmark(str(dnn_file))
    data['DNN n51'] = {'sequential': seq, 'greedy': greedy}
    
    return data


def save_comparison_csv(data, output_path_csv):
    """Save the extracted timestep comparison data to CSV."""
    algorithms = list(data.keys())
    rows = []

    for algorithm in algorithms:
        sequential = data[algorithm]['sequential']
        greedy = data[algorithm]['greedy']
        rows.append({
            'algorithm': algorithm,
            'sequential_timesteps': sequential,
            'greedy_timesteps': greedy,
            'speedup': sequential / greedy if greedy else None,
        })

    with open(output_path_csv, 'w', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=['algorithm', 'sequential_timesteps', 'greedy_timesteps', 'speedup']
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"CSV saved to {output_path_csv}")


def create_comparison_plot(data, output_path_png, output_path_pdf=None):
    """
    Create grouped bar chart comparing sequential vs greedy packing.
    
    Args:
        data: dict with algorithm names and timestep values
        output_path_png: where to save the PNG plot
        output_path_pdf: optional path where to save the PDF plot
    """
    algorithms = list(data.keys())
    sequential_values = [data[alg]['sequential'] for alg in algorithms]
    greedy_values = [data[alg]['greedy'] for alg in algorithms]
    
    # Set up bar positions
    x = np.arange(len(algorithms))
    width = 0.35
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Create bars
    bars1 = ax.bar(x - width/2, sequential_values, width, 
                   label='No Scheduling', 
                   color='#1f77b4', alpha=0.8, edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x + width/2, greedy_values, width,
                   label='Scheduling',
                   color='#ff7f0e', alpha=0.8, edgecolor='black', linewidth=1.2)
    
    # Add value labels on bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    
    # Customize plot
    ax.set_xlabel('Algorithm', fontsize=13, fontweight='bold')
    ax.set_ylabel('Timesteps', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms, fontsize=11)
    ax.legend(fontsize=11, loc='upper right', framealpha=0.95)
    
    # Add grid for readability
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.7)
    ax.set_axisbelow(True)
    
    # Format y-axis
    ax.tick_params(axis='y', labelsize=10)
    
    # Tight layout
    plt.tight_layout()
    
    # Save PNG (raster) and optional PDF (vector)
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Plot saved to {output_path_png}")

    if output_path_pdf is not None:
        plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')
        print(f"Plot saved to {output_path_pdf}")
    
    return fig, ax


def main():
    """Main execution function."""
    # Get base path
    script_dir = Path(__file__).parent
    base_path = script_dir.parent.parent  # Go up from scripts/
    
    print("Loading algorithm data...")
    data = load_all_algorithm_data(base_path)
    
    # Validate data
    print("\nExtracted data:")
    for alg, values in data.items():
        print(f"  {alg}:")
        print(f"    Sequential: {values['sequential']} timesteps")
        print(f"    Greedy Packing: {values['greedy']} timesteps")
        print(f"    Speedup: {values['sequential'] / values['greedy']:.1f}x")
    
    # Create output directory if needed
    output_dir = base_path / 'plots'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path_png = output_dir / 'algorithm_comparison_timesteps.png'
    output_path_pdf = output_dir / 'algorithm_comparison_timesteps.pdf'
    output_path_csv = output_dir / 'algorithm_comparison_timesteps.csv'
    
    print(f"\nGenerating plot...")
    create_comparison_plot(data, str(output_path_png), str(output_path_pdf))
    save_comparison_csv(data, str(output_path_csv))
    print("Done!")


if __name__ == '__main__':
    main()
