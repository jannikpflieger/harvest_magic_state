"""
Wirelength analysis and visualization for routing experiment results.
"""

import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def extract_wirelength_data(results_folder):
    """
    Extract average wirelength data from all experiment result files.

    Returns:
        dict: Nested dictionary structure: {depth: {algorithm: [wirelengths]}}
    """
    data = defaultdict(lambda: defaultdict(list))

    files = [f for f in os.listdir(results_folder) if f.endswith('comprehensive_results.json')]
    print(f"Found {len(files)} result files")

    for filename in files:
        try:
            parts = filename.split('_')
            depth = None
            run_num = None

            for i, part in enumerate(parts):
                if part.startswith('depth') and len(part) > 5:
                    depth = int(part[5:])
                elif part.startswith('run') and len(part) > 3:
                    run_num = int(part[3:])

            if depth is None:
                print(f"Could not parse depth from filename: {filename}")
                continue

            filepath = os.path.join(results_folder, filename)
            with open(filepath, 'r') as f:
                result_data = json.load(f)

            algorithm_results = result_data.get('algorithm_results', {})

            for algorithm in ['steiner_packing', 'steiner_pathfinder', 'steiner_tree']:
                if algorithm in algorithm_results:
                    metrics = algorithm_results[algorithm].get('metrics', {})
                    total_wirelength = metrics.get('total_wirelength')

                    if total_wirelength is not None:
                        data[depth][algorithm].append(total_wirelength)

        except (ValueError, IndexError, KeyError) as e:
            print(f"Error processing file {filename}: {e}")
            continue

    return dict(data)


def create_wirelength_plot(data, output_filename='wirelength_comparison.png'):
    """Create a bar plot showing total wirelength by depth and algorithm."""
    depths = sorted(data.keys())
    algorithms = ['steiner_packing', 'steiner_pathfinder', 'steiner_tree']

    plot_data = []
    for depth in depths:
        row_data = {'Depth': depth}
        for algorithm in algorithms:
            wirelengths = data[depth].get(algorithm, [])
            if wirelengths:
                avg_wirelength = np.mean(wirelengths)
                std_wirelength = np.std(wirelengths)
                row_data[f'{algorithm}_avg'] = avg_wirelength
                row_data[f'{algorithm}_std'] = std_wirelength
                row_data[f'{algorithm}_count'] = len(wirelengths)
                print(f"Depth {depth}, {algorithm}: {avg_wirelength:.2f} ± {std_wirelength:.2f} (n={len(wirelengths)})")
            else:
                row_data[f'{algorithm}_avg'] = 0
                row_data[f'{algorithm}_std'] = 0
                row_data[f'{algorithm}_count'] = 0
                print(f"Depth {depth}, {algorithm}: No data")
        plot_data.append(row_data)

    df = pd.DataFrame(plot_data)

    fig, ax = plt.subplots(figsize=(12, 8))

    x = np.arange(len(depths))
    width = 0.25

    bars1 = ax.bar(x - width, df['steiner_packing_avg'], width,
                   yerr=df['steiner_packing_std'], label='Steiner Packing',
                   alpha=0.8, capsize=5)
    bars2 = ax.bar(x, df['steiner_pathfinder_avg'], width,
                   yerr=df['steiner_pathfinder_std'], label='Steiner Pathfinder',
                   alpha=0.8, capsize=5)
    bars3 = ax.bar(x + width, df['steiner_tree_avg'], width,
                   yerr=df['steiner_tree_std'], label='Steiner Tree',
                   alpha=0.8, capsize=5)

    ax.set_xlabel('Circuit Depth', fontsize=12, fontweight='bold')
    ax.set_ylabel('Total Wirelength', fontsize=12, fontweight='bold')
    ax.set_title('Total Wirelength Comparison Across Circuit Depths\n(100 Qubits, Multiple Algorithms)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(depths)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    def add_value_labels(bars, values, stds):
        for bar, val, std in zip(bars, values, stds):
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{val:.1f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height + std),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)

    add_value_labels(bars1, df['steiner_packing_avg'], df['steiner_packing_std'])
    add_value_labels(bars2, df['steiner_pathfinder_avg'], df['steiner_pathfinder_std'])
    add_value_labels(bars3, df['steiner_tree_avg'], df['steiner_tree_std'])

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved as '{output_filename}'")

    return df


def extract_qasm_wirelength_data(results_folder):
    """
    Extract wirelength data from QASM experiment result files, grouped by num_qubits.

    Supports two formats:
      - Individual *_comprehensive_results.json files (one per circuit)
      - Combined *_experimental_results.json files (from run_qasm_experiment)

    Returns:
        dict: {num_qubits: {algorithm: [wirelengths]}}
    """
    data = defaultdict(lambda: defaultdict(list))

    all_files = [f for f in os.listdir(results_folder) if f.endswith('.json')]

    # Try combined experimental results files first
    exp_files = [f for f in all_files if f.endswith('experimental_results.json')]
    for filename in exp_files:
        try:
            filepath = os.path.join(results_folder, filename)
            with open(filepath, 'r') as f:
                result_data = json.load(f)

            results_by_qubits = result_data.get('results_by_qubits', {})
            for nq_str, qdata in results_by_qubits.items():
                num_qubits = int(nq_str)
                for result in qdata.get('results', []):
                    algorithm_results = result.get('algorithm_results', {})
                    for algorithm in ['steiner_packing', 'steiner_pathfinder', 'steiner_tree']:
                        if algorithm in algorithm_results:
                            metrics = algorithm_results[algorithm].get('metrics', {})
                            total_wirelength = metrics.get('total_wirelength')
                            if total_wirelength is not None:
                                data[num_qubits][algorithm].append(total_wirelength)
        except (ValueError, KeyError, json.JSONDecodeError) as e:
            print(f"Error processing experimental file {filename}: {e}")

    # Also check individual comprehensive results files
    comp_files = [f for f in all_files if f.endswith('comprehensive_results.json')]
    print(f"Found {len(exp_files)} experimental + {len(comp_files)} comprehensive result files")

    for filename in comp_files:
        try:
            filepath = os.path.join(results_folder, filename)
            with open(filepath, 'r') as f:
                result_data = json.load(f)

            circuit_info = result_data.get('circuit_info', {})
            num_qubits = circuit_info.get('num_qubits')

            if num_qubits is None:
                num_qubits = result_data.get('layout_info', {}).get('total_patches')
            if num_qubits is None:
                print(f"Could not determine num_qubits from: {filename}")
                continue

            algorithm_results = result_data.get('algorithm_results', {})
            for algorithm in ['steiner_packing', 'steiner_pathfinder', 'steiner_tree']:
                if algorithm in algorithm_results:
                    metrics = algorithm_results[algorithm].get('metrics', {})
                    total_wirelength = metrics.get('total_wirelength')
                    if total_wirelength is not None:
                        data[num_qubits][algorithm].append(total_wirelength)

        except (ValueError, KeyError, json.JSONDecodeError) as e:
            print(f"Error processing file {filename}: {e}")
            continue

    return dict(data)


def create_qasm_wirelength_plot(data, output_filename='qasm_wirelength_comparison.png', title_prefix=''):
    """Create a bar plot showing total wirelength by number of qubits and algorithm."""
    qubit_counts = sorted(data.keys())
    algorithms = ['steiner_packing', 'steiner_pathfinder', 'steiner_tree']

    plot_data = []
    for nq in qubit_counts:
        row_data = {'Qubits': nq}
        for algorithm in algorithms:
            wirelengths = data[nq].get(algorithm, [])
            if wirelengths:
                avg_wl = np.mean(wirelengths)
                std_wl = np.std(wirelengths)
                row_data[f'{algorithm}_avg'] = avg_wl
                row_data[f'{algorithm}_std'] = std_wl
                row_data[f'{algorithm}_count'] = len(wirelengths)
                print(f"Qubits {nq}, {algorithm}: {avg_wl:.2f} ± {std_wl:.2f} (n={len(wirelengths)})")
            else:
                row_data[f'{algorithm}_avg'] = 0
                row_data[f'{algorithm}_std'] = 0
                row_data[f'{algorithm}_count'] = 0
                print(f"Qubits {nq}, {algorithm}: No data")
        plot_data.append(row_data)

    df = pd.DataFrame(plot_data)

    fig, ax = plt.subplots(figsize=(max(12, len(qubit_counts) * 0.8), 8))

    x = np.arange(len(qubit_counts))
    width = 0.25

    bars1 = ax.bar(x - width, df['steiner_packing_avg'], width,
                   yerr=df['steiner_packing_std'], label='Steiner Packing',
                   alpha=0.8, capsize=5)
    bars2 = ax.bar(x, df['steiner_pathfinder_avg'], width,
                   yerr=df['steiner_pathfinder_std'], label='Steiner Pathfinder',
                   alpha=0.8, capsize=5)
    bars3 = ax.bar(x + width, df['steiner_tree_avg'], width,
                   yerr=df['steiner_tree_std'], label='Steiner Tree',
                   alpha=0.8, capsize=5)

    title = f'{title_prefix}Total Wirelength Comparison Across Circuit Sizes\n(QASM Benchmark Circuits)'
    ax.set_xlabel('Number of Qubits', fontsize=12, fontweight='bold')
    ax.set_ylabel('Total Wirelength', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(qubit_counts)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    def add_value_labels(bars, values, stds):
        for bar, val, std in zip(bars, values, stds):
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{val:.1f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height + std),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=7)

    add_value_labels(bars1, df['steiner_packing_avg'], df['steiner_packing_std'])
    add_value_labels(bars2, df['steiner_pathfinder_avg'], df['steiner_pathfinder_std'])
    add_value_labels(bars3, df['steiner_tree_avg'], df['steiner_tree_std'])

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved as '{output_filename}'")

    return df
