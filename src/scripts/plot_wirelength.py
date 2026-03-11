#!/usr/bin/env python3
"""
Create wirelength analysis plots from routing experiment results.

Usage:
    python scripts/plot_wirelength.py
"""

from harvest.evaluation.plotting import extract_wirelength_data, create_wirelength_plot


def main():
    results_folder = 'routing_experiment_results'

    print("Extracting total wirelength data from experiment results...")
    data = extract_wirelength_data(results_folder)

    print("\nData summary:")
    for depth in sorted(data.keys()):
        print(f"Depth {depth}:")
        for algorithm in ['steiner_packing', 'steiner_pathfinder', 'steiner_tree']:
            count = len(data[depth].get(algorithm, []))
            print(f"  {algorithm}: {count} runs")

    print("\nCreating total wirelength comparison plot...")
    df_results = create_wirelength_plot(data)

    print(f"\nResults summary:")
    print(df_results[['Depth', 'steiner_packing_avg', 'steiner_pathfinder_avg', 'steiner_tree_avg']])

    df_results.to_csv('results/wirelength_results_summary.csv', index=False)
    print("\nResults saved to 'results/wirelength_results_summary.csv'")


if __name__ == "__main__":
    main()
