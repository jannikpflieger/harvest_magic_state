#!/usr/bin/env python3
"""
Create Pauli evolution metric plots from circuit analysis results.

Generates per-circuit plots showing 4 metrics over layers (with moving windows):
  - Max Pauli Size
  - Avg Pauli Size
  - Max Pauli Count
  - Avg Pauli Count

Usage:
    python scripts/plot_pauli_metrics.py [--results-dir DIR] [--plots-dir DIR] [--file FILE] [--show]

Examples:
    # Process all circuit analysis JSON files (default directories)
    python scripts/plot_pauli_metrics.py

    # Process a specific JSON file
    python scripts/plot_pauli_metrics.py --file benchmark_circuits/circuit_analysis_results/some_circuit.json

    # Custom directories
    python scripts/plot_pauli_metrics.py --results-dir benchmark_circuits/circuit_analysis_results/ --plots-dir plots/
"""

import argparse
from pathlib import Path

from harvest.compilation.visualizer import (
    process_all_circuit_analyses,
    visualize_circuit_information,
)


def main():
    parser = argparse.ArgumentParser(
        description="Generate Pauli evolution metric plots from circuit analysis results."
    )
    parser.add_argument(
        "--results-dir",
        default="benchmark_circuits/circuit_analysis_results/",
        help="Directory containing circuit analysis JSON files (default: benchmark_circuits/circuit_analysis_results/)",
    )
    parser.add_argument(
        "--plots-dir",
        default="plots/",
        help="Directory to save generated plots (default: plots/)",
    )
    parser.add_argument(
        "--file",
        default=None,
        help="Process a single JSON file instead of the whole directory.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=None,
        help="Moving window size (default: auto-calculated based on number of layers).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively instead of only saving.",
    )

    args = parser.parse_args()

    if args.file:
        # Process a single file
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"Error: File not found: {file_path}")
            return

        plots_dir = Path(args.plots_dir)
        plots_dir.mkdir(parents=True, exist_ok=True)

        save_path = plots_dir / f"{file_path.stem}_analysis.png"
        print(f"Processing: {file_path}")
        visualize_circuit_information(
            str(file_path),
            window_size=args.window_size,
            save_file=str(save_path),
            show_plot=args.show,
        )
    else:
        # Batch-process all JSON files in the results directory
        results_dir = Path(args.results_dir)
        if not results_dir.exists():
            print(f"Error: Results directory not found: {results_dir}")
            return

        print(f"Processing all circuit analysis files in: {results_dir}")
        print(f"Saving plots to: {args.plots_dir}")
        process_all_circuit_analyses(
            results_dir=str(results_dir),
            plots_dir=args.plots_dir,
        )


if __name__ == "__main__":
    main()
