#!/usr/bin/env python3
"""
Benchmark Circuit Comparison Plot: Sequential vs Greedy Packing Scheduling

Summarizes all benchmark circuits we have results for, limited to circuits with
up to 100 qubits. Circuits from the qv and chemical families are skipped.

Outputs:
- CSV summary with one row per circuit
- Horizontal bar chart comparing sequential vs greedy packing timesteps
"""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


EXCLUDED_FAMILIES = {"qv", "chemical"}
MAX_QUBITS = 100


def extract_timesteps_from_result_file(file_path):
    """Extract sequential and greedy packing timesteps from a result JSON file."""
    with open(file_path, "r") as f:
        data = json.load(f)

    result_sets = []
    if "factory_vs_cultivation_results" in data:
        result_sets = data.get("factory_vs_cultivation_results", [])
    elif "results" in data:
        result_sets = data.get("results", [])

    sequential_ts = None
    greedy_ts = None

    for result in result_sets:
        scheduler_label = result.get("scheduler_label", "")
        scheduler_mode = result.get("scheduler_mode", "")

        if "factory_vs_cultivation_results" in data:
            source_label = result.get("source_label", "")
            matches_source = source_label == "Unlimited"
        else:
            layout_label = result.get("layout_label", "")
            matches_source = layout_label == "Circuit-Aware"

        if (
            matches_source
            and scheduler_label == "Sequential"
            and scheduler_mode == "steiner_tree"
        ):
            sequential_ts = result.get("num_timesteps")

        if (
            matches_source
            and scheduler_label == "Greedy Packing"
            and scheduler_mode == "steiner_packing"
        ):
            greedy_ts = result.get("num_timesteps")

    return sequential_ts, greedy_ts


def discover_benchmark_circuit_results(base_path):
    """Return benchmark comparison result files that match the requested filters."""
    results_dir = base_path / "routing_experiment_results" / "benchmark_comparison"
    if not results_dir.exists():
        raise FileNotFoundError(f"Missing results directory: {results_dir}")

    result_files = sorted(results_dir.glob("*.json"))
    selected = []

    for result_file in result_files:
        with open(result_file, "r") as f:
            data = json.load(f)

        metadata = data.get("circuit_metadata", {})
        num_qubits = metadata.get("num_qubits")
        family = metadata.get("family", "")

        if family in EXCLUDED_FAMILIES:
            continue
        if num_qubits is None or num_qubits > MAX_QUBITS:
            continue

        selected.append((result_file, metadata))

    return selected


def load_all_circuit_data(base_path):
    """Load sequential vs greedy timestep data for all eligible benchmark circuits."""
    rows = []

    for result_file, metadata in discover_benchmark_circuit_results(base_path):
        sequential_ts, greedy_ts = extract_timesteps_from_result_file(str(result_file))

        if sequential_ts is None or greedy_ts is None:
            print(f"Skipping {result_file.name}: could not find both timestep values")
            continue

        circuit_name = metadata.get("circuit_name", result_file.stem)
        family = metadata.get("family", "")
        num_qubits = metadata.get("num_qubits")
        speedup = sequential_ts / greedy_ts if greedy_ts else None

        rows.append(
            {
                "circuit_name": circuit_name,
                "family": family,
                "num_qubits": num_qubits,
                "sequential_timesteps": sequential_ts,
                "greedy_timesteps": greedy_ts,
                "speedup": speedup,
                "result_file": result_file.name,
            }
        )

    rows.sort(key=lambda row: (row["num_qubits"], row["family"], row["circuit_name"]))
    return rows


def save_comparison_csv(rows, output_path_csv):
    """Save the circuit-level comparison table to CSV."""
    fieldnames = [
        "circuit_name",
        "family",
        "num_qubits",
        "sequential_timesteps",
        "greedy_timesteps",
        "speedup",
        "result_file",
    ]

    with open(output_path_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"CSV saved to {output_path_csv}")


def create_comparison_plot(rows, output_path_png, output_path_pdf=None):
    """Create a horizontal bar chart comparing sequential vs greedy packing timesteps."""
    if not rows:
        raise ValueError("No benchmark circuit rows available for plotting")

    labels = [f"{row['circuit_name']} ({row['num_qubits']}q)" for row in rows]
    sequential_values = [row["sequential_timesteps"] for row in rows]
    greedy_values = [row["greedy_timesteps"] for row in rows]

    y = np.arange(len(rows))
    height = 0.38
    fig_height = max(8, 0.34 * len(rows) + 2)
    fig, ax = plt.subplots(figsize=(14, fig_height))

    bars1 = ax.barh(
        y - height / 2,
        sequential_values,
        height,
        label="No Scheduling",
        color="#1f77b4",
        alpha=0.82,
        edgecolor="black",
        linewidth=1.0,
    )
    bars2 = ax.barh(
        y + height / 2,
        greedy_values,
        height,
        label="Scheduling",
        color="#ff7f0e",
        alpha=0.82,
        edgecolor="black",
        linewidth=1.0,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Timesteps", fontsize=13, fontweight="bold")
    ax.set_ylabel("Circuit", fontsize=13, fontweight="bold")
    ax.set_title(
        "Sequential vs Greedy Packing Timesteps Across Benchmark Circuits",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(fontsize=11, loc="lower right", framealpha=0.95)
    ax.grid(axis="x", alpha=0.3, linestyle="--", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=10)

    max_value = max(max(sequential_values), max(greedy_values))
    ax.set_xlim(0, max_value * 1.12)

    for bar in list(bars1) + list(bars2):
        width = bar.get_width()
        ax.text(
            width + max_value * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{int(width)}",
            va="center",
            ha="left",
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(output_path_png, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Plot saved to {output_path_png}")

    if output_path_pdf is not None:
        plt.savefig(output_path_pdf, bbox_inches="tight", facecolor="white")
        print(f"Plot saved to {output_path_pdf}")

    return fig, ax


def main():
    """Main execution function."""
    script_dir = Path(__file__).parent
    base_path = script_dir.parent.parent

    print("Loading benchmark circuit data...")
    rows = load_all_circuit_data(base_path)

    print(f"\nLoaded {len(rows)} benchmark circuits after filtering")
    print("Extracted data:")
    for row in rows:
        print(
            f"  {row['circuit_name']} ({row['num_qubits']}q): "
            f"sequential={row['sequential_timesteps']}, "
            f"greedy={row['greedy_timesteps']}, "
            f"speedup={row['speedup']:.1f}x"
        )

    output_dir = base_path / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path_png = output_dir / "benchmark_comparison_timesteps_all.png"
    output_path_pdf = output_dir / "benchmark_comparison_timesteps_all.pdf"
    output_path_csv = output_dir / "benchmark_comparison_timesteps_all.csv"

    save_comparison_csv(rows, str(output_path_csv))

    print("\nGenerating plot...")
    create_comparison_plot(rows, str(output_path_png), str(output_path_pdf))
    print("Done!")


if __name__ == "__main__":
    main()