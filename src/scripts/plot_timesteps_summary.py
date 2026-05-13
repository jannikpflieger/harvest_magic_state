#!/usr/bin/env python3
"""
Plot mean timesteps across all benchmark circuits for each scheduler.

Reads all per-circuit JSON files from a benchmark sweep directory and produces
a summary bar chart of mean timesteps (±std) per scheduler, plus a per-circuit
scatter plot showing the speedup of each parallel scheduler over sequential.

Usage (from repo root):
    PYTHONPATH=src python src/scripts/plot_timesteps_summary.py
    PYTHONPATH=src python src/scripts/plot_timesteps_summary.py --sweep-dir routing_experiment_results/benchmark_sweep_20260426_173135
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCHEDULER_ORDER = ["Sequential", "Greedy Packing", "Pathfinder", "Adaptive"]

COLOURS = {
    "Sequential":    "#4C72B0",
    "Greedy Packing":"#55A868",
    "Pathfinder":    "#C44E52",
    "Adaptive":      "#FF7F0E",
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_sweep(sweep_dir: str) -> list:
    """Return a list of per-circuit dicts with timestep data for each scheduler."""
    records = []
    for path in sorted(Path(sweep_dir).glob("*.json")):
        try:
            with open(path) as f:
                d = json.load(f)
        except Exception:
            continue

        if d.get("status") not in ("success", None):
            continue

        row = {
            "circuit_name": d.get("circuit_name", path.stem),
            "family": d.get("family", ""),
            "num_qubits": d.get("num_qubits", 0),
            "num_pauli_evolutions": d.get("num_pauli_evolutions", 0),
        }

        for r in d.get("scheduler_results", []):
            if r.get("success") and r.get("completed", True):
                row[r["scheduler_name"]] = r["num_timesteps"]

        # Only keep circuits where we have data for all four schedulers
        if all(s in row for s in SCHEDULER_ORDER):
            records.append(row)

    return records


# ---------------------------------------------------------------------------
# Plot 1: Mean ± std bar chart
# ---------------------------------------------------------------------------

def plot_mean_timesteps(records: list, output_path: str) -> None:
    """Bar chart: mean timesteps per scheduler across all circuits."""
    means, stds, medians = {}, {}, {}
    for sched in SCHEDULER_ORDER:
        vals = [r[sched] for r in records]
        means[sched]   = np.mean(vals)
        stds[sched]    = np.std(vals)
        medians[sched] = np.median(vals)

    x = np.arange(len(SCHEDULER_ORDER))
    bar_vals   = [means[s]   for s in SCHEDULER_ORDER]
    bar_errs   = [stds[s]    for s in SCHEDULER_ORDER]
    bar_meds   = [medians[s] for s in SCHEDULER_ORDER]
    bar_colors = [COLOURS[s] for s in SCHEDULER_ORDER]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        x, bar_vals,
        yerr=bar_errs,
        capsize=5,
        color=bar_colors,
        alpha=0.85,
        edgecolor="white",
        error_kw={"elinewidth": 1.5, "ecolor": "black", "alpha": 0.6},
    )

    # Annotate mean and median on bars
    y_max = max(bar_vals) + max(bar_errs)
    for i, (bar, mean_v, med_v) in enumerate(zip(bars, bar_vals, bar_meds)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(bar_errs) * 0.05,
            f"μ={mean_v:.0f}\nmed={med_v:.0f}",
            ha="center", va="bottom", fontsize=8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(SCHEDULER_ORDER, fontsize=11)
    ax.set_ylabel("Timesteps", fontsize=11)
    ax.set_title(
        f"Mean Timesteps by Scheduler  ({len(records)} circuits)\n"
        "Error bars = ±1 std dev",
        fontsize=11, fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Plot 2: Speedup scatter (timesteps relative to sequential) per circuit
# ---------------------------------------------------------------------------

def plot_speedup_scatter(records: list, output_path: str) -> None:
    """Scatter: T_seq / T_sched for each circuit, sorted by T_seq."""
    records_sorted = sorted(records, key=lambda r: r["Sequential"])
    n = len(records_sorted)
    x = np.arange(n)

    parallel_scheds = ["Greedy Packing", "Pathfinder", "Adaptive"]

    fig, ax = plt.subplots(figsize=(max(8, n * 0.04 + 4), 5))

    for sched in parallel_scheds:
        speedups = [r["Sequential"] / max(r[sched], 1) for r in records_sorted]
        ax.scatter(x, speedups, s=8, alpha=0.6, label=sched, color=COLOURS[sched])

    ax.axhline(1.0, color="gray", linewidth=1, linestyle="--", label="Sequential (baseline)")
    ax.set_xlabel(f"Circuit (sorted by sequential timesteps, n={n})", fontsize=10)
    ax.set_ylabel("Speedup  (T_sequential / T_scheduler)", fontsize=10)
    ax.set_title(
        "Per-Circuit Timestep Speedup vs. Sequential",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Plot 3: Combined — mean bars + per-circuit lines (log scale)
# ---------------------------------------------------------------------------

def plot_combined(records: list, output_path: str) -> None:
    """Two-panel figure: mean bars (left) + per-circuit scatter (right)."""
    records_sorted = sorted(records, key=lambda r: r["Sequential"])
    n = len(records_sorted)
    x = np.arange(n)

    fig, (ax_bar, ax_sc) = plt.subplots(
        1, 2, figsize=(14, 5),
        gridspec_kw={"width_ratios": [1, 2.5]},
    )

    # --- Left: mean bar chart ---
    means = {s: np.mean([r[s] for r in records]) for s in SCHEDULER_ORDER}
    stds  = {s: np.std( [r[s] for r in records]) for s in SCHEDULER_ORDER}
    xi = np.arange(len(SCHEDULER_ORDER))

    bars = ax_bar.bar(
        xi,
        [means[s] for s in SCHEDULER_ORDER],
        yerr=[stds[s] for s in SCHEDULER_ORDER],
        capsize=4,
        color=[COLOURS[s] for s in SCHEDULER_ORDER],
        alpha=0.85,
        edgecolor="white",
        error_kw={"elinewidth": 1.2, "ecolor": "black", "alpha": 0.5},
    )
    for bar, sched in zip(bars, SCHEDULER_ORDER):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + stds[sched] * 0.05,
            f"{means[sched]:.0f}",
            ha="center", va="bottom", fontsize=9,
        )
    ax_bar.set_xticks(xi)
    ax_bar.set_xticklabels(
        [s.replace(" ", "\n") for s in SCHEDULER_ORDER], fontsize=9,
    )
    ax_bar.set_ylabel("Timesteps", fontsize=10)
    ax_bar.set_title(f"Mean ± std\n({n} circuits)", fontsize=10, fontweight="bold")
    ax_bar.grid(axis="y", alpha=0.3, linestyle="--")
    ax_bar.set_axisbelow(True)

    # --- Right: per-circuit scatter (log scale) ---
    for sched in SCHEDULER_ORDER:
        vals = [r[sched] for r in records_sorted]
        ax_sc.scatter(x, vals, s=6, alpha=0.55, label=sched, color=COLOURS[sched])

    ax_sc.set_yscale("log")
    ax_sc.set_xlabel(f"Circuit index (sorted by sequential T, n={n})", fontsize=9)
    ax_sc.set_ylabel("Timesteps (log scale)", fontsize=10)
    ax_sc.set_title("Per-Circuit Timesteps", fontsize=10, fontweight="bold")
    ax_sc.legend(fontsize=8, framealpha=0.9, markerscale=2)
    ax_sc.grid(alpha=0.25, linestyle="--")
    ax_sc.set_axisbelow(True)

    fig.suptitle(
        "Scheduler Timestep Comparison Across Benchmark Circuits",
        fontsize=12, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(records: list) -> None:
    seq_baseline = np.mean([r["Sequential"] for r in records])
    print(f"\n{'Scheduler':20s} {'Mean T':>8s} {'Median T':>9s} {'Std T':>7s} {'vs Seq':>8s}")
    print("-" * 58)
    for sched in SCHEDULER_ORDER:
        vals = [r[sched] for r in records]
        mean_v   = np.mean(vals)
        median_v = np.median(vals)
        std_v    = np.std(vals)
        ratio    = seq_baseline / mean_v if mean_v > 0 else float("nan")
        print(
            f"{sched:20s} {mean_v:8.1f} {median_v:9.1f} {std_v:7.1f} {ratio:7.2f}x"
        )
    print(f"\nTotal circuits included: {len(records)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot timestep summary across benchmark sweep.")
    parser.add_argument(
        "--sweep-dir", default=None,
        help="Path to sweep results directory (default: latest benchmark_sweep_* folder).",
    )
    parser.add_argument(
        "--out-dir", default="plots",
        help="Output directory for plots (default: plots/).",
    )
    args = parser.parse_args()

    # Resolve sweep directory
    if args.sweep_dir:
        sweep_dir = args.sweep_dir
    else:
        results_root = Path("routing_experiment_results")
        candidates = sorted(results_root.glob("benchmark_sweep_*"), reverse=True)
        if not candidates:
            raise RuntimeError("No benchmark_sweep_* directories found in routing_experiment_results/")
        sweep_dir = str(candidates[0])

    print(f"Loading from: {sweep_dir}")
    records = load_sweep(sweep_dir)
    if not records:
        raise RuntimeError("No complete circuit records found (need all 4 schedulers succeeded).")

    print_summary(records)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sweep_tag = Path(sweep_dir).name

    plot_mean_timesteps(records, str(out_dir / f"timesteps_mean_{sweep_tag}.png"))
    plot_speedup_scatter(records, str(out_dir / f"timesteps_speedup_{sweep_tag}.png"))
    plot_combined(records, str(out_dir / f"timesteps_combined_{sweep_tag}.png"))


if __name__ == "__main__":
    main()
