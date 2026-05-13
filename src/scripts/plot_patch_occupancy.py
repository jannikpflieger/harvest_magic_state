#!/usr/bin/env python3
"""
Patch occupancy plots — four separate PDF figures.

Reads either:
  - a finished summary.csv from evaluate_patch_occupancy.py, OR
  - the individual per-circuit JSON files in the same directory
    (useful if the sweep is still running or the CSV wasn't written).

Produces four PDFs in the same directory as the input:
  1. bar_averages.pdf       — grouped bars: mean routing/magic unused% per algorithm
  2. boxplot.pdf            — box-and-whisker distributions per algorithm
  3. strip.pdf              — per-circuit strip plot (one dot per circuit)
  4. heatmap_routing.pdf    — circuit × algorithm heatmap of routing unused%

Usage::

    PYTHONPATH=src python src/scripts/plot_patch_occupancy.py \\
        --results-dir results/patch_occupancy_<timestamp>

    # Or point at a specific summary.csv:
    PYTHONPATH=src python src/scripts/plot_patch_occupancy.py \\
        --csv results/patch_occupancy_<timestamp>/summary.csv
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
_src = str(PROJECT_ROOT / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALGO_ORDER = ["Sequential", "Greedy", "Pathfinder"]
ALGO_COLOURS = {
    "Sequential":  "#4C72B0",
    "Greedy":      "#55A868",
    "Pathfinder":  "#8172B2",
}
ALGO_LABELS = {
    "Sequential":  "No scheduling",
    "Greedy":      "Greedy",
    "Pathfinder":  "Pathfinder",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_from_json_dir(results_dir: Path):
    """Load per-circuit JSON files and return a flat list of row dicts."""
    rows = []
    for jf in sorted(results_dir.glob("*.json")):
        if jf.name in ("summary.json",):
            continue
        try:
            doc = json.loads(jf.read_text())
        except Exception:
            continue
        if doc.get("status") != "success":
            continue
        base = {
            "circuit_name":         doc["circuit_name"],
            "family":               doc.get("family", ""),
            "num_qubits":           doc.get("num_qubits", 0),
            "num_pauli_evolutions": doc.get("num_pauli_evolutions", 0),
            "num_routing_cells":    doc.get("num_routing_cells", 0),
            "num_magic_terminals":  doc.get("num_magic_terminals", 0),
        }
        for r in doc.get("scheduler_results", []):
            if not r.get("success"):
                continue
            rows.append({
                **base,
                "scheduler":           r["scheduler_name"],
                "routing_unused_frac": float(r.get("routing_unused_frac", 0)),
                "magic_unused_frac":   float(r.get("magic_unused_frac", 0)),
                "routing_unused":      int(r.get("routing_unused", 0)),
                "magic_unused":        int(r.get("magic_unused", 0)),
                "routing_total":       int(r.get("routing_total", 0)),
                "magic_total":         int(r.get("magic_total", 0)),
                "num_timesteps":       int(r.get("num_timesteps", 0)),
            })
    return rows


def _load_from_csv(csv_path: Path):
    import csv
    rows = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if not row.get("scheduler"):
                continue
            rows.append({
                "circuit_name":         row["circuit_name"],
                "family":               row.get("family", ""),
                "num_qubits":           int(row.get("num_qubits", 0) or 0),
                "num_pauli_evolutions": int(row.get("num_pauli_evolutions", 0) or 0),
                "num_routing_cells":    int(row.get("num_routing_cells", 0) or 0),
                "num_magic_terminals":  int(row.get("num_magic_terminals", 0) or 0),
                "scheduler":            row["scheduler"],
                "routing_unused_frac":  float(row.get("routing_unused_frac", 0) or 0),
                "magic_unused_frac":    float(row.get("magic_unused_frac", 0) or 0),
                "routing_unused":       int(row.get("routing_unused", 0) or 0),
                "magic_unused":         int(row.get("magic_unused", 0) or 0),
                "routing_total":        int(row.get("routing_total", 0) or 0),
                "magic_total":          int(row.get("magic_total", 0) or 0),
                "num_timesteps":        int(row.get("num_timesteps", 0) or 0),
            })
    return rows


def load_data(results_dir: Path, csv_path: Path = None):
    if csv_path and csv_path.exists():
        print(f"Loading from CSV: {csv_path}")
        rows = _load_from_csv(csv_path)
    else:
        print(f"Loading from JSON files in: {results_dir}")
        rows = _load_from_json_dir(results_dir)
    print(f"  {len(rows)} (circuit, scheduler) rows loaded")
    return rows


def _by_algo(rows):
    """Group rows by scheduler name, returning only algorithms that appear in ALGO_ORDER."""
    groups = {a: [] for a in ALGO_ORDER}
    for r in rows:
        sched = r["scheduler"]
        if sched in groups:
            groups[sched].append(r)
    return groups


# ---------------------------------------------------------------------------
# Plot 1 — Grouped bar chart of averages
# ---------------------------------------------------------------------------

def plot_bar_averages(rows, out_path: Path):
    """Side-by-side bars: mean routing unused% and mean magic unused% per algorithm."""
    groups = _by_algo(rows)
    algos = [a for a in ALGO_ORDER if groups[a]]

    r_means = [np.mean([x["routing_unused_frac"] for x in groups[a]]) * 100 for a in algos]
    m_means = [np.mean([x["magic_unused_frac"]   for x in groups[a]]) * 100 for a in algos]
    r_stds  = [np.std( [x["routing_unused_frac"] for x in groups[a]]) * 100 for a in algos]
    m_stds  = [np.std( [x["magic_unused_frac"]   for x in groups[a]]) * 100 for a in algos]

    x = np.arange(len(algos))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars_r = ax.bar(x - width / 2, r_means, width,
                    yerr=r_stds, capsize=4,
                    color=[ALGO_COLOURS[a] for a in algos],
                    alpha=0.85, label="Routing cells unused", hatch="//",
                    edgecolor="white")
    bars_m = ax.bar(x + width / 2, m_means, width,
                    yerr=m_stds, capsize=4,
                    color=[ALGO_COLOURS[a] for a in algos],
                    alpha=0.55, label="Magic state patches unused",
                    edgecolor="white")

    # value labels
    for bar in list(bars_r) + list(bars_m):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                f"{h:.1f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([ALGO_LABELS[a] for a in algos], fontsize=10)
    ax.set_xlabel("Scheduling algorithm", fontsize=11)
    ax.set_ylabel("Mean unused (%)", fontsize=11)
    ax.set_title("Average Patch Occupancy by Algorithm\n(error bars = ±1 std)", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.set_ylim(0, max(max(r_means), max(m_means)) * 1.25 + 5)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, framealpha=0.9)

    n = len(groups[algos[0]])
    ax.annotate(f"n = {n} circuits", xy=(0.99, 0.97), xycoords="axes fraction",
                ha="right", va="top", fontsize=8, color="gray")

    plt.tight_layout()
    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 2 — Box-and-whisker distributions
# ---------------------------------------------------------------------------

def plot_boxplot(rows, out_path: Path):
    """Box plots of routing unused% and magic unused% distributions per algorithm."""
    groups = _by_algo(rows)
    algos = [a for a in ALGO_ORDER if groups[a]]

    r_data = [[x["routing_unused_frac"] * 100 for x in groups[a]] for a in algos]
    m_data = [[x["magic_unused_frac"]   * 100 for x in groups[a]] for a in algos]
    labels = [ALGO_LABELS[a] for a in algos]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    def _draw(ax, data, title, ylabel):
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True,
                        medianprops=dict(color="black", linewidth=2),
                        whiskerprops=dict(linewidth=1.2),
                        capprops=dict(linewidth=1.2),
                        flierprops=dict(marker="o", markersize=3,
                                        linestyle="none", alpha=0.4))
        for patch, algo in zip(bp["boxes"], algos):
            patch.set_facecolor(ALGO_COLOURS[algo])
            patch.set_alpha(0.75)
        ax.set_xlabel("Scheduling algorithm", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)

    _draw(ax1, r_data, "Routing Cells Unused", "Unused (%)")
    _draw(ax2, m_data, "Magic State Patches Unused", "Unused (%)")

    fig.suptitle("Patch Occupancy Distribution by Algorithm", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 3 — Per-circuit strip plot
# ---------------------------------------------------------------------------

def plot_strip(rows, out_path: Path):
    """One dot per (circuit, algorithm) pair, jittered horizontally."""
    rng = np.random.default_rng(42)
    groups = _by_algo(rows)
    algos = [a for a in ALGO_ORDER if groups[a]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    def _draw(ax, metric_key, title, ylabel):
        for i, algo in enumerate(algos):
            vals = [x[metric_key] * 100 for x in groups[algo]]
            jitter = rng.uniform(-0.18, 0.18, size=len(vals))
            ax.scatter(np.full(len(vals), i) + jitter, vals,
                       color=ALGO_COLOURS[algo], alpha=0.35, s=12, linewidths=0)
            # overlay median line
            med = np.median(vals)
            ax.hlines(med, i - 0.3, i + 0.3, colors=ALGO_COLOURS[algo],
                      linewidth=2.5, zorder=5)
        ax.set_xticks(range(len(algos)))
        ax.set_xticklabels([ALGO_LABELS[a] for a in algos], fontsize=10)
        ax.set_xlabel("Scheduling algorithm", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)
        n = len(groups[algos[0]])
        ax.annotate(f"n = {n}", xy=(0.99, 0.97), xycoords="axes fraction",
                    ha="right", va="top", fontsize=8, color="gray")
        ax.annotate("— median", xy=(0.01, 0.97), xycoords="axes fraction",
                    ha="left", va="top", fontsize=8, color="gray")

    _draw(ax1, "routing_unused_frac", "Routing Cells Unused — per circuit", "Unused (%)")
    _draw(ax2, "magic_unused_frac",   "Magic State Patches Unused — per circuit", "Unused (%)")

    fig.suptitle("Per-Circuit Patch Occupancy by Algorithm", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 4 — Heatmap (routing unused%)
# ---------------------------------------------------------------------------

def plot_heatmap(rows, out_path: Path):
    """Heatmap: circuits (rows) × algorithms (columns), colour = routing unused%."""
    groups = _by_algo(rows)
    algos = [a for a in ALGO_ORDER if groups[a]]

    # Build circuit list from the first available algorithm
    ref_algo = algos[0]
    circuits = sorted({r["circuit_name"] for r in groups[ref_algo]},
                      key=lambda c: next(
                          (r["num_qubits"] for r in groups[ref_algo]
                           if r["circuit_name"] == c), 0))

    # matrix[i][j] = routing_unused_frac for circuit i, algo j
    lookup = {
        algo: {r["circuit_name"]: r["routing_unused_frac"] for r in groups[algo]}
        for algo in algos
    }
    matrix = np.array([
        [lookup[algo].get(c, np.nan) * 100 for algo in algos]
        for c in circuits
    ])

    # Limit to at most 80 circuits for readability; if more, aggregate by family
    if len(circuits) > 80:
        _plot_heatmap_family_aggregated(rows, algos, out_path)
        return

    fig_h = max(5, len(circuits) * 0.22 + 1.5)
    fig, ax = plt.subplots(figsize=(5, fig_h))

    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=100)
    plt.colorbar(im, ax=ax, label="Routing cells unused (%)", fraction=0.04, pad=0.02)

    ax.set_xticks(range(len(algos)))
    ax.set_xticklabels([ALGO_LABELS[a] for a in algos], fontsize=9)
    ax.set_xlabel("Scheduling algorithm", fontsize=11)
    ax.set_yticks(range(len(circuits)))
    ax.set_yticklabels(circuits, fontsize=5)
    ax.set_title("Routing Cells Unused (%) — per circuit", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def _plot_heatmap_family_aggregated(rows, algos, out_path: Path):
    """When there are too many circuits, aggregate by circuit family."""
    from collections import defaultdict

    # Gather all families
    family_algo_vals = defaultdict(lambda: defaultdict(list))
    for r in rows:
        fam = r["family"] or "other"
        if r["scheduler"] in algos:
            family_algo_vals[fam][r["scheduler"]].append(r["routing_unused_frac"] * 100)

    families = sorted(family_algo_vals.keys())
    matrix = np.array([
        [np.mean(family_algo_vals[fam][algo]) if family_algo_vals[fam][algo] else np.nan
         for algo in algos]
        for fam in families
    ])

    fig_h = max(4, len(families) * 0.55 + 1.5)
    fig, ax = plt.subplots(figsize=(5, fig_h))

    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=100)
    cbar = plt.colorbar(im, ax=ax, label="Mean routing cells unused (%)",
                        fraction=0.06, pad=0.02)

    ax.set_xticks(range(len(algos)))
    ax.set_xticklabels([ALGO_LABELS[a] for a in algos], fontsize=10)
    ax.set_xlabel("Scheduling algorithm", fontsize=11)
    ax.set_yticks(range(len(families)))
    ax.set_yticklabels(families, fontsize=9)
    ax.set_title("Mean Routing Cells Unused (%) by Family",
                 fontsize=11, fontweight="bold")

    # annotate cells
    for i in range(len(families)):
        for j in range(len(algos)):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                        fontsize=8, color="black" if val < 60 else "white")

    plt.tight_layout()
    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"  Saved (family-aggregated): {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate patch occupancy plots from evaluate_patch_occupancy results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results-dir", type=str, default=None,
        help="Directory with per-circuit JSON files (and optional summary.csv).",
    )
    parser.add_argument(
        "--csv", type=str, default=None,
        help="Path to summary.csv directly (overrides --results-dir for data loading).",
    )
    parser.add_argument(
        "--out-dir", type=str, default=None,
        help="Where to save PDFs. Defaults to same directory as the data.",
    )
    args = parser.parse_args()

    # Resolve paths
    if args.csv:
        csv_path = Path(args.csv)
        results_dir = csv_path.parent
    elif args.results_dir:
        results_dir = Path(args.results_dir)
        csv_path = results_dir / "summary.csv"
    else:
        # Auto-detect most recent patch_occupancy run
        runs = sorted(
            (PROJECT_ROOT / "results").glob("patch_occupancy_*"),
            key=lambda p: p.name, reverse=True
        )
        if not runs:
            print("No patch_occupancy results found. Run evaluate_patch_occupancy.py first.")
            sys.exit(1)
        results_dir = runs[0]
        csv_path = results_dir / "summary.csv"
        print(f"Auto-detected: {results_dir}")

    out_dir = Path(args.out_dir) if args.out_dir else results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load
    rows = load_data(results_dir, csv_path if csv_path.exists() else None)
    if not rows:
        print("No data found — nothing to plot.")
        sys.exit(1)

    algos_present = {r["scheduler"] for r in rows}
    print(f"  Algorithms present: {sorted(algos_present)}")
    print(f"  Unique circuits: {len({r['circuit_name'] for r in rows})}")
    print(f"  Saving PDFs to: {out_dir}")

    # Generate all four plots
    print("\nGenerating plots...")
    plot_bar_averages(rows, out_dir / "bar_averages.pdf")
    plot_boxplot(     rows, out_dir / "boxplot.pdf")
    plot_strip(       rows, out_dir / "strip.pdf")
    plot_heatmap(     rows, out_dir / "heatmap_routing.pdf")

    print("\nDone.")


if __name__ == "__main__":
    main()
