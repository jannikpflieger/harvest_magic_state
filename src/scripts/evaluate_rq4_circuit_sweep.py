#!/usr/bin/env python3
"""
RQ4 Circuit Sweep: average speedup and timestep ratio across all circuits.

Iterates over every QASM file in a directory (or an explicit list of files),
runs all three schedulers on each circuit under the same layout, and reports:

    - logical_timestep_ratio  (relative to sequential baseline, lower is better)
    - parallelism_gain / speedup  (relative to sequential baseline, higher is better)

Layout sizing (auto, per circuit — same policy as the pruner sweep):
    single_spacing / double_spacing:  side = ceil(sqrt(num_qubits))
    blocks_of_four:                   side = ceil(sqrt(num_qubits / 4))

Per-circuit and averaged results are saved as CSV and PNG plots.

Outputs (in --output-dir)
--------------------------
    sweep_per_circuit.csv        – one row per (circuit, scheduler)
    sweep_averages.csv           – one row per scheduler (mean over circuits)
    plots/
        per_circuit_timestep_ratio.png   – grouped bar per circuit (timestep ratio)
        per_circuit_speedup.png          – grouped bar per circuit (speedup)
        average_timestep_ratio.png       – averaged bar per scheduler
        average_speedup.png              – averaged bar per scheduler
        average_combined.png             – 2-panel: timestep ratio | speedup

Usage
-----
    # Directory sweep (auto-discovers all .qasm files):
    cd src
    python scripts/evaluate_rq4_circuit_sweep.py \\
        --qasm-dir ../benchmark_circuits/qasm/qaoa \\
        --output-dir ../results/rq4_sweep

    # Explicit list of files:
    python scripts/evaluate_rq4_circuit_sweep.py \\
        --qasm a.qasm b.qasm c.qasm \\
        --output-dir ../results/rq4_sweep
"""

import argparse
import csv
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Reuse shared infrastructure from the RQ4 comparison script
from scripts.evaluate_rq4_scheduler_comparison import (
    LAYOUT_PRESET_MAP,
    SCHEDULER_COLORS,
    SCHEDULERS,
    _direction_hint,
    _display_labels,
    add_derived_metrics,
    build_layout_engine,
    find_qasm_files,
    load_and_prepare_circuit,
    make_magic_source_fn,
    run_single_scheduler,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("RQ4-Sweep")

# ── Colours & order ────────────────────────────────────────────────────────

SCHED_ORDER = [lbl for _, lbl in SCHEDULERS]     # ["Sequential", "Greedy", "Pathfinder"]
SCHED_COLORS = SCHEDULER_COLORS                   # shared palette


# ── Family filtering (mirrors pruner sweep policy) ──────────────────────

def _family_from_path(qasm_path: str, qasm_root: Optional[str]) -> str:
    """Return the top-level subdirectory name under *qasm_root* (e.g. 'qaoa', 'qv')."""
    if qasm_root is None:
        return Path(qasm_path).parent.name
    try:
        rel = Path(qasm_path).resolve().relative_to(Path(qasm_root).resolve())
        return rel.parts[0] if rel.parts else "unknown"
    except Exception:
        return Path(qasm_path).parent.name


# ── Auto layout sizing (mirrors pruner sweep policy) ─────────────────────

def _auto_grid(num_qubits: int, layout_preset: str) -> Tuple[int, int]:
    """Compute minimal square grid dimensions for *num_qubits*.

    Mirrors the policy in evaluate_pruner_layout_sweep.py:
      single/double spacing → side = ceil(sqrt(n))
      blocks_of_four        → side = ceil(sqrt(n / 4))
    """
    n = max(1, int(num_qubits))
    if layout_preset == "blocks_of_four":
        side = max(1, math.ceil(math.sqrt(n / 4.0)))
    else:
        side = max(1, math.ceil(math.sqrt(n)))
    return side, side


def _run_circuit_auto(qasm_path: str, layout_preset: str,
                      magic_source_mode: str, magic_prep_cycles: int,
                      ) -> Tuple[str, List[Dict]]:
    """Run all schedulers on one circuit with auto-sized layout."""
    from pathlib import Path as _P
    circuit_name = _P(qasm_path).stem
    try:
        circuit, dag = load_and_prepare_circuit(qasm_path)
    except Exception as exc:
        logger.error(f"  Failed to load/prepare circuit: {exc}", exc_info=True)
        return circuit_name, [
            {"mode": m, "label": l, "success": False, "error": str(exc),
             "circuit_name": circuit_name}
            for m, l in SCHEDULERS
        ]

    rows, cols = _auto_grid(circuit.num_qubits, layout_preset)
    logger.info(f"  Auto grid: {rows}×{cols}  (layout={layout_preset}, qubits={circuit.num_qubits})")

    layout_engine = build_layout_engine(rows, cols, layout_preset)
    magic_source_fn = make_magic_source_fn(magic_source_mode, magic_prep_cycles, layout_engine)

    results: List[Dict] = []
    for mode, label in SCHEDULERS:
        r = run_single_scheduler(dag, layout_engine, mode, label, magic_source_fn)
        r["circuit_name"] = circuit_name
        r["num_qubits"] = circuit.num_qubits
        r["grid_rows"] = rows
        r["grid_cols"] = cols
        results.append(r)

    add_derived_metrics(results)
    return circuit_name, results


# ── Plotting helpers ────────────────────────────────────────────────────────

def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved plot → {path}")


def _single_circuit_plot(
    circuit_name: str,
    sched_results: List[Dict],
    output_path: Path,
) -> None:
    """2-panel bar chart for one circuit: timestep ratio | speedup across schedulers."""
    labels = SCHED_ORDER
    colors = [SCHED_COLORS.get(l, "#888888") for l in labels]
    x = np.arange(len(labels))

    ts_vals = []
    sp_vals = []
    for lbl in labels:
        row = next((r for r in sched_results if r.get("label") == lbl), None)
        ts_vals.append(row.get("logical_timestep_ratio") or 0.0 if row and row.get("success") else 0.0)
        sp_vals.append(row.get("parallelism_gain") or 0.0       if row and row.get("success") else 0.0)

    panels = [
        (ts_vals, "Timestep Ratio\n(sequential = 1.0)", "lower"),
        (sp_vals, "Speedup\n(sequential = 1.0)",        "higher"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=False)
    for ax, (vals, ylabel, direction) in zip(axes, panels):
        bars = ax.bar(x, vals, width=0.5, color=colors, alpha=0.88, edgecolor="white")
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
        y_max = max(vals) if any(v > 0 for v in vals) else 1
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + y_max * 0.02,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=8, fontweight="bold",
            )
        _direction_hint(ax, direction)
        ax.set_xticks(x)
        ax.set_xticklabels(_display_labels(labels), fontsize=10)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)

    num_qubits = next((r.get("num_qubits") for r in sched_results if r.get("num_qubits")), "?")
    fig.suptitle(
        f"{circuit_name}  ({num_qubits} qubits)",
        fontsize=10, fontweight="bold",
    )
    plt.tight_layout()
    _save(fig, output_path)


def _per_circuit_bar(
    circuit_names: List[str],
    data: Dict[str, List[Optional[float]]],   # {scheduler_label: [value per circuit]}
    ylabel: str,
    title: str,
    direction: str,
    output_path: Path,
) -> None:
    """Grouped bar chart: circuits on x-axis, scheduler groups within each cluster."""
    n_circuits = len(circuit_names)
    n_sched = len(SCHED_ORDER)
    group_w = 0.75
    bar_w = group_w / n_sched
    x = np.arange(n_circuits)

    # Compact x-labels: strip common prefix/suffix noise
    short_names = [_short_name(c) for c in circuit_names]

    fig, ax = plt.subplots(figsize=(max(8, n_circuits * 1.2 + 2), 5))

    for i, lbl in enumerate(SCHED_ORDER):
        values = data.get(lbl, [None] * n_circuits)
        offsets = x + (i - (n_sched - 1) / 2) * bar_w
        color = SCHED_COLORS.get(lbl, "#888888")
        bars = ax.bar(offsets, [v if v is not None else 0 for v in values],
                      width=bar_w * 0.9, color=color, alpha=0.88,
                      edgecolor="white", label=lbl)

        # Value labels above bars
        for bar, val in zip(bars, values):
            if val is None:
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(v for v in values if v) * 0.01,
                f"{val:.2f}",
                ha="center", va="bottom", fontsize=7, fontweight="bold",
            )

    _direction_hint(ax, direction)
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    plt.tight_layout()
    _save(fig, output_path)


def _average_bar(
    avg_values: Dict[str, Optional[float]],
    std_values: Dict[str, Optional[float]],
    ylabel: str,
    title: str,
    subtitle: str,
    direction: str,
    output_path: Path,
) -> None:
    """Single bar chart of per-scheduler averages with std-dev error bars."""
    labels = SCHED_ORDER
    disp = _display_labels(labels)
    colors = [SCHED_COLORS.get(l, "#888888") for l in labels]
    vals = [avg_values.get(l) or 0.0 for l in labels]
    errs = [std_values.get(l) or 0.0 for l in labels]

    fig, ax = plt.subplots(figsize=(6, 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, vals, width=0.5, color=colors, alpha=0.88,
                  edgecolor="white",
                  yerr=errs, capsize=5, error_kw={"elinewidth": 1.4, "alpha": 0.7})

    y_max = max(vals) if vals else 1
    for bar, val, err in zip(bars, vals, errs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + err + y_max * 0.01,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    _direction_hint(ax, direction)
    ax.set_xticks(x)
    ax.set_xticklabels(disp, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(f"{title}\n{subtitle}", fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    plt.tight_layout()
    _save(fig, output_path)


def _combined_average_plot(
    avg_ts: Dict[str, Optional[float]],
    std_ts: Dict[str, Optional[float]],
    avg_sp: Dict[str, Optional[float]],
    std_sp: Dict[str, Optional[float]],
    subtitle: str,
    output_path: Path,
) -> None:
    """2-panel combined plot: averaged timestep ratio | averaged speedup."""
    labels = SCHED_ORDER
    disp = _display_labels(labels)
    colors = [SCHED_COLORS.get(l, "#888888") for l in labels]
    x = np.arange(len(labels))

    panels = [
        (avg_ts, std_ts, "Timestep Ratio\n(sequential = 1.0)",  "lower"),
        (avg_sp, std_sp, "Speedup\n(sequential = 1.0)",         "higher"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=False)

    for ax, (avgs, stds, ylabel, direction) in zip(axes, panels):
        vals = [avgs.get(l) or 0.0 for l in labels]
        errs = [stds.get(l) or 0.0 for l in labels]

        bars = ax.bar(x, vals, width=0.5, color=colors, alpha=0.88,
                      edgecolor="white",
                      yerr=errs, capsize=5,
                      error_kw={"elinewidth": 1.4, "alpha": 0.7})
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.2, alpha=0.8)

        y_max = max(vals) if vals else 1
        for bar, val, err in zip(bars, vals, errs):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + err + y_max * 0.01,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold",
            )

        _direction_hint(ax, direction)
        ax.set_xticks(x)
        ax.set_xticklabels(disp, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)

    handles = [plt.Line2D([0], [0], color="gray", linestyle="--", linewidth=1.2)]
    fig.legend(handles, ["sequential baseline"], loc="lower center",
               ncol=1, fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(
        f"RQ4 — Average Speedup & Timestep Ratio Across Circuits\n{subtitle}",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    _save(fig, output_path)


# ── Utility ─────────────────────────────────────────────────────────────────

def _short_name(name: str, max_len: int = 22) -> str:
    """Shorten a circuit name for use as an x-axis tick label."""
    if len(name) <= max_len:
        return name
    # Keep suffix (usually carries meaningful info like qubit count)
    return "…" + name[-(max_len - 1):]


def _compute_averages(
    per_circuit: Dict[str, Dict[str, Optional[float]]],
    # {circuit_name: {scheduler_label: value}}
    metric: str,
    results_flat: List[Dict],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Return (mean_per_scheduler, std_per_scheduler) for *metric*."""
    by_sched: Dict[str, List[float]] = defaultdict(list)
    for r in results_flat:
        if r.get("success") and r.get(metric) is not None:
            by_sched[r["label"]].append(float(r[metric]))
    means = {lbl: float(np.mean(vals)) for lbl, vals in by_sched.items() if vals}
    stds  = {lbl: float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
             for lbl, vals in by_sched.items() if vals}
    return means, stds


# ── CSV helpers ──────────────────────────────────────────────────────────────

def save_per_circuit_csv(results_flat: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["circuit_name", "num_qubits", "grid_rows", "grid_cols",
              "label", "logical_timestep_ratio", "parallelism_gain",
              "num_time_steps", "success"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(results_flat)
    logger.info(f"  Saved CSV  → {path}")


def save_averages_csv(
    avg_ts: Dict[str, float], std_ts: Dict[str, float],
    avg_sp: Dict[str, float], std_sp: Dict[str, float],
    n_circuits: int,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["scheduler", "mean_timestep_ratio", "std_timestep_ratio",
                    "mean_speedup", "std_speedup", "n_circuits"])
        for lbl in SCHED_ORDER:
            w.writerow([
                lbl,
                f"{avg_ts.get(lbl, ''):.6f}" if lbl in avg_ts else "",
                f"{std_ts.get(lbl, ''):.6f}" if lbl in std_ts else "",
                f"{avg_sp.get(lbl, ''):.6f}" if lbl in avg_sp else "",
                f"{std_sp.get(lbl, ''):.6f}" if lbl in std_sp else "",
                n_circuits,
            ])
    logger.info(f"  Saved CSV  → {path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RQ4 Circuit Sweep: average speedup and timestep ratio across circuits.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--qasm-dir", metavar="PATH",
                     help="Directory to scan recursively for .qasm files.")
    src.add_argument("--qasm", nargs="+", metavar="PATH",
                     help="Explicit list of .qasm files.")

    p.add_argument("--max-files", metavar="INT", type=int, default=None,
                   help="Limit the number of circuits processed.")
    p.add_argument("--max-qubits", metavar="INT", type=int, default=100,
                   help="Skip circuits with more than this many qubits (0 = no limit).")
    p.add_argument("--exclude-families", nargs="*", default=["qv", "chemical", "dtc"],
                   metavar="F",
                   help="Top-level circuit families to exclude (by subdirectory name).")
    p.add_argument("--layout", default="single_spacing",
                   choices=list(LAYOUT_PRESET_MAP.keys()),
                   help="Layout type (grid size is auto-computed per circuit).")
    p.add_argument("--magic-source", choices=["unlimited", "factory"],
                   default="unlimited")
    p.add_argument("--magic-prep-cycles", metavar="INT", type=int, default=15)
    p.add_argument("--output-dir", metavar="PATH", default=None,
                   help="Output directory (default: results/rq4_sweep_<timestamp>).")
    return p.parse_args()


def main() -> None:
    from datetime import datetime
    args = parse_args()
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("../results") / f"rq4_sweep_{args.layout}_{ts}"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    circuits_plots_dir = plots_dir / "circuits"
    plots_dir.mkdir(parents=True, exist_ok=True)
    circuits_plots_dir.mkdir(parents=True, exist_ok=True)

    # ── Collect QASM files ────────────────────────────────────────────────
    exclude_families = set(args.exclude_families or [])
    max_qubits = args.max_qubits if args.max_qubits > 0 else None

    if args.qasm_dir:
        qasm_files = sorted(find_qasm_files(args.qasm_dir))
        # Apply family exclusion (only meaningful when scanning a directory tree)
        qasm_files = [
            q for q in qasm_files
            if _family_from_path(q, args.qasm_dir) not in exclude_families
        ]
    else:
        qasm_files = list(args.qasm)

    if args.max_files is not None:
        qasm_files = qasm_files[: args.max_files]

    if not qasm_files:
        raise SystemExit("No QASM files found.")

    logger.info(
        f"\nRQ4 Circuit Sweep"
        f"\n  Layout:          {args.layout}  (auto-sized per circuit)"
        f"\n  Circuits (pre-qubit filter): {len(qasm_files)}"
        f"\n  Exclude families: {sorted(exclude_families) or 'none'}"
        f"\n  Max qubits:       {max_qubits or 'unlimited'}"
        f"\n  Output:           {output_dir}\n"
    )

    # ── Run all circuits ──────────────────────────────────────────────────
    all_results: List[Dict] = []
    circuit_names: List[str] = []

    # Initialise per-scheduler series here so they stay in sync after each circuit
    ts_by_sched: Dict[str, List[Optional[float]]] = {l: [] for l in SCHED_ORDER}
    sp_by_sched: Dict[str, List[Optional[float]]] = {l: [] for l in SCHED_ORDER}

    excl_str = ", ".join(sorted(exclude_families)) if exclude_families else "none"

    skipped_too_large = 0
    for qasm_path in qasm_files:
        logger.info(f"\n{'='*60}")
        logger.info(f"Circuit: {qasm_path}")

        # Quick qubit-count check before running the expensive schedulers
        if max_qubits is not None:
            try:
                circuit_check, _ = load_and_prepare_circuit(qasm_path)
                n_q = circuit_check.num_qubits
            except Exception:
                n_q = 0
            if n_q > max_qubits:
                logger.info(f"  Skipped: {n_q} qubits > max_qubits={max_qubits}")
                skipped_too_large += 1
                continue

        circuit_name, sched_results = _run_circuit_auto(
            qasm_path=qasm_path,
            layout_preset=args.layout,
            magic_source_mode=args.magic_source,
            magic_prep_cycles=args.magic_prep_cycles,
        )
        circuit_names.append(circuit_name)
        all_results.extend(sched_results)

        # ── After each circuit: update per-scheduler series ───────────────
        for lbl in SCHED_ORDER:
            row = next((r for r in sched_results if r.get("label") == lbl), None)
            ts_by_sched[lbl].append(
                row.get("logical_timestep_ratio") if row and row.get("success") else None
            )
            sp_by_sched[lbl].append(
                row.get("parallelism_gain") if row and row.get("success") else None
            )

        # ── After each circuit: flush incremental CSV ─────────────────────
        save_per_circuit_csv(all_results, output_dir / "sweep_per_circuit.csv")

        # ── After each circuit: individual plot for this circuit ───────────
        _single_circuit_plot(
            circuit_name, sched_results,
            output_path=circuits_plots_dir / f"{circuit_name}.png",
        )

    if skipped_too_large:
        logger.info(f"\nSkipped {skipped_too_large} circuit(s) exceeding max_qubits={max_qubits}.")

    # ── Compute averages (once, at the end) ───────────────────────────────
    avg_ts, std_ts = _compute_averages({}, "logical_timestep_ratio", all_results)
    avg_sp, std_sp = _compute_averages({}, "parallelism_gain",       all_results)

    n_circuits = len(circuit_names)
    subtitle = (
        f"{n_circuits} circuit{'s' if n_circuits != 1 else ''}  |  "
        f"{args.layout} layout  (auto-sized)  |  "
        f"excl: {excl_str}  |  max {max_qubits or '∞'} q"
    )

    # ── Save averages CSV (once, at the end) ──────────────────────────────
    save_averages_csv(avg_ts, std_ts, avg_sp, std_sp, n_circuits,
                      output_dir / "sweep_averages.csv")

    # ── Grouped per-circuit bars (once, at the end) ────────────────────────
    if n_circuits > 1:
        _per_circuit_bar(
            circuit_names, ts_by_sched,
            ylabel="Timestep Ratio  (sequential = 1.0)",
            title="RQ4 — Timestep Ratio per Circuit",
            direction="lower",
            output_path=plots_dir / "per_circuit_timestep_ratio.png",
        )
        _per_circuit_bar(
            circuit_names, sp_by_sched,
            ylabel="Speedup  (sequential = 1.0)",
            title="RQ4 — Speedup per Circuit",
            direction="higher",
            output_path=plots_dir / "per_circuit_speedup.png",
        )

    # ── Average plots (once, at the end) ──────────────────────────────────
    _average_bar(
        avg_ts, std_ts,
        ylabel="Mean Timestep Ratio  (sequential = 1.0)",
        title="RQ4 — Average Timestep Ratio",
        subtitle=subtitle,
        direction="lower",
        output_path=plots_dir / "average_timestep_ratio.png",
    )
    _average_bar(
        avg_sp, std_sp,
        ylabel="Mean Speedup  (sequential = 1.0)",
        title="RQ4 — Average Speedup",
        subtitle=subtitle,
        direction="higher",
        output_path=plots_dir / "average_speedup.png",
    )
    _combined_average_plot(
        avg_ts, std_ts, avg_sp, std_sp,
        subtitle=subtitle,
        output_path=plots_dir / "average_combined.png",
    )

    # ── Summary table ──────────────────────────────────────────────────────
    logger.info(f"\n{'='*65}")
    logger.info("AVERAGE RESULTS")
    logger.info(f"{'='*65}")
    logger.info(f"  {'Scheduler':<20}  {'Mean Timestep Ratio':>20}  {'Mean Speedup':>14}")
    logger.info(f"  {'-'*58}")
    for lbl in SCHED_ORDER:
        ts_str = f"{avg_ts[lbl]:.4f} ± {std_ts.get(lbl, 0.0):.4f}" if lbl in avg_ts else "N/A"
        sp_str = f"{avg_sp[lbl]:.4f} ± {std_sp.get(lbl, 0.0):.4f}" if lbl in avg_sp else "N/A"
        logger.info(f"  {lbl:<20}  {ts_str:>20}  {sp_str:>14}")
    logger.info(f"{'='*65}")


if __name__ == "__main__":
    main()
