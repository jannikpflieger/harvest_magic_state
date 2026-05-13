#!/usr/bin/env python3
"""
RQ4: How do different lattice-surgery routing/scheduling strategies affect
execution time and routing cost under the same circuit, layout, and
magic-state assumptions?

Compares three scheduler modes on the same circuit DAG and layout:
    1. steiner_tree      – Sequential baseline
    2. steiner_packing   – Greedy parallel packing
    3. steiner_pathfinder – Pathfinder-style negotiated packing

For each scheduler mode the script collects:
    - num_time_steps, total_wirelength, avg_wirelength_per_net
    - avg_operations_per_timestep, max_operations_per_timestep
    - successful_operations, failed_operations, success_rate
    - total_runtime_ms, magic_terminal_utilization
    - active_routing_volume_proxy  (= total_wirelength, named for clarity)

Derived metrics (relative to steiner_tree baseline):
    - logical_timestep_ratio
    - wirelength_ratio
    - active_volume_ratio
    - parallelism_gain

Outputs (all in --output-dir):
    rq4_raw_results.json
    rq4_summary.csv
    rq4_normalized_summary.csv
    rq4_timesteps_bar.png
    rq4_wirelength_bar.png
    rq4_parallelism_bar.png
    rq4_normalized_comparison.png

Usage examples
--------------
Single circuit:
    cd src
    python scripts/evaluate_rq4_scheduler_comparison.py \\
        --qasm ../benchmark_circuits/qasm/qaoa/example.qasm \\
        --rows 10 --cols 10 \\
        --output-dir ../results/rq4_qaoa

Batch:
    cd src
    python scripts/evaluate_rq4_scheduler_comparison.py \\
        --benchmark-dir ../benchmark_circuits/qasm/qaoa \\
        --max-files 10 \\
        --rows 10 --cols 10 \\
        --output-dir ../results/rq4_qaoa_batch \\
        --normalize
"""

import argparse
import csv
import json
import logging
import math
import os
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from harvest.compilation.circuit_analysis import convert_rx_ry_to_rz, pre_prep_circuit
from harvest.compilation.pauli_block_conversion import convert_to_PCB, create_dag
from harvest.compilation.qasm_loader import find_qasm_files, qasm_to_circuit
from harvest.layout.presets import (
    blocks_of_four_qubit_patches,
    nxm_ring_layout_single_qubits,
    nxm_ring_layout_single_qubits_large_spacing,
)
from harvest.routing.magic_state_factory import MagicStateFactory
from harvest.routing.processor import DAGProcessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("RQ4")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCHEDULERS: List[Tuple[str, str]] = [
    ("steiner_tree",       "Sequential"),
    ("steiner_packing",    "Greedy Packing"),
    ("steiner_pathfinder", "Pathfinder"),
]

LAYOUT_PRESET_MAP = {
    "single_spacing":  nxm_ring_layout_single_qubits,
    "double_spacing":  nxm_ring_layout_single_qubits_large_spacing,
    "blocks_of_four":  blocks_of_four_qubit_patches,
}

# Metric keys written to CSV (raw)
RAW_METRIC_KEYS = [
    "num_time_steps",
    "total_wirelength",
    "avg_wirelength_per_net",
    "avg_operations_per_timestep",
    "max_operations_per_timestep",
    "successful_operations",
    "failed_operations",
    "success_rate",
    "total_runtime_ms",
    "magic_terminal_utilization",
    "active_routing_volume_proxy",
    "total_consumed",
    "total_wait_cycles",
]

DERIVED_METRIC_KEYS = [
    "logical_timestep_ratio",
    "wirelength_ratio",
    "active_volume_ratio",
    "parallelism_gain",
]

# ---------------------------------------------------------------------------
# Circuit loading
# ---------------------------------------------------------------------------

def load_and_prepare_circuit(qasm_path: str):
    """Load a QASM file, apply pre-processing, and return (circuit, dag).

    The PCB conversion and DAG construction are done once so that all
    scheduler runs share the exact same DAG.
    """
    logger.info(f"Loading circuit: {qasm_path}")
    circuit = qasm_to_circuit(qasm_path)
    circuit = pre_prep_circuit(circuit)

    gate_counts = circuit.count_ops()
    if "rx" in gate_counts or "ry" in gate_counts:
        logger.info("  Converting rx/ry gates to rz equivalents ...")
        circuit = convert_rx_ry_to_rz(circuit)

    logger.info(
        f"  {circuit.num_qubits} qubits, depth {circuit.depth()}, "
        f"gates {dict(gate_counts)}"
    )
    logger.info("  Converting to PCB format ...")
    pcb = convert_to_PCB(circuit, verbose=False)
    dag = create_dag(pcb)
    n_ops = len(list(dag.op_nodes()))
    logger.info(f"  DAG: {n_ops} operation nodes")
    return circuit, dag


# ---------------------------------------------------------------------------
# Layout construction
# ---------------------------------------------------------------------------

def build_layout_engine(rows: int, cols: int, layout_preset: str):
    """Return a LayoutEngine for the given grid dimensions and preset."""
    preset_fn = LAYOUT_PRESET_MAP[layout_preset]
    logger.info(f"  Building layout: {layout_preset}  ({rows}x{cols})")
    return preset_fn(rows, cols)


# ---------------------------------------------------------------------------
# Magic source factory
# ---------------------------------------------------------------------------

def make_magic_source_fn(magic_source_mode: str, prep_cycles: int, layout_engine):
    """Return a callable that produces a fresh magic source for each run.

    For "unlimited" → returns None (DAGProcessor treats None as unlimited).
    For "factory"   → returns MagicStateFactory with the layout's magic terminals.
    """
    if magic_source_mode == "unlimited":
        return lambda: None

    # Probe the magic terminal list from a throw-away processor instance.
    # This ensures we use exactly the terminals the layout provides.
    probe = DAGProcessor(layout_engine=layout_engine)
    magic_terminals = list(probe.magic_terminals)
    logger.info(f"  Magic terminals probed: {len(magic_terminals)} terminals")

    def _factory():
        return MagicStateFactory(magic_terminals, prep_cycles)

    return _factory


# ---------------------------------------------------------------------------
# Single scheduler run
# ---------------------------------------------------------------------------

def run_single_scheduler(
    dag,
    layout_engine,
    mode: str,
    label: str,
    magic_source_fn,
) -> Dict:
    """Route *dag* with *mode* on *layout_engine*; return a metrics dict."""
    logger.info(f"  Running scheduler: {label}  (mode={mode})")
    magic_source = magic_source_fn()

    try:
        t_start = time.perf_counter()
        processor = DAGProcessor(layout_engine=layout_engine, magic_source=magic_source)
        results = processor.process_entire_dag(dag, visualize_each_step=False, mode=mode)
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        # --- Scheduling metadata -----------------------------------------
        meta = getattr(processor, "_scheduling_metadata", {})
        num_time_steps   = meta.get("total_elapsed_steps", len(results))
        nodes_completed  = meta.get("num_nodes_completed", len(results))
        nodes_total      = meta.get("num_nodes_total",     len(results))

        # --- Wirelength & per-step ops ------------------------------------
        total_wirelength = 0
        wirelengths: List[int] = []
        ops_per_step: Dict[int, int] = defaultdict(int)

        for idx, r in enumerate(results):
            wl = len(r.get("steiner_edges", set()))
            total_wirelength += wl
            if wl > 0:
                wirelengths.append(wl)
            # steiner_tree does not add time_step → use sequential index
            step = r.get("time_step", idx)
            ops_per_step[step] += 1

        n_ops_total = len(results)
        avg_wl = total_wirelength / max(n_ops_total, 1)
        avg_ops = n_ops_total / max(num_time_steps, 1)
        max_ops = max(ops_per_step.values()) if ops_per_step else 0

        failed_ops  = nodes_total - nodes_completed
        success_rate = nodes_completed / max(nodes_total, 1)

        # --- Magic terminal utilization -----------------------------------
        used_terminals  = len(processor.used_magic_terminals)
        total_terminals = len(processor.magic_terminals)
        magic_util = used_terminals / max(total_terminals, 1)

        # --- Factory stats (if applicable) --------------------------------
        total_consumed   = 0
        total_wait_cycles = 0
        source = processor.magic_source
        if source is not None and not getattr(source, "unlimited", True):
            stats = source.get_stats()
            total_consumed    = stats.get("total_consumed",    0)
            total_wait_cycles = stats.get("total_wait_cycles", 0)

        metrics = {
            "mode":                        mode,
            "label":                       label,
            "num_time_steps":              num_time_steps,
            "total_wirelength":            total_wirelength,
            "avg_wirelength_per_net":      round(avg_wl, 4),
            "avg_operations_per_timestep": round(avg_ops, 4),
            "max_operations_per_timestep": max_ops,
            "successful_operations":       nodes_completed,
            "failed_operations":           failed_ops,
            "success_rate":                round(success_rate, 6),
            "total_runtime_ms":            round(elapsed_ms, 2),
            "magic_terminal_utilization":  round(magic_util, 6),
            # active_routing_volume_proxy == total_wirelength, named for RQ4 clarity
            "active_routing_volume_proxy": total_wirelength,
            "total_consumed":              total_consumed,
            "total_wait_cycles":           total_wait_cycles,
            "success":                     True,
        }

        logger.info(
            f"    ✓  T={num_time_steps}  WL={total_wirelength}"
            f"  avg_ops/step={avg_ops:.2f}  runtime={elapsed_ms:.0f} ms"
        )
        return metrics

    except Exception as exc:
        logger.error(f"    ✗  FAILED: {exc}", exc_info=True)
        return {
            "mode":    mode,
            "label":   label,
            "success": False,
            "error":   str(exc),
        }


# ---------------------------------------------------------------------------
# Derived metrics (computed after all 3 runs)
# ---------------------------------------------------------------------------

def add_derived_metrics(scheduler_results: List[Dict]) -> List[Dict]:
    """Compute derived ratio metrics relative to the steiner_tree baseline.

    Modifies each dict in-place and also returns the list.
    Division-by-zero is guarded; baseline row gets ratio = 1.0 always.
    """
    seq = next((r for r in scheduler_results if r["mode"] == "steiner_tree"), None)
    if seq is None or not seq.get("success"):
        # Cannot compute ratios without a valid baseline
        for r in scheduler_results:
            for k in DERIVED_METRIC_KEYS:
                r[k] = None
        return scheduler_results

    seq_steps    = max(seq["num_time_steps"],              1)
    seq_wl       = max(seq["total_wirelength"],            1)
    seq_avol     = max(seq["active_routing_volume_proxy"], 1)
    seq_avg_ops  = max(seq["avg_operations_per_timestep"], 1e-9)

    for r in scheduler_results:
        if not r.get("success"):
            for k in DERIVED_METRIC_KEYS:
                r[k] = None
            continue
        r["logical_timestep_ratio"] = round(r["num_time_steps"]              / seq_steps,   6)
        r["wirelength_ratio"]       = round(r["total_wirelength"]            / seq_wl,      6)
        r["active_volume_ratio"]    = round(r["active_routing_volume_proxy"] / seq_avol,    6)
        r["parallelism_gain"]       = round(r["avg_operations_per_timestep"] / seq_avg_ops, 6)

    return scheduler_results


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def save_json(data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2, default=str)
    logger.info(f"  Saved JSON -> {path}")


def save_csv(rows: List[Dict], keys: List[str], path: Path,
             extra_keys: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    all_keys = ["circuit_name", "mode", "label"] + keys
    if extra_keys:
        all_keys += extra_keys
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"  Saved CSV  -> {path}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

SCHEDULER_COLORS = {
    "Sequential":     "#4C72B0",
    "Greedy Packing": "#DD8452",
    "Pathfinder":     "#55A868",
}
SCHEDULER_DISPLAY = ["Sequential", "Greedy Packing", "Pathfinder"]


def _bar_chart(
    labels: List[str],
    values: List[float],
    ylabel: str,
    title: str,
    subtitle: str,
    output_path: Path,
    annotate_direction: str = "neutral",   # "lower", "higher", or "neutral"
) -> None:
    """Save a single grouped bar chart PNG."""
    colors = [SCHEDULER_COLORS.get(l, "#888888") for l in labels]
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, values, width=0.5, color=colors, alpha=0.88, edgecolor="white")

    y_max = max((v for v in values if v is not None and v > 0), default=1)
    for bar, val in zip(bars, values):
        if val is None:
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + y_max * 0.01,
            f"{val:,.2f}" if isinstance(val, float) else f"{val:,}",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    if annotate_direction == "lower":
        ax.text(0.98, 0.97, "↓ lower is better", transform=ax.transAxes,
                ha="right", va="top", fontsize=9, color="#555555")
    elif annotate_direction == "higher":
        ax.text(0.98, 0.97, "↑ higher is better", transform=ax.transAxes,
                ha="right", va="top", fontsize=9, color="#555555")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(f"{title}\n{subtitle}", fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved plot -> {output_path}")


def _normalized_comparison_plot(
    scheduler_results: List[Dict],
    circuit_name: str,
    output_path: Path,
) -> None:
    """3-panel normalized bar chart with sequential = 1.0 baseline."""
    metrics_info = [
        ("logical_timestep_ratio", "Timestep Ratio\n(sequential = 1.0)", "lower"),
        ("active_volume_ratio",    "Active Volume Ratio\n(sequential = 1.0)", "lower"),
        ("parallelism_gain",       "Parallelism Gain\n(sequential = 1.0)", "higher"),
    ]

    labels = [r["label"] for r in scheduler_results if r.get("success")]
    n_metrics = len(metrics_info)
    fig, axes = plt.subplots(1, n_metrics, figsize=(4.5 * n_metrics, 5), sharey=False)
    if n_metrics == 1:
        axes = [axes]

    colors = [SCHEDULER_COLORS.get(l, "#888888") for l in labels]
    x = np.arange(len(labels))

    for ax, (metric_key, ylabel, direction) in zip(axes, metrics_info):
        values = [r.get(metric_key) for r in scheduler_results if r.get("success")]
        ax.bar(x, values, width=0.5, color=colors, alpha=0.88, edgecolor="white")

        # Baseline reference line at 1.0
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.2, alpha=0.8,
                   label="sequential baseline")

        y_max = max((v for v in values if v is not None), default=1.5)
        for xi, val in zip(x, values):
            if val is None:
                continue
            ax.text(xi, val + y_max * 0.02, f"{val:.3f}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)
        hint = "↓ lower is better" if direction == "lower" else "↑ higher is better"
        ax.text(0.98, 0.97, hint, transform=ax.transAxes,
                ha="right", va="top", fontsize=8, color="#555555")

    fig.suptitle(
        f"RQ4 — Normalized Scheduler Comparison\n{circuit_name}",
        fontsize=12, fontweight="bold",
    )
    handles = [plt.Line2D([0], [0], color="gray", linestyle="--", linewidth=1.2)]
    fig.legend(handles, ["sequential baseline"], loc="lower center",
               ncol=1, fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, -0.04))
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved plot -> {output_path}")


def generate_plots(
    scheduler_results: List[Dict],
    circuit_name: str,
    output_dir: Path,
    normalize: bool,
) -> None:
    """Generate all four RQ4 plots for a single circuit's results."""
    successful = [r for r in scheduler_results if r.get("success")]
    if not successful:
        logger.warning("  No successful scheduler results — skipping plots.")
        return

    labels = [r["label"] for r in successful]
    subtitle = f"Circuit: {circuit_name}"

    # 1. Timesteps bar
    _bar_chart(
        labels,
        [r["num_time_steps"] for r in successful],
        ylabel="Logical Timesteps",
        title="RQ4 — Logical Timesteps by Scheduler",
        subtitle=subtitle,
        output_path=output_dir / "rq4_timesteps_bar.png",
        annotate_direction="lower",
    )

    # 2. Wirelength / active routing volume bar
    _bar_chart(
        labels,
        [r["active_routing_volume_proxy"] for r in successful],
        ylabel="Active Routing Volume Proxy\n(total wirelength in routing edges)",
        title="RQ4 — Active Routing Volume Proxy by Scheduler",
        subtitle=subtitle,
        output_path=output_dir / "rq4_wirelength_bar.png",
        annotate_direction="lower",
    )

    # 3. Parallelism (avg ops / timestep)
    _bar_chart(
        labels,
        [r["avg_operations_per_timestep"] for r in successful],
        ylabel="Avg. Operations per Timestep",
        title="RQ4 — Parallelism by Scheduler",
        subtitle=subtitle,
        output_path=output_dir / "rq4_parallelism_bar.png",
        annotate_direction="higher",
    )

    # 4. Normalized comparison (only if derived metrics present)
    if all(r.get("logical_timestep_ratio") is not None for r in successful):
        _normalized_comparison_plot(
            successful,
            circuit_name=circuit_name,
            output_path=output_dir / "rq4_normalized_comparison.png",
        )
    elif normalize:
        logger.warning(
            "  --normalize requested but derived metrics are missing (baseline failed?)."
        )


# ---------------------------------------------------------------------------
# Aggregate batch plots
# ---------------------------------------------------------------------------

def _mean_safe(values: List[Optional[float]]) -> Optional[float]:
    valid = [v for v in values if v is not None]
    return sum(valid) / len(valid) if valid else None


def generate_batch_plots(
    per_circuit_results: List[Dict],
    output_dir: Path,
    normalize: bool,
) -> None:
    """Aggregate metrics across circuits and save batch-level plots."""
    by_scheduler: Dict[str, List[Dict]] = defaultdict(list)
    for entry in per_circuit_results:
        if entry.get("success"):
            by_scheduler[entry["label"]].append(entry)

    if not by_scheduler:
        logger.warning("No successful results in batch — skipping aggregate plots.")
        return

    labels_in_order = [lbl for _, lbl in SCHEDULERS if lbl in by_scheduler]
    n_circuits = max(len(v) for v in by_scheduler.values())
    subtitle = f"Averaged over {n_circuits} circuits"

    def _means(metric_key: str) -> List[float]:
        return [
            _mean_safe([r.get(metric_key) for r in by_scheduler.get(lbl, [])]) or 0
            for lbl in labels_in_order
        ]

    _bar_chart(
        labels_in_order, _means("num_time_steps"),
        ylabel="Avg. Logical Timesteps",
        title="RQ4 — Avg. Timesteps by Scheduler (Batch)",
        subtitle=subtitle,
        output_path=output_dir / "rq4_timesteps_bar.png",
        annotate_direction="lower",
    )
    _bar_chart(
        labels_in_order, _means("active_routing_volume_proxy"),
        ylabel="Avg. Active Routing Volume Proxy",
        title="RQ4 — Avg. Active Routing Volume Proxy (Batch)",
        subtitle=subtitle,
        output_path=output_dir / "rq4_wirelength_bar.png",
        annotate_direction="lower",
    )
    _bar_chart(
        labels_in_order, _means("avg_operations_per_timestep"),
        ylabel="Avg. Operations per Timestep",
        title="RQ4 — Avg. Parallelism (Batch)",
        subtitle=subtitle,
        output_path=output_dir / "rq4_parallelism_bar.png",
        annotate_direction="higher",
    )

    if normalize:
        # Build pseudo-result dicts with averaged derived metrics
        avg_results = []
        for lbl in labels_in_order:
            group = by_scheduler.get(lbl, [])
            mode = group[0]["mode"] if group else lbl
            avg = {"label": lbl, "mode": mode, "success": True}
            for k in DERIVED_METRIC_KEYS:
                avg[k] = _mean_safe([r.get(k) for r in group])
            avg_results.append(avg)
        _normalized_comparison_plot(
            avg_results,
            circuit_name=f"Batch average ({n_circuits} circuits)",
            output_path=output_dir / "rq4_normalized_comparison.png",
        )


# ---------------------------------------------------------------------------
# Single-circuit pipeline
# ---------------------------------------------------------------------------

def run_circuit(
    qasm_path: str,
    rows: int,
    cols: int,
    layout_preset: str,
    magic_source_mode: str,
    magic_prep_cycles: int,
) -> Tuple[str, List[Dict]]:
    """Full RQ4 pipeline for one circuit. Returns (circuit_name, results)."""
    circuit_name = Path(qasm_path).stem

    try:
        _, dag = load_and_prepare_circuit(qasm_path)
    except Exception as exc:
        logger.error(f"  Failed to load/prepare circuit: {exc}", exc_info=True)
        return circuit_name, [
            {"mode": m, "label": l, "success": False,
             "error": f"Circuit load failed: {exc}"}
            for m, l in SCHEDULERS
        ]

    layout_engine = build_layout_engine(rows, cols, layout_preset)
    magic_source_fn = make_magic_source_fn(magic_source_mode, magic_prep_cycles, layout_engine)

    scheduler_results: List[Dict] = []
    for mode, label in SCHEDULERS:
        result = run_single_scheduler(dag, layout_engine, mode, label, magic_source_fn)
        result["circuit_name"] = circuit_name
        scheduler_results.append(result)

    add_derived_metrics(scheduler_results)
    return circuit_name, scheduler_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RQ4: Compare scheduling strategies under identical circuit & layout.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Circuit source (mutually exclusive but not enforced here for flexibility)
    p.add_argument("--qasm",          metavar="PATH",
                   help="Path to a single QASM input file.")
    p.add_argument("--benchmark-dir", metavar="PATH",
                   help="Directory of QASM files to process in batch mode.")
    p.add_argument("--max-files",     metavar="INT", type=int, default=None,
                   help="Max number of QASM files to process from --benchmark-dir.")

    # Layout
    p.add_argument("--rows",   metavar="INT", type=int, default=5,
                   help="Layout rows (number of qubit rows).")
    p.add_argument("--cols",   metavar="INT", type=int, default=5,
                   help="Layout columns (number of qubit columns).")
    p.add_argument("--layout", choices=list(LAYOUT_PRESET_MAP.keys()),
                   default="single_spacing",
                   help="Layout preset to use.")

    # Output
    p.add_argument("--output-dir", metavar="PATH",
                   default="results/rq4_scheduler_comparison",
                   help="Directory for all output files.")
    p.add_argument("--normalize", action="store_true",
                   help="Also save the normalized comparison plot.")

    # Magic states
    p.add_argument("--magic-source", choices=["unlimited", "factory"],
                   default="unlimited",
                   help="Magic-state source model (default: unlimited).")
    p.add_argument("--magic-prep-cycles", metavar="INT", type=int, default=15,
                   help="Preparation cycles per terminal (only used with --magic-source factory).")

    # Reproducibility
    p.add_argument("--seed", metavar="INT", type=int, default=None,
                   help="Random seed (reserved for future use with random circuits).")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.qasm and not args.benchmark_dir:
        raise SystemExit("Error: provide at least one of --qasm or --benchmark-dir.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_mode = bool(args.benchmark_dir)

    # ------------------------------------------------------------------
    # Collect QASM files
    # ------------------------------------------------------------------
    qasm_files: List[str] = []
    if args.qasm:
        qasm_files.append(args.qasm)
    if args.benchmark_dir:
        found = sorted(find_qasm_files(args.benchmark_dir))
        if args.max_files is not None:
            found = found[: args.max_files]
        qasm_files.extend(found)
        logger.info(f"Batch mode: {len(found)} QASM files found in {args.benchmark_dir}")

    logger.info(
        f"\nRQ4 Scheduler Comparison"
        f"\n  Layout:       {args.layout}  ({args.rows}x{args.cols})"
        f"\n  Magic source: {args.magic_source}"
        f"\n  Circuits:     {len(qasm_files)}"
        f"\n  Output dir:   {output_dir}\n"
    )

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    all_per_circuit_results: List[Dict] = []

    for qasm_path in qasm_files:
        logger.info(f"\n{'='*65}")
        logger.info(f"Circuit: {qasm_path}")
        logger.info(f"{'='*65}")

        circuit_name, sched_results = run_circuit(
            qasm_path       = qasm_path,
            rows            = args.rows,
            cols            = args.cols,
            layout_preset   = args.layout,
            magic_source_mode   = args.magic_source,
            magic_prep_cycles   = args.magic_prep_cycles,
        )
        all_per_circuit_results.extend(sched_results)

    # ------------------------------------------------------------------
    # Save JSON
    # ------------------------------------------------------------------
    if batch_mode:
        json_payload = {
            "parameters": {
                "layout":           args.layout,
                "rows":             args.rows,
                "cols":             args.cols,
                "magic_source":     args.magic_source,
                "magic_prep_cycles": args.magic_prep_cycles,
                "num_circuits":     len(qasm_files),
                "timestamp":        ts,
            },
            "per_circuit_results": all_per_circuit_results,
        }
        save_json(json_payload, output_dir / "rq4_per_circuit_results.json")
    else:
        # Single circuit: also write rq4_raw_results.json for clarity
        seq_results = [r for r in all_per_circuit_results
                       if r.get("circuit_name") == Path(args.qasm).stem]
        payload = {
            "parameters": {
                "qasm":             args.qasm,
                "layout":           args.layout,
                "rows":             args.rows,
                "cols":             args.cols,
                "magic_source":     args.magic_source,
                "magic_prep_cycles": args.magic_prep_cycles,
                "timestamp":        ts,
            },
            "results": seq_results,
        }
        save_json(payload, output_dir / "rq4_raw_results.json")

    # ------------------------------------------------------------------
    # Save CSV: summary (all raw + derived metrics)
    # ------------------------------------------------------------------
    csv_keys  = RAW_METRIC_KEYS + DERIVED_METRIC_KEYS
    save_csv(all_per_circuit_results, csv_keys,
             output_dir / "rq4_summary.csv")

    # Normalized CSV: only ratio columns + identifiers
    save_csv(
        all_per_circuit_results,
        DERIVED_METRIC_KEYS,
        output_dir / "rq4_normalized_summary.csv",
    )

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    if batch_mode:
        generate_batch_plots(all_per_circuit_results, output_dir, normalize=True)
        # Also save per-first-circuit plots if only one circuit was processed
        if len(qasm_files) == 1:
            single_results = [r for r in all_per_circuit_results
                              if r.get("circuit_name") == Path(qasm_files[0]).stem]
            generate_plots(single_results, Path(qasm_files[0]).stem,
                           output_dir, normalize=args.normalize)
    else:
        single_results = [r for r in all_per_circuit_results
                          if r.get("circuit_name") == Path(args.qasm).stem]
        generate_plots(single_results, Path(args.qasm).stem,
                       output_dir, normalize=args.normalize)

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    logger.info(f"\n{'='*75}")
    logger.info("SUMMARY")
    logger.info(f"{'='*75}")
    for r in all_per_circuit_results:
        if r.get("success"):
            cname = r.get("circuit_name", "?")
            lbl   = r.get("label", r.get("mode", "?"))
            logger.info(
                f"  {cname:30s} | {lbl:20s} |"
                f" T={r['num_time_steps']:5d}"
                f"  WL={r['total_wirelength']:7d}"
                f"  avg_ops/step={r['avg_operations_per_timestep']:.2f}"
                f"  parallelism_gain={r.get('parallelism_gain', 'N/A')}"
            )
        else:
            logger.info(
                f"  {r.get('circuit_name','?'):30s} | {r.get('label','?'):20s} |"
                f" FAILED: {r.get('error','')}"
            )
    logger.info(f"{'='*75}")
    logger.info(f"\nAll outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
