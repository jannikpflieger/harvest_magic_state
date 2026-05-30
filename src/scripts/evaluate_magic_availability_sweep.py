#!/usr/bin/env python3
"""
Magic-state availability sweep on a single circuit.

Fixes a circuit and a layout grid size, then sweeps the number of available
magic-state patches from 1 up to the maximum on a single edge (top), plus a
final "full ring" data point where magic patches cover all four sides.

For every availability level all three schedulers are run.

Outputs (in --output-dir)
--------------------------
    magic_availability_sweep.csv          – one row per (num_magic, scheduler)
    plots/
        timesteps_vs_magic.png            – timesteps per scheduler vs num_magic
        speedup_vs_magic.png              – speedup vs num_magic
        combined_vs_magic.png             – 2-panel combination

Usage
-----
    cd src
    python scripts/evaluate_magic_availability_sweep.py \\
        --qasm ../benchmark_circuits/qasm/qaoa/qaoa_barabasi_albert_N49_3reps.qasm \\
        --output-dir ../results/magic_availability_qaoa_n49
"""

import argparse
import csv
import logging
import math
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from harvest.compilation.circuit_analysis import convert_rx_ry_to_rz, pre_prep_circuit
from harvest.compilation.pauli_block_conversion import convert_to_PCB, create_dag
from harvest.compilation.qasm_loader import qasm_to_circuit
from harvest.layout.presets import (
    nxm_fixed_magic_count_layout_single_qubits,
    nxm_ring_layout_single_qubits,
)
from scripts.evaluate_rq4_scheduler_comparison import (
    SCHEDULER_COLORS,
    SCHEDULERS,
    add_derived_metrics,
    make_magic_source_fn,
    run_single_scheduler,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("MagicAvailSweep")

SCHED_ORDER  = [lbl for _, lbl in SCHEDULERS]
SCHED_COLORS = SCHEDULER_COLORS


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

def _auto_side(num_qubits: int) -> int:
    """Minimal square side for *num_qubits* (single-spacing policy)."""
    return max(1, math.ceil(math.sqrt(max(1, num_qubits))))


def _max_top_magic(side: int) -> int:
    """Number of interior positions on the top edge for a side×side single-spacing grid."""
    W = 2 * side + 3
    return W - 2  # positions 1 .. W-2


def build_top_only_layout(side: int, num_magic: int):
    """Single-spacing grid with exactly *num_magic* patches on the top edge."""
    return nxm_fixed_magic_count_layout_single_qubits(side, side, num_magic=num_magic)


def build_full_ring_layout(side: int):
    """Single-spacing grid with magic patches on all four sides (full ring)."""
    return nxm_ring_layout_single_qubits(side, side)


def _count_magic_terminals(layout_engine) -> int:
    from harvest.routing.processor import DAGProcessor
    probe = DAGProcessor(layout_engine=layout_engine)
    return len(list(probe.magic_terminals))


# ---------------------------------------------------------------------------
# Circuit loading
# ---------------------------------------------------------------------------

def load_circuit_and_dag(qasm_path: str):
    logger.info(f"Loading circuit: {qasm_path}")
    circuit = qasm_to_circuit(qasm_path)
    circuit = pre_prep_circuit(circuit)
    gate_counts = circuit.count_ops()
    if "rx" in gate_counts or "ry" in gate_counts:
        circuit = convert_rx_ry_to_rz(circuit)
    logger.info(
        f"  {circuit.num_qubits} qubits, depth {circuit.depth()}, gates {dict(gate_counts)}"
    )
    from harvest.compilation.pauli_block_conversion import convert_to_PCB, create_dag
    pcb = convert_to_PCB(circuit, verbose=False)
    dag = create_dag(pcb)
    logger.info(f"  DAG: {len(list(dag.op_nodes()))} ops")
    return circuit, dag


# ---------------------------------------------------------------------------
# One sweep step
# ---------------------------------------------------------------------------

def run_one_point(
    dag,
    layout_engine,
    num_magic_label: int,
    magic_source_mode: str,
    magic_prep_cycles: int,
) -> List[Dict]:
    """Run all schedulers with *layout_engine* and return metrics list."""
    magic_source_fn = make_magic_source_fn(magic_source_mode, magic_prep_cycles, layout_engine)
    results: List[Dict] = []
    for mode, label in SCHEDULERS:
        r = run_single_scheduler(dag, layout_engine, mode, label, magic_source_fn)
        r["num_magic"] = num_magic_label
        results.append(r)
    add_derived_metrics(results)
    return results


# ---------------------------------------------------------------------------
# CSV saving
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    "num_magic", "label",
    "num_time_steps", "total_wirelength",
    "avg_operations_per_timestep", "success_rate",
    "total_runtime_ms", "magic_terminal_utilization",
    "logical_timestep_ratio", "parallelism_gain",
    "success",
]


def save_csv(all_results: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_results)
    logger.info(f"Saved CSV → {path}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved plot → {path}")


def _build_series(all_results: List[Dict], metric: str):
    """Return (x_labels, {sched_label: [values]}) in sweep order."""
    # Gather ordered x values (preserve insertion order via dict)
    x_order: dict = {}
    for r in all_results:
        x_order[r["num_magic"]] = None
    x_labels = list(x_order.keys())

    series: Dict[str, List[Optional[float]]] = {lbl: [] for lbl in SCHED_ORDER}
    for x in x_labels:
        for lbl in SCHED_ORDER:
            val = next(
                (r.get(metric) for r in all_results
                 if r["num_magic"] == x and r.get("label") == lbl and r.get("success", True)),
                None,
            )
            series[lbl].append(float(val) if val is not None else float("nan"))
    return x_labels, series


def _x_tick_labels(x_labels) -> List[str]:
    return [str(x) if x != "full_ring" else "Full ring" for x in x_labels]


def plot_timesteps(all_results: List[Dict], output_path: Path, circuit_name: str) -> None:
    x_labels, series = _build_series(all_results, "num_time_steps")
    ticks = _x_tick_labels(x_labels)

    fig, ax = plt.subplots(figsize=(max(8, len(x_labels) * 0.6), 5))
    for lbl in SCHED_ORDER:
        vals = series[lbl]
        ax.plot(range(len(x_labels)), vals, marker="o", label=lbl, color=SCHED_COLORS[lbl])

    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(ticks, rotation=45, ha="right")
    ax.set_xlabel("Number of available magic states")
    ax.set_ylabel("Number of time steps")
    ax.set_title(f"Time steps vs magic-state availability\n{circuit_name}")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    _save(fig, output_path)


def plot_speedup(all_results: List[Dict], output_path: Path, circuit_name: str) -> None:
    x_labels, series = _build_series(all_results, "parallelism_gain")
    ticks = _x_tick_labels(x_labels)

    fig, ax = plt.subplots(figsize=(max(8, len(x_labels) * 0.6), 5))
    for lbl in SCHED_ORDER:
        if lbl == "Sequential":
            continue   # ratio is always 1 for baseline
        vals = series[lbl]
        ax.plot(range(len(x_labels)), vals, marker="o", label=lbl, color=SCHED_COLORS[lbl])

    ax.axhline(1.0, color=SCHED_COLORS["Sequential"], linestyle="--",
               label="Sequential (baseline)", linewidth=1)
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(ticks, rotation=45, ha="right")
    ax.set_xlabel("Number of available magic states")
    ax.set_ylabel("Speedup vs Sequential")
    ax.set_title(f"Speedup vs magic-state availability\n{circuit_name}")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    _save(fig, output_path)


def plot_combined(all_results: List[Dict], output_path: Path, circuit_name: str) -> None:
    x_labels, ts_series  = _build_series(all_results, "num_time_steps")
    _,        sp_series  = _build_series(all_results, "parallelism_gain")
    ticks = _x_tick_labels(x_labels)
    xs = range(len(x_labels))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(14, len(x_labels) * 1.0), 5))
    fig.suptitle(f"Magic-state availability sweep — {circuit_name}", fontsize=11)

    for lbl in SCHED_ORDER:
        ax1.plot(xs, ts_series[lbl], marker="o", label=lbl, color=SCHED_COLORS[lbl])
    ax1.set_xticks(xs); ax1.set_xticklabels(ticks, rotation=45, ha="right")
    ax1.set_xlabel("Available magic states"); ax1.set_ylabel("Time steps")
    ax1.set_title("Time steps"); ax1.legend(); ax1.grid(axis="y", alpha=0.3)

    for lbl in SCHED_ORDER:
        if lbl == "Sequential":
            continue
        ax2.plot(xs, sp_series[lbl], marker="o", label=lbl, color=SCHED_COLORS[lbl])
    ax2.axhline(1.0, color=SCHED_COLORS["Sequential"], linestyle="--",
                label="Sequential", linewidth=1)
    ax2.set_xticks(xs); ax2.set_xticklabels(ticks, rotation=45, ha="right")
    ax2.set_xlabel("Available magic states"); ax2.set_ylabel("Speedup")
    ax2.set_title("Speedup vs Sequential"); ax2.legend(); ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    _save(fig, output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Magic-state availability sweep (single circuit).")
    p.add_argument("--qasm", required=True,
                   help="Path to a single QASM circuit file.")
    p.add_argument("--output-dir", default=None,
                   help="Output directory (default: auto-timestamped).")
    p.add_argument("--magic-source", default="factory",
                   choices=["unlimited", "factory"],
                   help="Magic source mode (default: factory).")
    p.add_argument("--magic-prep-cycles", type=int, default=2,
                   help="Preparation cycles for MagicStateFactory (default: 2).")
    p.add_argument("--no-full-ring", action="store_true",
                   help="Skip the full-ring data point.")
    return p.parse_args()


def main():
    args = parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else \
        Path(f"../results/magic_avail_sweep_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output: {out_dir}")

    # ---- Load circuit once ------------------------------------------------
    circuit, dag = load_circuit_and_dag(args.qasm)
    circuit_name = Path(args.qasm).stem
    side = _auto_side(circuit.num_qubits)
    max_top = _max_top_magic(side)
    logger.info(
        f"Circuit: {circuit_name}  ({circuit.num_qubits} qubits)  "
        f"grid={side}×{side}  top positions={max_top}"
    )

    all_results: List[Dict] = []

    # ---- Sweep top-only magic counts 1 .. max_top -------------------------
    for nm in range(1, max_top + 1):
        logger.info(f"\n── num_magic = {nm} (top only) ──")
        layout = build_top_only_layout(side, nm)
        n_terminals = _count_magic_terminals(layout)
        logger.info(f"  Magic terminals in layout: {n_terminals}")
        results = run_one_point(dag, layout, nm, args.magic_source, args.magic_prep_cycles)
        all_results.extend(results)
        save_csv(all_results, out_dir / "magic_availability_sweep.csv")

    # ---- Full-ring point --------------------------------------------------
    if not args.no_full_ring:
        logger.info(f"\n── full ring ──")
        layout = build_full_ring_layout(side)
        n_terminals = _count_magic_terminals(layout)
        logger.info(f"  Magic terminals in layout: {n_terminals}")
        results = run_one_point(dag, layout, "full_ring", args.magic_source, args.magic_prep_cycles)
        all_results.extend(results)
        save_csv(all_results, out_dir / "magic_availability_sweep.csv")

    # ---- Plots ------------------------------------------------------------
    plots_dir = out_dir / "plots"
    plot_timesteps(all_results, plots_dir / "timesteps_vs_magic.png", circuit_name)
    plot_speedup  (all_results, plots_dir / "speedup_vs_magic.png",   circuit_name)
    plot_combined (all_results, plots_dir / "combined_vs_magic.png",  circuit_name)

    logger.info(f"\nDone. Results in {out_dir}")


if __name__ == "__main__":
    main()
