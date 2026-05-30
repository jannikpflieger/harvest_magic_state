#!/usr/bin/env python3
"""
Benchmark-wide pruner/layout sweep.

Runs all QASM benchmark circuits with these filters:
- exclude selected families (default: qv)
- skip circuits with more than max-qubits (default: 100)

For each remaining circuit:
- preprocess + convert to PCB DAG
- run greedy scheduler (steiner_packing)
- evaluate all 3 layout styles
    1) Single Spacing
    2) Double Spacing
    3) Blocks of 4
- apply lattice pruner and collect before/after patch metrics

Layout sizing policy (minimal square):
- single/double spacing: rows = cols = ceil(sqrt(num_qubits))
- blocks-of-4: block_rows = block_cols = ceil(sqrt(num_qubits / 4))

Outputs in a dedicated results directory:
    results/pruner_layout_sweep_<timestamp>/
        summary.json
        per_layout_rows.csv
        per_circuit.json
        plots/
            patch_reduction_pct_by_layout.png
            magic_patch_reduction_pct_by_layout.png
            patch_reduction_pct_vs_qubits.png
            routing_patch_reduction_vs_qubits.png
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import re
import signal
import statistics
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
_src = str(PROJECT_ROOT / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from harvest.compilation.circuit_analysis import convert_rx_ry_to_rz, pre_prep_circuit
from harvest.compilation.pauli_block_conversion import convert_to_PCB, create_dag
from harvest.compilation.qasm_loader import find_qasm_files, qasm_to_circuit
from harvest.layout.presets import (
    blocks_of_four_qubit_patches,
    nxm_ring_layout_single_qubits,
    nxm_ring_layout_single_qubits_large_spacing,
)
from harvest.routing.processor import DAGProcessor

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s - %(message)s",
)
logger = logging.getLogger("PrunerLayoutSweep")

for _noisy in (
    "HarvestMagicState",
    "HarvestMagicState.DAGProcessor",
    "HarvestMagicState.Detailed",
):
    logging.getLogger(_noisy).setLevel(logging.WARNING)


CSV_FIELDS = [
    "circuit_name",
    "family",
    "num_qubits",
    "layout_name",
    "scheduler",
    "scheduler_mode",
    "layout_side_single_or_double",
    "layout_side_blocks",
    "layout_total_patches",
    "layout_data_patches",
    "layout_magic_patches",
    "patches_before_pruning",
    "patches_after_pruning",
    "patches_removed",
    "patches_removed_pct",
    "magic_patches_before_pruning",
    "magic_patches_after_pruning",
    "magic_patches_removed",
    "magic_patches_removed_pct",
    "data_patches_before_pruning",
    "data_patches_after_pruning",
    "data_patches_removed",
    "graph_nodes_before_pruning",
    "graph_nodes_after_pruning",
    "graph_nodes_removed",
    "routing_cells_before_pruning",
    "routing_cells_after_pruning",
    "routing_cells_removed",
    "routing_cells_removed_pct",
    "port_nodes_removed",
    "port_nodes_removed_total",
    "magic_port_nodes_removed",
    "data_port_nodes_removed",
    "runtime_s",
    "num_timesteps",
    "num_nodes_processed",
    "num_nodes_total",
    "completed",
]


# ---------------------------------------------------------------------------
# Timeout helper (Linux/macOS)
# ---------------------------------------------------------------------------
class _SchedulerTimeout(BaseException):
    pass


def _alarm_handler(signum, frame):  # noqa: ARG001
    raise _SchedulerTimeout()


@contextmanager
def _alarm(seconds: int):
    if seconds > 0:
        old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(seconds)
    try:
        yield
    finally:
        if seconds > 0:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class LayoutPlan:
    single_side: int
    blocks_side: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def auto_layout_plan(num_qubits: int) -> LayoutPlan:
    """Smallest square layouts that can host the circuit qubits.

    Matches the historical square-size policy used elsewhere:
    - side = ceil(sqrt(n)) for single/double spacing
    Extended for blocks-of-4:
    - block_side = ceil(sqrt(n / 4))
    """
    n = max(1, int(num_qubits))
    single_side = max(1, math.ceil(math.sqrt(n)))
    blocks_side = max(1, math.ceil(math.sqrt(n / 4.0)))
    return LayoutPlan(single_side=single_side, blocks_side=blocks_side)


def _family_from_path(qasm_path: str, qasm_root: str) -> str:
    try:
        rel = Path(qasm_path).resolve().relative_to(Path(qasm_root).resolve())
        return rel.parts[0] if rel.parts else "unknown"
    except Exception:
        return Path(qasm_path).parent.name


def load_dag(qasm_path: str):
    """Load QASM and convert to PCB DAG using the standard preprocessing flow."""
    circuit = qasm_to_circuit(qasm_path)
    circuit = pre_prep_circuit(circuit)

    gate_counts = circuit.count_ops()
    if "rx" in gate_counts or "ry" in gate_counts:
        circuit = convert_rx_ry_to_rz(circuit)

    pcb = convert_to_PCB(circuit, verbose=False)
    dag = create_dag(pcb)
    return dag, circuit.num_qubits


def run_layout_once(
    dag,
    num_qubits: int,
    layout_name: str,
    scheduler_timeout_s: int,
):
    """Run one (circuit, layout) tuple with greedy scheduler and pruning."""
    plan = auto_layout_plan(num_qubits)

    if layout_name == "Single Spacing":
        rows = cols = plan.single_side
        engine = nxm_ring_layout_single_qubits(rows, cols)
    elif layout_name == "Double Spacing":
        rows = cols = plan.single_side
        engine = nxm_ring_layout_single_qubits_large_spacing(rows, cols)
    elif layout_name == "Blocks of 4":
        rows = cols = plan.blocks_side
        engine = blocks_of_four_qubit_patches(rows, cols)
    else:
        raise ValueError(f"Unknown layout: {layout_name}")

    processor = DAGProcessor(layout_engine=engine)

    patches_before = len(processor.ports_by_patch)
    graph_nodes_before = len(processor.graph)
    routing_cells_before = sum(1 for n in processor.graph if isinstance(n, tuple))

    t0 = time.perf_counter()
    with _alarm(scheduler_timeout_s):
        results = processor.process_entire_dag(
            dag,
            visualize_each_step=False,
            mode="steiner_packing",
        )
    runtime_s = time.perf_counter() - t0

    prune_stats = processor.prune_after_scheduling(results)

    patches_after = len(processor.ports_by_patch)
    graph_nodes_after = len(processor.graph)
    routing_cells_after = sum(1 for n in processor.graph if isinstance(n, tuple))

    meta = getattr(processor, "_scheduling_metadata", {})

    patches_removed = patches_before - patches_after
    patches_removed_pct = (patches_removed / patches_before * 100.0) if patches_before else 0.0

    magic_before = int(prune_stats.get("magic_patches_before", 0))
    magic_after = int(prune_stats.get("magic_patches_after", 0))
    magic_removed = int(prune_stats.get("magic_patches_removed", 0))
    magic_removed_pct = (magic_removed / magic_before * 100.0) if magic_before else 0.0
    routing_cells_removed = int(prune_stats.get("routing_cells_removed", routing_cells_before - routing_cells_after))
    routing_cells_removed_pct = (routing_cells_removed / routing_cells_before * 100.0) if routing_cells_before else 0.0

    return {
        "layout_name": layout_name,
        "scheduler": "Greedy",
        "scheduler_mode": "steiner_packing",
        "layout_side_single_or_double": plan.single_side,
        "layout_side_blocks": plan.blocks_side,
        "layout_total_patches": len(engine.patches),
        "layout_data_patches": sum(1 for p in engine.patches.values() if p.kind != "magic"),
        "layout_magic_patches": sum(1 for p in engine.patches.values() if p.kind == "magic"),
        "patches_before_pruning": patches_before,
        "patches_after_pruning": patches_after,
        "patches_removed": patches_removed,
        "patches_removed_pct": patches_removed_pct,
        "magic_patches_before_pruning": magic_before,
        "magic_patches_after_pruning": magic_after,
        "magic_patches_removed": magic_removed,
        "magic_patches_removed_pct": magic_removed_pct,
        "data_patches_before_pruning": int(prune_stats.get("data_patches_before", 0)),
        "data_patches_after_pruning": int(prune_stats.get("data_patches_after", 0)),
        "data_patches_removed": int(prune_stats.get("data_patches_removed", 0)),
        "graph_nodes_before_pruning": graph_nodes_before,
        "graph_nodes_after_pruning": graph_nodes_after,
        "graph_nodes_removed": graph_nodes_before - graph_nodes_after,
        "routing_cells_before_pruning": routing_cells_before,
        "routing_cells_after_pruning": routing_cells_after,
        "routing_cells_removed": routing_cells_removed,
        "routing_cells_removed_pct": routing_cells_removed_pct,
        "port_nodes_removed": int(prune_stats.get("port_nodes_removed", 0)),
        "port_nodes_removed_total": int(prune_stats.get("port_nodes_removed_total", 0)),
        "magic_port_nodes_removed": int(prune_stats.get("magic_port_nodes_removed", 0)),
        "data_port_nodes_removed": int(prune_stats.get("data_port_nodes_removed", 0)),
        "runtime_s": round(runtime_s, 3),
        "num_timesteps": int(meta.get("total_elapsed_steps", len(results))),
        "num_nodes_processed": int(meta.get("num_nodes_completed", len(results))),
        "num_nodes_total": int(meta.get("num_nodes_total", len(results))),
        "completed": bool(meta.get("completed", True)),
    }


def plot_avg_reduction_by_layout(rows: List[Dict], out_path: Path, key: str, ylabel: str, title: str):
    layouts = ["Single Spacing", "Double Spacing", "Blocks of 4"]
    vals = []
    errs = []
    for layout in layouts:
        ls = [r[key] for r in rows if r.get("layout_name") == layout]
        if ls:
            vals.append(float(statistics.mean(ls)))
            errs.append(float(statistics.pstdev(ls)) if len(ls) > 1 else 0.0)
        else:
            vals.append(0.0)
            errs.append(0.0)

    x = np.arange(len(layouts))
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(x, vals, yerr=errs, capsize=5, color=["#4C72B0", "#DD8452", "#55A868"], alpha=0.9)

    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{v:.1f}%", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(layouts)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_reduction_vs_qubits(rows: List[Dict], out_path: Path):
    layout_colors = {
        "Single Spacing": "#4C72B0",
        "Double Spacing": "#DD8452",
        "Blocks of 4": "#55A868",
    }

    fig, ax = plt.subplots(figsize=(9.5, 6))
    for layout, color in layout_colors.items():
        xs = [r["num_qubits"] for r in rows if r.get("layout_name") == layout]
        ys = [r["patches_removed_pct"] for r in rows if r.get("layout_name") == layout]
        if xs:
            ax.scatter(xs, ys, s=22, alpha=0.7, label=layout, color=color)

    ax.set_xlabel("#qubits")
    ax.set_ylabel("Patch reduction (%)")
    ax.set_title("Patch Reduction vs Circuit Size (Greedy + Pruner)")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_routing_reduction_vs_qubits(rows: List[Dict], out_path: Path):
    """Routing-only reduction percentage versus circuit qubits.

    Uses exact routing percentage when available:
        routing_cells_removed_pct = routing_cells_removed / routing_cells_before_pruning
    For legacy rows without routing_cells_before_pruning, falls back to:
        routing_cells_removed / graph_nodes_before_pruning
    """
    layout_colors = {
        "Single Spacing": "#4C72B0",
        "Double Spacing": "#DD8452",
        "Blocks of 4": "#55A868",
    }

    fig, ax = plt.subplots(figsize=(9.5, 6))
    for layout, color in layout_colors.items():
        xs = [r["num_qubits"] for r in rows if r.get("layout_name") == layout]
        ys = []
        for r in rows:
            if r.get("layout_name") != layout:
                continue
            exact_pct = r.get("routing_cells_removed_pct")
            if exact_pct is not None and exact_pct != "":
                ys.append(float(exact_pct))
                continue

            # Legacy fallback (older result folders without routing_cells_before_pruning)
            graph_before = float(r.get("graph_nodes_before_pruning", 0) or 0)
            removed = float(r.get("routing_cells_removed", 0) or 0)
            ys.append((removed / graph_before * 100.0) if graph_before > 0 else 0.0)
        if xs:
            ax.scatter(xs, ys, s=22, alpha=0.7, label=layout, color=color)

    ax.set_xlabel("Circuit qubits")
    ax.set_ylabel("Routing patch reduction (%)")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_avg_routing_patches_removed_pct_by_layout(rows: List[Dict], out_path: Path):
    """Plot mean routing patch reduction percentage per circuit by layout."""
    layouts = ["Single Spacing", "Double Spacing", "Blocks of 4"]
    means = []
    stds = []
    for layout in layouts:
        vals = [float(r.get("patches_removed_pct", 0.0)) for r in rows if r.get("layout_name") == layout]
        if vals:
            means.append(float(statistics.mean(vals)))
            stds.append(float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0)
        else:
            means.append(0.0)
            stds.append(0.0)

    x = np.arange(len(layouts))
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=["#4C72B0", "#DD8452", "#55A868"], alpha=0.9)

    for bar, val in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(layouts)
    ax.set_ylabel("Mean routing patch reduction (%)")
    ax.set_title("Average Routing Patch Reduction by Layout (per-circuit)")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _safe_name(s: str) -> str:
    """Filesystem-safe stem used for per-circuit JSON filenames."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


def _build_summary(
    *,
    ts: str,
    qasm_dir: str,
    exclude: set,
    max_qubits: Optional[int],
    layouts: List[str],
    qasm_files: List[str],
    max_circuits: int,
    circuit_docs: List[Dict],
) -> Dict:
    return {
        "timestamp": ts,
        "qasm_dir": qasm_dir,
        "exclude_families": sorted(exclude),
        "max_qubits": max_qubits,
        "scheduler": "Greedy (steiner_packing)",
        "layouts": layouts,
        "total_circuits_after_family_filter": len(qasm_files),
        "max_circuits": max_circuits,
        "successful_circuits": sum(1 for d in circuit_docs if d.get("status") == "success"),
        "timed_out_circuits": sum(1 for d in circuit_docs if d.get("status") == "timed_out"),
        "skipped_too_large": sum(1 for d in circuit_docs if d.get("status") == "skipped_too_large"),
        "load_failed": sum(1 for d in circuit_docs if d.get("status") == "load_failed"),
    }


def _write_checkpoint(
    *,
    run_dir: Path,
    plots_dir: Path,
    ts: str,
    qasm_dir: str,
    exclude: set,
    max_qubits: Optional[int],
    layouts: List[str],
    qasm_files: List[str],
    max_circuits: int,
    circuit_docs: List[Dict],
    flat_rows: List[Dict],
) -> None:
    """Persist current progress: summary JSON, aggregate JSON, CSV, and plots."""
    summary = _build_summary(
        ts=ts,
        qasm_dir=qasm_dir,
        exclude=exclude,
        max_qubits=max_qubits,
        layouts=layouts,
        qasm_files=qasm_files,
        max_circuits=max_circuits,
        circuit_docs=circuit_docs,
    )
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (run_dir / "per_circuit.json").write_text(json.dumps(circuit_docs, indent=2, default=str))

    # CSV is always present (header-only when no completed layout rows yet)
    with open(run_dir / "per_layout_rows.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in flat_rows:
            w.writerow({k: r.get(k, "") for k in CSV_FIELDS})

    if flat_rows:
        plot_avg_reduction_by_layout(
            flat_rows,
            plots_dir / "patch_reduction_pct_by_layout.png",
            key="patches_removed_pct",
            ylabel="Patch reduction (%)",
            title="Average Total Patch Reduction by Layout",
        )
        plot_avg_reduction_by_layout(
            flat_rows,
            plots_dir / "magic_patch_reduction_pct_by_layout.png",
            key="magic_patches_removed_pct",
            ylabel="Magic patch reduction (%)",
            title="Average Magic Patch Reduction by Layout",
        )
        plot_reduction_vs_qubits(
            flat_rows,
            plots_dir / "patch_reduction_pct_vs_qubits.png",
        )
        plot_routing_reduction_vs_qubits(
            flat_rows,
            plots_dir / "routing_patch_reduction_vs_qubits.png",
        )
        plot_avg_routing_patches_removed_pct_by_layout(
            flat_rows,
            plots_dir / "routing_patches_removed_avg_pct_by_layout.png",
        )


def main():
    p = argparse.ArgumentParser(
        description="Sweep all benchmark circuits for pruner/layout impact with greedy scheduling.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--qasm-dir",
        type=str,
        default=str(PROJECT_ROOT / "benchmark_circuits" / "qasm"),
        help="Root of benchmark QASM tree.",
    )
    p.add_argument(
        "--max-qubits",
        type=int,
        default=100,
        help="Skip circuits with more than this many qubits. Use 0 for no limit.",
    )
    p.add_argument(
        "--exclude-families",
        nargs="*",
        default=["qv", "chemical"],
        metavar="F",
        help="Top-level circuit families to exclude.",
    )
    p.add_argument(
        "--scheduler-timeout",
        type=int,
        default=180,
        help="Timeout per (circuit, layout) run in seconds.",
    )
    p.add_argument(
        "--circuit-timeout",
        type=int,
        default=120,
        help="Total wall-clock timeout budget per circuit across all 3 layouts (seconds).",
    )
    p.add_argument(
        "--max-circuits",
        type=int,
        default=0,
        help="Process at most this many circuits after filtering (0 = all).",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=str(PROJECT_ROOT),
        help="Output root; script creates results/pruner_layout_sweep_<timestamp> under it.",
    )
    args = p.parse_args()

    max_qubits = args.max_qubits if args.max_qubits > 0 else None
    exclude = set(args.exclude_families or [])

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.out_dir) / "results" / f"pruner_layout_sweep_{ts}"
    plots_dir = run_dir / "plots"
    circuits_dir = run_dir / "circuits"
    run_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    circuits_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Output directory: %s", run_dir)

    qasm_files = find_qasm_files(args.qasm_dir)
    qasm_files = [
        q for q in qasm_files
        if _family_from_path(q, args.qasm_dir) not in exclude
    ]
    qasm_files = sorted(qasm_files)
    if args.max_circuits and args.max_circuits > 0:
        qasm_files = qasm_files[:args.max_circuits]

    logger.info("Discovered %d circuits after family exclusions", len(qasm_files))

    circuit_docs: List[Dict] = []
    flat_rows: List[Dict] = []

    layouts = ["Single Spacing", "Double Spacing", "Blocks of 4"]

    for i, qasm_path in enumerate(qasm_files, 1):
        circuit_name = Path(qasm_path).stem
        family = _family_from_path(qasm_path, args.qasm_dir)

        logger.info("[%d/%d] %s", i, len(qasm_files), circuit_name)

        # Load and filter
        try:
            dag, num_qubits = load_dag(qasm_path)
        except Exception as exc:
            logger.warning("  load_failed: %s", exc)
            circuit_doc = {
                "circuit_name": circuit_name,
                "family": family,
                "qasm_path": qasm_path,
                "status": "load_failed",
                "error": str(exc),
            }
            circuit_docs.append(circuit_doc)
            per_circuit_path = circuits_dir / f"{i:04d}_{_safe_name(family)}_{_safe_name(circuit_name)}.json"
            per_circuit_path.write_text(json.dumps(circuit_doc, indent=2, default=str))
            _write_checkpoint(
                run_dir=run_dir,
                plots_dir=plots_dir,
                ts=ts,
                qasm_dir=args.qasm_dir,
                exclude=exclude,
                max_qubits=max_qubits,
                layouts=layouts,
                qasm_files=qasm_files,
                max_circuits=args.max_circuits,
                circuit_docs=circuit_docs,
                flat_rows=flat_rows,
            )
            continue

        if max_qubits is not None and num_qubits > max_qubits:
            logger.info("  skipped_too_large: %dq > %d", num_qubits, max_qubits)
            circuit_doc = {
                "circuit_name": circuit_name,
                "family": family,
                "qasm_path": qasm_path,
                "num_qubits": num_qubits,
                "status": "skipped_too_large",
            }
            circuit_docs.append(circuit_doc)
            per_circuit_path = circuits_dir / f"{i:04d}_{_safe_name(family)}_{_safe_name(circuit_name)}.json"
            per_circuit_path.write_text(json.dumps(circuit_doc, indent=2, default=str))
            _write_checkpoint(
                run_dir=run_dir,
                plots_dir=plots_dir,
                ts=ts,
                qasm_dir=args.qasm_dir,
                exclude=exclude,
                max_qubits=max_qubits,
                layouts=layouts,
                qasm_files=qasm_files,
                max_circuits=args.max_circuits,
                circuit_docs=circuit_docs,
                flat_rows=flat_rows,
            )
            continue

        circuit_result = {
            "circuit_name": circuit_name,
            "family": family,
            "qasm_path": qasm_path,
            "num_qubits": num_qubits,
            "status": "success",
            "circuit_timeout_s": args.circuit_timeout,
            "layout_results": [],
        }

        circuit_start = time.perf_counter()
        timed_out_circuit = False

        for layout_name in layouts:
            elapsed = time.perf_counter() - circuit_start
            remaining_budget = args.circuit_timeout - elapsed
            if remaining_budget <= 0:
                timed_out_circuit = True
                logger.warning(
                    "  circuit timeout reached after %.1fs, skipping remaining layouts",
                    elapsed,
                )
                circuit_result["layout_results"].append({
                    "layout_name": layout_name,
                    "success": False,
                    "timed_out": True,
                    "error": f"circuit timeout after {args.circuit_timeout}s",
                })
                break

            layout_timeout = max(1, int(min(args.scheduler_timeout, remaining_budget)))
            try:
                rec = run_layout_once(
                    dag=dag,
                    num_qubits=num_qubits,
                    layout_name=layout_name,
                    scheduler_timeout_s=layout_timeout,
                )
                rec["circuit_name"] = circuit_name
                rec["family"] = family
                rec["num_qubits"] = num_qubits
                flat_rows.append(rec)
                circuit_result["layout_results"].append(rec)

                logger.info(
                    "  %-14s patches %d -> %d (%.1f%% removed), magic %d -> %d",
                    layout_name,
                    rec["patches_before_pruning"],
                    rec["patches_after_pruning"],
                    rec["patches_removed_pct"],
                    rec["magic_patches_before_pruning"],
                    rec["magic_patches_after_pruning"],
                )
            except _SchedulerTimeout:
                elapsed = time.perf_counter() - circuit_start
                logger.warning(
                    "  %-14s timeout after %ds (elapsed %.1fs / budget %ds)",
                    layout_name,
                    layout_timeout,
                    elapsed,
                    args.circuit_timeout,
                )
                circuit_result["layout_results"].append({
                    "layout_name": layout_name,
                    "success": False,
                    "timed_out": True,
                    "error": f"timeout after {layout_timeout}s",
                })

                # If the circuit-wide budget is exhausted, skip the rest.
                if elapsed >= args.circuit_timeout:
                    timed_out_circuit = True
                    logger.warning("  circuit timeout reached; moving to next circuit")
                    break
            except Exception as exc:
                logger.warning("  %-14s failed: %s", layout_name, exc)
                circuit_result["layout_results"].append({
                    "layout_name": layout_name,
                    "success": False,
                    "error": str(exc),
                })

        if timed_out_circuit:
            circuit_result["status"] = "timed_out"

        circuit_docs.append(circuit_result)

        per_circuit_path = circuits_dir / f"{i:04d}_{_safe_name(family)}_{_safe_name(circuit_name)}.json"
        per_circuit_path.write_text(json.dumps(circuit_result, indent=2, default=str))

        _write_checkpoint(
            run_dir=run_dir,
            plots_dir=plots_dir,
            ts=ts,
            qasm_dir=args.qasm_dir,
            exclude=exclude,
            max_qubits=max_qubits,
            layouts=layouts,
            qasm_files=qasm_files,
            max_circuits=args.max_circuits,
            circuit_docs=circuit_docs,
            flat_rows=flat_rows,
        )

    # Final checkpoint to guarantee plots/aggregates exist even if no rows were produced.
    _write_checkpoint(
        run_dir=run_dir,
        plots_dir=plots_dir,
        ts=ts,
        qasm_dir=args.qasm_dir,
        exclude=exclude,
        max_qubits=max_qubits,
        layouts=layouts,
        qasm_files=qasm_files,
        max_circuits=args.max_circuits,
        circuit_docs=circuit_docs,
        flat_rows=flat_rows,
    )

    logger.info("Run complete.")
    logger.info("summary: %s", run_dir / "summary.json")
    logger.info("rows   : %s", run_dir / "per_layout_rows.csv")
    logger.info("plots  : %s", plots_dir)


if __name__ == "__main__":
    main()
