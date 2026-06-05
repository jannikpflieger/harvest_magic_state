#!/usr/bin/env python3
"""
Evaluate lattice-pruner impact on layout size (in patches) for QAOA 100-qubit.

This script runs the same circuit with greedy scheduling (steiner_packing)
across the three built-in layout styles:

1. Single spacing  : nxm_ring_layout_single_qubits
2. Double spacing  : nxm_ring_layout_single_qubits_large_spacing
3. Blocks of 4     : blocks_of_four_qubit_patches

For each layout, it records patch counts:
- before pruning: number of patches with at least one port in ports_by_patch
- after pruning : number of patches with at least one port in pruned ports_by_patch

It also stores raw pruning stats and generates a bar plot comparing
"before" vs "after" patch counts per layout.

Usage:
    PYTHONPATH=src python src/scripts/evaluate_pruner_layout_size_qaoa100.py

Optional:
    --rows N / --cols N   Data-grid dimensions for single/double spacing (default 10x10)
    --out-dir PATH        Output root (default: repo root)
    --qasm-path PATH      Explicit QAOA-100 QASM path
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List

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
from harvest.compilation.qasm_loader import qasm_to_circuit
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
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
)
logger = logging.getLogger("PrunerLayoutSizeQAOA100")

# Keep internals quieter for long runs
for _noisy in (
    "HarvestMagicState",
    "HarvestMagicState.DAGProcessor",
    "HarvestMagicState.Detailed",
):
    logging.getLogger(_noisy).setLevel(logging.WARNING)


def _default_qasm_path() -> str:
    return str(
        (PROJECT_ROOT / "benchmark_circuits" / "qasm" / "qaoa" / "big_100q" /
         "qaoa_barabasi_albert_N100_3reps.qasm").resolve()
    )


def _load_qaoa_dag(qasm_path: str):
    """Load QAOA QASM and convert to DAG in the project preprocessing style."""
    logger.info("Loading QASM: %s", qasm_path)
    circuit = qasm_to_circuit(qasm_path)
    circuit = pre_prep_circuit(circuit)

    gate_counts = circuit.count_ops()
    if "rx" in gate_counts or "ry" in gate_counts:
        logger.info("Converting rx/ry to rz ...")
        circuit = convert_rx_ry_to_rz(circuit)

    logger.info(
        "Circuit loaded: qubits=%d depth=%d ops=%s",
        circuit.num_qubits,
        circuit.depth(),
        dict(circuit.count_ops()),
    )

    pcb = convert_to_PCB(circuit, verbose=False)
    dag = create_dag(pcb)
    logger.info("DAG ops: %d", len(list(dag.op_nodes())))
    return dag, circuit.num_qubits


def _evaluate_layout(
    dag,
    layout_name: str,
    make_layout: Callable,
) -> Dict:
    """Run greedy scheduling + pruning once for one layout and collect metrics."""
    logger.info("Running layout: %s", layout_name)
    engine = make_layout()

    # Layout-level patch counts (static geometry)
    total_patches = len(engine.patches)
    total_magic_patches = sum(1 for p in engine.patches.values() if p.kind == "magic")
    total_data_patches = total_patches - total_magic_patches

    processor = DAGProcessor(layout_engine=engine)

    # "Before" count in the routing view: patches with at least one available port
    patches_before = len(processor.ports_by_patch)
    nodes_before = len(processor.graph)

    t0 = time.perf_counter()
    results = processor.process_entire_dag(
        dag,
        visualize_each_step=False,
        mode="steiner_packing",  # Greedy
    )
    schedule_runtime_s = time.perf_counter() - t0

    # Apply post-scheduling pruner
    prune_stats = processor.prune_after_scheduling(results)

    patches_after = len(processor.ports_by_patch)
    nodes_after = len(processor.graph)

    # Scheduler metadata
    meta = getattr(processor, "_scheduling_metadata", {})
    timesteps = meta.get("total_elapsed_steps", len(results))
    completed = meta.get("completed", True)
    nodes_completed = meta.get("num_nodes_completed", len(results))
    nodes_total = meta.get("num_nodes_total", len(results))

    patch_reduction = patches_before - patches_after
    patch_reduction_pct = (patch_reduction / patches_before * 100.0) if patches_before else 0.0

    rec = {
        "layout_name": layout_name,
        "scheduler": "Greedy",
        "scheduler_mode": "steiner_packing",
        "layout_total_patches": total_patches,
        "layout_data_patches": total_data_patches,
        "layout_magic_patches": total_magic_patches,
        "patches_before_pruning": patches_before,
        "patches_after_pruning": patches_after,
        "patches_removed": patch_reduction,
        "patches_removed_pct": patch_reduction_pct,
        "magic_patches_before_pruning": prune_stats.get("magic_patches_before"),
        "magic_patches_after_pruning": prune_stats.get("magic_patches_after"),
        "magic_patches_removed": prune_stats.get("magic_patches_removed"),
        "data_patches_before_pruning": prune_stats.get("data_patches_before"),
        "data_patches_after_pruning": prune_stats.get("data_patches_after"),
        "data_patches_removed": prune_stats.get("data_patches_removed"),
        "graph_nodes_before_pruning": nodes_before,
        "graph_nodes_after_pruning": nodes_after,
        "graph_nodes_removed": nodes_before - nodes_after,
        "schedule_runtime_s": round(schedule_runtime_s, 3),
        "num_timesteps": timesteps,
        "num_nodes_processed": nodes_completed,
        "num_nodes_total": nodes_total,
        "completed": completed,
        "prune_stats": prune_stats,
    }

    logger.info(
        "  %s: patches %d -> %d (removed %d, %.1f%%), graph nodes %d -> %d",
        layout_name,
        patches_before,
        patches_after,
        patch_reduction,
        patch_reduction_pct,
        nodes_before,
        nodes_after,
    )
    return rec


def _plot_patch_counts(records: List[Dict], out_png: str):
    """Create a grouped bar chart for patches before vs after pruning."""
    order = ["Single Spacing", "Double Spacing", "Blocks of 4"]
    rec_map = {r["layout_name"]: r for r in records}
    layouts = [name for name in order if name in rec_map]

    before = [rec_map[name]["patches_before_pruning"] for name in layouts]
    after = [rec_map[name]["patches_after_pruning"] for name in layouts]

    x = np.arange(len(layouts))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 6))

    bars_before = ax.bar(x - width / 2, before, width, label="Before pruning", color="#4C72B0")
    bars_after = ax.bar(x + width / 2, after, width, label="After pruning", color="#55A868")

    # Value labels
    for bars in (bars_before, bars_after):
        for b in bars:
            h = b.get_height()
            ax.text(
                b.get_x() + b.get_width() / 2,
                h + max(1.0, 0.01 * max(before + after)),
                f"{int(h)}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    # Reduction annotations
    for i, name in enumerate(layouts):
        b = rec_map[name]["patches_before_pruning"]
        a = rec_map[name]["patches_after_pruning"]
        red = b - a
        red_pct = (red / b * 100.0) if b else 0.0
        ax.text(
            x[i],
            max(b, a) * 0.92,
            f"-{red} ({red_pct:.1f}%)",
            ha="center",
            va="top",
            fontsize=9,
            color="#333333",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(layouts)
    ax.set_ylabel("Number of patches with active ports")
    ax.set_xlabel("Layout style")
    ax.set_title("QAOA n100 (Greedy) — Patch Count Before vs After Pruning")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved plot: %s", out_png)


def _write_csv(records: List[Dict], out_csv: str):
    """Write flat summary CSV for quick external analysis."""
    fields = [
        "layout_name",
        "scheduler",
        "scheduler_mode",
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
        "data_patches_before_pruning",
        "data_patches_after_pruning",
        "data_patches_removed",
        "graph_nodes_before_pruning",
        "graph_nodes_after_pruning",
        "graph_nodes_removed",
        "schedule_runtime_s",
        "num_timesteps",
        "num_nodes_processed",
        "num_nodes_total",
        "completed",
    ]

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in records:
            writer.writerow({k: r.get(k) for k in fields})
    logger.info("Saved CSV: %s", out_csv)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate patch counts before/after pruning for QAOA-100 using greedy scheduler across 3 layout styles."
    )
    parser.add_argument("--rows", type=int, default=10, help="Rows for single/double spacing layouts (default: 10)")
    parser.add_argument("--cols", type=int, default=10, help="Cols for single/double spacing layouts (default: 10)")
    parser.add_argument("--qasm-path", type=str, default=_default_qasm_path(), help="Path to QAOA-100 QASM")
    parser.add_argument("--out-dir", type=str, default=None, help="Output root directory (default: repo root)")
    args = parser.parse_args()

    if args.rows <= 0 or args.cols <= 0:
        raise ValueError("--rows and --cols must be > 0")
    if (args.rows % 2 != 0) or (args.cols % 2 != 0):
        logger.warning(
            "Rows/cols are not even. Blocks-of-4 layout uses rows//2, cols//2 and may host fewer than rows*cols qubits."
        )

    qasm_path = os.path.abspath(args.qasm_path)
    if not os.path.exists(qasm_path):
        raise FileNotFoundError(f"QASM not found: {qasm_path}")

    out_root = Path(args.out_dir).resolve() if args.out_dir else PROJECT_ROOT
    results_dir = out_root / "routing_experiment_results"
    plots_dir = out_root / "plots"
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"pruner_layout_size_qaoa100_{timestamp}"
    json_path = results_dir / f"{stem}.json"
    csv_path = results_dir / f"{stem}.csv"
    plot_path = plots_dir / f"{stem}.png"

    dag, n_qubits = _load_qaoa_dag(qasm_path)
    if n_qubits != 100:
        logger.warning("Loaded circuit has %d qubits (expected 100 for this experiment).", n_qubits)

    rows = args.rows
    cols = args.cols

    layouts = [
        ("Single Spacing", lambda: nxm_ring_layout_single_qubits(rows, cols)),
        ("Double Spacing", lambda: nxm_ring_layout_single_qubits_large_spacing(rows, cols)),
        ("Blocks of 4", lambda: blocks_of_four_qubit_patches(rows // 2, cols // 2)),
    ]

    records: List[Dict] = []
    for layout_name, make_layout in layouts:
        rec = _evaluate_layout(dag, layout_name, make_layout)
        records.append(rec)

    summary = {
        "experiment": "pruner_layout_size_qaoa100",
        "timestamp": timestamp,
        "circuit": {
            "qasm_path": qasm_path,
            "qubits": n_qubits,
        },
        "scheduler": {
            "name": "Greedy",
            "mode": "steiner_packing",
        },
        "layout_grid": {
            "rows": rows,
            "cols": cols,
            "blocks_layout_rows": rows // 2,
            "blocks_layout_cols": cols // 2,
        },
        "results": records,
    }

    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved JSON: %s", json_path)

    _write_csv(records, str(csv_path))
    _plot_patch_counts(records, str(plot_path))

    logger.info("\n=== Summary (patches with active ports) ===")
    for r in records:
        logger.info(
            "%s: %d -> %d (removed %d, %.1f%%)",
            r["layout_name"],
            r["patches_before_pruning"],
            r["patches_after_pruning"],
            r["patches_removed"],
            r["patches_removed_pct"],
        )


if __name__ == "__main__":
    main()
