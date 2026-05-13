#!/usr/bin/env python3
"""
Adaptive scheduler selection experiment — QAOA 100-qubit, single-spaced layout.

Compares four schedulers on a fixed circuit+layout combination:
  1. Sequential        (steiner_tree)
  2. Greedy Packing    (steiner_packing)
  3. Pathfinder        (steiner_pathfinder)
  4. Adaptive          (selects scheduler based on circuit-structure features)

Outputs:
  - Printed comparison table
  - JSON  → routing_experiment_results/adaptive_experiment_<timestamp>.json
  - Plot  → plots/adaptive_scheduler_motivation_<timestamp>.{png,pdf}

Reproduce with::

    cd /path/to/harvest_magic_state
    python src/scripts/adaptive_scheduler_experiment.py

Optional flags::

    --low-par FLOAT    Override low_parallelism_threshold (default 2.0)
    --high-weight FLOAT  Override high_weight_threshold (default 6.0)
    --out-dir PATH     Output directory for JSON/plots (default: current dir)
"""

import argparse
import json
import logging
import os
import time
from datetime import datetime
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from harvest.compilation.circuit_analysis import convert_rx_ry_to_rz, pre_prep_circuit
from harvest.compilation.pauli_block_conversion import convert_to_PCB, create_dag
from harvest.compilation.qasm_loader import qasm_to_circuit
from harvest.evaluation.metrics import compute_active_volume
from harvest.layout.presets import nxm_ring_layout_single_qubits
from harvest.routing.adaptive_scheduler import AdaptiveConfig, schedule_adaptive
from harvest.routing.circuit_features import compute_circuit_features
from harvest.routing.processor import DAGProcessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("AdaptiveExperiment")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_QUBITS = 100
LAYOUT_ROWS = 10
LAYOUT_COLS = 10

# Base schedulers to compare against adaptive.
BASE_SCHEDULERS = [
    ("Sequential", "steiner_tree"),
    ("Greedy Packing", "steiner_packing"),
    ("Pathfinder", "steiner_pathfinder"),
]


# ---------------------------------------------------------------------------
# Single-run helper
# ---------------------------------------------------------------------------

def _run_single(
    dag,
    scheduler_name: str,
    scheduler_mode: str,
    layout_engine,
    *,
    selected_scheduler: str = "",
    selection_reason: str = "",
    adaptive_config: Optional[AdaptiveConfig] = None,
) -> Dict:
    """Route *dag* on *layout_engine* and collect metrics.

    For the adaptive scheduler, pass ``scheduler_mode="adaptive"`` and supply
    *adaptive_config*.  In that case the function calls
    :func:`~harvest.routing.adaptive_scheduler.schedule_adaptive` and records
    the chosen mode in *selected_scheduler*.
    """
    logger.info(f"Running scheduler: {scheduler_name!r}")
    try:
        layout_engine_instance = layout_engine()  # fresh instance per run

        processor = DAGProcessor(layout_engine=layout_engine_instance)

        t0 = time.perf_counter()
        if scheduler_mode == "adaptive":
            results, chosen_mode, reason = schedule_adaptive(
                processor, dag, config=adaptive_config
            )
            selected_scheduler = chosen_mode
            selection_reason = reason
        else:
            results = processor.process_entire_dag(
                dag, visualize_each_step=False, mode=scheduler_mode
            )
            chosen_mode = scheduler_mode
        runtime_s = time.perf_counter() - t0

        meta = getattr(processor, "_scheduling_metadata", {})
        num_timesteps = meta.get("total_elapsed_steps", len(results))
        completed = meta.get("completed", True)
        nodes_completed = meta.get("num_nodes_completed", len(results))
        nodes_total = meta.get("num_nodes_total", len(results))

        active_vol = compute_active_volume(results, chosen_mode)
        total_wirelength = sum(len(r.get("steiner_edges", set())) for r in results)

        return {
            "scheduler_name": scheduler_name,
            "scheduler_mode": scheduler_mode,
            "selected_scheduler": selected_scheduler,
            "selection_reason": selection_reason,
            "num_timesteps": num_timesteps,
            "active_volume": active_vol,
            "total_wirelength": total_wirelength,
            "num_nodes_processed": nodes_completed,
            "num_nodes_total": nodes_total,
            "completed": completed,
            "runtime_s": round(runtime_s, 2),
            "success": True,
        }

    except Exception as exc:
        logger.error(f"  FAILED: {exc}", exc_info=True)
        return {
            "scheduler_name": scheduler_name,
            "scheduler_mode": scheduler_mode,
            "selected_scheduler": selected_scheduler,
            "selection_reason": selection_reason,
            "success": False,
            "error": str(exc),
        }


# ---------------------------------------------------------------------------
# Motivation plot
# ---------------------------------------------------------------------------

# Colour palette: blue family for base schedulers, orange for adaptive.
_COLOURS = {
    "Sequential":     "#4C72B0",
    "Greedy Packing": "#55A868",
    "Pathfinder":     "#8172B2",
    "Adaptive":       "#DD8452",
}


def create_motivation_plot(
    all_results: List[Dict],
    adaptive_info: Dict,
    output_prefix: str,
) -> None:
    """Two-subplot bar chart comparing schedulers on timesteps and active volume.

    Args:
        all_results:   List of result dicts from :func:`_run_single`.
        adaptive_info: Dict with keys ``"selected_mode"`` and ``"reason"``.
        output_prefix: File path prefix (without extension).  Both ``.png``
                       and ``.pdf`` will be written.
    """
    successful = [r for r in all_results if r.get("success")]
    if not successful:
        logger.warning("No successful results — skipping plot.")
        return

    names = [r["scheduler_name"] for r in successful]

    # Build display labels: adaptive bar gets sub-label showing chosen scheduler.
    display_labels = []
    for r in successful:
        if r["scheduler_mode"] == "adaptive" and r["selected_scheduler"]:
            short = {
                "steiner_tree": "Sequential",
                "steiner_packing": "Greedy\nPacking",
                "steiner_pathfinder": "Pathfinder",
            }.get(r["selected_scheduler"], r["selected_scheduler"])
            display_labels.append(f"Adaptive\n→ {short}")
        else:
            display_labels.append(r["scheduler_name"])

    timesteps = [r.get("num_timesteps", 0) for r in successful]
    active_vols = [r.get("active_volume", 0) for r in successful]
    colours = [_COLOURS.get(r["scheduler_name"], "#888888") for r in successful]

    x = np.arange(len(names))

    fig, (ax_ts, ax_av) = plt.subplots(1, 2, figsize=(12, 5))

    def _bar_plot(ax, values, ylabel, title):
        bars = ax.bar(x, values, color=colours, alpha=0.88, edgecolor="white", width=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(display_labels, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)
        y_max = max(values) if values else 1
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + y_max * 0.01,
                f"{val:,}",
                ha="center", va="bottom",
                fontsize=9, fontweight="bold",
            )

    _bar_plot(ax_ts, timesteps, "Timesteps", "Timesteps per Scheduler")
    _bar_plot(ax_av, active_vols, "Active Volume (cell-steps)", "Active Volume per Scheduler")

    # Highlight adaptive bar with a border in both subplots.
    for ax, values in [(ax_ts, timesteps), (ax_av, active_vols)]:
        bars = ax.patches  # same order as x
        if len(bars) > len(x):
            # matplotlib may add extra patches for grid lines; skip them
            pass
        for i, r in enumerate(successful):
            if r["scheduler_mode"] == "adaptive":
                ax.patches[i].set_edgecolor("#CC4400")
                ax.patches[i].set_linewidth(2.0)

    # Build subtitle from adaptive selection info.
    sel_mode = adaptive_info.get("selected_mode", "?")
    sel_short = {
        "steiner_tree": "Sequential",
        "steiner_packing": "Greedy Packing",
        "steiner_pathfinder": "Pathfinder",
    }.get(sel_mode, sel_mode)
    reason_short = adaptive_info.get("reason", "")
    # Trim reason to fit title — keep first sentence.
    reason_short = reason_short.split("→")[-1].strip() if "→" in reason_short else reason_short

    fig.suptitle(
        f"Scheduler Comparison — QAOA 100-qubit (Barabasi-Albert, 3 reps), "
        f"Single-Spaced Layout 10×10\n"
        f"Adaptive selected: {sel_short}  |  {reason_short}",
        fontsize=10,
        fontweight="bold",
        y=1.02,
    )

    # Colour legend patches for base schedulers + adaptive.
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=_COLOURS["Sequential"], label="Sequential"),
        Patch(facecolor=_COLOURS["Greedy Packing"], label="Greedy Packing"),
        Patch(facecolor=_COLOURS["Pathfinder"], label="Pathfinder"),
        Patch(facecolor=_COLOURS["Adaptive"], label="Adaptive",
              edgecolor="#CC4400", linewidth=2),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=4,
        fontsize=9,
        framealpha=0.9,
        bbox_to_anchor=(0.5, -0.04),
    )

    plt.tight_layout()
    for ext in ("png", "pdf"):
        path = f"{output_prefix}.{ext}"
        plt.savefig(path, dpi=180, bbox_inches="tight")
        logger.info(f"Saved plot → {path}")
    plt.close()


# ---------------------------------------------------------------------------
# Print table
# ---------------------------------------------------------------------------

def print_table(all_results: List[Dict]) -> None:
    """Print a compact comparison table to stdout."""
    col_w = [22, 20, 12, 15, 16, 10]
    sep = "+" + "+".join("-" * w for w in col_w) + "+"
    fmt = "| {:<20} | {:<18} | {:>10} | {:>13} | {:>14} | {:>8} |"

    print(sep)
    print(fmt.format(
        "Scheduler", "Selected (adaptive)", "Timesteps",
        "Active Volume", "Total Wirelength", "Time (s)",
    ))
    print(sep)
    for r in all_results:
        if r.get("success"):
            selected = r.get("selected_scheduler", "")
            if selected:
                # Map mode string to friendly label
                selected = {
                    "steiner_tree": "Sequential",
                    "steiner_packing": "Greedy Packing",
                    "steiner_pathfinder": "Pathfinder",
                }.get(selected, selected)
            completed_flag = "" if r.get("completed", True) else " ⚠ INCOMPLETE"
            print(fmt.format(
                r["scheduler_name"],
                selected,
                f"{r.get('num_timesteps', '?'):,}{completed_flag}",
                f"{r.get('active_volume', '?'):,}",
                f"{r.get('total_wirelength', '?'):,}",
                f"{r.get('runtime_s', '?'):.1f}",
            ))
        else:
            print(fmt.format(
                r["scheduler_name"], "", "FAILED", "", "", "",
            ))
    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Adaptive scheduler experiment on QAOA-100, single-spaced layout."
    )
    parser.add_argument(
        "--low-par", type=float, default=2.0,
        metavar="FLOAT",
        help="low_parallelism_threshold for adaptive heuristic (default: 2.0)",
    )
    parser.add_argument(
        "--high-par", type=float, default=5.0,
        metavar="FLOAT",
        help="high_parallelism_threshold for adaptive heuristic (default: 5.0)",
    )
    parser.add_argument(
        "--out-dir", type=str, default=None,
        metavar="PATH",
        help="Output directory for JSON and plots (default: repo root)",
    )
    args = parser.parse_args()

    # Resolve output paths relative to repo root (two levels above this script).
    repo_root = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )
    out_dir = args.out_dir if args.out_dir else repo_root

    results_dir = os.path.join(out_dir, "routing_experiment_results")
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load & preprocess QAOA 100-qubit circuit
    # ------------------------------------------------------------------
    qasm_path = os.path.normpath(
        os.path.join(
            os.path.dirname(__file__), "..", "..",
            "benchmark_circuits", "qasm", "qaoa", "big_100q",
            "qaoa_barabasi_albert_N100_3reps.qasm",
        )
    )
    logger.info(f"Loading QAOA 100-qubit circuit from:\n  {qasm_path}")
    circuit = qasm_to_circuit(qasm_path)
    circuit = pre_prep_circuit(circuit)
    logger.info(
        f"Circuit: {circuit.num_qubits} qubits, depth {circuit.depth()}, "
        f"gates {dict(circuit.count_ops())}"
    )

    gate_counts = circuit.count_ops()
    if "rx" in gate_counts or "ry" in gate_counts:
        logger.info("Converting rx/ry gates to rz equivalents ...")
        circuit = convert_rx_ry_to_rz(circuit)

    logger.info("Converting to PCB format ...")
    pcb = convert_to_PCB(circuit, verbose=False)
    dag = create_dag(pcb)
    logger.info(f"DAG: {len(list(dag.op_nodes()))} operation nodes")

    # ------------------------------------------------------------------
    # 2. Compute circuit features (printed once, used by adaptive scheduler)
    # ------------------------------------------------------------------
    features = compute_circuit_features(dag)
    logger.info(f"\nCircuit features: {features}")

    # ------------------------------------------------------------------
    # 3. Build adaptive config from CLI args
    # ------------------------------------------------------------------
    adaptive_config = AdaptiveConfig(
        low_parallelism_threshold=args.low_par,
        high_parallelism_threshold=args.high_par,
    )
    logger.info(
        f"Adaptive thresholds: low_par={adaptive_config.low_parallelism_threshold}, "
        f"high_weight={adaptive_config.high_weight_threshold}"
    )

    # ------------------------------------------------------------------
    # 4. Layout factory (callable → fresh instance per scheduler run)
    # ------------------------------------------------------------------
    def make_layout():
        return nxm_ring_layout_single_qubits(LAYOUT_ROWS, LAYOUT_COLS)

    layout_desc = f"single_spaced_{LAYOUT_ROWS}x{LAYOUT_COLS}"
    eng_sample = make_layout()
    logger.info(
        f"Layout: nxm_ring_layout_single_qubits({LAYOUT_ROWS}, {LAYOUT_COLS}) "
        f"→ grid {eng_sample.W}×{eng_sample.H}, "
        f"{sum(1 for p in eng_sample.patches.values() if p.kind == 'magic')} magic patches"
    )

    # ------------------------------------------------------------------
    # 5. Run all schedulers
    # ------------------------------------------------------------------
    all_results: List[Dict] = []
    adaptive_selected_mode = ""
    adaptive_reason = ""

    logger.info("\n" + "=" * 70)
    logger.info("Running base schedulers ...")
    logger.info("=" * 70)

    for sched_name, sched_mode in BASE_SCHEDULERS:
        result = _run_single(dag, sched_name, sched_mode, make_layout)
        all_results.append(result)
        if result["success"]:
            logger.info(
                f"  ✓ {sched_name}: T={result['num_timesteps']}, "
                f"active_vol={result['active_volume']:,}, "
                f"wirelength={result['total_wirelength']}, "
                f"time={result['runtime_s']:.1f}s"
            )

    logger.info("\n" + "=" * 70)
    logger.info("Running adaptive scheduler ...")
    logger.info("=" * 70)

    adaptive_result = _run_single(
        dag, "Adaptive", "adaptive", make_layout,
        adaptive_config=adaptive_config,
    )
    all_results.append(adaptive_result)

    if adaptive_result["success"]:
        adaptive_selected_mode = adaptive_result.get("selected_scheduler", "")
        adaptive_reason = adaptive_result.get("selection_reason", "")
        logger.info(
            f"  ✓ Adaptive → selected: {adaptive_selected_mode!r}\n"
            f"    Reason: {adaptive_reason}\n"
            f"    T={adaptive_result['num_timesteps']}, "
            f"active_vol={adaptive_result['active_volume']:,}, "
            f"wirelength={adaptive_result['total_wirelength']}, "
            f"time={adaptive_result['runtime_s']:.1f}s"
        )

    # ------------------------------------------------------------------
    # 6. Print table
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 70)
    print_table(all_results)

    # ------------------------------------------------------------------
    # 7. Save JSON
    # ------------------------------------------------------------------
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(results_dir, f"adaptive_experiment_{ts}.json")
    with open(json_path, "w") as fh:
        json.dump(
            {
                "parameters": {
                    "circuit": "qaoa_barabasi_albert_N100_3reps",
                    "num_qubits": N_QUBITS,
                    "layout": layout_desc,
                    "factory": "Unlimited",
                    "adaptive_config": {
                        "low_parallelism_threshold": adaptive_config.low_parallelism_threshold,
                        "high_weight_threshold": adaptive_config.high_weight_threshold,
                    },
                },
                "circuit_features": {
                    "num_qubits": features.num_qubits,
                    "num_pauli_products": features.num_pauli_products,
                    "avg_weight": features.avg_weight,
                    "max_weight": features.max_weight,
                    "t_count_ratio": features.t_count_ratio,
                    "avg_layer_width": features.avg_layer_width,
                    "dependency_depth": features.dependency_depth,
                },
                "adaptive_selection": {
                    "selected_mode": adaptive_selected_mode,
                    "reason": adaptive_reason,
                },
                "results": all_results,
            },
            fh,
            indent=2,
        )
    logger.info(f"\nSaved JSON → {json_path}")

    # ------------------------------------------------------------------
    # 8. Motivation plot
    # ------------------------------------------------------------------
    plot_prefix = os.path.join(plots_dir, f"adaptive_scheduler_motivation_{ts}")
    create_motivation_plot(
        all_results,
        {"selected_mode": adaptive_selected_mode, "reason": adaptive_reason},
        plot_prefix,
    )

    logger.info("\nDone.")


if __name__ == "__main__":
    main()
