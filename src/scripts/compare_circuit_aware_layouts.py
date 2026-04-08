#!/usr/bin/env python3
"""
Compare circuit-aware vs. baseline (non-circuit-aware) layout placement.

Both paths use the **same template** (same grid dimensions, magic ring,
routing lanes) so the only variable is *where* logical qubits are placed
on the data sites.

Path A (baseline): row-major or random qubit assignment
Path B (circuit-aware): interaction-graph-driven placement

Each layout is routed with all three scheduling modes and the results
are written to JSON + a bar chart.
"""

import json
import logging
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from harvest.compilation.pauli_block_conversion import convert_to_PCB, create_dag, create_random_circuit
from harvest.compilation.qasm_loader import qasm_to_circuit
from harvest.compilation.circuit_analysis import convert_rx_ry_to_rz, pre_prep_circuit
from harvest.routing.processor import DAGProcessor
from harvest.synthesis.synthesizer import StaticLayoutSynthesizer
from harvest.synthesis.placement import PlacementConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("CircuitAwareComparison")


# ------------------------------------------------------------------
# Experiment runner
# ------------------------------------------------------------------

def run_single(dag, layout_engine, scheduler_mode, layout_label, scheduler_label):
    """Route *dag* on *layout_engine* with *scheduler_mode* and return metrics dict."""
    logger.info(f"  {layout_label} + {scheduler_label}")
    try:
        processor = DAGProcessor(layout_engine=layout_engine)
        results = processor.process_entire_dag(dag, visualize_each_step=False, mode=scheduler_mode)

        time_steps = {r["time_step"] for r in results if "time_step" in r}
        num_timesteps = len(time_steps) if time_steps else len(results)

        total_wirelength = 0
        for r in results:
            total_wirelength += len(r.get("steiner_edges", set()))

        return {
            "layout_label": layout_label,
            "scheduler_label": scheduler_label,
            "scheduler_mode": scheduler_mode,
            "num_timesteps": num_timesteps,
            "num_nodes_processed": len(results),
            "total_wirelength": total_wirelength,
            "success": True,
            "magic_terminals_used": len(processor.used_magic_terminals),
            "magic_terminals_total": len(processor.magic_terminals),
        }
    except Exception as e:
        logger.error(f"    FAILED: {e}")
        return {
            "layout_label": layout_label,
            "scheduler_label": scheduler_label,
            "scheduler_mode": scheduler_mode,
            "success": False,
            "error": str(e),
        }


# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------

def create_comparison_plot(results, output_path):
    """Bar chart: timesteps and wirelength for each (layout, scheduler) pair."""
    all_schedulers = sorted({r["scheduler_label"] for r in results if r["success"]})
    layout_labels = sorted({r["layout_label"] for r in results if r["success"]})

    # build lookup
    lookup = {}
    for r in results:
        if r["success"]:
            lookup[(r["layout_label"], r["scheduler_label"])] = r

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for metric_idx, (metric, ylabel) in enumerate([
        ("num_timesteps", "Timesteps"),
        ("total_wirelength", "Total wirelength"),
    ]):
        schedulers = all_schedulers
        ax = axes[metric_idx]
        x = np.arange(len(schedulers))
        width = 0.8 / max(len(layout_labels), 1)
        for i, label in enumerate(layout_labels):
            vals = []
            for sched in schedulers:
                r = lookup.get((label, sched))
                vals.append(r[metric] if r else 0)
            offset = width * (i - (len(layout_labels) - 1) / 2)
            bars = ax.bar(x + offset, vals, width, label=label, alpha=0.85)
            for bar, v in zip(bars, vals):
                if v > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height(),
                        str(int(v)),
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )
        ax.set_xticks(x)
        ax.set_xticklabels(schedulers)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3, linestyle="--")

    fig.suptitle("Circuit-Aware vs Baseline Layout Placement", fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved plot to {output_path}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

# ------------------------------------------------------------------
# Shared comparison logic
# ------------------------------------------------------------------

def run_comparison(dag, label, seed=42):
    """
    Run circuit-aware vs baseline comparison on *dag*.
    Returns (all_results, reports_dict).
    """
    synth = StaticLayoutSynthesizer(
        placement_config=PlacementConfig(alpha=1.0, beta=0.5, max_swap_iterations=100, seed=seed),
    )

    engine_aware, report_aware = synth.synthesize(dag)
    engine_row, report_row = synth.synthesize_baseline(dag, mode="row_major")
    engine_col, report_col = synth.synthesize_baseline(dag, mode="column_major")

    logger.info(f"Circuit-aware cost    : {report_aware.placement_cost:.2f}")
    logger.info(f"Baseline (row_major)  : {report_row.placement_cost:.2f}")
    logger.info(f"Baseline (column_major): {report_col.placement_cost:.2f}")

    layouts = [
        ("Circuit-Aware", engine_aware),
        ("Row-Major Baseline", engine_row),
        ("Column-Major Baseline", engine_col),
    ]
    schedulers = [
        ("Sequential", "steiner_tree"),
        ("Greedy Packing", "steiner_packing"),
        ("Pathfinder", "steiner_pathfinder"),
    ]

    all_results = []
    for layout_label, engine in layouts:
        for sched_label, sched_mode in schedulers:
            result = run_single(dag, engine, sched_mode, layout_label, sched_label)
            all_results.append(result)

    reports = {
        "circuit_aware": {
            "template": report_aware.template_name,
            "grid": f"{report_aware.grid_width}x{report_aware.grid_height}",
            "cost": report_aware.placement_cost,
        },
        "row_major": {
            "template": report_row.template_name,
            "grid": f"{report_row.grid_width}x{report_row.grid_height}",
            "cost": report_row.placement_cost,
        },
        "column_major": {
            "template": report_col.template_name,
            "grid": f"{report_col.grid_width}x{report_col.grid_height}",
            "cost": report_col.placement_cost,
        },
    }
    return all_results, reports


def save_and_plot(all_results, reports, parameters, tag):
    """Save JSON results and create comparison plot."""
    os.makedirs("routing_experiment_results", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = f"routing_experiment_results/{tag}_{ts}.json"
    with open(json_path, "w") as f:
        json.dump(
            {"parameters": parameters, "reports": reports, "results": all_results},
            f,
            indent=2,
        )
    logger.info(f"Saved results to {json_path}")

    os.makedirs("plots", exist_ok=True)
    plot_path = f"plots/{tag}_{ts}.png"
    create_comparison_plot(all_results, plot_path)

    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    for r in all_results:
        if r["success"]:
            logger.info(
                f"  {r['layout_label']:25s} + {r['scheduler_label']:18s}: "
                f"{r['num_timesteps']:4d} steps, wirelength {r['total_wirelength']:5d}"
            )
        else:
            logger.info(f"  {r['layout_label']:25s} + {r['scheduler_label']:18s}: FAILED")
    logger.info("=" * 70)


# ------------------------------------------------------------------
# Main: random circuit
# ------------------------------------------------------------------

def main():
    num_qubits = 25
    depth = 100
    seed = 42

    logger.info(f"Creating random circuit: {num_qubits} qubits, depth {depth}, seed {seed}")
    circuit = create_random_circuit(num_qubits, depth, seed=seed)
    pcb = convert_to_PCB(circuit, verbose=False)
    dag = create_dag(pcb)

    logger.info("Synthesizing layouts ...")
    all_results, reports = run_comparison(dag, "random", seed=seed)
    save_and_plot(
        all_results, reports,
        {"num_qubits": num_qubits, "depth": depth, "seed": seed, "source": "random"},
        tag="circuit_aware_comparison",
    )


# ------------------------------------------------------------------
# Main: real QAOA 100-qubit benchmark circuit
# ------------------------------------------------------------------

def main_qaoa_100q():
    """Load the 100-qubit QAOA Barabasi-Albert benchmark and compare layouts."""
    qasm_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "benchmark_circuits", "qasm",
        "qaoa", "big_100q", "qaoa_barabasi_albert_N100_3reps.qasm",
    )
    qasm_path = os.path.normpath(qasm_path)

    logger.info(f"Loading QAOA 100-qubit circuit from {qasm_path}")
    circuit = qasm_to_circuit(qasm_path)
    circuit = pre_prep_circuit(circuit)
    logger.info(f"Loaded: {circuit.num_qubits} qubits, depth {circuit.depth()}, gates {dict(circuit.count_ops())}")

    # Convert rx/ry → rz if needed
    gate_counts = circuit.count_ops()
    if "rx" in gate_counts or "ry" in gate_counts:
        logger.info("Converting rx/ry gates to rz equivalents ...")
        circuit = convert_rx_ry_to_rz(circuit)

    logger.info("Converting to PCB format ...")
    pcb = convert_to_PCB(circuit, verbose=False)
    dag = create_dag(pcb)

    logger.info("Synthesizing layouts for QAOA 100q ...")
    all_results, reports = run_comparison(dag, "qaoa_100q", seed=42)
    save_and_plot(
        all_results, reports,
        {"circuit": "qaoa_barabasi_albert_N100_3reps", "num_qubits": circuit.num_qubits, "source": "benchmark"},
        tag="circuit_aware_qaoa_100q",
    )


if __name__ == "__main__":
    main_qaoa_100q()
