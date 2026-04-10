#!/usr/bin/env python3
"""
Compare magic-state preparation approaches on the 100-qubit QAOA circuit.

Runs **circuit-aware layout only** with three magic-state source
configurations × three scheduling modes (9 runs total):

    Sources:
        1. Unlimited  – no preparation constraint
        2. Factory    – deterministic 15-cycle cooldown
        3. Cultivation – geometric-distribution with mean ≈ 15 cycles

    Schedulers:
        1. Sequential        (steiner_tree)
        2. Greedy Packing    (steiner_packing)
        3. Pathfinder        (steiner_pathfinder)

Outputs:
    - JSON results  → routing_experiment_results/
    - Bar-chart PNG → plots/
"""

import json
import logging
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from harvest.compilation.pauli_block_conversion import convert_to_PCB, create_dag
from harvest.compilation.qasm_loader import qasm_to_circuit
from harvest.compilation.circuit_analysis import convert_rx_ry_to_rz, pre_prep_circuit
from harvest.routing.processor import DAGProcessor
from harvest.routing.magic_state_factory import MagicStateFactory
from harvest.routing.magic_state_cultivator import MagicStateCultivator, geometric_sampler
from harvest.synthesis.synthesizer import StaticLayoutSynthesizer
from harvest.synthesis.placement import PlacementConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("FactoryVsCultivation")


# ------------------------------------------------------------------
# Single run
# ------------------------------------------------------------------

def run_single(dag, layout_engine, scheduler_mode, scheduler_label,
               source_label, magic_source=None):
    """Route *dag* on *layout_engine* and return a metrics dict.

    Parameters
    ----------
    magic_source : MagicStateSource | None
        Pre-built source instance.  ``None`` → unlimited (old behaviour).
    """
    logger.info(f"  {scheduler_label} + {source_label}")
    try:
        processor = DAGProcessor(layout_engine=layout_engine, magic_source=magic_source)
        results = processor.process_entire_dag(dag, visualize_each_step=False, mode=scheduler_mode)

        meta = getattr(processor, '_scheduling_metadata', {})
        total_elapsed = meta.get('total_elapsed_steps', len(results))
        completed = meta.get('completed', True)
        nodes_completed = meta.get('num_nodes_completed', len(results))
        nodes_total = meta.get('num_nodes_total', len(results))

        total_wirelength = 0
        for r in results:
            total_wirelength += len(r.get("steiner_edges", set()))

        result = {
            "source_label": source_label,
            "scheduler_label": scheduler_label,
            "scheduler_mode": scheduler_mode,
            "num_timesteps": total_elapsed,
            "num_nodes_processed": nodes_completed,
            "num_nodes_total": nodes_total,
            "completed": completed,
            "total_wirelength": total_wirelength,
            "success": True,
            "magic_terminals_used": len(processor.used_magic_terminals),
            "magic_terminals_total": len(processor.magic_terminals),
        }

        source = processor.magic_source
        if source and not source.unlimited:
            stats = source.get_stats()
            result["magic_source_stats"] = stats
            result["magic_wait_cycles"] = stats["total_wait_cycles"]
        else:
            result["magic_wait_cycles"] = 0

        return result
    except Exception as e:
        logger.error(f"    FAILED: {e}")
        import traceback
        traceback.print_exc()
        return {
            "source_label": source_label,
            "scheduler_label": scheduler_label,
            "scheduler_mode": scheduler_mode,
            "success": False,
            "error": str(e),
        }


# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------

def create_comparison_plot(results, output_path):
    """Grouped bar chart: timesteps, wirelength, magic wait cycles."""
    schedulers = sorted({r["scheduler_label"] for r in results if r["success"]})
    source_labels = sorted({r["source_label"] for r in results if r["success"]})

    # Build lookup: (scheduler, source) → result
    lookup = {}
    for r in results:
        if r["success"]:
            lookup[(r["scheduler_label"], r["source_label"])] = r

    metrics = [
        ("num_timesteps", "Timesteps"),
        ("total_wirelength", "Total Wirelength"),
        ("magic_wait_cycles", "Magic Wait Cycles"),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(7 * len(metrics), 6))

    colors = plt.cm.Set2.colors

    for metric_idx, (metric, ylabel) in enumerate(metrics):
        ax = axes[metric_idx]
        x = np.arange(len(schedulers))
        width = 0.8 / max(len(source_labels), 1)

        for i, src_label in enumerate(source_labels):
            vals = []
            for sched in schedulers:
                r = lookup.get((sched, src_label))
                vals.append(r.get(metric, 0) if r else 0)
            offset = width * (i - (len(source_labels) - 1) / 2)
            bars = ax.bar(x + offset, vals, width, label=src_label,
                          color=colors[i % len(colors)], alpha=0.85)
            for bar, v in zip(bars, vals):
                if v > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height(),
                        str(int(v)),
                        ha="center", va="bottom", fontsize=7,
                    )

        ax.set_xticks(x)
        ax.set_xticklabels(schedulers, fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3, linestyle="--")

    fig.suptitle("Factory vs Cultivation: QAOA 100-qubit (Circuit-Aware Layout)",
                 fontweight="bold", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved plot to {output_path}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    # ---- Load QAOA 100-qubit circuit ----
    qasm_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "benchmark_circuits", "qasm",
        "qaoa", "big_100q", "qaoa_barabasi_albert_N100_3reps.qasm",
    )
    qasm_path = os.path.normpath(qasm_path)

    logger.info(f"Loading QAOA 100-qubit circuit from {qasm_path}")
    circuit = qasm_to_circuit(qasm_path)
    circuit = pre_prep_circuit(circuit)
    logger.info(f"Loaded: {circuit.num_qubits} qubits, depth {circuit.depth()}, "
                f"gates {dict(circuit.count_ops())}")

    gate_counts = circuit.count_ops()
    if "rx" in gate_counts or "ry" in gate_counts:
        logger.info("Converting rx/ry gates to rz equivalents ...")
        circuit = convert_rx_ry_to_rz(circuit)

    logger.info("Converting to PCB format ...")
    pcb = convert_to_PCB(circuit, verbose=False)
    dag = create_dag(pcb)

    # ---- Synthesize circuit-aware layout ----
    seed = 42
    synth = StaticLayoutSynthesizer(
        placement_config=PlacementConfig(alpha=1.0, beta=0.5, max_swap_iterations=100, seed=seed),
    )
    engine, report = synth.synthesize(dag)
    logger.info(f"Circuit-aware layout: {report.template_name} "
                f"({report.grid_width}x{report.grid_height}), cost {report.placement_cost:.2f}")

    # Discover magic terminals from the engine
    from harvest.routing.magic_terminal_selection import get_magic_terminals
    temp_proc = DAGProcessor(layout_engine=engine)
    magic_terminals = temp_proc.magic_terminals

    # ---- Define source configurations ----
    FACTORY_PREP_CYCLES = 15
    CULTIVATION_P = 1.0 / 15  # geometric mean ≈ 15 cycles
    CULTIVATION_SEED = 42

    def make_source(label):
        """Build a fresh magic-state source for the given label."""
        if label == "Unlimited":
            return None
        elif label == "Factory (15 cyc)":
            return MagicStateFactory(list(magic_terminals), FACTORY_PREP_CYCLES)
        elif label == "Cultivation (geo μ≈15)":
            return MagicStateCultivator(
                list(magic_terminals),
                readiness_sampler=geometric_sampler(CULTIVATION_P),
                seed=CULTIVATION_SEED,
            )
        else:
            raise ValueError(f"Unknown source label: {label}")

    source_labels = ["Unlimited", "Factory (15 cyc)", "Cultivation (geo μ≈15)"]

    schedulers = [
        ("Sequential", "steiner_tree"),
        ("Greedy Packing", "steiner_packing"),
        ("Pathfinder", "steiner_pathfinder"),
    ]

    # ---- Run experiments ----
    all_results = []
    for sched_label, sched_mode in schedulers:
        for src_label in source_labels:
            source = make_source(src_label)
            result = run_single(dag, engine, sched_mode, sched_label,
                                src_label, magic_source=source)
            all_results.append(result)

    # ---- Save results ----
    os.makedirs("routing_experiment_results", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = "factory_vs_cultivation_qaoa_100q"

    json_path = f"routing_experiment_results/{tag}_{ts}.json"
    with open(json_path, "w") as f:
        json.dump({
            "parameters": {
                "circuit": "qaoa_barabasi_albert_N100_3reps",
                "num_qubits": circuit.num_qubits,
                "layout": report.template_name,
                "grid": f"{report.grid_width}x{report.grid_height}",
                "placement_cost": report.placement_cost,
                "factory_prep_cycles": FACTORY_PREP_CYCLES,
                "cultivation_p": CULTIVATION_P,
                "cultivation_seed": CULTIVATION_SEED,
            },
            "results": all_results,
        }, f, indent=2, default=str)
    logger.info(f"Saved results to {json_path}")

    plot_path = f"plots/{tag}_{ts}.png"
    create_comparison_plot(all_results, plot_path)

    # ---- Summary table ----
    logger.info("\n" + "=" * 78)
    logger.info("SUMMARY")
    logger.info("=" * 78)
    for r in all_results:
        if r["success"]:
            wait = r.get("magic_wait_cycles", 0)
            wait_str = f", wait {wait:4d}" if wait else ""
            done = "" if r.get("completed", True) else (
                f" [INCOMPLETE {r.get('num_nodes_processed',0)}"
                f"/{r.get('num_nodes_total',0)} nodes]")
            logger.info(
                f"  {r['scheduler_label']:18s} + {r['source_label']:25s}: "
                f"{r['num_timesteps']:5d} steps, wl {r['total_wirelength']:5d}{wait_str}{done}"
            )
        else:
            logger.info(
                f"  {r['scheduler_label']:18s} + {r['source_label']:25s}: FAILED — {r.get('error','')}"
            )
    logger.info("=" * 78)


if __name__ == "__main__":
    main()
