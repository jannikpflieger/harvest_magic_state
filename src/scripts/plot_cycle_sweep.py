#!/usr/bin/env python3
"""
Sweep magic-state preparation cycle time and compare Factory vs Cultivation.

Runs the Ising n66 circuit with a circuit-aware top-only magic layout
using Greedy Packing (steiner_packing).  For each cycle time from 1 to
19, both a deterministic factory and a stochastic cultivator (geometric
distribution with matching mean) are evaluated.

Outputs:
    - Line-plot PNG → plots/
    - JSON results  → routing_experiment_results/
"""

import copy
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
from harvest.synthesis.circuit_summary import extract_circuit_summary
from harvest.synthesis.templates import select_template
from harvest.synthesis.placement import circuit_aware_placement
from harvest.synthesis.emitter import emit_layout

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("CycleSweep")

# ── Constants ─────────────────────────────────────────────────────────
CYCLE_RANGE = range(1, 20)          # 1 .. 19
SCHEDULER_MODE = "steiner_packing"  # Greedy Packing
CULTIVATION_SEED = 42
PLACEMENT_SEED = 42


# ------------------------------------------------------------------
# Single run
# ------------------------------------------------------------------

def run_single(dag, layout_engine, source_label, magic_source):
    """Route *dag* and return a metrics dict."""
    try:
        processor = DAGProcessor(layout_engine=layout_engine, magic_source=magic_source)
        results = processor.process_entire_dag(dag, visualize_each_step=False, mode=SCHEDULER_MODE)

        meta = getattr(processor, '_scheduling_metadata', {})
        total_elapsed = meta.get('total_elapsed_steps', len(results))
        completed = meta.get('completed', True)
        nodes_completed = meta.get('num_nodes_completed', len(results))
        nodes_total = meta.get('num_nodes_total', len(results))

        total_wirelength = sum(len(r.get("steiner_edges", set())) for r in results)

        result = {
            "source_label": source_label,
            "num_timesteps": total_elapsed,
            "num_nodes_processed": nodes_completed,
            "num_nodes_total": nodes_total,
            "completed": completed,
            "total_wirelength": total_wirelength,
            "success": True,
        }

        source = processor.magic_source
        if source and not source.unlimited:
            stats = source.get_stats()
            result["magic_wait_cycles"] = stats["total_wait_cycles"]
        else:
            result["magic_wait_cycles"] = 0

        return result
    except Exception as e:
        logger.error(f"    FAILED: {e}")
        import traceback
        traceback.print_exc()
        return {"source_label": source_label, "success": False, "error": str(e)}


# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------

def create_line_plot(factory_steps, cultivation_steps, cycle_times, output_path):
    """Line plot: timesteps vs cycle time for Factory and Cultivation."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(cycle_times, factory_steps, marker="o", linewidth=2,
            label="Factory", color="#1f77b4")
    ax.plot(cycle_times, cultivation_steps, marker="s", linewidth=2,
            linestyle="--", label="Cultivation", color="#ff7f0e")

    ax.set_xlabel("Cycle Time", fontsize=12)
    ax.set_ylabel("Timesteps", fontsize=12)
    ax.set_title("Factory vs Cultivation: Timesteps by Cycle Time\n"
                 "(Ising n66, Greedy Packing, Top-Only Layout)",
                 fontweight="bold", fontsize=13)
    ax.set_xticks(cycle_times)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved plot to {output_path}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    # ---- Load Ising n66 circuit ----
    qasm_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "benchmark_circuits", "qasm",
        "qasmbench-large", "ising_n66", "ising_n66.qasm",
    )
    qasm_path = os.path.normpath(qasm_path)

    logger.info(f"Loading Ising n66 circuit from {qasm_path}")
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

    # ---- Synthesize circuit-aware layout (top-only magic patches) ----
    placement_cfg = PlacementConfig(alpha=1.0, beta=0.5, max_swap_iterations=100, seed=PLACEMENT_SEED)
    summary = extract_circuit_summary(dag)
    template = select_template(
        n_qubits=summary.num_qubits,
        max_parallelism=summary.parallelism_profile.get("max_pauli_per_layer", 0),
    )
    # Filter magic sites to top edge only
    template_top = copy.deepcopy(template)
    template_top.magic_sites = [(x, y) for x, y in template_top.magic_sites if y == 0]
    placement = circuit_aware_placement(summary, template_top, placement_cfg)
    engine = emit_layout(template_top, placement)
    report_grid = f"{template_top.grid_width}x{template_top.grid_height}"
    logger.info(f"Layout: {template_top.name} ({report_grid}), "
                f"magic patches: {len(template_top.magic_sites)} (top only), "
                f"cost {placement.cost:.2f}")

    # Discover magic terminals
    temp_proc = DAGProcessor(layout_engine=engine)
    magic_terminals = list(temp_proc.magic_terminals)
    logger.info(f"Magic terminals: {len(magic_terminals)}")

    # ---- Sweep cycle times ----
    cycle_times = list(CYCLE_RANGE)
    factory_steps = []
    cultivation_steps = []
    all_results = []

    for cycle_time in cycle_times:
        logger.info(f"\n{'='*60}")
        logger.info(f"Cycle time = {cycle_time}")
        logger.info(f"{'='*60}")

        # Factory (deterministic)
        logger.info(f"  Factory (prep_cycles={cycle_time})")
        factory_src = MagicStateFactory(magic_terminals, cycle_time)
        factory_result = run_single(dag, engine, f"Factory ({cycle_time} cyc)", factory_src)
        factory_result["cycle_time"] = cycle_time
        all_results.append(factory_result)

        if factory_result["success"]:
            factory_steps.append(factory_result["num_timesteps"])
            logger.info(f"    → {factory_result['num_timesteps']} timesteps")
        else:
            factory_steps.append(None)

        # Cultivation (geometric, mean = cycle_time)
        p = 1.0 / cycle_time
        logger.info(f"  Cultivation (geometric p={p:.4f}, mean≈{cycle_time})")
        cultivation_src = MagicStateCultivator(
            magic_terminals,
            readiness_sampler=geometric_sampler(p),
            seed=CULTIVATION_SEED,
        )
        cultivation_result = run_single(dag, engine, f"Cultivation (geo μ≈{cycle_time})", cultivation_src)
        cultivation_result["cycle_time"] = cycle_time
        all_results.append(cultivation_result)

        if cultivation_result["success"]:
            cultivation_steps.append(cultivation_result["num_timesteps"])
            logger.info(f"    → {cultivation_result['num_timesteps']} timesteps")
        else:
            cultivation_steps.append(None)

    # ---- Save results ----
    os.makedirs("routing_experiment_results", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = "cycle_sweep_ising_n66"

    json_path = f"routing_experiment_results/{tag}_{ts}.json"
    with open(json_path, "w") as f:
        json.dump({
            "parameters": {
                "circuit": "ising_n66",
                "num_qubits": circuit.num_qubits,
                "layout": template_top.name,
                "grid": report_grid,
                "magic_topology": "top_only",
                "num_magic_patches": len(template_top.magic_sites),
                "placement_cost": placement.cost,
                "scheduler": SCHEDULER_MODE,
                "cycle_range": [CYCLE_RANGE.start, CYCLE_RANGE.stop - 1],
                "cultivation_seed": CULTIVATION_SEED,
            },
            "results": all_results,
        }, f, indent=2, default=str)
    logger.info(f"Saved results to {json_path}")

    # ---- Plot ----
    plot_path = f"plots/{tag}_{ts}.png"
    create_line_plot(factory_steps, cultivation_steps, cycle_times, plot_path)

    # ---- Summary ----
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY — Timesteps by Cycle Time")
    logger.info(f"{'Cycle':>6s}  {'Factory':>10s}  {'Cultivation':>12s}")
    logger.info("-" * 34)
    for i, ct in enumerate(cycle_times):
        f_val = factory_steps[i] if factory_steps[i] is not None else "FAIL"
        c_val = cultivation_steps[i] if cultivation_steps[i] is not None else "FAIL"
        logger.info(f"{ct:>6d}  {f_val:>10}  {c_val:>12}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
