#!/usr/bin/env python3
"""
Compare magic-state **access topologies** on the 100-qubit QAOA circuit.

Fixes a single bus architecture (circuit-aware, single-spaced) and varies
only the number / placement of magic-state patches:

    1. All-Around Ring  – magic patches on all 4 sides (default)
    2. One-Side (Top)   – magic patches only on the top edge
    3. Fixed Count (8)  – exactly 8 magic patches evenly spaced on the top edge

Each topology is run with three schedulers × two magic-source modes:

    Schedulers:
        1. Sequential        (steiner_tree)
        2. Greedy Packing    (steiner_packing)
        3. Pathfinder        (steiner_pathfinder)

    Sources:
        1. Unlimited              – no preparation constraint (pure routing geometry)
        2. Factory (15 cyc)       – deterministic 15-cycle cooldown
        3. Cultivation (geo μ≈19) – stochastic geometric-distribution cooldown

Total: 3 topologies × 3 schedulers × 3 sources = 27 experiments.

Outputs:
    - JSON results  → routing_experiment_results/
    - Bar-chart PNG → plots/
"""

import argparse
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
from harvest.synthesis.emitter import emit_layout

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("MagicAccessComparison")

# ── Constants ─────────────────────────────────────────────────────────
FACTORY_PREP_CYCLES = 15
CULTIVATION_MEAN = 19
CULTIVATION_P = 1.0 / CULTIVATION_MEAN  # geometric mean ≈ 19 cycles
CULTIVATION_SEED = 42
NUM_FIXED_MAGIC = 8
FIXED_MAGIC_SIDE = "top"
SEED = 42


# ── Build layout engines ─────────────────────────────────────────────

def _filter_magic_one_side(template, side="top"):
    """Return a template copy with magic sites only on *side*."""
    t = copy.deepcopy(template)
    W, H = t.grid_width, t.grid_height
    if side == "top":
        t.magic_sites = [(x, y) for x, y in t.magic_sites if y == 0]
    elif side == "bottom":
        t.magic_sites = [(x, y) for x, y in t.magic_sites if y == H - 1]
    elif side == "left":
        t.magic_sites = [(x, y) for x, y in t.magic_sites if x == 0]
    elif side == "right":
        t.magic_sites = [(x, y) for x, y in t.magic_sites if x == W - 1]
    else:
        raise ValueError(f"Unknown side: {side!r}")
    return t


def _filter_magic_fixed_count(template, num_magic, side="top"):
    """Return a template copy with exactly *num_magic* magic sites evenly
    distributed along *side*."""
    t = _filter_magic_one_side(template, side)
    available = len(t.magic_sites)
    actual = min(num_magic, available)
    if actual < num_magic:
        logger.warning(
            "Requested %d magic patches on %s but only %d available; capping.",
            num_magic, side, available,
        )
    if actual <= 0:
        t.magic_sites = []
    elif actual == 1:
        t.magic_sites = [t.magic_sites[available // 2]]
    else:
        chosen_idx = [round(i * (available - 1) / (actual - 1)) for i in range(actual)]
        t.magic_sites = [t.magic_sites[ci] for ci in chosen_idx]
    return t


def build_engines(template, placement):
    """Build three LayoutEngines that share the *same* data-qubit positions
    but differ in magic-state access topology.

    Returns dict  {label: (engine, num_magic_patches)}
    """
    # 1. All-around ring (original template)
    eng_ring = emit_layout(template, placement)
    n_ring = len(template.magic_sites)

    # 2. One-side (top)
    t_one = _filter_magic_one_side(template, side=FIXED_MAGIC_SIDE)
    eng_one = emit_layout(t_one, placement)
    n_one = len(t_one.magic_sites)

    # 3. Fixed count
    t_fix = _filter_magic_fixed_count(template, NUM_FIXED_MAGIC, side=FIXED_MAGIC_SIDE)
    eng_fix = emit_layout(t_fix, placement)
    n_fix = len(t_fix.magic_sites)

    return {
        f"All-Around Ring ({n_ring})": eng_ring,
        f"One-Side Top ({n_one})": eng_one,
        f"Fixed Count ({n_fix})": eng_fix,
    }


# ── Single run ────────────────────────────────────────────────────────

def run_single(dag, layout_engine, scheduler_mode, scheduler_label,
               source_label, topology_label, magic_source=None):
    """Route *dag* on *layout_engine* and return a metrics dict."""
    logger.info("  %s | %s | %s", topology_label, scheduler_label, source_label)
    try:
        processor = DAGProcessor(layout_engine=layout_engine,
                                 magic_source=magic_source)
        results = processor.process_entire_dag(
            dag, visualize_each_step=False, mode=scheduler_mode,
        )

        meta = getattr(processor, "_scheduling_metadata", {})
        total_elapsed = meta.get("total_elapsed_steps", len(results))
        completed = meta.get("completed", True)
        nodes_completed = meta.get("num_nodes_completed", len(results))
        nodes_total = meta.get("num_nodes_total", len(results))

        total_wirelength = sum(
            len(r.get("steiner_edges", set())) for r in results
        )

        result = {
            "topology_label": topology_label,
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
        logger.error("    FAILED: %s", e)
        import traceback
        traceback.print_exc()
        return {
            "topology_label": topology_label,
            "source_label": source_label,
            "scheduler_label": scheduler_label,
            "scheduler_mode": scheduler_mode,
            "success": False,
            "error": str(e),
        }


# ── Plotting ──────────────────────────────────────────────────────────

def create_comparison_plot(results, output_path):
    """Grouped bar chart: timesteps, wirelength, magic wait cycles.

    Groups are (topology × source), x-axis is scheduler.
    """
    ok = [r for r in results if r["success"]]
    if not ok:
        logger.warning("No successful runs to plot.")
        return

    schedulers = sorted({r["scheduler_label"] for r in ok})
    group_labels = sorted(
        {(r["topology_label"], r["source_label"]) for r in ok}
    )

    lookup = {}
    for r in ok:
        lookup[(r["topology_label"], r["source_label"], r["scheduler_label"])] = r

    metrics = [
        ("num_timesteps", "Timesteps"),
        ("total_wirelength", "Total Wirelength"),
        ("magic_wait_cycles", "Magic Wait Cycles"),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(7 * len(metrics), 7))
    colors = plt.cm.tab10.colors

    for metric_idx, (metric, ylabel) in enumerate(metrics):
        ax = axes[metric_idx]
        x = np.arange(len(schedulers))
        n_groups = len(group_labels)
        width = 0.8 / max(n_groups, 1)

        for gi, (topo, src) in enumerate(group_labels):
            vals = []
            for sched in schedulers:
                r = lookup.get((topo, src, sched))
                vals.append(r.get(metric, 0) if r else 0)
            offset = width * (gi - (n_groups - 1) / 2)
            label = f"{topo} / {src}"
            bars = ax.bar(x + offset, vals, width, label=label,
                          color=colors[gi % len(colors)], alpha=0.85)
            for bar, v in zip(bars, vals):
                if v > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height(), str(int(v)),
                            ha="center", va="bottom", fontsize=6)

        ax.set_xticks(x)
        ax.set_xticklabels(schedulers, fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend(fontsize=6, loc="upper left")
        ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Derive a short title from the results
    circ_label = ok[0].get("_circuit_name", "")
    fig.suptitle(
        f"Magic-State Access Topology Comparison: {circ_label}\n"
        "(Circuit-Aware Layout, Single-Spaced Bus)",
        fontweight="bold", fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("Saved plot to %s", output_path)


# ── Default circuit path ──────────────────────────────────────────────

_DEFAULT_QASM = os.path.normpath(os.path.join(
    os.path.dirname(__file__), "..", "..", "benchmark_circuits", "qasm",
    "qaoa", "big_100q", "qaoa_barabasi_albert_N100_3reps.qasm",
))


# ── Main ──────────────────────────────────────────────────────────────

def main(circuit_path: str | None = None, circuit_name: str | None = None):
    # ── Resolve circuit ──
    qasm_path = os.path.normpath(circuit_path or _DEFAULT_QASM)
    if circuit_name is None:
        circuit_name = os.path.splitext(os.path.basename(qasm_path))[0]

    logger.info("Loading circuit '%s' from %s", circuit_name, qasm_path)
    circuit = qasm_to_circuit(qasm_path)
    circuit = pre_prep_circuit(circuit)
    logger.info("Loaded: %d qubits, depth %d, gates %s",
                circuit.num_qubits, circuit.depth(), dict(circuit.count_ops()))

    gate_counts = circuit.count_ops()
    if "rx" in gate_counts or "ry" in gate_counts:
        logger.info("Converting rx/ry gates to rz equivalents ...")
        circuit = convert_rx_ry_to_rz(circuit)

    logger.info("Converting to PCB format ...")
    pcb = convert_to_PCB(circuit, verbose=False)
    dag = create_dag(pcb)

    # ── Synthesize circuit-aware layout (template + placement) ──
    synth = StaticLayoutSynthesizer(
        placement_config=PlacementConfig(
            alpha=1.0, beta=0.5, max_swap_iterations=100, seed=SEED,
        ),
    )
    engine_default, report = synth.synthesize(dag)
    logger.info("Circuit-aware layout: %s (%dx%d), cost %.2f",
                report.template_name, report.grid_width, report.grid_height,
                report.placement_cost)

    # Re-obtain the template and placement so we can build variants
    from harvest.synthesis.circuit_summary import extract_circuit_summary
    from harvest.synthesis.templates import select_template
    from harvest.synthesis.placement import circuit_aware_placement

    summary = extract_circuit_summary(dag)
    template = select_template(
        n_qubits=summary.num_qubits,
        max_parallelism=summary.parallelism_profile.get("max_pauli_per_layer", 0),
    )
    placement = circuit_aware_placement(
        summary, template,
        PlacementConfig(alpha=1.0, beta=0.5, max_swap_iterations=100, seed=SEED),
    )

    # ── Build 3 layout engines ──
    engines = build_engines(template, placement)
    for label, eng in engines.items():
        temp_proc = DAGProcessor(layout_engine=eng)
        logger.info("  %-30s  magic terminals: %d", label, len(temp_proc.magic_terminals))

    # ── Define schedulers & sources ──
    schedulers = [
        ("Sequential", "steiner_tree"),
        ("Greedy Packing", "steiner_packing"),
        ("Pathfinder", "steiner_pathfinder"),
    ]
    source_labels = [
        "Unlimited",
        f"Factory ({FACTORY_PREP_CYCLES} cyc)",
        f"Cultivation (geo μ≈{CULTIVATION_MEAN})",
    ]

    def _make_source(label, magic_terminals):
        """Build a fresh magic-state source for the given label."""
        if label.startswith("Unlimited"):
            return None
        elif label.startswith("Factory"):
            return MagicStateFactory(list(magic_terminals), FACTORY_PREP_CYCLES)
        elif label.startswith("Cultivation"):
            return MagicStateCultivator(
                list(magic_terminals),
                readiness_sampler=geometric_sampler(CULTIVATION_P),
                seed=CULTIVATION_SEED,
            )
        else:
            raise ValueError(f"Unknown source label: {label!r}")

    # ── Run experiments ──
    all_results = []
    for topo_label, eng in engines.items():
        # discover magic terminals for this engine
        temp_proc = DAGProcessor(layout_engine=eng)
        magic_terminals = temp_proc.magic_terminals

        for sched_label, sched_mode in schedulers:
            for src_label in source_labels:
                magic_source = _make_source(src_label, magic_terminals)

                result = run_single(
                    dag, eng, sched_mode, sched_label,
                    src_label, topo_label, magic_source=magic_source,
                )
                result["_circuit_name"] = circuit_name
                all_results.append(result)

    # ── Save results ──
    os.makedirs("routing_experiment_results", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = circuit_name.replace(" ", "_").replace("/", "_")
    tag = f"magic_access_comparison_{safe_name}"

    json_path = f"routing_experiment_results/{tag}_{ts}.json"
    with open(json_path, "w") as f:
        json.dump(
            {
                "parameters": {
                    "circuit": circuit_name,
                    "num_qubits": summary.num_qubits,
                    "layout": report.template_name,
                    "grid": f"{report.grid_width}x{report.grid_height}",
                    "placement_cost": report.placement_cost,
                    "factory_prep_cycles": FACTORY_PREP_CYCLES,
                    "cultivation_mean": CULTIVATION_MEAN,
                    "cultivation_p": CULTIVATION_P,
                    "cultivation_seed": CULTIVATION_SEED,
                    "num_fixed_magic": NUM_FIXED_MAGIC,
                    "fixed_magic_side": FIXED_MAGIC_SIDE,
                    "topologies": {k: len(DAGProcessor(layout_engine=v).magic_terminals)
                                   for k, v in engines.items()},
                },
                "results": all_results,
            },
            f,
            indent=2,
            default=str,
        )
    logger.info("Saved results to %s", json_path)

    plot_path = f"plots/{tag}_{ts}.png"
    create_comparison_plot(all_results, plot_path)

    # ── Summary table ──
    logger.info("\n" + "=" * 90)
    logger.info("SUMMARY")
    logger.info("=" * 90)
    for r in all_results:
        if r["success"]:
            wait = r.get("magic_wait_cycles", 0)
            wait_str = f", wait {wait:5d}" if wait else ""
            done = "" if r.get("completed", True) else (
                f" [INCOMPLETE {r.get('num_nodes_processed', 0)}"
                f"/{r.get('num_nodes_total', 0)} nodes]"
            )
            logger.info(
                "  %-30s | %-16s | %-22s : %5d steps, wl %5d%s%s",
                r["topology_label"], r["scheduler_label"],
                r["source_label"], r["num_timesteps"],
                r["total_wirelength"], wait_str, done,
            )
        else:
            logger.info(
                "  %-30s | %-16s | %-22s : FAILED — %s",
                r["topology_label"], r["scheduler_label"],
                r["source_label"], r.get("error", ""),
            )
    logger.info("=" * 90)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Magic-state access topology comparison")
    parser.add_argument("--circuit", type=str, default=None,
                        help="Path to a .qasm circuit file (default: QAOA 100q)")
    parser.add_argument("--name", type=str, default=None,
                        help="Short circuit name for labels/filenames")
    args = parser.parse_args()
    main(circuit_path=args.circuit, circuit_name=args.name)
