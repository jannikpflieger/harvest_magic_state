#!/usr/bin/env python3
"""
Active volume comparison for QAOA 100-qubit benchmark.

Active volume = Σ_t |{ grid cells active at time step t }|
             = for each time step t: count of (x,y) tuple nodes
               in the union of all steiner_nodes routed at step t.

Scheduler fixed to: Steiner Packing (Greedy)

Three layouts × two placements → six bars as a grouped bar chart
with one group per layout, two bars comparing placement strategies.

Layouts (all for 100 qubits):
  1. nxm_ring_layout_single_qubits(10, 10)               — "Single Spacing"
  2. nxm_ring_layout_single_qubits_large_spacing(10, 10) — "Double Spacing"
  3. blocks_of_four_qubit_patches(5, 5)                  — "4 Blocks"

Placement strategies (applied to each layout's grid):
  - "Row-Major"      : sequential assignment q_0->site_0, q_1->site_1, ...
  - "Circuit-Aware"  : greedy + swap placement minimising interaction-graph
                       weighted distance cost
"""

import json
import logging
import os
from datetime import datetime
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from harvest.compilation.pauli_block_conversion import convert_to_PCB, create_dag
from harvest.compilation.qasm_loader import qasm_to_circuit
from harvest.compilation.circuit_analysis import convert_rx_ry_to_rz, pre_prep_circuit
from harvest.layout.engine import LayoutEngine
from harvest.layout.presets import (
    nxm_ring_layout_single_qubits,
    nxm_ring_layout_single_qubits_large_spacing,
    blocks_of_four_qubit_patches,
)
from harvest.routing.processor import DAGProcessor
from harvest.synthesis.circuit_summary import extract_circuit_summary
from harvest.synthesis.templates import LayoutTemplate
from harvest.synthesis.placement import (
    PlacementConfig,
    circuit_aware_placement,
    baseline_placement,
)
from harvest.synthesis.emitter import emit_layout
from harvest.evaluation.metrics import compute_active_volume

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("ActiveVolume")

Coord = Tuple[int, int]

# ---------------------------------------------------------------------------
# Template extraction from a preset LayoutEngine
# ---------------------------------------------------------------------------

def template_from_layout_engine(eng: LayoutEngine, n_qubits: int,
                                 template_name: str) -> LayoutTemplate:
    """Build a LayoutTemplate from an existing preset LayoutEngine."""
    # Data sites in qubit order q_0 ... q_{n-1}
    data_sites: List[Coord] = []
    for i in range(n_qubits):
        patch = eng.patches[f"q_{i}"]
        coord = next(iter(patch.cells))
        data_sites.append(coord)

    # Magic sites
    magic_sites: List[Coord] = []
    for name, patch in eng.patches.items():
        if patch.kind == "magic":
            coord = next(iter(patch.cells))
            magic_sites.append(coord)

    # Pairwise Manhattan distances + centrality
    n = n_qubits
    distances = {}
    total_dist = {i: 0.0 for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            d = abs(data_sites[i][0] - data_sites[j][0]) + \
                abs(data_sites[i][1] - data_sites[j][1])
            distances[(i, j)] = float(d)
            total_dist[i] += d
            total_dist[j] += d

    centrality = {}
    for i in range(n):
        avg_d = total_dist[i] / max(n - 1, 1)
        centrality[i] = 1.0 / max(avg_d, 1e-6)

    return LayoutTemplate(
        name=template_name,
        grid_width=eng.W,
        grid_height=eng.H,
        data_sites=data_sites,
        magic_sites=magic_sites,
        routing_lanes=1,
        pairwise_distances=distances,
        site_centrality=centrality,
    )


# ---------------------------------------------------------------------------
# Single-run helper
# ---------------------------------------------------------------------------

def run_single(dag, layout_engine: LayoutEngine, scheduler_mode: str,
               layout_label: str, placement_label: str) -> dict:
    """Route *dag* on *layout_engine* with *scheduler_mode*."""
    logger.info(f"  [{layout_label}] + [{placement_label}]")
    try:
        processor = DAGProcessor(layout_engine=layout_engine)
        results = processor.process_entire_dag(dag, visualize_each_step=False,
                                               mode=scheduler_mode)
        meta = getattr(processor, "_scheduling_metadata", {})
        num_timesteps = meta.get("total_elapsed_steps", len(results))
        nodes_completed = meta.get("num_nodes_completed", len(results))
        nodes_total = meta.get("num_nodes_total", len(results))
        completed = meta.get("completed", True)
        active_vol = compute_active_volume(results, scheduler_mode)
        total_wirelength = sum(len(r.get("steiner_edges", set())) for r in results)
        return {
            "layout_label": layout_label,
            "placement_label": placement_label,
            "scheduler_mode": scheduler_mode,
            "active_volume": active_vol,
            "num_timesteps": num_timesteps,
            "total_wirelength": total_wirelength,
            "num_nodes_processed": nodes_completed,
            "num_nodes_total": nodes_total,
            "completed": completed,
            "success": True,
        }
    except Exception as exc:
        logger.error(f"    FAILED: {exc}", exc_info=True)
        return {
            "layout_label": layout_label,
            "placement_label": placement_label,
            "scheduler_mode": scheduler_mode,
            "success": False,
            "error": str(exc),
        }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

PLACEMENT_STYLES = {
    "Row-Major":     {"color": "#4C72B0"},
    "Circuit-Aware": {"color": "#DD8452"},
}


def create_active_volume_plot(all_results: list, output_path: str) -> None:
    """Grouped bar chart: active volume per (layout, placement) pair."""
    successful = [r for r in all_results if r["success"]]
    if not successful:
        logger.error("No successful results — cannot create plot.")
        return

    layout_labels = ["Single Spacing", "Double Spacing", "4 Blocks"]
    placement_labels = ["Row-Major", "Circuit-Aware"]
    lookup = {(r["layout_label"], r["placement_label"]): r for r in successful}

    x = np.arange(len(layout_labels))
    n_bars = len(placement_labels)
    bar_width = 0.35

    all_vals = [
        lookup.get((l, p), {}).get("active_volume", 0)
        for l in layout_labels for p in placement_labels
    ]
    y_max = max(all_vals) if all_vals else 1

    fig, ax = plt.subplots(figsize=(9, 5))

    for bar_idx, placement_label in enumerate(placement_labels):
        style = PLACEMENT_STYLES[placement_label]
        offset = (bar_idx - (n_bars - 1) / 2) * bar_width
        values = [
            lookup.get((layout, placement_label), {}).get("active_volume", 0)
            for layout in layout_labels
        ]
        bars = ax.bar(
            x + offset, values, bar_width,
            label=placement_label,
            color=style["color"],
            alpha=0.88,
            edgecolor="white",
        )
        layout_ts = [
            lookup.get((layout, placement_label), {}).get("num_timesteps")
            for layout in layout_labels
        ]
        for bar, val, ts in zip(bars, values, layout_ts):
            if val > 0:
                # Active volume above bar
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + y_max * 0.01,
                    f"{val:,}",
                    ha="center", va="bottom",
                    fontsize=8, fontweight="bold",
                )
                # Timestep inside bar
                if ts is not None:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() * 0.5,
                        f"T={ts}",
                        ha="center", va="center",
                        fontsize=7, color="white", fontweight="bold",
                    )

    ax.set_xticks(x)
    ax.set_xticklabels(layout_labels, fontsize=11)
    ax.set_ylabel("Active Volume", fontsize=11)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved plot -> {output_path}")


def create_timestep_plot(all_results: list, output_path: str) -> None:
    """Grouped bar chart: number of timesteps per (layout, placement) pair."""
    successful = [r for r in all_results if r["success"]]
    if not successful:
        logger.error("No successful results — cannot create plot.")
        return

    layout_labels = ["Single Spacing", "Double Spacing", "4 Blocks"]
    placement_labels = ["Row-Major", "Circuit-Aware"]
    lookup = {(r["layout_label"], r["placement_label"]): r for r in successful}

    x = np.arange(len(layout_labels))
    n_bars = len(placement_labels)
    bar_width = 0.35

    all_vals = [
        lookup.get((l, p), {}).get("num_timesteps", 0)
        for l in layout_labels for p in placement_labels
    ]
    y_max = max(all_vals) if all_vals else 1

    fig, ax = plt.subplots(figsize=(9, 5))

    for bar_idx, placement_label in enumerate(placement_labels):
        style = PLACEMENT_STYLES[placement_label]
        offset = (bar_idx - (n_bars - 1) / 2) * bar_width
        values = [
            lookup.get((layout, placement_label), {}).get("num_timesteps", 0)
            for layout in layout_labels
        ]
        bars = ax.bar(
            x + offset, values, bar_width,
            label=placement_label,
            color=style["color"],
            alpha=0.88,
            edgecolor="white",
        )
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + y_max * 0.01,
                    f"{val:,}",
                    ha="center", va="bottom",
                    fontsize=8, fontweight="bold",
                )

    ax.set_xticks(x)
    ax.set_xticklabels(layout_labels, fontsize=11)
    ax.set_ylabel("Timesteps", fontsize=11)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved plot -> {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    N_QUBITS = 100
    SCHEDULER_MODE = "steiner_packing"

    # ------------------------------------------------------------------
    # 1. Load & preprocess QAOA 100-qubit circuit
    # ------------------------------------------------------------------
    qasm_path = os.path.normpath(
        os.path.join(
            os.path.dirname(__file__),
            "..", "..",
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

    # Extract circuit summary once for circuit-aware placement
    summary = extract_circuit_summary(dag)

    # ------------------------------------------------------------------
    # 2. Preset layout geometries for 100 qubits
    # ------------------------------------------------------------------
    preset_engines = [
        ("Single Spacing", nxm_ring_layout_single_qubits(10, 10)),
        ("Double Spacing", nxm_ring_layout_single_qubits_large_spacing(10, 10)),
        ("4 Blocks",       blocks_of_four_qubit_patches(5, 5)),
    ]

    placement_config = PlacementConfig(
        alpha=1.0, beta=0.5, max_swap_iterations=200, seed=42
    )

    # ------------------------------------------------------------------
    # 3. For each layout: build both placement variants then route
    # ------------------------------------------------------------------
    all_results = []

    for layout_label, preset_eng in preset_engines:
        logger.info(f"\n{'='*60}")
        logger.info(f"Layout: {layout_label}  (grid {preset_eng.W}x{preset_eng.H})")
        logger.info(f"{'='*60}")

        template = template_from_layout_engine(
            preset_eng, N_QUBITS, template_name=layout_label
        )
        logger.info(
            f"  Template: {len(template.data_sites)} data sites, "
            f"{len(template.magic_sites)} magic sites"
        )

        # Row-Major (identical to the preset's default ordering)
        logger.info("  Building Row-Major layout ...")
        placement_rm = baseline_placement(template, N_QUBITS, mode="row_major")
        engine_rm = emit_layout(template, placement_rm)
        logger.info(f"  Row-Major placement cost: {placement_rm.cost:.2f}")
        all_results.append(run_single(dag, engine_rm, SCHEDULER_MODE,
                                      layout_label, "Row-Major"))

        # Circuit-Aware
        logger.info("  Building Circuit-Aware layout ...")
        placement_ca = circuit_aware_placement(summary, template, placement_config)
        engine_ca = emit_layout(template, placement_ca)
        logger.info(f"  Circuit-Aware placement cost: {placement_ca.cost:.2f}")
        all_results.append(run_single(dag, engine_ca, SCHEDULER_MODE,
                                      layout_label, "Circuit-Aware"))

    # ------------------------------------------------------------------
    # 4. Save JSON
    # ------------------------------------------------------------------
    os.makedirs("routing_experiment_results", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = f"routing_experiment_results/active_volume_qaoa_100q_{ts}.json"
    with open(json_path, "w") as fh:
        json.dump(
            {
                "parameters": {
                    "circuit": "qaoa_barabasi_albert_N100_3reps",
                    "num_qubits": N_QUBITS,
                    "scheduler": SCHEDULER_MODE,
                    "factory": "Unlimited",
                },
                "results": all_results,
            },
            fh,
            indent=2,
        )
    logger.info(f"\nSaved results -> {json_path}")

    # ------------------------------------------------------------------
    # 5. Summary table
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 75)
    logger.info("SUMMARY  (scheduler: steiner_packing)")
    logger.info("=" * 75)
    for r in all_results:
        if r["success"]:
            done = "" if r.get("completed", True) else (
                f" [INCOMPLETE {r.get('num_nodes_processed',0)}"
                f"/{r.get('num_nodes_total',0)}]"
            )
            logger.info(
                f"  {r['layout_label']:20s} | {r['placement_label']:16s} | "
                f"active_vol={r['active_volume']:8,d} | "
                f"T={r['num_timesteps']:5d} | "
                f"wirelength={r['total_wirelength']:6d}{done}"
            )
        else:
            logger.info(
                f"  {r['layout_label']:20s} | {r['placement_label']:16s} | "
                f"FAILED: {r.get('error','')}"
            )
    logger.info("=" * 75)

    # ------------------------------------------------------------------
    # 6. Plot
    # ------------------------------------------------------------------
    os.makedirs("plots", exist_ok=True)
    plot_path = f"plots/active_volume_qaoa_100q_{ts}.pdf"
    create_active_volume_plot(all_results, plot_path)

    timestep_plot_path = f"plots/timesteps_qaoa_100q_{ts}.pdf"
    create_timestep_plot(all_results, timestep_plot_path)

    logger.info("\nDone.")


if __name__ == "__main__":
    main()
