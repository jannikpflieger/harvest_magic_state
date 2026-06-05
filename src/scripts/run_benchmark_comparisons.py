#!/usr/bin/env python3
"""
Batch benchmark comparison script.

Loads all circuits matching the plot_benchmark_landscape filters
(≤70 qubits, depth ≤2000, exclude qv, successful DAG analysis,
total_pauli_evolutions > 0) and runs both:

    1. Factory vs Cultivation comparison (9 experiments per circuit):
       3 schedulers × 3 magic-state sources

    2. Magic-state access topology comparison (27 experiments per circuit):
       3 topologies × 3 schedulers × 3 sources

Outputs:
    - Per-circuit JSON  → routing_experiment_results/benchmark_comparison/
    - Significance summary JSON (>10% differences) → same directory

Usage:
    python scripts/run_benchmark_comparisons.py
    python scripts/run_benchmark_comparisons.py --force          # re-run existing
    python scripts/run_benchmark_comparisons.py --max-qubits 30  # smaller subset
"""

import argparse
import copy
import json
import logging
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Resolve project root
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # .../harvest_magic_state
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from harvest.compilation.pauli_block_conversion import convert_to_PCB, create_dag
from harvest.compilation.qasm_loader import qasm_to_circuit
from harvest.compilation.circuit_analysis import convert_rx_ry_to_rz, pre_prep_circuit
from harvest.routing.processor import DAGProcessor
from harvest.routing.magic_state_factory import MagicStateFactory
from harvest.routing.magic_state_cultivator import MagicStateCultivator, geometric_sampler
from harvest.synthesis.synthesizer import StaticLayoutSynthesizer
from harvest.synthesis.placement import PlacementConfig, circuit_aware_placement
from harvest.synthesis.emitter import emit_layout
from harvest.synthesis.circuit_summary import extract_circuit_summary
from harvest.synthesis.templates import select_template

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("BenchmarkComparisons")

# ---------------------------------------------------------------------------
# Constants (matching existing comparison scripts)
# ---------------------------------------------------------------------------
DEFAULT_ANALYSIS_DIR = PROJECT_ROOT / "benchmark_circuits" / "circuit_analysis_results"
OUTPUT_DIR = PROJECT_ROOT / "routing_experiment_results" / "benchmark_comparison"
PLOT_DIR = OUTPUT_DIR / "plots"

FACTORY_PREP_CYCLES = 15
CULTIVATION_MEAN = 19
CULTIVATION_P = 1.0 / CULTIVATION_MEAN
CULTIVATION_SEED = 42
NUM_FIXED_MAGIC = 8
FIXED_MAGIC_SIDE = "top"
SEED = 42

MAX_QUBITS = 100
MAX_DEPTH = 2000
EXCLUDE_FAMILIES = {"qv", "chemical"}

SCHEDULERS = [
    ("Sequential", "steiner_tree"),
    ("Greedy Packing", "steiner_packing"),
    ("Pathfinder", "steiner_pathfinder"),
]

SOURCE_LABELS = [
    "Unlimited",
    f"Factory ({FACTORY_PREP_CYCLES} cyc)",
    f"Cultivation (geo μ≈{CULTIVATION_MEAN})",
]

SIGNIFICANCE_THRESHOLD = 0.10  # 10%


# ---------------------------------------------------------------------------
# Circuit discovery from analysis JSONs
# ---------------------------------------------------------------------------

def load_eligible_circuits(analysis_dir, max_qubits=MAX_QUBITS,
                           max_depth=MAX_DEPTH,
                           exclude_families=EXCLUDE_FAMILIES):
    """Load circuit metadata from analysis JSONs, applying landscape filters.

    Returns a list of dicts, each containing the full circuit analysis data
    plus the resolved QASM file path.
    """
    records = []
    analysis_path = Path(analysis_dir)

    for jf in sorted(analysis_path.glob("*.json")):
        try:
            data = json.loads(jf.read_text())
        except (json.JSONDecodeError, OSError):
            continue

        circ = data.get("circuit", {})

        # Skip transpiled duplicates
        fpath = circ.get("file_path", "")
        if "_transpiled" in fpath:
            continue

        if circ.get("dag_analysis_successful") is not True:
            continue

        t_count = circ.get("total_pauli_evolutions")
        t_depth = circ.get("num_layers")
        n_qubits = circ.get("num_qubits")

        if not all(v is not None and v > 0 for v in [t_count, t_depth, n_qubits]):
            continue

        if max_qubits is not None and n_qubits > max_qubits:
            continue

        orig_depth = circ.get("depth")
        if max_depth is not None and orig_depth is not None and orig_depth > max_depth:
            continue

        # Derive family from file path
        fpath_parts = Path(fpath).parts
        family = fpath_parts[2] if len(fpath_parts) > 2 else "unknown"
        if exclude_families and family in exclude_families:
            continue

        # Resolve absolute QASM path
        qasm_abs = PROJECT_ROOT / fpath
        if not qasm_abs.exists():
            logger.warning("QASM file not found: %s (skipping)", qasm_abs)
            continue

        evol_per_layer = circ.get("pauli_evolutions_per_layer", [])
        avg_pp = sum(evol_per_layer) / len(evol_per_layer) if evol_per_layer else 0.0

        records.append({
            "circuit_name": circ.get("circuit_name", jf.stem),
            "family": family,
            "num_qubits": n_qubits,
            "depth": orig_depth,
            "t_count": t_count,
            "t_depth": t_depth,
            "avg_pp_per_layer": avg_pp,
            "max_pauli_per_layer": circ.get("max_pauli_per_layer"),
            "median_pauli_per_layer": circ.get("median_pauli_per_layer"),
            "non_clifford_gates": circ.get("non_clifford_gates"),
            "gate_counts": circ.get("gate_counts", {}),
            "qasm_path": str(qasm_abs),
        })

    logger.info("Discovered %d eligible circuits", len(records))
    return records


# ---------------------------------------------------------------------------
# Magic-access topology helpers (from compare_magic_access.py)
# ---------------------------------------------------------------------------

def _filter_magic_one_side(template, side="top"):
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
    t = _filter_magic_one_side(template, side)
    available = len(t.magic_sites)
    actual = min(num_magic, available)
    if actual <= 0:
        t.magic_sites = []
    elif actual == 1:
        t.magic_sites = [t.magic_sites[available // 2]]
    else:
        chosen_idx = [round(i * (available - 1) / (actual - 1)) for i in range(actual)]
        t.magic_sites = [t.magic_sites[ci] for ci in chosen_idx]
    return t


def build_topology_engines(template, placement):
    """Build three LayoutEngines with different magic-access topologies."""
    eng_ring = emit_layout(template, placement)
    n_ring = len(template.magic_sites)

    t_one = _filter_magic_one_side(template, side=FIXED_MAGIC_SIDE)
    eng_one = emit_layout(t_one, placement)
    n_one = len(t_one.magic_sites)

    t_fix = _filter_magic_fixed_count(template, NUM_FIXED_MAGIC, side=FIXED_MAGIC_SIDE)
    eng_fix = emit_layout(t_fix, placement)
    n_fix = len(t_fix.magic_sites)

    return {
        f"All-Around Ring ({n_ring})": eng_ring,
        f"One-Side Top ({n_one})": eng_one,
        f"Fixed Count ({n_fix})": eng_fix,
    }


# ---------------------------------------------------------------------------
# Magic source builder
# ---------------------------------------------------------------------------

def make_source(label, magic_terminals):
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


# ---------------------------------------------------------------------------
# Single experiment run
# ---------------------------------------------------------------------------

def run_single(dag, layout_engine, scheduler_mode, scheduler_label,
               source_label, magic_source=None, topology_label=None):
    """Route *dag* on *layout_engine* and return a metrics dict."""
    tag = f"{topology_label} | " if topology_label else ""
    logger.info("    %s%s + %s", tag, scheduler_label, source_label)
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

        if topology_label:
            result["topology_label"] = topology_label

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
        traceback.print_exc()
        result = {
            "source_label": source_label,
            "scheduler_label": scheduler_label,
            "scheduler_mode": scheduler_mode,
            "success": False,
            "error": str(e),
        }
        if topology_label:
            result["topology_label"] = topology_label
        return result


# ---------------------------------------------------------------------------
# Per-circuit pipeline
# ---------------------------------------------------------------------------

def process_circuit(record):
    """Run all comparisons for a single circuit.

    Returns the full result dict to be written as JSON, or None on failure.
    """
    circuit_name = record["circuit_name"]
    qasm_path = record["qasm_path"]

    logger.info("Loading circuit '%s' from %s", circuit_name, qasm_path)
    circuit = qasm_to_circuit(qasm_path)
    circuit = pre_prep_circuit(circuit)

    gate_counts = circuit.count_ops()
    if "rx" in gate_counts or "ry" in gate_counts:
        circuit = convert_rx_ry_to_rz(circuit)

    pcb = convert_to_PCB(circuit, verbose=False)
    dag = create_dag(pcb)

    # ── Synthesize circuit-aware layout ──
    synth = StaticLayoutSynthesizer(
        placement_config=PlacementConfig(
            alpha=1.0, beta=0.5, max_swap_iterations=100, seed=SEED,
        ),
    )
    engine, report = synth.synthesize(dag)
    logger.info("  Layout: %s (%dx%d), cost %.2f",
                report.template_name, report.grid_width, report.grid_height,
                report.placement_cost)

    layout_info = {
        "template_name": report.template_name,
        "grid_width": report.grid_width,
        "grid_height": report.grid_height,
        "placement_cost": report.placement_cost,
    }

    # Discover magic terminals from the default engine
    temp_proc = DAGProcessor(layout_engine=engine)
    magic_terminals = temp_proc.magic_terminals

    # ──────────────────────────────────────────────────────────────────
    # 1. Factory vs Cultivation (9 experiments)
    # ──────────────────────────────────────────────────────────────────
    logger.info("  Running Factory vs Cultivation (9 experiments) ...")
    fvc_results = []
    for sched_label, sched_mode in SCHEDULERS:
        for src_label in SOURCE_LABELS:
            source = make_source(src_label, magic_terminals)
            result = run_single(dag, engine, sched_mode, sched_label,
                                src_label, magic_source=source)
            fvc_results.append(result)

    # ──────────────────────────────────────────────────────────────────
    # 2. Magic Access Topology comparison (27 experiments)
    # ──────────────────────────────────────────────────────────────────
    logger.info("  Running Magic Access Topology comparison (27 experiments) ...")

    # Re-obtain template and placement for topology variants
    summary = extract_circuit_summary(dag)
    template = select_template(
        n_qubits=summary.num_qubits,
        max_parallelism=summary.parallelism_profile.get("max_pauli_per_layer", 0),
    )
    placement = circuit_aware_placement(
        summary, template,
        PlacementConfig(alpha=1.0, beta=0.5, max_swap_iterations=100, seed=SEED),
    )

    engines = build_topology_engines(template, placement)
    topology_info = {}
    for topo_label, eng in engines.items():
        tp = DAGProcessor(layout_engine=eng)
        topology_info[topo_label] = len(tp.magic_terminals)

    ma_results = []
    for topo_label, eng in engines.items():
        tp = DAGProcessor(layout_engine=eng)
        topo_magic_terminals = tp.magic_terminals
        for sched_label, sched_mode in SCHEDULERS:
            for src_label in SOURCE_LABELS:
                source = make_source(src_label, topo_magic_terminals)
                result = run_single(dag, eng, sched_mode, sched_label,
                                    src_label, magic_source=source,
                                    topology_label=topo_label)
                ma_results.append(result)

    # ── Build output ──
    circuit_metadata = {
        "circuit_name": circuit_name,
        "family": record["family"],
        "num_qubits": record["num_qubits"],
        "depth": record["depth"],
        "t_count": record["t_count"],
        "t_depth": record["t_depth"],
        "avg_pp_per_layer": record["avg_pp_per_layer"],
        "max_pauli_per_layer": record.get("max_pauli_per_layer"),
        "median_pauli_per_layer": record.get("median_pauli_per_layer"),
        "non_clifford_gates": record.get("non_clifford_gates"),
        "gate_counts": record.get("gate_counts", {}),
    }

    return {
        "circuit_metadata": circuit_metadata,
        "layout_info": layout_info,
        "topology_info": topology_info,
        "parameters": {
            "factory_prep_cycles": FACTORY_PREP_CYCLES,
            "cultivation_mean": CULTIVATION_MEAN,
            "cultivation_p": CULTIVATION_P,
            "cultivation_seed": CULTIVATION_SEED,
            "num_fixed_magic": NUM_FIXED_MAGIC,
            "fixed_magic_side": FIXED_MAGIC_SIDE,
            "seed": SEED,
        },
        "factory_vs_cultivation_results": fvc_results,
        "magic_access_results": ma_results,
        "timestamp": datetime.now().isoformat(),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def create_fvc_plot(results, circuit_name, output_path):
    """Grouped bar chart for factory-vs-cultivation results."""
    ok = [r for r in results if r.get("success")]
    if not ok:
        return

    schedulers = sorted({r["scheduler_label"] for r in ok})
    source_labels = sorted({r["source_label"] for r in ok})
    lookup = {(r["scheduler_label"], r["source_label"]): r for r in ok}

    metrics = [
        ("num_timesteps", "Timesteps"),
        ("total_wirelength", "Total Wirelength"),
        ("magic_wait_cycles", "Magic Wait Cycles"),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(7 * len(metrics), 6))
    colors = plt.cm.Set2.colors

    for mi, (metric, ylabel) in enumerate(metrics):
        ax = axes[mi]
        x = np.arange(len(schedulers))
        width = 0.8 / max(len(source_labels), 1)
        for i, src in enumerate(source_labels):
            vals = [lookup.get((s, src), {}).get(metric, 0) for s in schedulers]
            offset = width * (i - (len(source_labels) - 1) / 2)
            bars = ax.bar(x + offset, vals, width, label=src,
                          color=colors[i % len(colors)], alpha=0.85)
            for bar, v in zip(bars, vals):
                if v > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height(), str(int(v)),
                            ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(schedulers, fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3, linestyle="--")

    fig.suptitle(f"Factory vs Cultivation: {circuit_name}",
                 fontweight="bold", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Plot → %s", output_path)


def create_ma_plot(results, circuit_name, output_path):
    """Grouped bar chart for magic-access topology results."""
    ok = [r for r in results if r.get("success")]
    if not ok:
        return

    schedulers = sorted({r["scheduler_label"] for r in ok})
    group_labels = sorted(
        {(r["topology_label"], r["source_label"]) for r in ok}
    )
    lookup = {(r["topology_label"], r["source_label"], r["scheduler_label"]): r
              for r in ok}

    metrics = [
        ("num_timesteps", "Timesteps"),
        ("total_wirelength", "Total Wirelength"),
        ("magic_wait_cycles", "Magic Wait Cycles"),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(7 * len(metrics), 7))
    colors = plt.cm.tab10.colors

    for mi, (metric, ylabel) in enumerate(metrics):
        ax = axes[mi]
        x = np.arange(len(schedulers))
        n_groups = len(group_labels)
        width = 0.8 / max(n_groups, 1)
        for gi, (topo, src) in enumerate(group_labels):
            vals = [lookup.get((topo, src, s), {}).get(metric, 0)
                    for s in schedulers]
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

    fig.suptitle(f"Magic-State Access Topology: {circuit_name}",
                 fontweight="bold", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Plot → %s", output_path)


# ---------------------------------------------------------------------------
# Significance analysis
# ---------------------------------------------------------------------------

def _pct_diff(a, b):
    """Symmetric percentage difference between two values."""
    base = max(abs(a), abs(b))
    if base == 0:
        return 0.0
    return abs(a - b) / base


def analyse_significance(all_circuit_results):
    """Find all pairwise >10% metric differences across scheduling dimensions.

    Compares:
      - Schedulers pairwise (within same source, and within same topology+source)
      - Sources pairwise (within same scheduler, and within same topology+scheduler)
      - Topologies pairwise (within same scheduler+source) [magic_access only]

    Returns a list of flagged differences.
    """
    metrics = ["num_timesteps", "total_wirelength", "magic_wait_cycles"]
    flags = []

    for circ_data in all_circuit_results:
        cname = circ_data["circuit_metadata"]["circuit_name"]

        # -- Factory vs Cultivation results --
        fvc = [r for r in circ_data["factory_vs_cultivation_results"] if r.get("success")]
        # Compare schedulers within each source
        for src in SOURCE_LABELS:
            group = [r for r in fvc if r["source_label"] == src]
            _pairwise_compare(flags, group, "scheduler_label",
                              metrics, cname, "factory_vs_cultivation",
                              context_key="source_label")
        # Compare sources within each scheduler
        for sched_label, _ in SCHEDULERS:
            group = [r for r in fvc if r["scheduler_label"] == sched_label]
            _pairwise_compare(flags, group, "source_label",
                              metrics, cname, "factory_vs_cultivation",
                              context_key="scheduler_label")

        # -- Magic Access results --
        ma = [r for r in circ_data["magic_access_results"] if r.get("success")]
        topo_labels = sorted({r["topology_label"] for r in ma})

        # Compare schedulers within each (topology, source)
        for topo in topo_labels:
            for src in SOURCE_LABELS:
                group = [r for r in ma
                         if r["topology_label"] == topo and r["source_label"] == src]
                _pairwise_compare(flags, group, "scheduler_label",
                                  metrics, cname, "magic_access",
                                  context_key="topology+source",
                                  context_val=f"{topo} / {src}")

        # Compare sources within each (topology, scheduler)
        for topo in topo_labels:
            for sched_label, _ in SCHEDULERS:
                group = [r for r in ma
                         if r["topology_label"] == topo and r["scheduler_label"] == sched_label]
                _pairwise_compare(flags, group, "source_label",
                                  metrics, cname, "magic_access",
                                  context_key="topology+scheduler",
                                  context_val=f"{topo} / {sched_label}")

        # Compare topologies within each (scheduler, source)
        for sched_label, _ in SCHEDULERS:
            for src in SOURCE_LABELS:
                group = [r for r in ma
                         if r["scheduler_label"] == sched_label and r["source_label"] == src]
                _pairwise_compare(flags, group, "topology_label",
                                  metrics, cname, "magic_access",
                                  context_key="scheduler+source",
                                  context_val=f"{sched_label} / {src}")

    return flags


def _pairwise_compare(flags, group, vary_key, metrics, circuit_name,
                      experiment_type, context_key=None, context_val=None):
    """Pairwise comparison of results in *group* along *vary_key*."""
    if len(group) < 2:
        return
    for i in range(len(group)):
        for j in range(i + 1, len(group)):
            a, b = group[i], group[j]
            for metric in metrics:
                va = a.get(metric, 0)
                vb = b.get(metric, 0)
                if va == 0 and vb == 0:
                    continue
                pct = _pct_diff(va, vb)
                if pct >= SIGNIFICANCE_THRESHOLD:
                    entry = {
                        "circuit_name": circuit_name,
                        "experiment_type": experiment_type,
                        "compared_dimension": vary_key,
                        "metric": metric,
                        "a_label": a[vary_key],
                        "a_value": va,
                        "b_label": b[vary_key],
                        "b_value": vb,
                        "pct_diff": round(pct * 100, 2),
                    }
                    if context_key:
                        entry["context"] = context_val or a.get(context_key, "")
                    flags.append(entry)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Batch factory/cultivation & magic-access comparison "
                    "for benchmark circuits.",
    )
    parser.add_argument(
        "--analysis-dir", type=str, default=str(DEFAULT_ANALYSIS_DIR),
        help="Directory with circuit analysis JSON files.",
    )
    parser.add_argument(
        "--max-qubits", type=int, default=MAX_QUBITS,
        help=f"Max qubit count (default: {MAX_QUBITS}). 0 to disable.",
    )
    parser.add_argument(
        "--max-depth", type=int, default=MAX_DEPTH,
        help=f"Max circuit depth (default: {MAX_DEPTH}). 0 to disable.",
    )
    parser.add_argument(
        "--exclude-families", type=str, nargs="*", default=list(EXCLUDE_FAMILIES),
        help="Circuit families to exclude (default: qv).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run circuits that already have output JSON.",
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(OUTPUT_DIR),
        help="Directory for per-circuit result JSONs.",
    )
    args = parser.parse_args()

    max_q = args.max_qubits if args.max_qubits > 0 else None
    max_d = args.max_depth if args.max_depth > 0 else None
    excl = set(args.exclude_families) if args.exclude_families else None
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # ── Phase 1: Discover circuits ──
    records = load_eligible_circuits(args.analysis_dir, max_qubits=max_q,
                                     max_depth=max_d, exclude_families=excl)
    if not records:
        logger.error("No eligible circuits found. Exiting.")
        return

    # ── Phase 2: Process each circuit ──
    all_results = []
    total = len(records)
    for idx, record in enumerate(records, 1):
        cname = record["circuit_name"]
        out_file = out_dir / f"{cname}.json"
        fvc_plot = plot_dir / f"{cname}_fvc.png"
        ma_plot = plot_dir / f"{cname}_magic_access.png"

        if (out_file.exists() and fvc_plot.exists() and ma_plot.exists()
                and not args.force):
            logger.info("[%d/%d] %s — already exists, skipping (use --force to re-run)",
                        idx, total, cname)
            try:
                all_results.append(json.loads(out_file.read_text()))
            except (json.JSONDecodeError, OSError):
                pass
            continue

        # If JSON exists but plots are missing, reload and just generate plots
        if out_file.exists() and not args.force:
            logger.info("[%d/%d] %s — JSON exists, generating missing plots ...",
                        idx, total, cname)
            try:
                result = json.loads(out_file.read_text())
                if not fvc_plot.exists():
                    create_fvc_plot(result["factory_vs_cultivation_results"],
                                   cname, str(fvc_plot))
                if not ma_plot.exists():
                    create_ma_plot(result["magic_access_results"],
                                  cname, str(ma_plot))
                all_results.append(result)
            except (json.JSONDecodeError, OSError, KeyError) as e:
                logger.warning("  Could not reload %s: %s", cname, e)
            continue

        logger.info("[%d/%d] Processing %s (%d qubits, depth %s) ...",
                    idx, total, cname, record["num_qubits"], record.get("depth", "?"))
        try:
            result = process_circuit(record)
            if result is None:
                continue

            with open(out_file, "w") as f:
                json.dump(result, f, indent=2, default=str)
            logger.info("  Saved → %s", out_file)

            create_fvc_plot(result["factory_vs_cultivation_results"],
                           cname, str(fvc_plot))
            create_ma_plot(result["magic_access_results"],
                          cname, str(ma_plot))

            all_results.append(result)
        except Exception as e:
            logger.error("  FAILED on %s: %s", cname, e)
            traceback.print_exc()
            continue

    # ── Phase 3: Significance analysis ──
    if not all_results:
        logger.warning("No results to analyse for significance.")
        return

    logger.info("Running significance analysis (threshold >%.0f%%) ...",
                SIGNIFICANCE_THRESHOLD * 100)
    flags = analyse_significance(all_results)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = out_dir / f"significance_summary_{ts}.json"
    summary = {
        "generated": datetime.now().isoformat(),
        "threshold_pct": SIGNIFICANCE_THRESHOLD * 100,
        "num_circuits_analysed": len(all_results),
        "num_flags": len(flags),
        "flags": flags,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Significance summary (%d flags) → %s", len(flags), summary_path)

    # Print top-level stats
    if flags:
        by_circuit = {}
        for fl in flags:
            by_circuit.setdefault(fl["circuit_name"], []).append(fl)
        logger.info("Circuits with significant differences: %d / %d",
                    len(by_circuit), len(all_results))
        for cname, cflags in sorted(by_circuit.items()):
            logger.info("  %-40s  %d flags", cname, len(cflags))
    else:
        logger.info("No significant differences (>%.0f%%) found.",
                    SIGNIFICANCE_THRESHOLD * 100)


if __name__ == "__main__":
    main()
