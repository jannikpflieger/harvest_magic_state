#!/usr/bin/env python3
"""
RQ1: How does HARVEST scale as circuit size and Pauli-based computation
complexity increase?

For a set of benchmark (QASM) or randomly generated circuits the script runs
a *fixed* scheduler (default: steiner_packing) and collects:

  Circuit-level metrics (from the QASM/circuit analysis stage):
    num_qubits, original_depth, total_gates, gate_counts, non_clifford_gates,
    pcb_conversion_successful, pcb_depth, pcb_total_gates,
    total_pauli_evolutions, num_dag_layers, max_pauli_per_layer,
    median_pauli_per_layer, avg_pauli_evolution_size, max_pauli_evolution_size

  Schedule-level metrics (from DAGProcessor / scheduler):
    num_time_steps, total_wirelength, avg_wirelength_per_net, max_wirelength,
    successful_operations, failed_operations, success_rate,
    avg_operations_per_timestep, max_operations_per_timestep,
    total_runtime_ms, magic_terminal_utilization, total_wait_cycles

  Derived metrics:
    active_routing_volume_proxy  = total_wirelength
    space_time_volume_proxy      = num_time_steps * layout_area
    timestep_per_pauli           = num_time_steps / total_pauli_evolutions
    wirelength_per_pauli         = total_wirelength / total_pauli_evolutions
    parallelism_efficiency_proxy = total_pauli_evolutions / num_time_steps
    routing_overhead_factor      = num_time_steps / num_dag_layers

Outputs (all in --output-dir):
    rq1_raw_results.json
    rq1_summary.csv
    rq1_failed_circuits.csv
    rq1_timesteps_vs_pauli_ops.png
    rq1_timesteps_vs_num_qubits.png
    rq1_wirelength_vs_pauli_ops.png
    rq1_space_time_proxy_vs_pauli_ops.png
    rq1_scheduler_runtime_vs_pauli_ops.png
    rq1_routing_overhead_vs_dag_layers.png
    rq1_parallelism_efficiency_vs_dag_width.png  (optional)

Usage examples
--------------
Single circuit:
    cd src
    python scripts/evaluate_rq1_scalability.py \\
        --qasm ../benchmark_circuits/qasm/qaoa/example.qasm \\
        --rows 10 --cols 10 \\
        --output-dir ../results/rq1_test

Batch:
    cd src
    python scripts/evaluate_rq1_scalability.py \\
        --benchmark-dir ../benchmark_circuits/qasm \\
        --max-files 50 \\
        --scheduler steiner_packing \\
        --output-dir ../results/rq1_benchmark_scalability

Random circuits:
    cd src
    python scripts/evaluate_rq1_scalability.py \\
        --random \\
        --qubits 10,20,40,60,80,100 \\
        --depths 10,25,50,100 \\
        --runs 5 \\
        --scheduler steiner_packing \\
        --output-dir ../results/rq1_random_scalability
"""

import argparse
import csv
import json
import logging
import math
import os
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from harvest.compilation.circuit_analysis import (
    analyze_dag_layers,
    convert_rx_ry_to_rz,
    count_non_clifford_gates,
    pre_prep_circuit,
)
from harvest.compilation.pauli_block_conversion import (
    convert_to_PCB,
    create_dag,
    create_random_circuit,
)
from harvest.compilation.qasm_loader import find_qasm_files, qasm_to_circuit
from harvest.layout.presets import (
    blocks_of_four_qubit_patches,
    nxm_ring_layout_single_qubits,
    nxm_ring_layout_single_qubits_large_spacing,
)
from harvest.routing.magic_state_factory import MagicStateFactory
from harvest.routing.processor import DAGProcessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("RQ1")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LAYOUT_PRESET_MAP = {
    "single_spacing": nxm_ring_layout_single_qubits,
    "double_spacing": nxm_ring_layout_single_qubits_large_spacing,
    "blocks_of_four": blocks_of_four_qubit_patches,
}

# Spec-aligned CSV columns — one row per circuit instance.
ALL_CSV_KEYS = [
    "circuit_id",
    "circuit_family",
    "num_qubits",
    "original_depth",
    "total_gates",
    "non_clifford_gates",
    "pcb_depth",
    "pcb_total_gates",
    "total_pauli_evolutions",
    "num_dag_layers",
    "max_pauli_per_layer",
    "avg_pauli_evolution_size",
    "scheduler",
    "layout_rows",
    "layout_cols",
    "layout_area_proxy",
    "num_time_steps",
    "total_wirelength",
    "avg_wirelength_per_net",
    "success_rate",
    "avg_operations_per_timestep",
    "total_runtime_ms",
    "active_routing_volume_proxy",
    "space_time_volume_proxy",
    "timestep_per_pauli",
    "wirelength_per_pauli",
    "parallelism_efficiency_proxy",
    "routing_overhead_factor",
]

FAILED_CSV_KEYS = [
    "circuit_id",
    "circuit_family",
    "num_qubits",
    "error_phase",
    "error_message",
]

# Categorical colour palette for circuit families
_FAMILY_PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2",
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RQ1: HARVEST scalability with circuit size and PBC complexity.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Circuit sources
    p.add_argument(
        "--mode", choices=["qasm", "random"], default=None,
        help="Circuit source mode: 'qasm' (read files) or 'random' (generate).",
    )
    p.add_argument(
        "--benchmark-dir", metavar="PATH",
        help="Directory of QASM files to process in batch mode.",
    )
    p.add_argument(
        "--qasm", metavar="PATH",
        help="Path to a single QASM input file.",
    )
    p.add_argument(
        "--max-files", metavar="INT", type=int, default=None,
        help="Maximum number of QASM files to process from --benchmark-dir.",
    )
    p.add_argument(
        "--random", action="store_true",
        help="Alias for --mode random (kept for backwards compatibility).",
    )
    p.add_argument(
        "--qubits", metavar="LIST", default="10,20,40,60,80,100",
        help="Comma-separated qubit counts for random experiments.",
    )
    p.add_argument(
        "--depths", metavar="LIST", default="10,25,50,100",
        help="Comma-separated circuit depths for random experiments.",
    )
    p.add_argument(
        "--runs", metavar="INT", type=int, default=3,
        help="Number of random runs per qubit/depth setting.",
    )

    # Layout
    p.add_argument(
        "--rows", metavar="INT", type=int, default=None,
        help="Layout rows. Auto-sized from qubit count if omitted.",
    )
    p.add_argument(
        "--cols", metavar="INT", type=int, default=None,
        help="Layout columns. Auto-sized from qubit count if omitted.",
    )
    p.add_argument(
        "--auto-layout", action="store_true",
        help="Auto-size layout from circuit qubit count (default when --rows/--cols omitted).",
    )
    p.add_argument(
        "--layout", choices=list(LAYOUT_PRESET_MAP.keys()),
        default="single_spacing",
        help="Layout preset to use.",
    )

    # Scheduler
    p.add_argument(
        "--scheduler",
        choices=["steiner_tree", "steiner_packing", "steiner_pathfinder"],
        default="steiner_packing",
        help="Scheduler mode (fixed for all circuits).",
    )

    # Magic states
    p.add_argument(
        "--magic-source", choices=["unlimited", "factory"],
        default="unlimited",
        help="Magic-state source model.",
    )
    p.add_argument(
        "--magic-prep-cycles", metavar="INT", type=int, default=15,
        help="Preparation cycles per terminal (only used with --magic-source factory).",
    )

    # Output
    p.add_argument(
        "--output-dir", metavar="PATH",
        default="results/rq1_scalability",
        help="Directory for all output files.",
    )

    # Reproducibility
    p.add_argument(
        "--seed", metavar="INT", type=int, default=None,
        help="Base random seed for reproducibility.",
    )

    return p.parse_args()


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def parse_int_list(s: str) -> List[int]:
    """Parse a comma-separated string of integers into a list."""
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def auto_layout_size(num_qubits: int) -> Tuple[int, int]:
    """Return (rows, cols) = ceil(sqrt(n)) x ceil(sqrt(n)), minimum (1, 1)."""
    side = max(1, math.ceil(math.sqrt(num_qubits)))
    return side, side


def _is_random_mode(args) -> bool:
    return args.random or args.mode == "random"


def _circuit_family_from_path(path: str) -> str:
    """Derive a circuit family label from the QASM file's parent directory."""
    parent = Path(path).parent.name
    return parent if parent else "qasm"


# ---------------------------------------------------------------------------
# Circuit loading / generation
# ---------------------------------------------------------------------------

def load_or_generate_circuits(args) -> List[Dict]:
    """
    Return a list of circuit descriptor dicts:
        {
            'name':     str,          # human-readable circuit name
            'source':   str,          # 'qasm' | 'random'
            'path':     str | None,   # QASM path (None for random)
            'circuit':  QuantumCircuit | None,  # pre-built circuit (random)
            'num_qubits': int | None, # known qubit count (random)
        }
    """
    entries: List[Dict] = []

    if _is_random_mode(args):
        qubit_counts = parse_int_list(args.qubits)
        depths = parse_int_list(args.depths)
        base_seed = args.seed  # may be None

        for n in qubit_counts:
            for d in depths:
                for run_idx in range(args.runs):
                    seed = None
                    if base_seed is not None:
                        seed = base_seed + n * 10000 + d * 100 + run_idx
                    name = f"random_n{n}_d{d}_r{run_idx}"
                    logger.debug(f"  Generating {name}  (seed={seed})")
                    try:
                        circuit = create_random_circuit(n, d, seed=seed)
                        entries.append({
                            "name":           name,
                            "circuit_family": "random",
                            "source":         "random",
                            "path":           None,
                            "circuit":        circuit,
                            "num_qubits":     n,
                        })
                    except Exception as exc:
                        logger.warning(f"  Could not generate {name}: {exc}")
        logger.info(f"Generated {len(entries)} random circuits.")
        return entries

    # --- QASM mode ---
    qasm_files: List[str] = []
    if args.qasm:
        qasm_files.append(args.qasm)
    if args.benchmark_dir:
        found = sorted(find_qasm_files(args.benchmark_dir))
        if args.max_files is not None:
            found = found[: args.max_files]
        qasm_files.extend(found)
        logger.info(
            f"Found {len(found)} QASM files in {args.benchmark_dir}"
            + (f" (capped at {args.max_files})" if args.max_files else "")
        )

    for path in qasm_files:
        entries.append({
            "name":           Path(path).stem,
            "circuit_family": _circuit_family_from_path(path),
            "source":         "qasm",
            "path":           path,
            "circuit":        None,
            "num_qubits":     None,
        })

    return entries


# ---------------------------------------------------------------------------
# Circuit analysis
# ---------------------------------------------------------------------------

def analyze_circuit_for_rq1(entry: Dict) -> Tuple[Dict, object]:
    """
    Load/prepare a circuit and extract circuit-level metrics.

    Parameters
    ----------
    entry : dict
        A descriptor from load_or_generate_circuits().

    Returns
    -------
    metrics : dict   All circuit-level metrics.
    dag      : DAGCircuit  Ready for scheduling.

    Raises
    ------
    Any exception propagates to the caller, which records it as a failure.
    """
    name = entry["name"]
    source = entry["source"]

    if source == "qasm":
        logger.info(f"  Loading QASM: {entry['path']}")
        circuit = qasm_to_circuit(entry["path"])
    else:
        circuit = entry["circuit"]

    # Pre-processing
    circuit = pre_prep_circuit(circuit)
    gate_counts = dict(circuit.count_ops())
    if "rx" in gate_counts or "ry" in gate_counts:
        logger.info("  Converting rx/ry gates to rz equivalents …")
        circuit = convert_rx_ry_to_rz(circuit)
        gate_counts = dict(circuit.count_ops())

    num_qubits = circuit.num_qubits
    original_depth = circuit.depth()
    total_gates = sum(gate_counts.values())
    non_clifford_gates = count_non_clifford_gates(circuit)

    logger.info(
        f"  {num_qubits} qubits, depth {original_depth}, "
        f"gates {total_gates}, non-Clifford {non_clifford_gates}"
    )

    # PCB conversion
    logger.info("  Converting to PCB format …")
    pcb = convert_to_PCB(circuit, verbose=False)
    pcb_gate_counts = dict(pcb.count_ops())
    pcb_depth = pcb.depth()
    pcb_total_gates = sum(pcb_gate_counts.values())
    pcb_conversion_successful = True

    # DAG construction and layer analysis
    logger.info("  Building DAG …")
    dag = create_dag(pcb)
    dag_stats = analyze_dag_layers(dag)

    num_dag_layers = dag_stats["num_layers"]
    total_pauli_evolutions = dag_stats["total_pauli_evolutions"]
    max_pauli_per_layer = dag_stats["max_pauli_per_layer"]
    median_pauli_per_layer = dag_stats["median_pauli_per_layer"]
    avg_pauli_evolution_size = dag_stats["avg_pauli_evolution_size"]
    max_pauli_evolution_size = dag_stats["max_pauli_evolution_size"]

    logger.info(
        f"  DAG: {num_dag_layers} layers, "
        f"{total_pauli_evolutions} Pauli evolutions total"
    )

    metrics = {
        # Identification
        "circuit_id":                name,
        "circuit_name":              name,
        "source":                    source,
        "num_qubits":                num_qubits,
        "original_depth":            original_depth,
        "total_gates":               total_gates,
        "gate_counts":               json.dumps(gate_counts),  # CSV-safe
        "non_clifford_gates":        non_clifford_gates,
        "pcb_conversion_successful": pcb_conversion_successful,
        "pcb_depth":                 pcb_depth,
        "pcb_total_gates":           pcb_total_gates,
        "total_pauli_evolutions":    total_pauli_evolutions,
        "num_dag_layers":            num_dag_layers,
        "max_pauli_per_layer":       max_pauli_per_layer,
        "median_pauli_per_layer":    median_pauli_per_layer,
        "avg_pauli_evolution_size":  round(avg_pauli_evolution_size, 4),
        "max_pauli_evolution_size":  max_pauli_evolution_size,
    }
    return metrics, dag


# ---------------------------------------------------------------------------
# Layout construction
# ---------------------------------------------------------------------------

def build_layout_engine(rows: int, cols: int, layout_preset: str):
    """Return a LayoutEngine for the given grid dimensions and preset."""
    preset_fn = LAYOUT_PRESET_MAP[layout_preset]
    logger.info(f"  Building layout: {layout_preset}  ({rows}×{cols})")
    return preset_fn(rows, cols)


# ---------------------------------------------------------------------------
# Magic source construction
# ---------------------------------------------------------------------------

def make_magic_source(layout_engine, magic_source_mode: str, prep_cycles: int):
    """
    Return a magic source object (or None for unlimited).

    For "unlimited" → None (DAGProcessor treats None as unlimited).
    For "factory"   → MagicStateFactory probed from the layout's magic terminals.
    """
    if magic_source_mode == "unlimited":
        return None

    probe = DAGProcessor(layout_engine=layout_engine)
    magic_terminals = list(probe.magic_terminals)
    logger.info(f"  Magic terminals probed: {len(magic_terminals)}")
    return MagicStateFactory(magic_terminals, prep_cycles)


# ---------------------------------------------------------------------------
# Scheduler execution
# ---------------------------------------------------------------------------

def run_scheduler_for_rq1(dag, processor: DAGProcessor, mode: str) -> Dict:
    """
    Route *dag* using *processor* in the given *mode*.  Returns a metrics dict.
    """
    logger.info(f"  Running scheduler: {mode} …")
    t_start = time.perf_counter()
    results = processor.process_entire_dag(dag, visualize_each_step=False, mode=mode)
    elapsed_ms = (time.perf_counter() - t_start) * 1000.0

    # Scheduling metadata (populated by scheduler functions)
    meta = getattr(processor, "_scheduling_metadata", {})
    num_time_steps  = meta.get("total_elapsed_steps", len(results))
    nodes_completed = meta.get("num_nodes_completed", len(results))
    nodes_total     = meta.get("num_nodes_total",     len(results))

    # Per-result aggregation
    total_wirelength = 0
    wirelengths: List[int] = []
    ops_per_step: Dict[int, int] = defaultdict(int)
    total_wait_cycles = 0

    for idx, r in enumerate(results):
        wl = len(r.get("steiner_edges", set()))
        total_wirelength += wl
        if wl > 0:
            wirelengths.append(wl)
        step = r.get("time_step", idx)
        ops_per_step[step] += 1
        total_wait_cycles += r.get("magic_wait_cycles", 0)

    n_ops = len(results)
    avg_wl = total_wirelength / max(n_ops, 1)
    max_wl = max(wirelengths) if wirelengths else 0
    avg_ops = n_ops / max(num_time_steps, 1)
    max_ops = max(ops_per_step.values()) if ops_per_step else 0

    failed_ops = nodes_total - nodes_completed
    success_rate = nodes_completed / max(nodes_total, 1)

    used_terminals  = len(processor.used_magic_terminals)
    total_terminals = len(processor.magic_terminals)
    magic_util = used_terminals / max(total_terminals, 1)

    metrics = {
        "num_time_steps":              num_time_steps,
        "total_wirelength":            total_wirelength,
        "avg_wirelength_per_net":      round(avg_wl, 4),
        "max_wirelength":              max_wl,
        "successful_operations":       nodes_completed,
        "failed_operations":           failed_ops,
        "success_rate":                round(success_rate, 6),
        "avg_operations_per_timestep": round(avg_ops, 4),
        "max_operations_per_timestep": max_ops,
        "total_runtime_ms":            round(elapsed_ms, 2),
        "magic_terminal_utilization":  round(magic_util, 6),
        "total_wait_cycles":           total_wait_cycles,
    }

    logger.info(
        f"    ✓  T={num_time_steps}  WL={total_wirelength}"
        f"  ops/step={avg_ops:.2f}  runtime={elapsed_ms:.0f} ms"
    )
    return metrics


# ---------------------------------------------------------------------------
# Derived metrics
# ---------------------------------------------------------------------------

def compute_derived_metrics(
    circuit_m: Dict,
    schedule_m: Dict,
    rows: int,
    cols: int,
    eng,
) -> Dict:
    """Compute layout dimensions and scalability-focused derived metrics."""
    layout_area_proxy = eng.W * eng.H

    num_time_steps        = schedule_m["num_time_steps"]
    total_wirelength      = schedule_m["total_wirelength"]
    total_pauli_evols     = circuit_m["total_pauli_evolutions"]
    num_dag_layers        = circuit_m["num_dag_layers"]

    def safe_div(a, b):
        return round(a / b, 6) if b > 0 else None

    return {
        "layout_rows":                   rows,
        "layout_cols":                   cols,
        "layout_area_proxy":             layout_area_proxy,
        # Named proxy fields
        "active_routing_volume_proxy":   total_wirelength,
        "space_time_volume_proxy":       num_time_steps * layout_area_proxy,
        # Normalised scalability metrics
        "timestep_per_pauli":            safe_div(num_time_steps,    total_pauli_evols),
        "wirelength_per_pauli":          safe_div(total_wirelength,   total_pauli_evols),
        "parallelism_efficiency_proxy":  safe_div(total_pauli_evols,  num_time_steps),
        "routing_overhead_factor":       safe_div(num_time_steps,     num_dag_layers),
    }


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def save_json(data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2, default=str)
    logger.info(f"  Saved JSON → {path}")


def save_results(
    all_results: List[Dict],
    failed_results: List[Dict],
    output_dir: Path,
    args_dict: Dict,
) -> None:
    """Write JSON, summary CSV, and failed-circuits CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # rq1_raw_results.json
    save_json(
        {"parameters": args_dict, "results": all_results},
        output_dir / "rq1_raw_results.json",
    )

    # rq1_summary.csv
    csv_path = output_dir / "rq1_summary.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=ALL_CSV_KEYS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_results)
    logger.info(f"  Saved CSV  → {csv_path}  ({len(all_results)} rows)")

    # rq1_failed_circuits.csv
    if failed_results:
        failed_path = output_dir / "rq1_failed_circuits.csv"
        with open(failed_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=FAILED_CSV_KEYS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(failed_results)
        logger.info(f"  Saved failed CSV → {failed_path}  ({len(failed_results)} rows)")
    else:
        logger.info("  No failed circuits.")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _family_colors(families: List[str]) -> Dict[str, str]:
    """Map each unique family label to a consistent colour from the palette."""
    unique = sorted(set(families))
    return {f: _FAMILY_PALETTE[i % len(_FAMILY_PALETTE)] for i, f in enumerate(unique)}


def _scatter_colored(
    ax,
    results: List[Dict],
    x_key: str,
    y_key: str,
    xlabel: str,
    ylabel: str,
    title: str,
    log_log: bool = False,
    ref_line_y_eq_x: bool = False,
) -> None:
    """
    One scatter point per circuit, coloured by circuit_family.
    Adds a combined linear trend line over all valid points.
    Optionally adds a y = x reference line.
    """
    families = [r.get("circuit_family", "?") for r in results]
    color_map = _family_colors(families)

    by_family: Dict[str, Tuple[List, List]] = defaultdict(lambda: ([], []))
    all_xs: List[float] = []
    all_ys: List[float] = []

    for r in results:
        x = r.get(x_key)
        y = r.get(y_key)
        fam = r.get("circuit_family", "?")
        if x is None or y is None:
            continue
        try:
            xf, yf = float(x), float(y)
        except (TypeError, ValueError):
            continue
        if math.isnan(xf) or math.isnan(yf):
            continue
        if log_log and (xf <= 0 or yf <= 0):
            continue
        by_family[fam][0].append(xf)
        by_family[fam][1].append(yf)
        all_xs.append(xf)
        all_ys.append(yf)

    if not all_xs:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", fontsize=12, color="gray")
        ax.set_title(title, fontsize=11, fontweight="bold")
        return

    # Plot each family as a distinct series
    for fam in sorted(by_family):
        xs, ys = by_family[fam]
        ax.scatter(
            xs, ys,
            s=45, alpha=0.80,
            color=color_map[fam],
            edgecolors="white", linewidth=0.4,
            label=fam, zorder=3,
        )

    # y = x reference line (DAG layers is a lower-bound proxy for timesteps)
    if ref_line_y_eq_x:
        lo = min(min(all_xs), min(all_ys))
        hi = max(max(all_xs), max(all_ys))
        ref = np.linspace(lo, hi, 200)
        ax.plot(ref, ref, color="black", linewidth=1.2, linestyle=":",
                alpha=0.55, label="y = x  (ideal lower bound)", zorder=2)

    # Combined linear trend line over all points
    if len(all_xs) >= 2:
        try:
            coeffs = np.polyfit(all_xs, all_ys, 1)
            x_range = np.linspace(min(all_xs), max(all_xs), 200)
            ax.plot(x_range, np.polyval(coeffs, x_range),
                    color="#888888", linewidth=1.5, linestyle="--",
                    alpha=0.8, label=f"linear fit (slope={coeffs[0]:.2e})", zorder=2)
        except (np.linalg.LinAlgError, ValueError):
            pass

    # Log-log (power-law) trend line
    if log_log and len(all_xs) >= 2 and all(v > 0 for v in all_xs) and all(v > 0 for v in all_ys):
        try:
            log_x = np.log(all_xs)
            log_y = np.log(all_ys)
            coeffs_ll = np.polyfit(log_x, log_y, 1)
            x_range = np.linspace(min(all_xs), max(all_xs), 200)
            ax.plot(
                x_range,
                np.exp(np.polyval(coeffs_ll, np.log(x_range))),
                color="#55A868", linewidth=1.4, linestyle=":",
                alpha=0.8, label=f"power-law fit (exp={coeffs_ll[0]:.2f})", zorder=2,
            )
        except (np.linalg.LinAlgError, ValueError):
            pass

    ax.legend(fontsize=8, loc="upper left", framealpha=0.85)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)


def create_plots(all_results: List[Dict], output_dir: Path) -> None:
    """Generate all RQ1 scatter plots -- one point per circuit, coloured by family."""
    if not all_results:
        logger.warning("No results to plot.")
        return

    # Each entry: (x_key, y_key, xlabel, ylabel, filename, log_log, ref_y_eq_x)
    plots = [
        (
            "total_pauli_evolutions",
            "num_time_steps",
            "Total Pauli Evolutions",
            "Logical Timesteps",
            "rq1_timesteps_vs_pauli_evolutions.png",
            True,
            False,
        ),
        (
            "total_pauli_evolutions",
            "total_wirelength",
            "Total Pauli Evolutions",
            "Total Wirelength (routing edges)",
            "rq1_wirelength_vs_pauli_evolutions.png",
            True,
            False,
        ),
        (
            "total_pauli_evolutions",
            "space_time_volume_proxy",
            "Total Pauli Evolutions",
            "Space-Time Volume Proxy\n(timesteps x layout area)",
            "rq1_space_time_proxy_vs_pauli_evolutions.png",
            True,
            False,
        ),
        (
            "num_dag_layers",
            "num_time_steps",
            "DAG Layers",
            "Logical Timesteps",
            "rq1_timesteps_vs_dag_layers.png",
            False,
            True,   # y = x reference line (DAG layers = ideal lower bound)
        ),
        (
            "total_pauli_evolutions",
            "routing_overhead_factor",
            "Total Pauli Evolutions",
            "Routing Overhead Factor\n(timesteps / DAG layers)",
            "rq1_routing_overhead_factor.png",
            False,
            False,
        ),
        (
            "total_pauli_evolutions",
            "total_runtime_ms",
            "Total Pauli Evolutions",
            "Scheduler Runtime (ms)",
            "rq1_scheduler_runtime_vs_pauli_evolutions.png",
            True,
            False,
        ),
    ]

    for x_key, y_key, xlabel, ylabel, fname, log_log, ref_y_eq_x in plots:
        stem = fname.replace("rq1_", "RQ1 -- ").replace(".png", "").replace("_", " ")
        fig, ax = plt.subplots(figsize=(7, 5))
        _scatter_colored(
            ax, all_results,
            x_key=x_key, y_key=y_key,
            xlabel=xlabel, ylabel=ylabel,
            title=stem,
            log_log=log_log,
            ref_line_y_eq_x=ref_y_eq_x,
        )
        plt.tight_layout()
        out = output_dir / fname
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=200, bbox_inches="tight")
        plt.close()
        logger.info(f"  Saved plot -> {out}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    use_random = _is_random_mode(args)
    if not use_random and not args.qasm and not args.benchmark_dir:
        raise SystemExit(
            "Error: provide at least one of --mode random, --random, "
            "--qasm, or --benchmark-dir."
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger.info(
        f"\nRQ1 Scalability Evaluation"
        f"\n  Mode:         {'random' if use_random else 'qasm'}"
        f"\n  Scheduler:    {args.scheduler}"
        f"\n  Layout:       {args.layout}"
        f"\n  Magic source: {args.magic_source}"
        f"\n  Output dir:   {output_dir}"
        f"\n  Timestamp:    {ts}\n"
    )

    # ------------------------------------------------------------------
    # Collect circuit descriptors
    # ------------------------------------------------------------------
    circuit_inputs = load_or_generate_circuits(args)
    if not circuit_inputs:
        raise SystemExit("Error: no circuits found or generated.")

    logger.info(f"Processing {len(circuit_inputs)} circuit(s) ...\n")

    # ------------------------------------------------------------------
    # Per-circuit loop
    # ------------------------------------------------------------------
    all_results: List[Dict] = []
    failed_results: List[Dict] = []

    for entry in circuit_inputs:
        circuit_id     = entry["name"]
        circuit_family = entry["circuit_family"]
        source         = entry["source"]

        logger.info(f"\n{'='*65}")
        logger.info(f"Circuit: {circuit_id}  [{circuit_family}]")
        logger.info(f"{'='*65}")

        # ---- Step 1: Circuit analysis ----
        try:
            circuit_metrics, dag = analyze_circuit_for_rq1(entry)
        except Exception as exc:
            logger.error(f"  Analysis FAILED: {exc}", exc_info=True)
            failed_results.append({
                "circuit_id":     circuit_id,
                "circuit_family": circuit_family,
                "num_qubits":     entry.get("num_qubits", "?"),
                "error_phase":    "analysis",
                "error_message":  str(exc),
            })
            continue

        num_qubits = circuit_metrics["num_qubits"]
        total_pauli_evols = circuit_metrics["total_pauli_evolutions"]

        # Skip degenerate circuits with no Pauli evolutions
        if total_pauli_evols == 0:
            logger.warning(
                f"  Skipping {circuit_id}: 0 Pauli evolutions after PCB conversion."
            )
            failed_results.append({
                "circuit_id":     circuit_id,
                "circuit_family": circuit_family,
                "num_qubits":     num_qubits,
                "error_phase":    "analysis",
                "error_message":  "0 Pauli evolutions after PCB conversion",
            })
            continue

        # ---- Step 2: Layout ----
        if args.rows is not None and args.cols is not None:
            rows, cols = args.rows, args.cols
        else:
            rows, cols = auto_layout_size(num_qubits)
            # Ensure layout has enough cells for all qubits
            while rows * cols < num_qubits:
                cols += 1
            logger.info(f"  Auto-sized layout: {rows}x{cols}")

        try:
            layout_engine = build_layout_engine(rows, cols, args.layout)
        except Exception as exc:
            logger.error(f"  Layout build FAILED: {exc}", exc_info=True)
            failed_results.append({
                "circuit_id":     circuit_id,
                "circuit_family": circuit_family,
                "num_qubits":     num_qubits,
                "error_phase":    "layout",
                "error_message":  str(exc),
            })
            continue

        # ---- Step 3: Magic source ----
        try:
            magic_source = make_magic_source(
                layout_engine, args.magic_source, args.magic_prep_cycles
            )
        except Exception as exc:
            logger.error(f"  Magic source build FAILED: {exc}", exc_info=True)
            failed_results.append({
                "circuit_id":     circuit_id,
                "circuit_family": circuit_family,
                "num_qubits":     num_qubits,
                "error_phase":    "magic_source",
                "error_message":  str(exc),
            })
            continue

        # ---- Step 4: Scheduling ----
        try:
            processor = DAGProcessor(
                layout_engine=layout_engine,
                magic_source=magic_source,
            )
            schedule_metrics = run_scheduler_for_rq1(dag, processor, args.scheduler)
        except Exception as exc:
            logger.error(f"  Scheduling FAILED: {exc}", exc_info=True)
            failed_results.append({
                "circuit_id":     circuit_id,
                "circuit_family": circuit_family,
                "num_qubits":     num_qubits,
                "error_phase":    "scheduling",
                "error_message":  str(exc),
            })
            continue

        # ---- Step 5: Derived metrics ----
        derived = compute_derived_metrics(
            circuit_metrics, schedule_metrics, rows, cols, layout_engine
        )

        # ---- Step 6: Flatten to a single result row ----
        row: Dict = {}
        row.update(circuit_metrics)
        row["circuit_family"] = circuit_family
        row["scheduler"]      = args.scheduler
        row.update(derived)          # layout_rows, layout_cols, layout_area_proxy, ...
        row.update(schedule_metrics)

        all_results.append(row)

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    args_dict = {
        "mode":             "random" if use_random else "qasm",
        "scheduler":        args.scheduler,
        "layout":           args.layout,
        "magic_source":     args.magic_source,
        "magic_prep_cycles": args.magic_prep_cycles,
        "seed":             args.seed,
        "timestamp":        ts,
        "num_circuits_attempted": len(circuit_inputs),
        "num_circuits_succeeded": len(all_results),
        "num_circuits_failed":    len(failed_results),
    }
    save_results(all_results, failed_results, output_dir, args_dict)

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    create_plots(all_results, output_dir)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    logger.info(
        f"\n{'='*65}"
        f"\nRQ1 complete."
        f"\n  Circuits processed:  {len(all_results)}"
        f"\n  Circuits failed:     {len(failed_results)}"
        f"\n  Output directory:    {output_dir}"
        f"\n{'='*65}"
    )


if __name__ == "__main__":
    main()
