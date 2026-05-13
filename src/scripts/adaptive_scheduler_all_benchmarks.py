#!/usr/bin/env python3
"""
Adaptive scheduler benchmark sweep — all circuits.

Runs the same 4-scheduler comparison (Sequential / Greedy Packing / Pathfinder /
Adaptive) on every benchmark circuit and produces per-circuit motivation plots
identical in style to the QAOA-100 figure.

Outputs per run::

    routing_experiment_results/benchmark_sweep_<timestamp>/<circuit_name>.json
    routing_experiment_results/benchmark_sweep_<timestamp>/summary.json
    plots/benchmark_sweep_<timestamp>/<circuit_name>_motivation.{png,pdf}

Reproduce with::

    cd /path/to/harvest_magic_state
    PYTHONPATH=src python src/scripts/adaptive_scheduler_all_benchmarks.py

Optional flags::

    --qasm-dir PATH           Root of .qasm benchmark tree (default: benchmark_circuits/qasm)
    --max-qubits N            Skip circuits with more than N logical qubits (default: 100)
    --exclude-families F …    Space-separated family names to skip (default: qv)
    --scheduler-timeout S     Per-scheduler wall-clock timeout, seconds (default: 120)
    --no-pathfinder           Skip the pathfinder scheduler entirely
    --low-par FLOAT           Adaptive low_parallelism_threshold (default: 2.0)
    --high-weight FLOAT       Adaptive high_weight_threshold (default: 6.0)
    --out-dir PATH            Output root (default: repo root)
"""

import argparse
import json
import logging
import math
import os
import signal
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Path bootstrap — works when invoked as:
#   PYTHONPATH=src python src/scripts/adaptive_scheduler_all_benchmarks.py
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
_src = str(PROJECT_ROOT / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from harvest.compilation.circuit_analysis import convert_rx_ry_to_rz, pre_prep_circuit
from harvest.compilation.pauli_block_conversion import convert_to_PCB, create_dag
from harvest.compilation.qasm_loader import qasm_to_circuit, find_qasm_files
from harvest.evaluation.metrics import compute_active_volume
from harvest.layout.presets import nxm_ring_layout_single_qubits
from harvest.routing.adaptive_scheduler import AdaptiveConfig, schedule_adaptive
from harvest.routing.processor import DAGProcessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
)
logger = logging.getLogger("BenchmarkSweep")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_SCHEDULERS: List[Tuple[str, str]] = [
    ("Sequential",    "steiner_tree"),
    ("Greedy Packing","steiner_packing"),
    ("Pathfinder",    "steiner_pathfinder"),
]

_COLOURS: Dict[str, str] = {
    "Sequential":     "#4C72B0",
    "Greedy Packing": "#55A868",
    "Pathfinder":     "#8172B2",
    "Adaptive":       "#DD8452",
}

_MODE_SHORT: Dict[str, str] = {
    "steiner_tree":        "Sequential",
    "steiner_packing":     "Greedy\nPacking",
    "steiner_pathfinder":  "Pathfinder",
}


# ---------------------------------------------------------------------------
# Per-scheduler timeout (SIGALRM — Linux/macOS only)
# ---------------------------------------------------------------------------

class _SchedulerTimeout(BaseException):
    """Raised by SIGALRM handler.  Inherits BaseException so it is NOT caught
    by bare ``except Exception`` clauses inside the scheduler code."""


def _alarm_handler(signum, frame):  # noqa: ARG001
    raise _SchedulerTimeout()


@contextmanager
def _alarm(seconds: int):
    """Context manager that raises :class:`_SchedulerTimeout` after *seconds*."""
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
# Layout helpers
# ---------------------------------------------------------------------------

def compute_grid_dims(n_qubits: int) -> Tuple[int, int]:
    """Return (rows, cols) for a near-square single-spaced grid fitting *n_qubits*."""
    rows = math.ceil(math.sqrt(n_qubits))
    cols = math.ceil(n_qubits / rows)
    return rows, cols


def _family_from_path(qasm_path: str, qasm_root: str) -> str:
    """Return the top-level circuit family (first directory under *qasm_root*)."""
    try:
        rel = Path(qasm_path).relative_to(qasm_root)
        return rel.parts[0]
    except ValueError:
        return Path(qasm_path).parent.name


# ---------------------------------------------------------------------------
# Circuit loading pipeline
# ---------------------------------------------------------------------------

def load_circuit_dag(qasm_path: str):
    """Load a QASM file and return ``(dag, num_qubits, num_pauli_evolutions)``."""
    from qiskit.circuit.library import PauliEvolutionGate

    circuit = qasm_to_circuit(qasm_path)
    circuit = pre_prep_circuit(circuit)

    gate_counts = circuit.count_ops()
    if "rx" in gate_counts or "ry" in gate_counts:
        circuit = convert_rx_ry_to_rz(circuit)

    pcb_circuit = convert_to_PCB(circuit)
    dag = create_dag(pcb_circuit)

    n_pe = sum(
        1 for op in dag.topological_op_nodes()
        if isinstance(op.op, PauliEvolutionGate)
    )
    return dag, circuit.num_qubits, n_pe


# ---------------------------------------------------------------------------
# Single scheduler run
# ---------------------------------------------------------------------------

def _run_single(
    dag,
    scheduler_name: str,
    scheduler_mode: str,
    layout_factory,
    *,
    adaptive_config: Optional[AdaptiveConfig] = None,
) -> Dict:
    """Route *dag* with one scheduler and collect metrics.

    Creates a fresh :class:`DAGProcessor` instance so that successive runs on
    the same *dag* do not interfere with each other.
    """
    layout = layout_factory()
    processor = DAGProcessor(layout_engine=layout)

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
        selected_scheduler = ""
        selection_reason = ""
    runtime_s = time.perf_counter() - t0

    meta = getattr(processor, "_scheduling_metadata", {})
    num_timesteps = meta.get("total_elapsed_steps", len(results))
    completed = meta.get("completed", True)

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
        "completed": completed,
        "runtime_s": round(runtime_s, 2),
        "success": True,
    }


# ---------------------------------------------------------------------------
# Motivation plot
# ---------------------------------------------------------------------------

def create_motivation_plot(
    all_results: List[Dict],
    adaptive_info: Dict,
    output_prefix: str,
    circuit_title: str,
) -> None:
    """Two-subplot bar chart comparing schedulers on timesteps and active volume.

    Args:
        all_results:    Result dicts from :func:`_run_single`.
        adaptive_info:  Dict with ``"selected_mode"`` and ``"reason"``.
        output_prefix:  File path prefix (no extension). ``.png`` and ``.pdf``
                        are written.
        circuit_title:  Human-readable circuit description for the figure title.
    """
    successful = [r for r in all_results if r.get("success")]
    if len(successful) < 2:
        logger.warning("  Too few successful results — skipping plot.")
        return

    # Build display labels (multi-line for adaptive to show per-layer summary).
    display_labels = []
    for r in successful:
        if r["scheduler_mode"] == "adaptive":
            reason = r.get("selection_reason", "")
            display_labels.append(f"Adaptive\n(per-layer)")
        else:
            display_labels.append(r["scheduler_name"])

    timesteps = [r.get("num_timesteps", 0) for r in successful]
    active_vols = [r.get("active_volume", 0) for r in successful]
    colours = [_COLOURS.get(r["scheduler_name"], "#888888") for r in successful]
    x = np.arange(len(successful))

    fig, (ax_ts, ax_av) = plt.subplots(1, 2, figsize=(12, 5))

    def _bar_plot(ax, values, ylabel, title):
        bars = ax.bar(x, values, color=colours, alpha=0.88,
                      edgecolor="white", width=0.6)
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
                ha="center", va="bottom", fontsize=9, fontweight="bold",
            )

    _bar_plot(ax_ts, timesteps, "Timesteps", "Timesteps per Scheduler")
    _bar_plot(ax_av, active_vols, "Active Volume (cell-steps)", "Active Volume per Scheduler")

    # Highlight the adaptive bar with a coloured border.
    for ax in (ax_ts, ax_av):
        for i, r in enumerate(successful):
            if r["scheduler_mode"] == "adaptive":
                ax.patches[i].set_edgecolor("#CC4400")
                ax.patches[i].set_linewidth(2.0)

    sel_mode = adaptive_info.get("selected_mode", "?")
    sel_short = {
        "steiner_tree":       "Sequential",
        "steiner_packing":    "Greedy Packing",
        "steiner_pathfinder": "Pathfinder",
    }.get(sel_mode, sel_mode)
    reason = adaptive_info.get("reason", "")
    reason = reason.split("→")[-1].strip() if "→" in reason else reason

    fig.suptitle(
        f"Scheduler Comparison — {circuit_title}\n"
        f"Adaptive selected: {sel_short}  |  {reason}",
        fontsize=10, fontweight="bold", y=1.02,
    )

    from matplotlib.patches import Patch
    fig.legend(
        handles=[
            Patch(facecolor=_COLOURS["Sequential"],     label="Sequential"),
            Patch(facecolor=_COLOURS["Greedy Packing"], label="Greedy Packing"),
            Patch(facecolor=_COLOURS["Pathfinder"],     label="Pathfinder"),
            Patch(facecolor=_COLOURS["Adaptive"],       label="Adaptive",
                  edgecolor="#CC4400", linewidth=2),
        ],
        loc="lower center", ncol=4, fontsize=9,
        framealpha=0.9, bbox_to_anchor=(0.5, -0.04),
    )

    plt.tight_layout()
    for ext in ("png", "pdf"):
        out_path = f"{output_prefix}.{ext}"
        plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Print table
# ---------------------------------------------------------------------------

def print_table(circuit_name: str, all_results: List[Dict]) -> None:
    """Print a compact per-circuit comparison table to stdout."""
    col_w = [22, 20, 12, 15, 16, 10]
    sep = "+" + "+".join("-" * w for w in col_w) + "+"
    fmt = "| {:<20} | {:<18} | {:>10} | {:>13} | {:>14} | {:>8} |"
    print(f"\n  Circuit: {circuit_name}")
    print(sep)
    print(fmt.format("Scheduler", "Selected (adaptive)",
                     "Timesteps", "Active Volume", "Total Wirelength", "Time (s)"))
    print(sep)
    for r in all_results:
        if r.get("success"):
            sel = r.get("selected_scheduler", "")
            if sel:
                sel = {"steiner_tree": "Sequential",
                       "steiner_packing": "Greedy Packing",
                       "steiner_pathfinder": "Pathfinder"}.get(sel, sel)
            flag = "" if r.get("completed", True) else " ⚠"
            print(fmt.format(
                r["scheduler_name"], sel,
                f"{r.get('num_timesteps','?'):,}{flag}",
                f"{r.get('active_volume','?'):,}",
                f"{r.get('total_wirelength','?'):,}",
                f"{r.get('runtime_s','?'):.1f}",
            ))
        elif r.get("timed_out"):
            print(fmt.format(r["scheduler_name"], "", "TIMEOUT", "", "", ""))
        else:
            print(fmt.format(r["scheduler_name"], "", "FAILED", "", "", ""))
    print(sep)


# ---------------------------------------------------------------------------
# Per-circuit processing
# ---------------------------------------------------------------------------

def run_circuit(
    qasm_path: str,
    qasm_root: str,
    adaptive_config: AdaptiveConfig,
    timeout_sec: int,
    no_pathfinder: bool,
    max_qubits: Optional[int],
    out_results_dir: str,
    out_plots_dir: str,
) -> Dict:
    """Process one circuit: load → run schedulers → save JSON + plot.

    Returns a summary dict with at minimum ``"circuit_name"`` and ``"status"``.
    """
    circuit_name = Path(qasm_path).stem
    family = _family_from_path(qasm_path, qasm_root)

    # ---- Load ----------------------------------------------------------------
    try:
        dag, num_qubits, n_pe = load_circuit_dag(qasm_path)
    except Exception as exc:
        logger.warning(f"  Load failed: {exc}")
        return {
            "circuit_name": circuit_name,
            "family": family,
            "qasm_path": str(qasm_path),
            "status": "load_failed",
            "error": str(exc),
        }

    # ---- Qubit filter --------------------------------------------------------
    if max_qubits is not None and num_qubits > max_qubits:
        logger.info(f"  Skip ({num_qubits}q > max_qubits={max_qubits})")
        return {
            "circuit_name": circuit_name,
            "family": family,
            "qasm_path": str(qasm_path),
            "num_qubits": num_qubits,
            "status": "skipped_too_large",
        }

    # ---- Clifford filter -----------------------------------------------------
    if n_pe == 0:
        logger.info(f"  Skip: 0 PauliEvolutions (Clifford-only)")
        return {
            "circuit_name": circuit_name,
            "family": family,
            "qasm_path": str(qasm_path),
            "num_qubits": num_qubits,
            "num_pauli_evolutions": 0,
            "status": "clifford_only",
        }

    # ---- Layout --------------------------------------------------------------
    rows, cols = compute_grid_dims(num_qubits)
    layout_factory = lambda r=rows, c=cols: nxm_ring_layout_single_qubits(r, c)
    logger.info(f"  {num_qubits}q | {n_pe} T-ops | layout {rows}×{cols}")

    # ---- Scheduler list ------------------------------------------------------
    schedulers = [s for s in BASE_SCHEDULERS
                  if not (no_pathfinder and s[1] == "steiner_pathfinder")]
    schedulers.append(("Adaptive", "adaptive"))

    all_results: List[Dict] = []
    adaptive_info: Dict = {}

    for sched_name, sched_mode in schedulers:
        t_start = time.perf_counter()
        try:
            with _alarm(timeout_sec):
                if sched_mode == "adaptive":
                    r = _run_single(dag, sched_name, sched_mode, layout_factory,
                                    adaptive_config=adaptive_config)
                    adaptive_info = {
                        "selected_mode": r.get("selected_scheduler"),
                        "reason": r.get("selection_reason", ""),
                    }
                else:
                    r = _run_single(dag, sched_name, sched_mode, layout_factory)
        except _SchedulerTimeout:
            elapsed = round(time.perf_counter() - t_start, 1)
            logger.warning(f"    {sched_name}: TIMEOUT after {elapsed}s")
            r = {
                "scheduler_name": sched_name,
                "scheduler_mode": sched_mode,
                "selected_scheduler": "",
                "selection_reason": "",
                "success": False,
                "timed_out": True,
                "error": f"timeout after {timeout_sec}s",
            }
        except Exception as exc:
            logger.error(f"    {sched_name}: ERROR — {exc}", exc_info=True)
            r = {
                "scheduler_name": sched_name,
                "scheduler_mode": sched_mode,
                "success": False,
                "error": str(exc),
            }

        all_results.append(r)
        if r.get("success"):
            logger.info(
                f"    {sched_name}: T={r['num_timesteps']:,}  "
                f"AV={r['active_volume']:,}  t={r['runtime_s']}s"
            )

    # ---- Print table ---------------------------------------------------------
    print_table(circuit_name, all_results)

    # ---- Save per-circuit JSON -----------------------------------------------
    circ_doc = {
        "circuit_name": circuit_name,
        "family": family,
        "qasm_path": str(qasm_path),
        "num_qubits": num_qubits,
        "num_pauli_evolutions": n_pe,
        "layout_rows": rows,
        "layout_cols": cols,
        "scheduler_results": all_results,
        "adaptive_info": adaptive_info,
        "status": "success",
    }
    json_path = os.path.join(out_results_dir, f"{circuit_name}.json")
    Path(json_path).write_text(json.dumps(circ_doc, indent=2, default=str))

    # ---- Save motivation plot ------------------------------------------------
    n_successful = sum(1 for r in all_results if r.get("success"))
    if n_successful >= 2:
        os.makedirs(out_plots_dir, exist_ok=True)
        plot_prefix = os.path.join(out_plots_dir, f"{circuit_name}_motivation")
        circuit_title = (
            f"{circuit_name}  ({num_qubits}q, {n_pe} T-ops)  "
            f"Layout {rows}×{cols}"
        )
        create_motivation_plot(all_results, adaptive_info, plot_prefix, circuit_title)
        logger.info(f"  Saved plot → {plot_prefix}.png")

    return circ_doc


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Adaptive scheduler sweep over all benchmark circuits.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--qasm-dir", type=str,
        default=str(PROJECT_ROOT / "benchmark_circuits" / "qasm"),
        help="Root directory of .qasm benchmark files.",
    )
    parser.add_argument(
        "--max-qubits", type=int, default=100,
        help="Skip circuits with more than N logical qubits. 0 = no limit.",
    )
    parser.add_argument(
        "--exclude-families", nargs="*", default=["qv"],
        metavar="FAMILY",
        help="Top-level circuit family directories to skip entirely.",
    )
    parser.add_argument(
        "--scheduler-timeout", type=int, default=120,
        metavar="S",
        help="Per-scheduler wall-clock timeout in seconds. 0 = no timeout.",
    )
    parser.add_argument(
        "--no-pathfinder", action="store_true",
        help="Skip the steiner_pathfinder scheduler (speeds up the sweep).",
    )
    parser.add_argument(
        "--low-par", type=float, default=2.0, metavar="FLOAT",
        help="Adaptive low_parallelism_threshold.",
    )
    parser.add_argument(
        "--high-par", type=float, default=5.0, metavar="FLOAT",
        help="Adaptive high_parallelism_threshold: avg_layer_width ≥ this → pathfinder.",
    )
    parser.add_argument(
        "--out-dir", type=str, default=str(PROJECT_ROOT),
        metavar="PATH",
        help="Output root directory for results and plots.",
    )
    args = parser.parse_args()

    adaptive_config = AdaptiveConfig(
        low_parallelism_threshold=args.low_par,
        high_parallelism_threshold=args.high_par,
    )
    max_qubits = args.max_qubits if args.max_qubits > 0 else None
    timeout_sec = args.scheduler_timeout

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir)
    out_results_dir = out_dir / "routing_experiment_results" / f"benchmark_sweep_{timestamp}"
    out_plots_dir = out_dir / "plots" / f"benchmark_sweep_{timestamp}"
    out_results_dir.mkdir(parents=True, exist_ok=True)
    out_plots_dir.mkdir(parents=True, exist_ok=True)

    qasm_root = args.qasm_dir

    # ---- Discover circuits ---------------------------------------------------
    all_qasm = find_qasm_files(qasm_root)
    # Remove transpiled variants
    all_qasm = [f for f in all_qasm if "_transpiled" not in Path(f).name]

    # Apply family exclusion
    exclude = set(args.exclude_families) if args.exclude_families else set()
    if exclude:
        before = len(all_qasm)
        all_qasm = [
            f for f in all_qasm
            if _family_from_path(str(f), qasm_root) not in exclude
        ]
        logger.info(
            f"Family filter (exclude={sorted(exclude)}): "
            f"kept {len(all_qasm)}/{before} circuits"
        )

    all_qasm = sorted(all_qasm)
    logger.info(
        f"\n{'='*60}\n"
        f"  Benchmark sweep — {len(all_qasm)} circuits\n"
        f"  max_qubits={max_qubits}  timeout={timeout_sec}s  "
        f"no_pathfinder={args.no_pathfinder}\n"
        f"  Output: {out_results_dir}\n"
        f"{'='*60}"
    )

    # ---- Sweep ---------------------------------------------------------------
    all_summaries: List[Dict] = []
    n_success = n_skipped = n_failed = 0

    for i, qasm_path in enumerate(all_qasm, 1):
        logger.info(f"\n[{i}/{len(all_qasm)}] {Path(qasm_path).name}")
        result = run_circuit(
            str(qasm_path), qasm_root,
            adaptive_config=adaptive_config,
            timeout_sec=timeout_sec,
            no_pathfinder=args.no_pathfinder,
            max_qubits=max_qubits,
            out_results_dir=str(out_results_dir),
            out_plots_dir=str(out_plots_dir),
        )
        all_summaries.append(result)

        status = result.get("status", "unknown")
        if status == "success":
            n_success += 1
        elif status in ("clifford_only", "skipped_too_large"):
            n_skipped += 1
        else:
            n_failed += 1

    # ---- Combined summary JSON -----------------------------------------------
    summary = {
        "timestamp": timestamp,
        "qasm_dir": qasm_root,
        "max_qubits": max_qubits,
        "exclude_families": sorted(exclude),
        "scheduler_timeout_s": timeout_sec,
        "no_pathfinder": args.no_pathfinder,
        "adaptive_low_par": args.low_par,
        "adaptive_high_par": args.high_par,
        "n_circuits_total": len(all_qasm),
        "n_circuits_processed": n_success,
        "n_circuits_skipped": n_skipped,
        "n_circuits_failed": n_failed,
        "circuits": all_summaries,
    }
    summary_path = out_results_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))

    logger.info(
        f"\n{'='*60}\n"
        f"  Sweep complete\n"
        f"  Processed : {n_success}\n"
        f"  Skipped   : {n_skipped}  (Clifford-only or >max_qubits)\n"
        f"  Failed    : {n_failed}\n"
        f"  Summary   : {summary_path}\n"
        f"  Plots     : {out_plots_dir}\n"
        f"{'='*60}"
    )


if __name__ == "__main__":
    main()
