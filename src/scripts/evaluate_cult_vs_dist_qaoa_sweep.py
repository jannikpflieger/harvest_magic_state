#!/usr/bin/env python3
"""
Sweep over QAOA circuits (≤ 100 qubits) comparing distillation vs cultivation
under varying preparation time (μ) and number of available magic states.

Two nested loops per circuit
-----------------------------
  Outer : num_magic  – number of magic-state terminals active at once (1 … 20)
  Inner : mu         – preparation time in cycles (1 … 20)

Magic-state sources
-------------------
  distillation : deterministic MagicStateFactory with prep_cycles = mu
  cultivation  : stochastic MagicStateCultivator with geometric mean ≈ mu
                 (p = 1/mu, so the expected wait is mu cycles)

Layout
------
  Single-spaced ring layout, circuit-aware qubit placement via
  StaticLayoutSynthesizer.  The layout is built *once* per circuit; only
  the magic source changes across the inner loops.

  The number of active magic terminals is controlled by selecting
  ``num_magic`` evenly-spaced terminals from the full magic ring.  If the
  layout has fewer magic terminals than requested, all available terminals
  are used.

Scheduler
---------
  steiner_packing (Greedy parallel packing) – same as in other sweep scripts.

CSV output
----------
  ``cultivation_vs_distillation_sweep.csv`` is written to --output-dir and
  flushed after every individual run so partial results are preserved.

Usage
-----
  cd src

  # Full sweep (all QAOA circuits ≤ 100 q):
  python scripts/evaluate_cult_vs_dist_qaoa_sweep.py

  # Quick smoke-test (first 3 circuits only):
  python scripts/evaluate_cult_vs_dist_qaoa_sweep.py \\
      --max-circuits 3 \\
      --output-dir ../results/cult_vs_dist_test

  # Custom circuit directory:
  python scripts/evaluate_cult_vs_dist_qaoa_sweep.py \\
      --qasm-dir ../benchmark_circuits/qasm/qaoa \\
      --output-dir ../results/cult_vs_dist_full
"""

import argparse
import csv
import logging
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from harvest.compilation.circuit_analysis import convert_rx_ry_to_rz, pre_prep_circuit
from harvest.compilation.pauli_block_conversion import convert_to_PCB, create_dag
from harvest.compilation.qasm_loader import find_qasm_files, qasm_to_circuit
from harvest.layout.presets import (
    blocks_of_four_qubit_patches,
    nxm_ring_layout_single_qubits_large_spacing,
)
from harvest.routing.magic_state_cultivator import MagicStateCultivator, geometric_sampler
from harvest.routing.magic_state_factory import MagicStateFactory
from harvest.routing.processor import DAGProcessor
from harvest.synthesis.placement import PlacementConfig
from harvest.synthesis.synthesizer import StaticLayoutSynthesizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
# Suppress very verbose per-step detail loggers that slow down large sweeps
logging.getLogger("HarvestMagicState.Detailed").setLevel(logging.WARNING)
logging.getLogger("HarvestMagicState.DAGProcessor").setLevel(logging.WARNING)
logger = logging.getLogger("CultVsDistSweep")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALL_SCHEDULERS = {
    "steiner_packing":    "Greedy",
    "steiner_pathfinder": "Pathfinder",
    "harvest":            "Harvest",
}
SCHEDULERS = list(ALL_SCHEDULERS.items())  # default: all schedulers

LAYOUT_PRESET_MAP = {
    "double_spacing": nxm_ring_layout_single_qubits_large_spacing,
    "blocks_of_four": blocks_of_four_qubit_patches,
}
MU_RANGE = range(10, 21)          # μ = 10 … 20
NUM_MAGIC_RANGE = range(16, 3, -1)  # available magic terminals = 16 … 4 (descending)
CULTIVATION_SEED = 42
MIN_QUBITS = 30
MAX_QUBITS = 50

CSV_FIELDS = [
    "circuit_name",
    "num_qubits",
    "layout_preset",        # "circuit_aware", "double_spacing", or "blocks_of_four"
    "layout_grid",          # e.g. "23x23"
    "num_magic_in_layout",  # total magic terminals in the layout
    "num_magic",            # active magic terminals used in this run
    "mu",
    "scheduler",            # "Greedy" or "Pathfinder"
    "source_type",          # "distillation" or "cultivation"
    "num_timesteps",
    "total_wirelength",
    "magic_wait_cycles",
    "num_nodes_processed",
    "num_nodes_total",
    "completed",
    "success",
    "error",
]


# ---------------------------------------------------------------------------
# Circuit loading
# ---------------------------------------------------------------------------

def load_circuit_dag(qasm_path: str):
    """Load, preprocess, and convert a QASM circuit to (circuit, dag)."""
    circuit = qasm_to_circuit(qasm_path)
    circuit = pre_prep_circuit(circuit)
    ops = circuit.count_ops()
    if "rx" in ops or "ry" in ops:
        circuit = convert_rx_ry_to_rz(circuit)
    pcb = convert_to_PCB(circuit, verbose=False)
    dag = create_dag(pcb)
    return circuit, dag


# ---------------------------------------------------------------------------
# Terminal selection
# ---------------------------------------------------------------------------

def select_terminals_evenly(all_terminals: List[str], n: int) -> List[str]:
    """Return *n* evenly-spaced entries from *all_terminals*.

    If n >= len(all_terminals), all terminals are returned.
    """
    total = len(all_terminals)
    if n >= total:
        return list(all_terminals)
    if n == 1:
        return [all_terminals[total // 2]]
    chosen = [all_terminals[round(i * (total - 1) / (n - 1))] for i in range(n)]
    return chosen


# ---------------------------------------------------------------------------
# Single routing run
# ---------------------------------------------------------------------------

def run_single(dag, layout_engine, scheduler_mode: str, source_label: str, magic_source) -> Dict:
    """Route *dag* on *layout_engine* with *magic_source*; return metrics dict."""
    try:
        processor = DAGProcessor(layout_engine=layout_engine, magic_source=magic_source)
        results = processor.process_entire_dag(
            dag, visualize_each_step=False, mode=scheduler_mode
        )

        meta = getattr(processor, "_scheduling_metadata", {})
        total_elapsed   = meta.get("total_elapsed_steps",  len(results))
        completed       = meta.get("completed",             True)
        nodes_completed = meta.get("num_nodes_completed",   len(results))
        nodes_total     = meta.get("num_nodes_total",       len(results))

        total_wirelength = sum(len(r.get("steiner_edges", set())) for r in results)

        magic_wait = 0
        src = processor.magic_source
        if src is not None and not getattr(src, "unlimited", True):
            stats = src.get_stats()
            magic_wait = stats.get("total_wait_cycles", 0)

        return {
            "source_type":          source_label,
            "num_timesteps":        total_elapsed,
            "total_wirelength":     total_wirelength,
            "magic_wait_cycles":    magic_wait,
            "num_nodes_processed":  nodes_completed,
            "num_nodes_total":      nodes_total,
            "completed":            completed,
            "success":              True,
            "error":                "",
        }

    except Exception as exc:
        logger.error(f"    FAILED ({source_label}): {exc}", exc_info=True)
        return {
            "source_type":          source_label,
            "num_timesteps":        None,
            "total_wirelength":     None,
            "magic_wait_cycles":    None,
            "num_nodes_processed":  None,
            "num_nodes_total":      None,
            "completed":            False,
            "success":              False,
            "error":                str(exc),
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Sweep cultivation vs distillation over QAOA circuits (≤ 100 q)."
    )
    parser.add_argument(
        "--qasm-dir",
        default=os.path.join(
            os.path.dirname(__file__), "..", "..", "benchmark_circuits", "qasm", "qaoa"
        ),
        help="Directory (searched recursively) for QAOA .qasm files.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(
            os.path.dirname(__file__), "..", "..", "results",
            f"cult_vs_dist_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        ),
        help="Directory to write results to.",
    )
    parser.add_argument(
        "--max-circuits",
        type=int,
        default=None,
        help="Limit the number of circuits processed (useful for quick tests).",
    )
    parser.add_argument(
        "--min-qubits",
        type=int,
        default=None,
        help="Override MIN_QUBITS constant (inclusive lower bound on circuit size).",
    )
    parser.add_argument(
        "--max-qubits",
        type=int,
        default=None,
        help="Override MAX_QUBITS constant (inclusive upper bound on circuit size).",
    )
    parser.add_argument(
        "--layout-preset",
        choices=["circuit_aware", "double_spacing", "blocks_of_four"],
        default="circuit_aware",
        help="Layout strategy: 'circuit_aware' uses StaticLayoutSynthesizer (default); "
             "'double_spacing' and 'blocks_of_four' use fixed preset functions with auto-sized grid.",
    )
    parser.add_argument(
        "--name-filter",
        default=None,
        help="Only process circuits whose filename contains this substring (e.g. 'ising').",
    )
    parser.add_argument(
        "--num-magic-min",
        type=int,
        default=None,
        help="Override the lower bound of NUM_MAGIC_RANGE (inclusive).",
    )
    parser.add_argument(
        "--num-magic-max",
        type=int,
        default=None,
        help="Override the upper bound of NUM_MAGIC_RANGE (inclusive).",
    )
    parser.add_argument(
        "--mu-min",
        type=int,
        default=None,
        help="Override the lower bound of MU_RANGE (inclusive).",
    )
    parser.add_argument(
        "--mu-max",
        type=int,
        default=None,
        help="Override the upper bound of MU_RANGE (inclusive).",
    )
    parser.add_argument(
        "--use-all-magic",
        action="store_true",
        default=False,
        help="Use all available magic terminals in the layout (ignores --num-magic-min/max).",
    )
    parser.add_argument(
        "--num-magic-to-layout-max",
        action="store_true",
        default=False,
        help="Sweep num_magic from 1 up to the maximum available in the layout for each circuit.",
    )
    parser.add_argument(
        "--scheduler-modes",
        nargs="+",
        choices=list(ALL_SCHEDULERS.keys()),
        default=None,
        help="Scheduler mode(s) to run. Defaults to all schedulers.",
    )
    args = parser.parse_args()

    # Allow CLI to restrict schedulers
    schedulers = (
        [(m, ALL_SCHEDULERS[m]) for m in args.scheduler_modes]
        if args.scheduler_modes
        else SCHEDULERS
    )

    # Allow CLI to override num_magic range (ascending when custom bounds are given)
    if args.num_magic_min is not None or args.num_magic_max is not None:
        nm_min = args.num_magic_min if args.num_magic_min is not None else NUM_MAGIC_RANGE[-1]
        nm_max = args.num_magic_max if args.num_magic_max is not None else NUM_MAGIC_RANGE[0]
        num_magic_range = range(nm_min, nm_max + 1)
    else:
        num_magic_range = NUM_MAGIC_RANGE

    # Allow CLI to override mu range
    if args.mu_min is not None or args.mu_max is not None:
        mu_min = args.mu_min if args.mu_min is not None else MU_RANGE.start
        mu_max = args.mu_max if args.mu_max is not None else MU_RANGE.stop - 1
        mu_range = range(mu_min, mu_max + 1)
    else:
        mu_range = MU_RANGE

    # Allow CLI to override the module-level qubit-filter constants
    min_qubits = args.min_qubits if args.min_qubits is not None else MIN_QUBITS
    max_qubits = args.max_qubits if args.max_qubits is not None else MAX_QUBITS

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "cultivation_vs_distillation_sweep.csv"

    # ── Collect and filter circuits ────────────────────────────────────────
    qasm_dir = Path(args.qasm_dir)
    all_qasm = find_qasm_files(str(qasm_dir))
    if args.name_filter:
        all_qasm = [p for p in all_qasm if args.name_filter in Path(p).name]
    logger.info(f"Found {len(all_qasm)} QASM files under {qasm_dir}"
                + (f" (filtered by '{args.name_filter}')" if args.name_filter else ""))

    accepted: List[Tuple[str, object, object]] = []
    for qasm_path in all_qasm:
        try:
            circuit, dag = load_circuit_dag(qasm_path)
        except Exception as exc:
            logger.warning(f"  Skipping {Path(qasm_path).name}: failed to load – {exc}")
            continue

        if not (min_qubits <= circuit.num_qubits <= max_qubits):
            logger.debug(
                f"  Skipped ({circuit.num_qubits}q, outside [{min_qubits}\u2026{max_qubits}]): "
                f"{Path(qasm_path).name}"
            )
            continue

        accepted.append((qasm_path, circuit, dag))
        logger.info(
            f"  Accepted: {Path(qasm_path).name}  ({circuit.num_qubits}q, "
            f"{len(list(dag.op_nodes()))} DAG ops)"
        )

    if args.max_circuits is not None:
        accepted = accepted[: args.max_circuits]

    logger.info(f"\n{'='*70}")
    logger.info(f"Processing {len(accepted)} circuits.")
    logger.info(f"  mu range       : {mu_range.start} … {mu_range.stop - 1}")
    logger.info(f"  num_magic range: {'all available' if args.use_all_magic else '1 … layout max (per circuit)' if args.num_magic_to_layout_max else f'{list(num_magic_range)[0]} … {list(num_magic_range)[-1]}'}")
    logger.info(f"  qubit filter   : {min_qubits} … {max_qubits}")
    logger.info(f"  name filter    : {args.name_filter or '(none)'}")
    logger.info(f"  layout preset  : {args.layout_preset}")
    logger.info(f"  schedulers     : {[lbl for _, lbl in schedulers]}")
    logger.info(f"  output CSV     : {csv_path}")
    logger.info(f"{'='*70}\n")

    # Shared synthesizer instance (used only for "circuit_aware" preset)
    synth = StaticLayoutSynthesizer(
        placement_config=PlacementConfig(
            alpha=1.0, beta=0.5, max_swap_iterations=100, seed=42
        ),
    ) if args.layout_preset == "circuit_aware" else None

    # ── Open CSV and sweep ─────────────────────────────────────────────────
    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        csv_file.flush()

        total_circuits = len(accepted)
        for circ_idx, (qasm_path, circuit, dag) in enumerate(accepted, start=1):
            circuit_name = Path(qasm_path).stem
            num_qubits   = circuit.num_qubits

            logger.info(
                f"\n[Circuit {circ_idx}/{total_circuits}] {circuit_name}  ({num_qubits}q)"
            )

            # Build layout (once per circuit)
            try:
                if args.layout_preset == "circuit_aware":
                    engine, report = synth.synthesize(dag)
                    layout_grid = f"{report.grid_width}x{report.grid_height}"
                    logger.info(
                        f"  Layout: {layout_grid}, {report.num_magic_sites} magic sites  "
                        f"(placement cost {report.placement_cost:.2f})"
                    )
                else:
                    n = circuit.num_qubits
                    if args.layout_preset == "blocks_of_four":
                        side = max(1, math.ceil(math.sqrt(n / 4.0)))
                    else:
                        side = max(1, math.ceil(math.sqrt(n)))
                    engine = LAYOUT_PRESET_MAP[args.layout_preset](side, side)
                    layout_grid = f"{side}x{side}"
                    logger.info(f"  Layout: {layout_grid}  (preset={args.layout_preset})")
            except Exception as exc:
                logger.error(f"  Layout build failed: {exc}", exc_info=True)
                continue

            # Probe all magic terminals from the synthesised layout
            probe_proc = DAGProcessor(layout_engine=engine)
            all_magic = list(probe_proc.magic_terminals)
            num_magic_in_layout = len(all_magic)
            logger.info(f"  Magic terminals in layout: {num_magic_in_layout}")

            # ── Outer loop: number of active magic terminals ───────────────
            if args.use_all_magic:
                effective_nm_range = [num_magic_in_layout]
            elif args.num_magic_to_layout_max:
                effective_nm_range = range(1, num_magic_in_layout + 1)
            else:
                effective_nm_range = num_magic_range
            for num_magic in effective_nm_range:
                selected_terms = select_terminals_evenly(all_magic, num_magic)
                actual_magic   = len(selected_terms)

                logger.info(
                    f"  num_magic={num_magic}  "
                    f"(using {actual_magic} terminal(s))"
                )

                # ── Inner loop: preparation time μ ─────────────────────────
                for mu in mu_range:
                    logger.info(f"    mu={mu}")

                    row_base = {
                        "circuit_name":        circuit_name,
                        "num_qubits":          num_qubits,
                        "layout_preset":       args.layout_preset,
                        "layout_grid":         layout_grid,
                        "num_magic_in_layout": num_magic_in_layout,
                        "num_magic":           num_magic,
                        "mu":                  mu,
                    }

                    for sched_mode, sched_label in schedulers:
                        logger.info(f"      scheduler={sched_label}")

                        # Distillation — deterministic, fixed prep_cycles = mu
                        factory = MagicStateFactory(selected_terms, mu)
                        dist_result = run_single(
                            dag, engine, sched_mode, "distillation", factory
                        )
                        writer.writerow({**row_base, "scheduler": sched_label, **dist_result})
                        csv_file.flush()

                        # Cultivation — stochastic, geometric distribution, mean ≈ mu
                        cultivator = MagicStateCultivator(
                            selected_terms,
                            readiness_sampler=geometric_sampler(1.0 / mu),
                            seed=CULTIVATION_SEED,
                        )
                        cult_result = run_single(
                            dag, engine, sched_mode, "cultivation", cultivator
                        )
                        writer.writerow({**row_base, "scheduler": sched_label, **cult_result})
                        csv_file.flush()

    logger.info(f"\nDone. Results saved to {csv_path}")


if __name__ == "__main__":
    main()
