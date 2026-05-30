#!/usr/bin/env python3
"""
Layout-type × Placement × Scheduler sweep over QAOA circuits.

For each circuit the script runs the full cross-product of:
  · 3 layout types  : single_spacing, double_spacing, blocks_of_four
  · 2 placements    : row_major, circuit_aware
  · 3 schedulers    : Sequential (steiner_tree), Greedy (steiner_packing),
                       Pathfinder (steiner_pathfinder)
= 18 runs per circuit.

Layout types are represented as LayoutTemplates whose data-site coordinates
match the corresponding preset functions in harvest/layout/presets.py.
Circuit-aware placement uses the same optimiser as StaticLayoutSynthesizer.
Magic states are treated as unlimited (no factory).

Results are written to CSV after every individual routing run so partial
results are preserved on interrupt.

Usage
-----
    cd src

    python scripts/evaluate_layout_placement_scheduler_sweep.py \\
        --qasm-dir ../benchmark_circuits/qasm/qaoa \\
        --output-dir ../results/layout_placement_sweep \\
        --min-qubits 20 --max-qubits 30
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
from harvest.routing.processor import DAGProcessor
from harvest.synthesis.circuit_summary import extract_circuit_summary
from harvest.synthesis.emitter import emit_layout
from harvest.synthesis.placement import (
    PlacementConfig,
    baseline_placement,
    circuit_aware_placement,
)
from harvest.synthesis.templates import (
    LayoutTemplate,
    _compute_distances_and_centrality,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logging.getLogger("HarvestMagicState.Detailed").setLevel(logging.WARNING)
logging.getLogger("HarvestMagicState.DAGProcessor").setLevel(logging.WARNING)
logger = logging.getLogger("LayoutPlacementSweep")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCHEDULERS: List[Tuple[str, str]] = [
    ("steiner_tree",       "Sequential"),
    ("steiner_packing",    "Greedy"),
    ("steiner_pathfinder", "Pathfinder"),
]

PLACEMENTS = ["row_major", "circuit_aware"]
LAYOUT_TYPES = ["single_spacing", "double_spacing", "blocks_of_four"]

MIN_QUBITS = 20
MAX_QUBITS = 30

PLACEMENT_CONFIG = PlacementConfig(alpha=1.0, beta=0.5, max_swap_iterations=100, seed=42)

CSV_FIELDS = [
    "circuit_name",
    "num_qubits",
    "layout_type",
    "layout_grid",
    "num_magic_in_layout",
    "placement",
    "placement_cost",
    "scheduler",
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
# Template builders
# Each function mirrors the geometry of the corresponding preset in
# harvest/layout/presets.py, expressed as a LayoutTemplate so that both
# row_major and circuit_aware placement can be applied uniformly.
# ---------------------------------------------------------------------------

Coord = Tuple[int, int]


def _perimeter_magic_sites(W: int, H: int) -> List[Coord]:
    """Return perimeter magic-site coordinates (ring convention from presets.py)."""
    sites: List[Coord] = []
    for x in range(1, W - 1):
        sites.append((x, 0))
        sites.append((x, H - 1))
    for y in range(1, H - 1):
        sites.append((0, y))
        sites.append((W - 1, y))
    return sites


def single_spacing_template(n_qubits: int) -> LayoutTemplate:
    """Mirrors nxm_ring_layout_single_qubits: 1-patch spacing between qubits."""
    cols = max(1, math.ceil(math.sqrt(n_qubits)))
    rows = max(1, math.ceil(n_qubits / cols))
    W = 2 * cols + 3
    H = 2 * rows + 3

    data_sites: List[Coord] = []
    for r in range(rows):
        for c in range(cols):
            if len(data_sites) >= n_qubits:
                break
            data_sites.append((2 * c + 2, 2 * r + 2))

    distances, centrality = _compute_distances_and_centrality(data_sites)
    return LayoutTemplate(
        name=f"single_spacing_{n_qubits}q",
        grid_width=W,
        grid_height=H,
        data_sites=data_sites,
        magic_sites=_perimeter_magic_sites(W, H),
        routing_lanes=1,
        pairwise_distances=distances,
        site_centrality=centrality,
    )


def double_spacing_template(n_qubits: int) -> LayoutTemplate:
    """Mirrors nxm_ring_layout_single_qubits_large_spacing: 2-patch spacing."""
    cols = max(1, math.ceil(math.sqrt(n_qubits)))
    rows = max(1, math.ceil(n_qubits / cols))
    W = 3 * cols + 2
    H = 3 * rows + 2

    data_sites: List[Coord] = []
    for r in range(rows):
        for c in range(cols):
            if len(data_sites) >= n_qubits:
                break
            data_sites.append((3 * c + 2, 3 * r + 2))

    distances, centrality = _compute_distances_and_centrality(data_sites)
    return LayoutTemplate(
        name=f"double_spacing_{n_qubits}q",
        grid_width=W,
        grid_height=H,
        data_sites=data_sites,
        magic_sites=_perimeter_magic_sites(W, H),
        routing_lanes=2,
        pairwise_distances=distances,
        site_centrality=centrality,
    )


def blocks_of_four_template(n_qubits: int) -> LayoutTemplate:
    """Mirrors blocks_of_four_qubit_patches: 2×2 qubit blocks with 1-patch spacing."""
    num_blocks = max(1, math.ceil(n_qubits / 4))
    block_cols = max(1, math.ceil(math.sqrt(num_blocks)))
    block_rows = max(1, math.ceil(num_blocks / block_cols))
    W = 3 * block_cols + 3
    H = 3 * block_rows + 3

    data_sites: List[Coord] = []
    for br in range(block_rows):
        for bc in range(block_cols):
            ox = 3 * bc + 2
            oy = 3 * br + 2
            for dx, dy in [(0, 0), (1, 0), (0, 1), (1, 1)]:
                if len(data_sites) >= n_qubits:
                    break
                data_sites.append((ox + dx, oy + dy))

    distances, centrality = _compute_distances_and_centrality(data_sites)
    return LayoutTemplate(
        name=f"blocks_of_four_{n_qubits}q",
        grid_width=W,
        grid_height=H,
        data_sites=data_sites,
        magic_sites=_perimeter_magic_sites(W, H),
        routing_lanes=1,
        pairwise_distances=distances,
        site_centrality=centrality,
    )


TEMPLATE_BUILDERS = {
    "single_spacing": single_spacing_template,
    "double_spacing": double_spacing_template,
    "blocks_of_four": blocks_of_four_template,
}

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
# Single routing run
# ---------------------------------------------------------------------------

def run_single(dag, layout_engine, scheduler_mode: str) -> Dict:
    """Route *dag* on *layout_engine* with *scheduler_mode*; return metrics dict."""
    try:
        processor = DAGProcessor(layout_engine=layout_engine)
        results = processor.process_entire_dag(
            dag, visualize_each_step=False, mode=scheduler_mode
        )

        meta = getattr(processor, "_scheduling_metadata", {})
        total_elapsed   = meta.get("total_elapsed_steps",  len(results))
        completed       = meta.get("completed",             True)
        nodes_completed = meta.get("num_nodes_completed",   len(results))
        nodes_total     = meta.get("num_nodes_total",       len(results))
        total_wirelength = sum(len(r.get("steiner_edges", set())) for r in results)

        return {
            "num_timesteps":       total_elapsed,
            "total_wirelength":    total_wirelength,
            "magic_wait_cycles":   0,
            "num_nodes_processed": nodes_completed,
            "num_nodes_total":     nodes_total,
            "completed":           completed,
            "success":             True,
            "error":               "",
        }

    except Exception as exc:
        logger.error(f"    FAILED: {exc}", exc_info=True)
        return {
            "num_timesteps":       None,
            "total_wirelength":    None,
            "magic_wait_cycles":   None,
            "num_nodes_processed": None,
            "num_nodes_total":     None,
            "completed":           False,
            "success":             False,
            "error":               str(exc),
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Sweep layout type × placement × scheduler over QAOA circuits."
    )
    parser.add_argument(
        "--qasm-dir",
        default=os.path.join(
            os.path.dirname(__file__), "..", "..", "benchmark_circuits", "qasm", "qaoa"
        ),
        help="Directory (searched recursively) for .qasm files.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(
            os.path.dirname(__file__), "..", "..", "results", "layout_placement_sweep",
        ),
        help="Base directory for results. A timestamp (YYYYMMDD_HHMMSS) is always appended.",
    )
    parser.add_argument(
        "--min-qubits", type=int, default=None,
        help=f"Inclusive lower qubit-count bound (default: {MIN_QUBITS}).",
    )
    parser.add_argument(
        "--max-qubits", type=int, default=None,
        help=f"Inclusive upper qubit-count bound (default: {MAX_QUBITS}).",
    )
    parser.add_argument(
        "--max-circuits", type=int, default=None,
        help="Limit the number of circuits processed (useful for quick tests).",
    )
    parser.add_argument(
        "--name-filter", default=None,
        help="Only process circuits whose filename contains this substring.",
    )
    args = parser.parse_args()

    min_qubits = args.min_qubits if args.min_qubits is not None else MIN_QUBITS
    max_qubits = args.max_qubits if args.max_qubits is not None else MAX_QUBITS

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"{args.output_dir}_{ts}")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "layout_placement_scheduler_sweep.csv"

    # ── Collect and filter circuits ────────────────────────────────────────
    qasm_dir = Path(args.qasm_dir)
    all_qasm = find_qasm_files(str(qasm_dir))
    if args.name_filter:
        all_qasm = [p for p in all_qasm if args.name_filter in Path(p).name]
    logger.info(f"Found {len(all_qasm)} QASM files under {qasm_dir}")

    accepted: List[Tuple[str, object, object]] = []
    for qasm_path in sorted(all_qasm):
        try:
            circuit, dag = load_circuit_dag(qasm_path)
        except Exception as exc:
            logger.warning(f"  Skipping {Path(qasm_path).name}: failed to load – {exc}")
            continue

        if not (min_qubits <= circuit.num_qubits <= max_qubits):
            logger.debug(
                f"  Skipped ({circuit.num_qubits}q, outside [{min_qubits}…{max_qubits}]): "
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

    total_runs = len(accepted) * len(LAYOUT_TYPES) * len(PLACEMENTS) * len(SCHEDULERS)
    logger.info(f"\n{'='*70}")
    logger.info(
        f"Processing {len(accepted)} circuits × {len(LAYOUT_TYPES)} layout types "
        f"× {len(PLACEMENTS)} placements × {len(SCHEDULERS)} schedulers = {total_runs} total runs"
    )
    logger.info(f"  qubit filter : {min_qubits} … {max_qubits}")
    logger.info(f"  layout types : {LAYOUT_TYPES}")
    logger.info(f"  placements   : {PLACEMENTS}")
    logger.info(f"  schedulers   : {[lbl for _, lbl in SCHEDULERS]}")
    logger.info(f"  output CSV   : {csv_path}")
    logger.info(f"{'='*70}\n")

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

            # Extract circuit summary once (used by circuit_aware placement)
            try:
                summary = extract_circuit_summary(dag)
            except Exception as exc:
                logger.error(f"  Circuit summary extraction failed: {exc}", exc_info=True)
                continue

            # ── Layout type loop ───────────────────────────────────────────
            for layout_type in LAYOUT_TYPES:
                logger.info(f"  Layout type : {layout_type}")

                try:
                    template = TEMPLATE_BUILDERS[layout_type](num_qubits)
                except Exception as exc:
                    logger.error(f"    Template build failed: {exc}", exc_info=True)
                    continue

                layout_grid        = f"{template.grid_width}x{template.grid_height}"
                num_magic_in_layout = len(template.magic_sites)
                logger.info(f"    Grid: {layout_grid},  {num_magic_in_layout} magic sites")

                # ── Placement loop ─────────────────────────────────────────
                for placement_mode in PLACEMENTS:
                    logger.info(f"    Placement   : {placement_mode}")

                    try:
                        if placement_mode == "circuit_aware":
                            placement_result = circuit_aware_placement(
                                summary, template, PLACEMENT_CONFIG
                            )
                        else:
                            placement_result = baseline_placement(
                                template, num_qubits, mode="row_major"
                            )
                        engine          = emit_layout(template, placement_result)
                        placement_cost  = placement_result.cost
                    except Exception as exc:
                        logger.error(f"      Placement failed: {exc}", exc_info=True)
                        for _, sched_label in SCHEDULERS:
                            writer.writerow({
                                "circuit_name":        circuit_name,
                                "num_qubits":          num_qubits,
                                "layout_type":         layout_type,
                                "layout_grid":         layout_grid,
                                "num_magic_in_layout": num_magic_in_layout,
                                "placement":           placement_mode,
                                "placement_cost":      None,
                                "scheduler":           sched_label,
                                "success":             False,
                                "error":               str(exc),
                            })
                            csv_file.flush()
                        continue

                    # ── Scheduler loop ─────────────────────────────────────
                    for sched_mode, sched_label in SCHEDULERS:
                        logger.info(f"      Scheduler : {sched_label}")
                        result = run_single(dag, engine, sched_mode)

                        row = {
                            "circuit_name":        circuit_name,
                            "num_qubits":          num_qubits,
                            "layout_type":         layout_type,
                            "layout_grid":         layout_grid,
                            "num_magic_in_layout": num_magic_in_layout,
                            "placement":           placement_mode,
                            "placement_cost":      placement_cost,
                            "scheduler":           sched_label,
                            **result,
                        }
                        writer.writerow(row)
                        csv_file.flush()

                        if result["success"]:
                            done = (
                                "" if result.get("completed", True)
                                else f" [INCOMPLETE {result.get('num_nodes_processed', 0)}"
                                     f"/{result.get('num_nodes_total', 0)} nodes]"
                            )
                            logger.info(
                                f"        → {result['num_timesteps']:5d} steps, "
                                f"wirelength {result['total_wirelength']:5d}{done}"
                            )

    logger.info(f"\nDone. Results saved to {csv_path}")


if __name__ == "__main__":
    main()
