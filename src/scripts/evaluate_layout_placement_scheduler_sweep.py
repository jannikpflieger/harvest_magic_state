#!/usr/bin/env python3
"""
Layout-type × Placement sweep over all benchmark circuits (Greedy scheduler only).

For each circuit (excl. qv / chemical families) the script runs the full
cross-product of:
  · 3 layout types  : single_spacing, double_spacing, blocks_of_four
  · 2 placements    : row_major, circuit_aware
= 6 runs per circuit (Greedy / steiner_packing only).

Layout types are represented as LayoutTemplates whose data-site coordinates
match the corresponding preset functions in harvest/layout/presets.py.
Circuit-aware placement uses the same optimiser as StaticLayoutSynthesizer.
Magic states are treated as unlimited (no factory).

A 3-minute (180 s) SIGALRM timeout guards each individual routing run.

Outputs
-------
  results/layout_placement_sweep_<timestamp>/
      summary.json
      per_layout_rows.csv           – one row per (circuit, layout, placement)
      circuits/
          <n>_<family>_<circuit>.json   – per-circuit JSON
      schedules/
          <n>_<family>_<circuit>_<layout>_<placement>.json

Results are written after every circuit so partial results are preserved on
interrupt.

Usage
-----
    cd src

    python scripts/evaluate_layout_placement_scheduler_sweep.py
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import multiprocessing
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
_src = str(PROJECT_ROOT / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

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

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s - %(message)s",
)
logger = logging.getLogger("LayoutPlacementSweep")

for _noisy in (
    "HarvestMagicState",
    "HarvestMagicState.DAGProcessor",
    "HarvestMagicState.Detailed",
):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Available schedulers: (mode, label)
ALL_SCHEDULERS: List[Tuple[str, str]] = [
    ("steiner_tree",       "Sequential"),
    ("steiner_packing",    "Greedy"),
    ("steiner_pathfinder", "Pathfinder"),
    ("harvest",            "Harvest"),
]
DEFAULT_SCHEDULERS = ["steiner_pathfinder", "steiner_tree"]

PLACEMENTS   = ["row_major", "circuit_aware"]
#LAYOUT_TYPES = ["single_spacing", "double_spacing", "blocks_of_four"]
LAYOUT_TYPES = ["single_spacing"]

DEFAULT_EXCLUDE_FAMILIES    = ["qv", "chemical"]
DEFAULT_MAX_QUBITS          = 100
DEFAULT_SCHEDULER_TIMEOUT_S = 60  # 1 minute

PLACEMENT_CONFIG = PlacementConfig(alpha=1.0, beta=0.5, max_swap_iterations=100, seed=42)

CSV_FIELDS = [
    "circuit_name",
    "family",
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
    "runtime_s",
    "success",
    "timed_out",
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
# Timeout helper – subprocess-based hard kill (works even inside JAX/XLA)
# ---------------------------------------------------------------------------

class _SchedulerTimeout(Exception):
    pass


def _run_worker(queue, dag, layout_engine, layout_type, placement, num_qubits,
                scheduler_mode, scheduler_label):
    """Subprocess worker: runs the routing and puts (status, ...) in queue."""
    try:
        t0 = time.perf_counter()
        processor = DAGProcessor(layout_engine=layout_engine)
        results = processor.process_entire_dag(
            dag, visualize_each_step=False, mode=scheduler_mode
        )
        runtime_s = time.perf_counter() - t0

        meta             = getattr(processor, "_scheduling_metadata", {})
        total_elapsed    = meta.get("total_elapsed_steps",  len(results))
        completed        = meta.get("completed",             True)
        nodes_completed  = meta.get("num_nodes_completed",   len(results))
        nodes_total      = meta.get("num_nodes_total",       len(results))
        total_wirelength = sum(len(r.get("steiner_edges", set())) for r in results)

        metrics = {
            "scheduler":           scheduler_label,
            "num_timesteps":       total_elapsed,
            "total_wirelength":    total_wirelength,
            "magic_wait_cycles":   0,
            "num_nodes_processed": nodes_completed,
            "num_nodes_total":     nodes_total,
            "completed":           completed,
            "runtime_s":           round(runtime_s, 3),
            "success":             True,
            "timed_out":           False,
            "error":               "",
        }
        schedule = _build_schedule_plan(results, layout_type, placement, num_qubits)
        queue.put(("ok", metrics, schedule))
    except Exception as exc:  # noqa: BLE001
        queue.put(("error", str(exc), None))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _family_from_path(qasm_path: str, qasm_root: str) -> str:
    try:
        rel = Path(qasm_path).resolve().relative_to(Path(qasm_root).resolve())
        return rel.parts[0] if rel.parts else "unknown"
    except Exception:
        return Path(qasm_path).parent.name


def _safe_name(s: str) -> str:
    """Filesystem-safe stem for per-circuit/schedule filenames."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


def _qubit_idx_from_terminal(port: str) -> Optional[int]:
    parts = port.split(":")
    if len(parts) >= 2 and parts[1].startswith("q_"):
        try:
            return int(parts[1][2:])
        except ValueError:
            return None
    return None


def _build_schedule_plan(
    results: List[Dict],
    layout_type: str,
    placement: str,
    num_qubits: int,
) -> Dict:
    timestep_map: Dict[int, List[Dict]] = {}
    for r in results:
        if not r.get("success", True):
            continue
        step = int(r.get("time_step", 0))
        steiner_nodes = r.get("steiner_nodes", set())
        routing_cells = sorted([list(n) for n in steiner_nodes if isinstance(n, tuple)])
        port_nodes    = sorted([n for n in steiner_nodes if isinstance(n, str)])
        qubit_terminals = r.get("qubit_terminals", [])
        qubit_indices = sorted(set(
            idx
            for t in qubit_terminals
            for idx in [_qubit_idx_from_terminal(t)]
            if idx is not None
        ))
        route = {
            "gate_name":      r.get("gate_name", ""),
            "qubit_indices":  qubit_indices,
            "magic_terminal": r.get("magic_terminal"),
            "qubit_ports":    qubit_terminals,
            "routing_cells":  routing_cells,
            "port_nodes":     port_nodes,
        }
        if step not in timestep_map:
            timestep_map[step] = []
        timestep_map[step].append(route)

    timesteps = [
        {"step": step, "routes": routes}
        for step, routes in sorted(timestep_map.items())
    ]
    return {
        "layout_type":   layout_type,
        "placement":     placement,
        "num_qubits":    num_qubits,
        "num_timesteps": len(timesteps),
        "routes_total":  sum(len(ts["routes"]) for ts in timesteps),
        "timesteps":     timesteps,
    }


# ---------------------------------------------------------------------------
# Single routing run
# ---------------------------------------------------------------------------

def run_single(
    dag,
    layout_engine,
    layout_type: str,
    placement: str,
    num_qubits: int,
    timeout_s: int,
    scheduler_mode: str,
    scheduler_label: str,
) -> Tuple[Dict, Optional[Dict]]:
    """
    Route *dag* on *layout_engine* in a subprocess with a hard kill timeout.
    Returns (metrics_dict, schedule_plan).
    Raises _SchedulerTimeout on timeout.
    """
    ctx = multiprocessing.get_context("fork")
    q: multiprocessing.Queue = ctx.Queue()
    p = ctx.Process(
        target=_run_worker,
        args=(q, dag, layout_engine, layout_type, placement, num_qubits,
              scheduler_mode, scheduler_label),
        daemon=True,
    )
    p.start()
    p.join(timeout_s)
    if p.is_alive():
        p.terminate()
        p.join(3)
        if p.is_alive():
            p.kill()
            p.join(2)
        raise _SchedulerTimeout()
    if p.exitcode != 0:
        raise RuntimeError(f"Worker exited with code {p.exitcode}")
    try:
        status, metrics, schedule = q.get_nowait()
    except Exception as exc:
        raise RuntimeError("Worker produced no result") from exc
    if status == "error":
        raise RuntimeError(metrics)  # metrics holds the error string
    return metrics, schedule


# ---------------------------------------------------------------------------
# Summary / checkpoint helpers
# ---------------------------------------------------------------------------

def _build_summary(
    *,
    ts: str,
    qasm_dir: str,
    exclude: set,
    max_qubits: Optional[int],
    qasm_files: List[str],
    max_circuits: int,
    circuit_docs: List[Dict],
) -> Dict:
    return {
        "timestamp":                   ts,
        "qasm_dir":                    qasm_dir,
        "exclude_families":            sorted(exclude),
        "max_qubits":                  max_qubits,
        "schedulers":                  circuit_docs[0].get("schedulers", []) if circuit_docs else [],
        "layout_types":                LAYOUT_TYPES,
        "placements":                  PLACEMENTS,
        "total_circuits_after_filter": len(qasm_files),
        "max_circuits":                max_circuits,
        "successful_circuits":         sum(1 for d in circuit_docs if d.get("status") == "success"),
        "timed_out_circuits":          sum(1 for d in circuit_docs if d.get("status") == "timed_out"),
        "skipped_too_large":           sum(1 for d in circuit_docs if d.get("status") == "skipped_too_large"),
        "load_failed":                 sum(1 for d in circuit_docs if d.get("status") == "load_failed"),
    }


def _write_checkpoint(
    *,
    run_dir: Path,
    ts: str,
    qasm_dir: str,
    exclude: set,
    max_qubits: Optional[int],
    qasm_files: List[str],
    max_circuits: int,
    circuit_docs: List[Dict],
    flat_rows: List[Dict],
) -> None:
    """Persist current progress: summary JSON, per_circuit JSON, CSV."""
    summary = _build_summary(
        ts=ts,
        qasm_dir=qasm_dir,
        exclude=exclude,
        max_qubits=max_qubits,
        qasm_files=qasm_files,
        max_circuits=max_circuits,
        circuit_docs=circuit_docs,
    )
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (run_dir / "per_circuit.json").write_text(
        json.dumps(circuit_docs, indent=2, default=str)
    )
    with open(run_dir / "per_layout_rows.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in flat_rows:
            w.writerow({k: r.get(k, "") for k in CSV_FIELDS})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description=(
            "Sweep all benchmark circuits (excl. qv/chemical) for "
            "layout-type × placement impact with Greedy scheduling."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--qasm-dir",
        type=str,
        default=str(PROJECT_ROOT / "benchmark_circuits" / "qasm"),
        help="Root of benchmark QASM tree.",
    )
    p.add_argument(
        "--max-qubits",
        type=int,
        default=DEFAULT_MAX_QUBITS,
        help="Skip circuits with more qubits than this. 0 = no limit.",
    )
    p.add_argument(
        "--min-qubits",
        type=int,
        default=0,
        help="Skip circuits with fewer qubits than this. 0 = no limit.",
    )
    p.add_argument(
        "--exclude-families",
        nargs="*",
        default=DEFAULT_EXCLUDE_FAMILIES,
        metavar="F",
        help="Top-level circuit families to exclude.",
    )
    p.add_argument(
        "--schedulers",
        nargs="+",
        default=DEFAULT_SCHEDULERS,
        choices=[m for m, _ in ALL_SCHEDULERS],
        metavar="MODE",
        help=(
            "Scheduler modes to run. Choices: "
            + ", ".join(m for m, _ in ALL_SCHEDULERS)
            + f". Default: {DEFAULT_SCHEDULERS}"
        ),
    )
    p.add_argument(
        "--scheduler-timeout",
        type=int,
        default=DEFAULT_SCHEDULER_TIMEOUT_S,
        help="Timeout per (circuit, layout, placement, scheduler) run in seconds.",
    )
    p.add_argument(
        "--max-circuits",
        type=int,
        default=0,
        help="Process at most this many circuits after filtering (0 = all).",
    )
    p.add_argument(
        "--name-filter",
        default=None,
        help="Only process circuits whose filename contains this substring.",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "results" / "layout_placement_sweep"),
        help="Base output path; a timestamp suffix is always appended.",
    )
    p.add_argument(
        "--resume-dir",
        type=str,
        default=None,
        metavar="DIR",
        help=(
            "Path to a previous (interrupted) run directory. "
            "All circuits whose JSON already exists in DIR/circuits/ will be "
            "skipped. Their rows are seeded into the new run's CSV so the "
            "final results are complete."
        ),
    )
    args = p.parse_args()

    max_qubits = args.max_qubits if args.max_qubits > 0 else None
    min_qubits = args.min_qubits if args.min_qubits > 0 else None
    exclude    = set(args.exclude_families or [])
    label_for  = {m: lbl for m, lbl in ALL_SCHEDULERS}
    schedulers = [(m, label_for[m]) for m in args.schedulers]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir       = Path(f"{args.output_dir}_{ts}")
    circuits_dir  = run_dir / "circuits"
    schedules_dir = run_dir / "schedules"
    run_dir.mkdir(parents=True, exist_ok=True)
    circuits_dir.mkdir(parents=True, exist_ok=True)
    schedules_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Output directory: %s", run_dir)

    qasm_dir   = args.qasm_dir
    qasm_files = find_qasm_files(qasm_dir)
    qasm_files = [
        q for q in qasm_files
        if _family_from_path(q, qasm_dir) not in exclude
    ]
    if args.name_filter:
        qasm_files = [q for q in qasm_files if args.name_filter in Path(q).name]
    qasm_files = sorted(qasm_files)

    if args.max_circuits and args.max_circuits > 0:
        qasm_files = qasm_files[: args.max_circuits]

    logger.info(
        "Discovered %d circuits after family exclusions (%s)",
        len(qasm_files),
        sorted(exclude),
    )

    # ── Resume: seed results from a previous interrupted run ───────────────
    circuit_docs: List[Dict] = []
    flat_rows:    List[Dict] = []
    skip_circuit_names: set = set()

    if args.resume_dir:
        resume_dir = Path(args.resume_dir)
        old_circuits_dir = resume_dir / "circuits"
        old_csv_path     = resume_dir / "per_layout_rows.csv"

        if old_circuits_dir.is_dir():
            for jf in sorted(old_circuits_dir.glob("*.json")):
                try:
                    doc = json.loads(jf.read_text())
                    skip_circuit_names.add(doc["circuit_name"])
                    circuit_docs.append(doc)
                except Exception as exc:
                    logger.warning("  resume: could not read %s: %s", jf.name, exc)

        if old_csv_path.is_file():
            with open(old_csv_path, newline="") as f:
                for row in csv.DictReader(f):
                    flat_rows.append(row)

        logger.info(
            "Resume: seeded %d circuit docs and %d CSV rows from %s; skipping %d circuits.",
            len(circuit_docs), len(flat_rows), resume_dir, len(skip_circuit_names),
        )

    for circ_idx, qasm_path in enumerate(qasm_files, 1):
        circuit_name = Path(qasm_path).stem
        family       = _family_from_path(qasm_path, qasm_dir)

        logger.info(
            "[%d/%d] %s  (family: %s)",
            circ_idx, len(qasm_files), circuit_name, family,
        )

        # --- Resume: skip already-processed circuits ----------------------
        if circuit_name in skip_circuit_names:
            logger.info("  skipping (already in resume dir)")
            continue

        # --- Load circuit -------------------------------------------------
        try:
            circuit, dag = load_circuit_dag(qasm_path)
        except Exception as exc:
            logger.warning("  load_failed: %s", exc)
            circuit_doc = {
                "circuit_name": circuit_name,
                "family":       family,
                "qasm_path":    qasm_path,
                "status":       "load_failed",
                "error":        str(exc),
            }
            circuit_docs.append(circuit_doc)
            per_path = (
                circuits_dir
                / f"{circ_idx:04d}_{_safe_name(family)}_{_safe_name(circuit_name)}.json"
            )
            per_path.write_text(json.dumps(circuit_doc, indent=2, default=str))
            _write_checkpoint(
                run_dir=run_dir, ts=ts, qasm_dir=qasm_dir,
                exclude=exclude, max_qubits=max_qubits,
                qasm_files=qasm_files, max_circuits=args.max_circuits,
                circuit_docs=circuit_docs, flat_rows=flat_rows,
            )
            continue

        num_qubits = circuit.num_qubits

        if min_qubits is not None and num_qubits < min_qubits:
            logger.info("  skipped_too_small: %dq < %d", num_qubits, min_qubits)
            circuit_doc = {
                "circuit_name": circuit_name,
                "family":       family,
                "qasm_path":    qasm_path,
                "num_qubits":   num_qubits,
                "status":       "skipped_too_small",
            }
            circuit_docs.append(circuit_doc)
            per_path = (
                circuits_dir
                / f"{circ_idx:04d}_{_safe_name(family)}_{_safe_name(circuit_name)}.json"
            )
            per_path.write_text(json.dumps(circuit_doc, indent=2, default=str))
            _write_checkpoint(
                run_dir=run_dir, ts=ts, qasm_dir=qasm_dir,
                exclude=exclude, max_qubits=max_qubits,
                qasm_files=qasm_files, max_circuits=args.max_circuits,
                circuit_docs=circuit_docs, flat_rows=flat_rows,
            )
            continue

        if max_qubits is not None and num_qubits > max_qubits:
            logger.info("  skipped_too_large: %dq > %d", num_qubits, max_qubits)
            circuit_doc = {
                "circuit_name": circuit_name,
                "family":       family,
                "qasm_path":    qasm_path,
                "num_qubits":   num_qubits,
                "status":       "skipped_too_large",
            }
            circuit_docs.append(circuit_doc)
            per_path = (
                circuits_dir
                / f"{circ_idx:04d}_{_safe_name(family)}_{_safe_name(circuit_name)}.json"
            )
            per_path.write_text(json.dumps(circuit_doc, indent=2, default=str))
            _write_checkpoint(
                run_dir=run_dir, ts=ts, qasm_dir=qasm_dir,
                exclude=exclude, max_qubits=max_qubits,
                qasm_files=qasm_files, max_circuits=args.max_circuits,
                circuit_docs=circuit_docs, flat_rows=flat_rows,
            )
            continue

        # --- Extract circuit summary (for circuit_aware placement) ---------
        try:
            summary = extract_circuit_summary(dag)
        except Exception as exc:
            logger.error("  circuit summary failed: %s", exc, exc_info=True)
            circuit_doc = {
                "circuit_name": circuit_name,
                "family":       family,
                "qasm_path":    qasm_path,
                "num_qubits":   num_qubits,
                "status":       "summary_failed",
                "error":        str(exc),
            }
            circuit_docs.append(circuit_doc)
            per_path = (
                circuits_dir
                / f"{circ_idx:04d}_{_safe_name(family)}_{_safe_name(circuit_name)}.json"
            )
            per_path.write_text(json.dumps(circuit_doc, indent=2, default=str))
            _write_checkpoint(
                run_dir=run_dir, ts=ts, qasm_dir=qasm_dir,
                exclude=exclude, max_qubits=max_qubits,
                qasm_files=qasm_files, max_circuits=args.max_circuits,
                circuit_docs=circuit_docs, flat_rows=flat_rows,
            )
            continue

        circuit_result = {
            "circuit_name": circuit_name,
            "family":       family,
            "qasm_path":    qasm_path,
            "num_qubits":   num_qubits,
            "status":       "success",
            "schedulers":   [m for m, _ in schedulers],
            "run_results":  [],
        }

        # --- Layout × placement sweep ------------------------------------
        for layout_type in LAYOUT_TYPES:
            logger.info("  layout_type: %s", layout_type)

            try:
                template = TEMPLATE_BUILDERS[layout_type](num_qubits)
            except Exception as exc:
                logger.error("    template build failed: %s", exc)
                for placement_mode in PLACEMENTS:
                    for _, sched_label in schedulers:
                        row = {
                            "circuit_name":        circuit_name,
                            "family":              family,
                            "num_qubits":          num_qubits,
                            "layout_type":         layout_type,
                            "layout_grid":         "",
                            "num_magic_in_layout": None,
                            "placement":           placement_mode,
                            "placement_cost":      None,
                            "scheduler":           sched_label,
                            "success":             False,
                            "timed_out":           False,
                            "error":               f"template build: {exc}",
                        }
                        flat_rows.append(row)
                        circuit_result["run_results"].append(row)
                continue

            layout_grid         = f"{template.grid_width}x{template.grid_height}"
            num_magic_in_layout = len(template.magic_sites)

            for placement_mode in PLACEMENTS:
                logger.info("    placement: %s", placement_mode)

                # Build engine from placement
                try:
                    if placement_mode == "circuit_aware":
                        placement_result = circuit_aware_placement(
                            summary, template, PLACEMENT_CONFIG
                        )
                    else:
                        placement_result = baseline_placement(
                            template, num_qubits, mode="row_major"
                        )
                    engine         = emit_layout(template, placement_result)
                    placement_cost = placement_result.cost
                except Exception as exc:
                    logger.warning("      placement failed: %s", exc)
                    for _, sched_label in schedulers:
                        row = {
                            "circuit_name":        circuit_name,
                            "family":              family,
                            "num_qubits":          num_qubits,
                            "layout_type":         layout_type,
                            "layout_grid":         layout_grid,
                            "num_magic_in_layout": num_magic_in_layout,
                            "placement":           placement_mode,
                            "placement_cost":      None,
                            "scheduler":           sched_label,
                            "success":             False,
                            "timed_out":           False,
                            "error":               f"placement: {exc}",
                        }
                        flat_rows.append(row)
                        circuit_result["run_results"].append(row)
                    continue

                # Route with timeout – scheduler loop
                for sched_mode, sched_label in schedulers:
                    logger.info("      scheduler: %s", sched_label)
                    try:
                        metrics, schedule = run_single(
                            dag, engine,
                            layout_type=layout_type,
                            placement=placement_mode,
                            num_qubits=num_qubits,
                            timeout_s=args.scheduler_timeout,
                            scheduler_mode=sched_mode,
                            scheduler_label=sched_label,
                        )
                        row = {
                            "circuit_name":        circuit_name,
                            "family":              family,
                            "num_qubits":          num_qubits,
                            "layout_type":         layout_type,
                            "layout_grid":         layout_grid,
                            "num_magic_in_layout": num_magic_in_layout,
                            "placement":           placement_mode,
                            "placement_cost":      placement_cost,
                            **metrics,
                        }
                        flat_rows.append(row)
                        circuit_result["run_results"].append(row)

                        # Write per-run schedule JSON
                        if schedule is not None:
                            schedule["circuit_name"] = circuit_name
                            sched_name = (
                                f"{circ_idx:04d}_{_safe_name(family)}_"
                                f"{_safe_name(circuit_name)}_"
                                f"{_safe_name(layout_type)}_"
                                f"{_safe_name(placement_mode)}_"
                                f"{_safe_name(sched_label)}.json"
                            )
                            (schedules_dir / sched_name).write_text(
                                json.dumps(schedule, indent=2, default=str)
                            )

                        done_flag = (
                            "" if metrics.get("completed", True)
                            else (
                                f" [INCOMPLETE "
                                f"{metrics.get('num_nodes_processed', 0)}"
                                f"/{metrics.get('num_nodes_total', 0)} nodes]"
                            )
                        )
                        logger.info(
                            "        → %5d steps, wirelength %5d, %.2fs%s",
                            metrics["num_timesteps"],
                            metrics["total_wirelength"],
                            metrics["runtime_s"],
                            done_flag,
                        )

                    except _SchedulerTimeout:
                        logger.warning(
                            "        timeout after %ds: %s / %s / %s / %s",
                            args.scheduler_timeout,
                            circuit_name, layout_type, placement_mode, sched_label,
                        )
                        row = {
                            "circuit_name":        circuit_name,
                            "family":              family,
                            "num_qubits":          num_qubits,
                            "layout_type":         layout_type,
                            "layout_grid":         layout_grid,
                            "num_magic_in_layout": num_magic_in_layout,
                            "placement":           placement_mode,
                            "placement_cost":      placement_cost,
                            "scheduler":           sched_label,
                            "success":             False,
                            "timed_out":           True,
                            "error":               f"timeout after {args.scheduler_timeout}s",
                        }
                        flat_rows.append(row)
                        circuit_result["run_results"].append(row)

                    except Exception as exc:
                        logger.warning("        failed: %s", exc, exc_info=True)
                        row = {
                            "circuit_name":        circuit_name,
                            "family":              family,
                            "num_qubits":          num_qubits,
                            "layout_type":         layout_type,
                            "layout_grid":         layout_grid,
                            "num_magic_in_layout": num_magic_in_layout,
                            "placement":           placement_mode,
                            "placement_cost":      placement_cost,
                            "scheduler":           sched_label,
                            "success":             False,
                            "timed_out":           False,
                            "error":               str(exc),
                        }
                        flat_rows.append(row)
                        circuit_result["run_results"].append(row)

        if any(r.get("timed_out") for r in circuit_result["run_results"]):
            circuit_result["status"] = "timed_out"

        circuit_docs.append(circuit_result)

        per_path = (
            circuits_dir
            / f"{circ_idx:04d}_{_safe_name(family)}_{_safe_name(circuit_name)}.json"
        )
        per_path.write_text(json.dumps(circuit_result, indent=2, default=str))

        _write_checkpoint(
            run_dir=run_dir, ts=ts, qasm_dir=qasm_dir,
            exclude=exclude, max_qubits=max_qubits,
            qasm_files=qasm_files, max_circuits=args.max_circuits,
            circuit_docs=circuit_docs, flat_rows=flat_rows,
        )

    # Final checkpoint
    _write_checkpoint(
        run_dir=run_dir, ts=ts, qasm_dir=qasm_dir,
        exclude=exclude, max_qubits=max_qubits,
        qasm_files=qasm_files, max_circuits=args.max_circuits,
        circuit_docs=circuit_docs, flat_rows=flat_rows,
    )

    logger.info("Run complete.")
    logger.info("summary  : %s", run_dir / "summary.json")
    logger.info("rows     : %s", run_dir / "per_layout_rows.csv")
    logger.info("circuits : %s", circuits_dir)
    logger.info("schedules: %s", schedules_dir)


if __name__ == "__main__":
    main()
