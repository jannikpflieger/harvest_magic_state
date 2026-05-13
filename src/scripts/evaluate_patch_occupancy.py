#!/usr/bin/env python3
"""
Patch occupancy experiment — all benchmark circuits.

Measures how many routing cells and magic patches are NEVER used during routing
for each of the three base scheduling algorithms:
  Sequential (steiner_tree), Greedy (steiner_packing), Pathfinder (steiner_pathfinder)

Settings:
  - Layout  : circuit-aware, single-spaced  (StaticLayoutSynthesizer(num_lanes=1))
  - Magic   : always available (magic_prep_cycles=None / unlimited)
  - Tracked : routing cells (free (x,y) cells) + magic patches never selected

Outputs per run::

    results/patch_occupancy_<timestamp>/<circuit_name>.json  — per-circuit data
    results/patch_occupancy_<timestamp>/summary.json         — run metadata
    results/patch_occupancy_<timestamp>/summary.csv          — flat rows for analysis

Reproduce with::

    cd /path/to/harvest_magic_state
    PYTHONPATH=src python src/scripts/evaluate_patch_occupancy.py

Optional flags::

    --qasm-dir PATH           Root of .qasm benchmark tree (default: benchmark_circuits/qasm)
    --max-qubits N            Skip circuits with more than N logical qubits (default: 100)
    --exclude-families F …    Space-separated family names to skip (default: qv)
    --scheduler-timeout S     Per-scheduler wall-clock timeout in seconds (default: 120)
    --no-pathfinder           Skip the Pathfinder scheduler
    --out-dir PATH            Output root directory (default: repo root)
"""

import argparse
import csv
import json
import logging
import os
import signal
import statistics
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Path bootstrap — works when invoked as:
#   PYTHONPATH=src python src/scripts/evaluate_patch_occupancy.py
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
_src = str(PROJECT_ROOT / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from harvest.compilation.circuit_analysis import convert_rx_ry_to_rz, pre_prep_circuit
from harvest.compilation.pauli_block_conversion import convert_to_PCB, create_dag
from harvest.compilation.qasm_loader import qasm_to_circuit, find_qasm_files
from harvest.routing.magic_terminal_selection import get_magic_terminals
from harvest.routing.processor import DAGProcessor
from harvest.synthesis.synthesizer import StaticLayoutSynthesizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
)
logger = logging.getLogger("PatchOccupancy")

# Suppress the very verbose per-operation routing loggers so the run stays
# fast and the output stays readable.
for _noisy in ("HarvestMagicState", "HarvestMagicState.DAGProcessor",
               "HarvestMagicState.Detailed"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Algorithm list
# ---------------------------------------------------------------------------

ALL_SCHEDULERS: List[Tuple[str, str]] = [
    ("Sequential", "steiner_tree"),
    ("Greedy",     "steiner_packing"),
    ("Pathfinder", "steiner_pathfinder"),
]


# ---------------------------------------------------------------------------
# Per-scheduler timeout (SIGALRM — Linux/macOS only)
# ---------------------------------------------------------------------------

class _SchedulerTimeout(BaseException):
    """Raised by SIGALRM handler. Inherits BaseException so bare except clauses
    inside scheduler code do not silently catch it."""


def _alarm_handler(signum, frame):  # noqa: ARG001
    raise _SchedulerTimeout()


@contextmanager
def _alarm(seconds: int):
    """Raise :class:`_SchedulerTimeout` after *seconds* wall-clock seconds."""
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
# Circuit loading
# ---------------------------------------------------------------------------

def load_circuit_dag(qasm_path: str):
    """Load a QASM file → (dag, num_qubits, num_pauli_evolutions)."""
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


def _family_from_path(qasm_path: str, qasm_root: str) -> str:
    """Return top-level circuit family (first directory component under qasm_root)."""
    try:
        rel = Path(qasm_path).relative_to(qasm_root)
        return rel.parts[0]
    except ValueError:
        return Path(qasm_path).parent.name


# ---------------------------------------------------------------------------
# Utilization computation
# ---------------------------------------------------------------------------

def compute_patch_utilization(
    results: List[Dict],
    all_routing_cells: Set,
    all_magic: Set,
) -> Dict:
    """Compute routing-cell and magic-patch utilization from a list of result dicts.

    Args:
        results          : List of per-operation result dicts from process_entire_dag.
        all_routing_cells: Set of (x, y) tuple nodes from the initial routing graph.
        all_magic        : Set of magic patch names (kind == "magic") in the layout.

    Returns:
        Dict with keys:
            routing_total, routing_used, routing_unused, routing_unused_frac
            magic_total, magic_used, magic_unused, magic_unused_frac
    """
    used_routing: Set = set()
    for r in results:
        for node in r.get("steiner_nodes", set()):
            if isinstance(node, tuple):
                used_routing.add(node)

    used_magic: Set = {
        r["magic_terminal"]
        for r in results
        if r.get("magic_terminal")
    }

    unused_routing = all_routing_cells - used_routing
    unused_magic = all_magic - used_magic

    rt = len(all_routing_cells)
    mt = len(all_magic)

    return {
        "routing_total":        rt,
        "routing_used":         len(used_routing),
        "routing_unused":       len(unused_routing),
        "routing_unused_frac":  len(unused_routing) / rt if rt > 0 else 0.0,
        "magic_total":          mt,
        "magic_used":           len(used_magic),
        "magic_unused":         len(unused_magic),
        "magic_unused_frac":    len(unused_magic) / mt if mt > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Per-circuit processing
# ---------------------------------------------------------------------------

def run_circuit(
    qasm_path: str,
    qasm_root: str,
    schedulers: List[Tuple[str, str]],
    timeout_sec: int,
    max_qubits: Optional[int],
    out_dir: str,
) -> Dict:
    """Process one circuit: load → synthesize layout → run each scheduler → save JSON.

    Returns a summary dict describing the outcome.
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

    # ---- Synthesise layout (single-spaced, circuit-aware, 1 routing lane) ---
    try:
        synth = StaticLayoutSynthesizer(num_lanes=1)
        engine, report = synth.synthesize(dag)
    except Exception as exc:
        logger.warning(f"  Synthesis failed: {exc}")
        return {
            "circuit_name": circuit_name,
            "family": family,
            "qasm_path": str(qasm_path),
            "num_qubits": num_qubits,
            "status": "synthesis_failed",
            "error": str(exc),
        }

    # ---- Compute baseline sets (from the unmodified engine) ------------------
    # build_routing_graph() creates a fresh graph dict from engine state each
    # call and never modifies engine.patches / engine.occ, so this is safe to
    # call before any processor touches the layout.
    init_graph, init_ports_by_patch, _, _ = engine.build_routing_graph()
    all_routing_cells: Set = {n for n in init_graph if isinstance(n, tuple)}
    # magic terminals are M-type port node IDs (e.g. "P:mT1:M_S:M") —
    # same format as the 'magic_terminal' field in each result dict.
    all_magic: Set = set(get_magic_terminals(init_ports_by_patch))

    logger.info(
        f"  {num_qubits}q | {n_pe} T-ops | "
        f"{len(all_routing_cells)} routing cells | {len(all_magic)} magic terminals | "
        f"layout {report.grid_width}×{report.grid_height} ({report.template_name})"
    )

    # ---- Run each scheduler --------------------------------------------------
    scheduler_results: List[Dict] = []

    for sched_name, sched_mode in schedulers:
        try:
            with _alarm(timeout_sec):
                # Fresh processor each run — engine.build_routing_graph() is
                # called inside __init__ and produces an independent graph copy.
                processor = DAGProcessor(layout_engine=engine, magic_prep_cycles=None)
                t0 = time.perf_counter()
                results = processor.process_entire_dag(
                    dag, visualize_each_step=False, mode=sched_mode
                )
                runtime_s = round(time.perf_counter() - t0, 2)

            meta = getattr(processor, "_scheduling_metadata", {})
            util = compute_patch_utilization(results, all_routing_cells, all_magic)

            r = {
                "scheduler_name":    sched_name,
                "scheduler_mode":    sched_mode,
                "success":           True,
                "completed":         meta.get("completed", True),
                "num_timesteps":     meta.get("total_elapsed_steps", len(results)),
                "runtime_s":         runtime_s,
                **util,
            }
            logger.info(
                f"    {sched_name:<12}: T={r['num_timesteps']:>5,} | "
                f"routing unused {r['routing_unused']:>4}/{r['routing_total']:<4} "
                f"({r['routing_unused_frac']:5.1%}) | "
                f"magic terminal unused {r['magic_unused']:>3}/{r['magic_total']:<3} "
                f"({r['magic_unused_frac']:5.1%})"
            )

        except _SchedulerTimeout:
            logger.warning(f"    {sched_name}: TIMEOUT after {timeout_sec}s")
            r = {
                "scheduler_name": sched_name,
                "scheduler_mode": sched_mode,
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

        scheduler_results.append(r)

    # ---- Save per-circuit JSON -----------------------------------------------
    doc = {
        "circuit_name":       circuit_name,
        "family":             family,
        "qasm_path":          str(qasm_path),
        "num_qubits":         num_qubits,
        "num_pauli_evolutions": n_pe,
        "layout_template":    report.template_name,
        "layout_grid":        f"{report.grid_width}x{report.grid_height}",
        "num_routing_cells":  len(all_routing_cells),
        "num_magic_terminals": len(all_magic),
        "scheduler_results":  scheduler_results,
        "status":             "success",
    }
    json_path = os.path.join(out_dir, f"{circuit_name}.json")
    Path(json_path).write_text(json.dumps(doc, indent=2, default=str))

    return doc


# ---------------------------------------------------------------------------
# Aggregated output
# ---------------------------------------------------------------------------

def print_summary(all_docs: List[Dict], schedulers: List[Tuple[str, str]]) -> None:
    """Print a per-algorithm summary block to stdout."""
    from collections import defaultdict

    by_alg: Dict[str, List[Dict]] = defaultdict(list)
    for doc in all_docs:
        if doc.get("status") != "success":
            continue
        for r in doc.get("scheduler_results", []):
            if r.get("success"):
                by_alg[r["scheduler_name"]].append(r)

    col = [16, 9, 22, 16, 20, 14]
    sep = "+" + "+".join("-" * (c + 2) for c in col) + "+"
    hdr = (
        f"| {'Algorithm':<{col[0]}} | {'Circuits':>{col[1]}} "
        f"| {'Routing Unused (avg)':>{col[2]}} | {'Routing Unused%':>{col[3]}} "
        f"| {'Magic Unused (avg)':>{col[4]}} | {'Magic Unused%':>{col[5]}} |"
    )

    print("\n=== PATCH OCCUPANCY SUMMARY ===")
    print(sep)
    print(hdr)
    print(sep)

    for sched_name, _ in schedulers:
        entries = by_alg[sched_name]
        if not entries:
            print(
                f"| {sched_name:<{col[0]}} | {'0':>{col[1]}} "
                f"| {'N/A':>{col[2]}} | {'N/A':>{col[3]}} "
                f"| {'N/A':>{col[4]}} | {'N/A':>{col[5]}} |"
            )
            continue

        avg_ru  = statistics.mean(e["routing_unused"]      for e in entries)
        avg_ruf = statistics.mean(e["routing_unused_frac"] for e in entries)
        avg_mu  = statistics.mean(e["magic_unused"]        for e in entries)
        avg_muf = statistics.mean(e["magic_unused_frac"]   for e in entries)

        print(
            f"| {sched_name:<{col[0]}} | {len(entries):>{col[1]}} "
            f"| {avg_ru:>{col[2]}.1f} | {avg_ruf:>{col[3]}.1%} "
            f"| {avg_mu:>{col[4]}.1f} | {avg_muf:>{col[5]}.1%} |"
        )

    print(sep)
    print()


def save_summary_csv(
    all_docs: List[Dict],
    out_dir: str,
) -> None:
    """Write a flat CSV where each row is one (circuit, scheduler) pair."""
    rows: List[Dict] = []
    for doc in all_docs:
        if doc.get("status") != "success":
            continue
        base = {
            "circuit_name":          doc["circuit_name"],
            "family":                doc["family"],
            "num_qubits":            doc["num_qubits"],
            "num_pauli_evolutions":  doc["num_pauli_evolutions"],
            "layout_template":       doc.get("layout_template", ""),
            "layout_grid":           doc.get("layout_grid", ""),
            "num_routing_cells":     doc.get("num_routing_cells", ""),
            "num_magic_terminals":   doc.get("num_magic_terminals", ""),
        }
        for r in doc.get("scheduler_results", []):
            if not r.get("success"):
                continue
            row = {
                **base,
                "scheduler":            r["scheduler_name"],
                "completed":            r.get("completed", ""),
                "num_timesteps":        r.get("num_timesteps", ""),
                "runtime_s":            r.get("runtime_s", ""),
                "routing_total":        r.get("routing_total", ""),
                "routing_used":         r.get("routing_used", ""),
                "routing_unused":       r.get("routing_unused", ""),
                "routing_unused_frac":  round(r.get("routing_unused_frac", 0.0), 6),
                "magic_total":          r.get("magic_total", ""),
                "magic_used":           r.get("magic_used", ""),
                "magic_unused":         r.get("magic_unused", ""),
                "magic_unused_frac":    round(r.get("magic_unused_frac", 0.0), 6),
            }
            rows.append(row)

    if not rows:
        logger.warning("No successful results to write to CSV.")
        return

    csv_path = os.path.join(out_dir, "summary.csv")
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Summary CSV saved → {csv_path}")


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Measure routing-cell and magic-patch occupancy for all benchmark circuits. "
            "Uses single-spaced circuit-aware layout with unlimited magic states."
        ),
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
        metavar="F",
        help="Circuit families to exclude (matched against top-level qasm subdir).",
    )
    parser.add_argument(
        "--scheduler-timeout", type=int, default=120, metavar="SEC",
        help="Per-scheduler wall-clock timeout in seconds.",
    )
    parser.add_argument(
        "--no-pathfinder", action="store_true",
        help="Skip the Pathfinder scheduler.",
    )
    parser.add_argument(
        "--out-dir", type=str, default=str(PROJECT_ROOT),
        help="Root directory for output (results/ subfolder will be created here).",
    )
    args = parser.parse_args()

    max_q = args.max_qubits if args.max_qubits > 0 else None
    exclude = set(args.exclude_families or [])

    schedulers = [
        s for s in ALL_SCHEDULERS
        if not (args.no_pathfinder and s[1] == "steiner_pathfinder")
    ]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.out_dir, "results", f"patch_occupancy_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"Output directory: {out_dir}")
    logger.info(f"Schedulers: {[s[0] for s in schedulers]}")
    logger.info(f"Layout: circuit-aware, num_lanes=1 (single-spaced), magic always available")

    # ---- Discover circuits ---------------------------------------------------
    qasm_files = find_qasm_files(args.qasm_dir)
    qasm_files = [
        f for f in qasm_files
        if _family_from_path(f, args.qasm_dir) not in exclude
    ]
    logger.info(f"Found {len(qasm_files)} circuits after exclusions")

    # ---- Process -------------------------------------------------------------
    all_docs: List[Dict] = []
    for i, qasm_path in enumerate(sorted(qasm_files), 1):
        circuit_name = Path(qasm_path).stem
        logger.info(f"[{i}/{len(qasm_files)}] {circuit_name}")
        doc = run_circuit(
            qasm_path=qasm_path,
            qasm_root=args.qasm_dir,
            schedulers=schedulers,
            timeout_sec=args.scheduler_timeout,
            max_qubits=max_q,
            out_dir=out_dir,
        )
        all_docs.append(doc)

    # ---- Aggregate and save --------------------------------------------------
    n_success  = sum(1 for d in all_docs if d.get("status") == "success")
    n_skipped  = sum(1 for d in all_docs if d.get("status", "").startswith("skipped"))
    n_clifford = sum(1 for d in all_docs if d.get("status") == "clifford_only")
    n_failed   = sum(
        1 for d in all_docs
        if d.get("status") in ("load_failed", "synthesis_failed")
    )

    summary = {
        "timestamp":            timestamp,
        "total_circuits_found": len(qasm_files),
        "circuits_processed":   n_success,
        "circuits_skipped":     n_skipped,
        "circuits_clifford":    n_clifford,
        "circuits_failed":      n_failed,
        "schedulers":           [s[0] for s in schedulers],
        "magic_always_available": True,
        "layout": "circuit_aware_single_spaced_num_lanes_1",
        "max_qubits_filter":    max_q,
        "excluded_families":    list(exclude),
    }
    Path(os.path.join(out_dir, "summary.json")).write_text(
        json.dumps(summary, indent=2)
    )

    save_summary_csv(all_docs, out_dir)
    print_summary(all_docs, schedulers)

    logger.info(
        f"Finished. Processed {n_success} circuits "
        f"(skipped {n_skipped} too-large, {n_clifford} Clifford-only, {n_failed} failed)."
    )


if __name__ == "__main__":
    main()
