#!/usr/bin/env python3
"""
Benchmark Circuit Landscape Plot

Scatter plot summarising the benchmark circuit suite:
  x-axis:  T-count  (total Pauli evolutions after PCB conversion)
  y-axis:  T-depth  (number of Pauli-evolution layers)
  size:    number of logical qubits
  colour:  average Pauli products per layer  (parallelism proxy)
  label:   circuit name

Missing analysis results (e.g. for qasmbench-small/medium/large circuits)
are computed on-the-fly with a per-circuit timeout.

Usage:
    python scripts/plot_benchmark_landscape.py
    python scripts/plot_benchmark_landscape.py --skip-labels
    python scripts/plot_benchmark_landscape.py --timeout 120 --no-run-missing
"""

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

# ---------------------------------------------------------------------------
# Resolve project root so the script works from any cwd
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent          # .../harvest_magic_state
sys.path.insert(0, str(PROJECT_ROOT / "src"))

DEFAULT_ANALYSIS_DIR = PROJECT_ROOT / "benchmark_circuits" / "circuit_analysis_results"
DEFAULT_QASM_DIR     = PROJECT_ROOT / "benchmark_circuits" / "qasm"
DEFAULT_OUTPUT       = PROJECT_ROOT / "plots" / "benchmark_landscape"


# ---------------------------------------------------------------------------
# Timeout helper  (Linux-only, uses SIGALRM)
# ---------------------------------------------------------------------------
class AnalysisTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise AnalysisTimeout("analysis timed out")


# ---------------------------------------------------------------------------
# 1. Discover missing analyses
# ---------------------------------------------------------------------------
def _circuit_id_from_path(qasm_path: str) -> str:
    """Reproduce the circuit_id naming convention used by analyze_single_circuit."""
    rel = os.path.relpath(qasm_path, PROJECT_ROOT)
    return rel.replace("/", "_").replace(".qasm", "")


def _quick_qubit_count(qasm_file: str) -> int | None:
    """Peek at a QASM file to get the qubit count without full parsing."""
    try:
        with open(qasm_file) as f:
            for line in f:
                line = line.strip()
                if line.startswith("qreg ") or line.startswith("qubit["):
                    # qreg q[N];  or  qubit[N] q;
                    import re
                    m = re.search(r"\[(\d+)\]", line)
                    if m:
                        return int(m.group(1))
    except OSError:
        pass
    return None


def _quick_depth(qasm_file: str) -> int | None:
    """Load a QASM circuit just far enough to read its depth."""
    try:
        from qiskit import QuantumCircuit
        qc = QuantumCircuit.from_qasm_file(qasm_file)
        return qc.depth()
    except Exception:
        return None


def discover_missing_analyses(qasm_dir: str, analysis_dir: str,
                              max_qubits: int | None = None,
                              max_depth: int | None = None,
                              exclude_families: set[str] | None = None):
    """Return list of .qasm files that have no analysis JSON yet.

    Skips *_transpiled.qasm files to avoid duplicates.
    If *max_qubits* is set, skips circuits with more qubits (quick header peek).
    If *max_depth* is set, skips circuits whose depth exceeds the limit.
    If *exclude_families* is set, skips circuits whose parent folder matches.
    """
    from harvest.compilation.qasm_loader import find_qasm_files

    all_qasm = find_qasm_files(qasm_dir)

    # Filter out transpiled variants
    all_qasm = [f for f in all_qasm if not f.endswith("_transpiled.qasm")]

    # Filter out excluded circuit families (by parent directory name)
    if exclude_families:
        before = len(all_qasm)
        all_qasm = [f for f in all_qasm
                    if not any(excl in Path(f).parts for excl in exclude_families)]
        print(f"Family filter (exclude {exclude_families}): kept {len(all_qasm)}/{before} circuits")

    # Filter by qubit count (cheap header scan)
    if max_qubits is not None:
        before = len(all_qasm)
        kept = []
        for qf in all_qasm:
            nq = _quick_qubit_count(qf)
            if nq is None or nq <= max_qubits:
                kept.append(qf)
        all_qasm = kept
        print(f"Qubit filter (≤{max_qubits}): kept {len(all_qasm)}/{before} circuits")

    existing = {p.stem for p in Path(analysis_dir).glob("*.json")}

    # Only consider circuits that are actually missing an analysis JSON
    candidates = []
    for qf in all_qasm:
        cid = _circuit_id_from_path(qf)
        if cid not in existing:
            candidates.append(qf)

    # Filter missing circuits by depth (needs Qiskit load — only for missing ones)
    missing = []
    if max_depth is not None and candidates:
        print(f"Checking depth for {len(candidates)} missing circuits (limit ≤{max_depth})…")
        for qf in candidates:
            d = _quick_depth(qf)
            if d is None or d <= max_depth:
                missing.append(qf)
            else:
                print(f"  Depth filter: skipping {os.path.basename(qf)} (depth={d})")
        print(f"Depth filter: kept {len(missing)}/{len(candidates)} missing circuits")
    else:
        missing = candidates

    print(f"Total non-transpiled QASM files: {len(all_qasm)}")
    print(f"Existing analysis JSONs:         {len(existing)}")
    print(f"Missing analyses:                {len(missing)}")
    return missing


# ---------------------------------------------------------------------------
# 2. Analyse a single circuit with timeout
# ---------------------------------------------------------------------------
def analyze_with_timeout(qasm_file: str, output_dir: str, timeout_sec: int):
    """Run analyze_single_circuit with a SIGALRM timeout.

    Returns the result dict, or None if the analysis timed out or failed.
    """
    from harvest.compilation.circuit_analysis import analyze_single_circuit

    old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(timeout_sec)
    try:
        result = analyze_single_circuit(qasm_file, output_dir=output_dir)
        signal.alarm(0)  # cancel alarm
        return result
    except AnalysisTimeout:
        name = os.path.basename(qasm_file)
        print(f"  TIMEOUT after {timeout_sec}s — skipping {name}")
        return None
    except Exception as exc:
        signal.alarm(0)
        name = os.path.basename(qasm_file)
        print(f"  ERROR analysing {name}: {exc}")
        return None
    finally:
        signal.signal(signal.SIGALRM, old_handler)


# ---------------------------------------------------------------------------
# 3. Load landscape data from analysis JSONs
# ---------------------------------------------------------------------------
def load_landscape_data(analysis_dir: str, max_qubits: int | None = None,
                        max_depth: int | None = None,
                        exclude_families: set[str] | None = None):
    """Read all analysis JSONs and return a list of dicts with the 5 metrics.

    Only circuits with successful DAG analysis and total_pauli_evolutions > 0
    are included.  If *max_qubits* is set, circuits with more qubits are skipped.
    If *max_depth* is set, circuits whose original depth exceeds the limit are
    skipped.  If *exclude_families* is set, circuits from those families are
    skipped.
    """
    records = []
    for jf in sorted(Path(analysis_dir).glob("*.json")):
        try:
            data = json.loads(jf.read_text())
        except (json.JSONDecodeError, OSError):
            continue

        circ = data.get("circuit", {})

        # Skip transpiled duplicates that may already exist
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

        # Derive circuit family from file path
        fpath_parts = Path(fpath).parts  # e.g. ('benchmark_circuits','qasm','qaoa',...)
        fam_check = fpath_parts[2] if len(fpath_parts) > 2 else ""
        if exclude_families and fam_check in exclude_families:
            continue

        evol_per_layer = circ.get("pauli_evolutions_per_layer", [])
        if evol_per_layer:
            avg_pp = sum(evol_per_layer) / len(evol_per_layer)
        else:
            avg_pp = 0.0

        # Derive circuit family from file path
        parts = Path(fpath).parts  # e.g. ('benchmark_circuits','qasm','qaoa',...)
        family = parts[2] if len(parts) > 2 else "unknown"

        records.append({
            "circuit_name": circ.get("circuit_name", jf.stem),
            "family": family,
            "t_count": t_count,
            "t_depth": t_depth,
            "num_qubits": n_qubits,
            "avg_pp_per_layer": avg_pp,
        })

    print(f"Loaded {len(records)} circuits with valid PCB metrics")
    return records


# ---------------------------------------------------------------------------
# 4. Create the landscape scatter plot
# ---------------------------------------------------------------------------
def create_landscape_plot(records, output_path, log_scale=True, skip_labels=False):
    """Produce the benchmark-landscape scatter plot."""
    if not records:
        print("No data to plot.")
        return

    t_count   = np.array([r["t_count"]        for r in records])
    t_depth   = np.array([r["t_depth"]         for r in records])
    n_qubits  = np.array([r["num_qubits"]      for r in records], dtype=float)
    avg_pp    = np.array([r["avg_pp_per_layer"] for r in records])
    names     = [r["circuit_name"] for r in records]

    # --- size scaling: area proportional to qubits, with a visible floor ----
    size_scale = 800.0 / max(n_qubits.max(), 1)
    sizes = n_qubits * size_scale
    sizes = np.clip(sizes, 15, None)  # minimum visible size

    fig, ax = plt.subplots(figsize=(12, 8))

    sc = ax.scatter(
        t_count, t_depth,
        s=sizes,
        c=avg_pp,
        cmap="viridis",
        alpha=0.75,
        edgecolors="grey",
        linewidths=0.4,
    )

    # --- colour bar ---
    cbar = fig.colorbar(sc, ax=ax, pad=0.02, shrink=0.85)
    cbar.set_label("Avg Pauli products / layer", fontsize=10)

    # --- axis labels ---
    ax.set_xlabel("T-count  (total Pauli evolutions)", fontsize=11)
    ax.set_ylabel("T-depth  (Pauli-evolution layers)", fontsize=11)
    ax.set_title("Benchmark Circuit Landscape", fontsize=13, fontweight="bold")

    if log_scale:
        ax.set_xscale("log")
        ax.set_yscale("log")

    ax.grid(True, which="both", alpha=0.25, linestyle="--")

    # --- size legend (reference circles) ---
    ref_qubits = _pick_reference_sizes(n_qubits)
    legend_handles = []
    for q in ref_qubits:
        s = max(q * size_scale, 15)
        h = mlines.Line2D(
            [], [],
            marker="o",
            color="w",
            markeredgecolor="grey",
            markersize=np.sqrt(s),
            linestyle="None",
            label=f"{int(q)} qubits",
        )
        legend_handles.append(h)
    ax.legend(
        handles=legend_handles,
        title="Logical qubits",
        loc="upper left",
        fontsize=8,
        title_fontsize=9,
        framealpha=0.8,
    )

    # --- circuit-name labels ---
    if not skip_labels:
        for i, name in enumerate(names):
            ax.annotate(
                _short_name(name),
                (t_count[i], t_depth[i]),
                fontsize=5,
                alpha=0.7,
                textcoords="offset points",
                xytext=(4, 3),
            )

    plt.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out) + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(str(out) + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}.png  and  {out}.pdf")


def _pick_reference_sizes(qubits_arr):
    """Choose 3-4 nice round reference qubit counts for the size legend."""
    lo, hi = int(qubits_arr.min()), int(qubits_arr.max())
    candidates = [2, 5, 10, 20, 50, 100, 200, 500]
    chosen = [c for c in candidates if lo <= c <= hi]
    if not chosen:
        chosen = [lo, hi]
    # Keep at most 4
    while len(chosen) > 4:
        chosen.pop(len(chosen) // 2)
    return chosen


def _short_name(name: str) -> str:
    """Shorten long circuit names for annotation labels."""
    # Strip common prefixes
    for prefix in ("qaoa_barabasi_albert_", "square_heisenberg_"):
        if name.startswith(prefix):
            return name[len(prefix):]
    if len(name) > 25:
        return name[:22] + "…"
    return name


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate a benchmark-circuit landscape scatter plot."
    )
    parser.add_argument(
        "--analysis-dir", type=str, default=str(DEFAULT_ANALYSIS_DIR),
        help="Directory with circuit analysis JSON files.",
    )
    parser.add_argument(
        "--qasm-dir", type=str, default=str(DEFAULT_QASM_DIR),
        help="Root directory of .qasm benchmark files.",
    )
    parser.add_argument(
        "--output", type=str, default=str(DEFAULT_OUTPUT),
        help="Output path (without extension). .png and .pdf are appended.",
    )
    parser.add_argument(
        "--no-run-missing", action="store_true",
        help="Skip analysing circuits that lack a result JSON.",
    )
    parser.add_argument(
        "--timeout", type=int, default=60,
        help="Per-circuit analysis timeout in seconds (default: 60).",
    )
    parser.add_argument(
        "--log-scale", action="store_true", default=True,
        help="Use log-log axes (default).",
    )
    parser.add_argument(
        "--linear", action="store_true",
        help="Use linear axes instead of log-log.",
    )
    parser.add_argument(
        "--skip-labels", action="store_true",
        help="Omit circuit-name annotations (cleaner plot).",
    )
    parser.add_argument(
        "--max-qubits", type=int, default=70,
        help="Skip circuits with more than this many qubits (default: 70). "
             "Set to 0 to disable the filter.",
    )
    parser.add_argument(
        "--max-depth", type=int, default=1000,
        help="Skip circuits whose original depth exceeds this limit (default: 1000). "
             "Set to 0 to disable the filter.",
    )
    parser.add_argument(
        "--exclude-families", type=str, nargs="*", default=["qv"],
        help='Circuit families (subfolder names) to exclude entirely (default: ["qv"]).',
    )

    args = parser.parse_args()
    use_log = args.log_scale and not args.linear
    max_q = args.max_qubits if args.max_qubits > 0 else None
    max_d = args.max_depth if args.max_depth > 0 else None
    excl = set(args.exclude_families) if args.exclude_families else None

    # Phase 1: fill gaps
    if not args.no_run_missing:
        missing = discover_missing_analyses(args.qasm_dir, args.analysis_dir,
                                             max_qubits=max_q, max_depth=max_d,
                                             exclude_families=excl)
        if missing:
            print(f"\nAnalysing {len(missing)} missing circuits (timeout {args.timeout}s each)…")
            for i, qf in enumerate(missing, 1):
                name = os.path.basename(qf)
                print(f"  [{i}/{len(missing)}] {name} …", end=" ", flush=True)
                t0 = time.time()
                result = analyze_with_timeout(qf, args.analysis_dir, args.timeout)
                if result and result.get("analysis_status") == "SUCCESS":
                    elapsed = time.time() - t0
                    print(f"OK ({elapsed:.1f}s)")
                elif result is None:
                    pass  # timeout/error already printed
                else:
                    print(f"status={result.get('analysis_status')}")
        else:
            print("All circuits already analysed.")

    # Phase 2: load data
    records = load_landscape_data(args.analysis_dir, max_qubits=max_q, max_depth=max_d,
                                   exclude_families=excl)

    # Phase 3: plot
    create_landscape_plot(records, args.output, log_scale=use_log, skip_labels=args.skip_labels)


if __name__ == "__main__":
    main()
