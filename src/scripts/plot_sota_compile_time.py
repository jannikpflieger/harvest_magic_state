#!/usr/bin/env python3
"""Compiler wall-clock time versus circuit size, from stored raw records.

Reads ``compiler_runtime_s`` of every successful trial (nothing is rerun) and
plots one panel per comparison regime: x = T count of the Clifford+T source
(from the input ``manifest.json``), y = compile time per circuit.  A compiler
with several configurations (Litinski blocks, O3LS boards, LSQCA floorplans,
TACO block counts) contributes the geometric mean over its configurations for
each circuit, i.e. the time of one compile; stochastic tools (PureMagic) the
arithmetic mean over their seeded trials.  Each compiler is drawn as faint
per-circuit points plus a line through geometric means of log-spaced T-count
bins (only bins that contain measurements).

Runtime scope follows each adapter (see ``docs/sota_baselines.md``): matched-IR
HARVEST/Silva times cover scheduling (+ HARVEST pruning); native HARVEST also
includes QASM loading, PCB conversion and layout; PureMagic and TACO include
their subprocesses; Litinski is a closed-form evaluation; O3LS, LSQCA and the
TACO scheduler are HARVEST-side reimplementations, not the original tools.

Example::

    PYTHONPATH=src python src/scripts/plot_sota_compile_time.py \\
        --panel "Matched IR=results/sota_comparison_feynman_matched_ir:matched_ir" \\
        --panel "Native end-to-end=results/sota_comparison_feynman_native_e2e:native_end_to_end" \\
        --inputs results/sota_feynman_cliffordt_inputs results/sota_extended_cliffordt_inputs \\
        --exclude-baselines dascot --output-dir results/sota_compile_time
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/harvest-matplotlib")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from harvest.baselines.metrics import geometric_mean
from harvest.baselines.tradeoff import TradeoffDataError, load_raw_payloads
from scripts.compare_sota_baselines import COLORS, DISPLAY
from scripts.plot_sota_space_time_tradeoff import INK, MARKERS, MUTED, PLOT_ORDER, _style


def load_t_counts(input_dirs: Sequence[Path]) -> Dict[str, int]:
    """T count per circuit from the lowering manifests (source of truth for size)."""
    counts: Dict[str, int] = {}
    for directory in input_dirs:
        manifest = Path(directory) / "manifest.json"
        if not manifest.exists():
            raise TradeoffDataError(f"{directory} has no manifest.json")
        for record in json.loads(manifest.read_text()):
            counts[Path(record["output"]).stem] = int(record["t_count"])
    return counts


def collect(
    result_dir: Path, mode: str, exclude: Sequence[str]
) -> Tuple[Dict[Tuple[str, str], List[float]], List[str]]:
    """{(compiler, circuit): [per-configuration mean runtime]} plus notes."""
    per_config: Dict[Tuple[str, str, str], List[float]] = defaultdict(list)
    notes: List[str] = []
    missing_runtime = 0
    for payload in load_raw_payloads(result_dir):
        result = payload["result"]
        if result.get("comparison_mode") != mode or result.get("baseline") in exclude:
            continue
        if result.get("completed") is not True:
            continue
        runtime = result.get("compiler_runtime_s")
        if runtime is None or runtime <= 0:
            missing_runtime += 1
            continue
        compiler = DISPLAY.get((result["baseline"], result["variant"]), result["variant"])
        per_config[(compiler, result["variant"], result["circuit"])].append(float(runtime))
    if missing_runtime:
        notes.append(f"{result_dir}: {missing_runtime} completed record(s) without a runtime skipped")
    grouped: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for (compiler, _variant, circuit), runtimes in per_config.items():
        grouped[(compiler, circuit)].append(mean(runtimes))  # mean over seeded trials
    return grouped, notes


def binned_trend(xs: Sequence[float], ys: Sequence[float], bins: int) -> List[Tuple[float, float]]:
    """Geometric means of (x, y) inside log-spaced x bins that hold measurements."""
    lo, hi = math.log10(min(xs)), math.log10(max(xs))
    if hi - lo < 1e-9:
        return [(geometric_mean(xs), geometric_mean(ys))]
    width = (hi - lo) / bins
    buckets: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
    for x, y in zip(xs, ys):
        buckets[min(bins - 1, int((math.log10(x) - lo) / width))].append((x, y))
    return [
        (geometric_mean(x for x, _ in pts), geometric_mean(y for _, y in pts))
        for _, pts in sorted(buckets.items())
    ]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--panel", action="append", required=True, metavar="TITLE=DIR:MODE",
                        help="One panel per comparison regime (regimes are never pooled).")
    parser.add_argument("--inputs", nargs="+", type=Path, required=True,
                        help="Input directories whose manifest.json gives each circuit's T count.")
    parser.add_argument("--exclude-baselines", default="")
    parser.add_argument("--bins", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-stem", default="compile_time")
    parser.add_argument("--width", type=float, default=6.9, help="inches (two columns)")
    parser.add_argument("--height", type=float, default=2.6)
    args = parser.parse_args(argv)

    exclude = [b.strip() for b in args.exclude_baselines.split(",") if b.strip()]
    t_counts = load_t_counts(args.inputs)
    panels = []
    for spec in args.panel:
        title, rest = spec.split("=", 1)
        directory, mode = rest.rsplit(":", 1)
        grouped, notes = collect(Path(directory), mode, exclude)
        unknown = sorted({c for _, c in grouped} - set(t_counts))
        if unknown:
            raise TradeoffDataError(f"no T count for {unknown}; add their input directory")
        panels.append((title, mode, grouped, notes))

    _style()
    fig, axes = plt.subplots(1, len(panels), figsize=(args.width, args.height), sharey=True,
                             squeeze=False)
    rows = []
    legend_compilers: List[str] = []
    for ax, (title, mode, grouped, notes) in zip(axes[0], panels):
        for note in notes:
            print(f"note: {note}")
        compilers = [c for c in PLOT_ORDER if any(k[0] == c for k in grouped)]
        compilers += sorted({k[0] for k in grouped} - set(compilers))
        print(f"== {title} ({mode}): compile time per circuit")
        for compiler in compilers:
            points = sorted(
                (t_counts[circuit], geometric_mean(runtimes), circuit, len(runtimes))
                for (name, circuit), runtimes in grouped.items()
                if name == compiler
            )
            xs, ys = [p[0] for p in points], [p[1] for p in points]
            color, marker = COLORS.get(compiler, MUTED), MARKERS.get(compiler, "o")
            emphasized = compiler == "HARVEST"
            ax.scatter(xs, ys, s=10 if emphasized else 8, marker=marker, facecolor=color,
                       edgecolor="none", alpha=0.25, zorder=2)
            trend = binned_trend(xs, ys, args.bins)
            ax.plot([x for x, _ in trend], [y for _, y in trend], color=color,
                    linewidth=1.6 if emphasized else 1.1, marker=marker,
                    markersize=3.8 if emphasized else 3.2, markeredgecolor="white",
                    markeredgewidth=0.4, zorder=4 if emphasized else 3)
            if compiler not in legend_compilers:
                legend_compilers.append(compiler)
            print(f"  {compiler:14s} n={len(ys):3d}  geomean={geometric_mean(ys):9.4f}s  "
                  f"median={median(ys):9.4f}s  max={max(ys):9.2f}s  total={sum(ys):9.1f}s")
            for x, y, circuit, n_configs in points:
                rows.append({"panel": title, "comparison_mode": mode, "compiler": compiler,
                             "circuit": circuit, "t_count": x, "compile_time_s": y,
                             "configurations": n_configs})
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(title, fontsize=8, color=INK)
        ax.set_xlabel("T count of the circuit")
        ax.grid(True, which="major", color="#E6E6E6", linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    seconds = FuncFormatter(lambda v, _: f"{v:g} s" if v >= 1 else f"{v * 1000:g} ms")
    axes[0][0].yaxis.set_major_formatter(seconds)
    axes[0][0].set_ylabel("Compile time per circuit")
    handles = [
        plt.Line2D([], [], color=COLORS.get(c, MUTED), marker=MARKERS.get(c, "o"),
                   markeredgecolor="white", markeredgewidth=0.4, markersize=4, linewidth=1.1,
                   label=c)
        for c in legend_compilers
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False,
               ncol=min(len(handles), 7), handletextpad=0.3, columnspacing=1.0)
    fig.tight_layout(pad=0.3)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / args.output_stem
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    with stem.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for suffix in (".pdf", ".png", ".csv"):
        print(f"wrote {stem.with_suffix(suffix)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
