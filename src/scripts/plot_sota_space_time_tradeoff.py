#!/usr/bin/env python3
"""Matched-IR space/time trade-off figure, normalized by per-benchmark lower bounds.

Only ``comparison_mode == "matched_ir"`` records are used; DASCOT never
appears (it has no matched-IR path).  For every benchmark:

* ``time_lb``  = dependency critical path of the matched Pauli-product DAG,
  one logical cycle per schedulable Pauli-product operation;
* ``space_lb`` = ``num_qubits + has_magic_state_operation`` (one data patch per
  logical qubit, plus one magic-state patch if the stream consumes a magic
  state; routing space is not counted).

See ``harvest.baselines.tradeoff`` for the exact definitions.  Each point is
``(logical_patches / space_lb, schedule_length / time_lb)``; stochastic
systems (PureMagic) contribute the arithmetic mean schedule length over their
successful trials of one configuration, matching ``summary.csv``.

With several result directories (a resource sweep), every successful
configuration becomes a point, the non-dominated set in
``(logical_patches, schedule_length)`` is computed per compiler and circuit,
and only Pareto-optimal measured points are connected.

Example::

    PYTHONPATH=src python src/scripts/plot_sota_space_time_tradeoff.py \\
        results/sota_comparison_matched_ir
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/harvest-matplotlib")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

from harvest.baselines.tradeoff import (
    TradeoffDataError,
    TradeoffPoint,
    aggregate_points,
    compiler_geomeans,
    compute_bounds,
    load_raw_payloads,
    mark_pareto,
    normalize_point,
    shared_circuits,
)

# Colors are imported from the comparison driver so a compiler has the same
# color in every SOTA figure.  The existing palette has a weak red/green and
# red/orange pair, so identity is also carried by marker shape and labels.
from scripts.compare_sota_baselines import COLORS, DISPLAY

CSV_FIELDS = [
    "circuit",
    "compiler",
    "variant",
    "logical_patches",
    "schedule_length",
    "space_lb",
    "time_lb",
    "normalized_space",
    "normalized_time",
    "normalized_space_time",
    # provenance beyond the requested columns
    "config_id",
    "source_dir",
    "num_qubits",
    "schedule_length_min",
    "schedule_length_max",
    "trials_successful",
    "trials_total",
    "pareto_optimal",
    "in_shared_subset",
]
PLOT_ORDER = ["HARVEST", "Silva EAF", "PureMagic Bus", "PureMagic"]
MARKERS = {"HARVEST": "o", "Silva EAF": "s", "PureMagic Bus": "^", "PureMagic": "D"}
ISO_PRODUCTS = (2, 4, 8, 16)
# Geomean label offsets (points, ha, va), chosen to clear per-benchmark marks.
LABEL_PLACEMENT = {
    "HARVEST": (0, 7, "center", "bottom"),
    "Silva EAF": (0, 6.5, "center", "bottom"),
    "PureMagic Bus": (0, 6.5, "center", "bottom"),
    "PureMagic": (-7, 0, "right", "center"),
}
INK = "#222222"
MUTED = "#777777"
GUIDE = "#BBBBBB"


def _display(baseline: str, variant: str) -> str:
    return DISPLAY.get((baseline, variant), variant)


def load_points(
    result_dirs: Sequence[Path], include_unshared: bool
) -> Dict[str, object]:
    """Load, validate, bound, and normalize every matched-IR point."""
    from scripts.compare_sota_baselines import load_benchmark

    all_points: List[TradeoffPoint] = []
    failures = []
    notes: List[str] = []
    bounds_by_circuit = {}
    for result_dir in result_dirs:
        payloads = load_raw_payloads(result_dir)
        points, dir_failures, dir_notes = aggregate_points(
            payloads, str(result_dir), _display
        )
        if not points:
            raise TradeoffDataError(f"{result_dir}: no successful matched_ir result")
        bounds = compute_bounds(
            payloads,
            result_dir,
            PROJECT_ROOT,
            load_benchmark,
            {p.circuit for p in points},
        )
        for circuit, bound in bounds.items():
            previous = bounds_by_circuit.setdefault(circuit, bound)
            if (previous.time_lb, previous.space_lb) != (bound.time_lb, bound.space_lb):
                raise TradeoffDataError(
                    f"{circuit}: lower bounds differ between result directories"
                )
        all_points.extend(normalize_point(p, bounds[p.circuit]) for p in points)
        failures.extend(dir_failures)
        notes.extend(dir_notes)

    compilers = [c for c in PLOT_ORDER if any(p.compiler == c for p in all_points)]
    compilers += sorted({p.compiler for p in all_points} - set(compilers))
    shared = shared_circuits(all_points, compilers)
    for p in all_points:
        p.in_shared_subset = p.circuit in shared
    mark_pareto(all_points)
    plotted = all_points if include_unshared else [p for p in all_points if p.in_shared_subset]
    if not plotted:
        raise TradeoffDataError("no circuit is shared by all plotted systems")
    return {
        "all_points": all_points,
        "plotted": plotted,
        "compilers": compilers,
        "shared": shared,
        "failures": failures,
        "notes": notes,
        "bounds": bounds_by_circuit,
    }


def write_csv(path: Path, points: Sequence[TradeoffPoint]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for p in sorted(points, key=lambda q: (q.circuit, q.compiler, q.config_id)):
            writer.writerow(p.to_row())


def print_summary(data: Dict[str, object], geomeans: Dict[str, Dict[str, float]]) -> None:
    all_points: List[TradeoffPoint] = data["all_points"]  # type: ignore[assignment]
    compilers: List[str] = data["compilers"]  # type: ignore[assignment]
    shared: List[str] = data["shared"]  # type: ignore[assignment]
    print("== Matched-IR space/time trade-off ==")
    for note in data["notes"]:  # type: ignore[union-attr]
        print(f"note: {note}")
    print(f"systems: {', '.join(compilers)}")
    print(f"circuits shared by all systems ({len(shared)}): {', '.join(shared) or '-'}")
    by_circuit = defaultdict(set)
    for p in all_points:
        by_circuit[p.circuit].add(p.compiler)
    for circuit in sorted(set(by_circuit) - set(shared)):
        missing = sorted(set(compilers) - by_circuit[circuit])
        print(f"  not shared: {circuit} (no successful result for {', '.join(missing)})")
    for f in data["failures"]:  # type: ignore[union-attr]
        status = "FAILED" if f.trials_failed == f.trials_total else "partially failed"
        print(
            f"  {status}: {f.compiler} on {f.circuit} "
            f"({f.trials_failed}/{f.trials_total} trials): {f.errors[0][:110]}"
        )
    print("lower bounds:")
    for circuit, b in sorted(data["bounds"].items()):  # type: ignore[union-attr]
        print(
            f"  {circuit}: n={b.num_qubits} ops={b.num_operations} "
            f"time_lb={b.time_lb} space_lb={b.space_lb} [{b.source}]"
        )
    below = [p for p in all_points if p.normalized_time < 1.0]
    for p in below:
        print(f"  note: {p.compiler} on {p.circuit} beats time_lb ({p.normalized_time:.2f}x)")
    for compiler in compilers:
        pts = [p for p in all_points if p.compiler == compiler and p.in_shared_subset]
        tight = sum(math.isclose(p.normalized_time, 1.0) for p in pts)
        print(f"  {compiler}: meets time_lb exactly on {tight}/{len(pts)} shared circuits")
    print("geometric means over shared circuits (space x, time x, space-time x):")
    for compiler, g in geomeans.items():
        print(
            f"  {compiler:14s} {g['normalized_space']:.2f}  "
            f"{g['normalized_time']:.2f}  {g['normalized_space_time']:.2f}  (n={g['n']})"
        )


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Linux Libertine O",
                "Libertinus Serif",
                "Times New Roman",
                "STIXGeneral",
                "DejaVu Serif",
            ],
            "mathtext.fontset": "stix",
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 600,
        }
    )


def _limits(values: Sequence[float], log_scale: bool) -> tuple:
    lo, hi = min(min(values), 1.0), max(values)
    if log_scale:
        return lo / 1.25, hi * 1.35
    pad = 0.06 * (hi - lo or 1.0)
    return max(0.0, lo - pad), hi + pad


def _ticks(lo: float, hi: float) -> List[float]:
    candidates = [1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]
    return [t for t in candidates if lo <= t <= hi]


def plot(
    data: Dict[str, object],
    geomeans: Dict[str, Dict[str, float]],
    stem: Path,
    log_scale: Optional[bool],
    width_in: float,
    height_in: float,
) -> List[Path]:
    _style()
    points: List[TradeoffPoint] = data["plotted"]  # type: ignore[assignment]
    compilers: List[str] = data["compilers"]  # type: ignore[assignment]
    xs = [p.normalized_space for p in points] + [g["normalized_space"] for g in geomeans.values()]
    ys = [p.normalized_time for p in points] + [g["normalized_time"] for g in geomeans.values()]
    if log_scale is None:  # log only if the data span at least ~4x on some axis
        log_scale = max(max(xs) / min(xs), max(ys) / min(ys)) >= 4.0
    xlim, ylim = _limits(xs, log_scale), _limits(ys, log_scale)

    fig, ax = plt.subplots(figsize=(width_in, height_in))
    if log_scale:
        ax.set_xscale("log")
        ax.set_yscale("log")

    # Guides: the ideal point lies at (1, 1); iso-space-time contours x*y=c.
    ax.axvline(1.0, color=GUIDE, linewidth=0.6, zorder=0)
    ax.axhline(1.0, color=GUIDE, linewidth=0.6, zorder=0)
    for product in ISO_PRODUCTS:
        x0, x1 = max(xlim[0], product / ylim[1]), min(xlim[1], product / ylim[0])
        if x0 >= x1:
            continue
        n = 200
        if log_scale:
            grid = [x0 * (x1 / x0) ** (i / (n - 1)) for i in range(n)]
        else:
            grid = [x0 + (x1 - x0) * i / (n - 1) for i in range(n)]
        ax.plot(grid, [product / x for x in grid], color=GUIDE, linewidth=0.5,
                linestyle=(0, (1, 2)), zorder=0)
        # Label each contour where it enters the plot (top edge, else left edge).
        if product / ylim[1] >= xlim[0]:
            anchor, offset = (product / ylim[1], ylim[1]), (2, -7)
        else:
            anchor, offset = (xlim[0], product / xlim[0]), (3, -2)
        ax.annotate(f"{product}×", anchor, xytext=offset, textcoords="offset points",
                    fontsize=6, color=MUTED, ha="left", va="center")

    multi_config = any(
        len({p.config_id + p.source_dir for p in points if (p.compiler, p.circuit) == key}) > 1
        for key in {(p.compiler, p.circuit) for p in points}
    )

    for compiler in compilers:
        color = COLORS.get(compiler, MUTED)
        marker = MARKERS.get(compiler, "o")
        own = [p for p in points if p.compiler == compiler]
        emphasized = compiler == "HARVEST"
        ax.scatter(
            [p.normalized_space for p in own],
            [p.normalized_time for p in own],
            s=14 if emphasized else 11,
            marker=marker,
            facecolor=color,
            edgecolor="none",
            alpha=0.35,
            zorder=3 if emphasized else 2,
            label=None,
        )
        if multi_config:
            by_circuit = defaultdict(list)
            for p in own:
                if p.pareto_optimal:
                    by_circuit[p.circuit].append(p)
            for frontier in by_circuit.values():
                frontier.sort(key=lambda q: (q.logical_patches, q.schedule_length))
                if len(frontier) > 1:
                    ax.plot([q.normalized_space for q in frontier],
                            [q.normalized_time for q in frontier],
                            color=color, linewidth=0.7, alpha=0.5, zorder=2)

    # Geometric-mean markers, directly labelled.
    for compiler in compilers:
        g = geomeans.get(compiler)
        if g is None:
            continue
        emphasized = compiler == "HARVEST"
        ax.scatter(
            [g["normalized_space"]],
            [g["normalized_time"]],
            s=58 if emphasized else 40,
            marker=MARKERS.get(compiler, "o"),
            facecolor=COLORS.get(compiler, MUTED),
            edgecolor="white" if not emphasized else INK,
            linewidth=0.9 if emphasized else 0.7,
            zorder=6 if emphasized else 5,
        )
        dx, dy, ha, va = LABEL_PLACEMENT.get(compiler, (0, 6.5, "center", "bottom"))
        ax.annotate(
            compiler,
            (g["normalized_space"], g["normalized_time"]),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=7.5 if emphasized else 7,
            fontweight="bold" if emphasized else "normal",
            color=INK,
            zorder=7,
        )

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    if log_scale:
        fmt = FuncFormatter(lambda v, _: f"{v:g}×")
        for axis, lim in ((ax.xaxis, xlim), (ax.yaxis, ylim)):
            axis.set_major_locator(FixedLocator(_ticks(*lim)))
            axis.set_minor_locator(NullLocator())
            axis.set_major_formatter(fmt)
    ax.set_xlabel("Logical space over lower bound")
    ax.set_ylabel("Logical time over dependency bound")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.annotate("ideal", (1.0, 1.0), xytext=(3, -8), textcoords="offset points",
                fontsize=6, color=MUTED, ha="left")

    # Compact legend for marker shapes (geomean markers are directly labelled).
    handles = [
        plt.Line2D([], [], linestyle="none", marker=MARKERS.get(c, "o"),
                   markerfacecolor=COLORS.get(c, MUTED), markeredgecolor="none",
                   markersize=4.5, label=c)
        for c in compilers
    ]
    handles.append(plt.Line2D([], [], linestyle="none", marker="o", markerfacecolor=MUTED,
                              markeredgecolor="none", alpha=0.35, markersize=3.5,
                              label="per benchmark"))
    # The band just above the y=1 guide is empty in the measured data.
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.06, 0.25),
              frameon=False, handletextpad=0.2, borderaxespad=0.0,
              labelspacing=0.25, ncol=1)

    fig.tight_layout(pad=0.2)
    outputs = [stem.with_suffix(".pdf"), stem.with_suffix(".png")]
    fig.savefig(outputs[0], bbox_inches="tight")
    fig.savefig(outputs[1], dpi=600, bbox_inches="tight")
    plt.close(fig)
    return outputs


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "result_dirs",
        nargs="*",
        type=Path,
        default=[PROJECT_ROOT / "results" / "sota_comparison_matched_ir"],
        help="One or more matched_ir result directories (several = resource sweep).",
    )
    parser.add_argument("--output-dir", type=Path,
                        help="Defaults to the first result directory.")
    parser.add_argument("--output-stem", default="space_time_tradeoff")
    parser.add_argument("--include-unshared", action="store_true",
                        help="Also plot circuits that not every system completed.")
    scale = parser.add_mutually_exclusive_group()
    scale.add_argument("--log", dest="log_scale", action="store_true", default=None)
    scale.add_argument("--linear", dest="log_scale", action="store_false")
    parser.add_argument("--width", type=float, default=3.33, help="inches (one column)")
    parser.add_argument("--height", type=float, default=2.5, help="inches")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    data = load_points(args.result_dirs, args.include_unshared)
    geomeans = compiler_geomeans(data["all_points"], data["shared"])  # type: ignore[arg-type]
    print_summary(data, geomeans)
    output_dir = args.output_dir or args.result_dirs[0]
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / args.output_stem
    csv_path = stem.with_suffix(".csv")
    write_csv(csv_path, data["plotted"])  # type: ignore[arg-type]
    outputs = plot(data, geomeans, stem, args.log_scale, args.width, args.height)
    for path in [*outputs, csv_path]:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
