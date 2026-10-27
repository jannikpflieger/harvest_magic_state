#!/usr/bin/env python3
"""Per-circuit time and space in absolute units (cycles, tiles) for a few compilers.

Reads the per-circuit CSV written by ``plot_sota_space_time_tradeoff.py`` (one
row per compiler configuration and circuit; for each compiler the
lowest-space-time configuration per circuit is used) and draws three panels:

(a) schedule length per circuit, one dot per compiler, HARVEST joined to
    ``--time-rival`` with the rival / HARVEST ratio;
(b) footprint (logical patches) per circuit, HARVEST joined to
    ``--space-rival`` with the ratio;
(c) tiles vs cycles, one point per circuit and compiler.

``--relative`` draws only (c), with every compiler's tiles and cycles divided
by HARVEST's on the same circuit (HARVEST = 1x on both axes).

Panels (a) and (b) are linear from zero, so distances are proportional to
the values; (c) is zoomed to the data range.  Every compiler is drawn in every panel.

Example::

    PYTHONPATH=src python src/scripts/plot_pairwise_dots.py \\
        results/sota_ising_400_440/interim/space_time_tradeoff_interim.csv \\
        --compilers "HARVEST,Silva EAF,PureMagic" --time-rival "Silva EAF" \\
        --space-rival PureMagic --output results/sota_ising_400_440/interim/pairwise
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/harvest-matplotlib")
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.compare_sota_baselines import COLORS
from scripts.plot_sota_space_time_tradeoff import MARKERS, _style

INK, MUTED, GRID = "#222222", "#666666", "#E3E3E3"


def load(csv_path: Path, compilers, exclude=()):
    best = {}
    for r in csv.DictReader(csv_path.open()):
        if (r["compiler"] not in compilers or r.get("in_shared_subset") != "True"
                or r["circuit"] in exclude):
            continue
        key = (r["circuit"], r["compiler"])
        st = float(r["normalized_space_time"])
        if key not in best or st < best[key][2]:
            best[key] = (float(r["schedule_length"]), float(r["logical_patches"]), st)
    circuits = sorted({c for c, _ in best}, key=lambda c: (len(c), c))
    circuits = [c for c in circuits if all((c, k) in best for k in compilers)]
    if not circuits:
        raise SystemExit("no circuit has a result for every compiler")
    return circuits, best


def geomean(values):
    return math.exp(sum(math.log(v) for v in values) / len(values))


def dot_panel(ax, circuits, best, compilers, index, rival, xlabel, title):
    ys = list(range(len(circuits)))[::-1]
    ratios = []
    top = max(best[(c, k)][index] for c in circuits for k in compilers)
    # HARVEST on the row, the rival just above, any other compiler just below,
    # so near-equal values never hide each other.
    others = [k for k in compilers if k not in ("HARVEST", rival)]
    offset = {"HARVEST": 0.0, rival: 0.22, **{k: -0.22 for k in others}}
    for y, c in zip(ys, circuits):
        h, r = best[(c, "HARVEST")][index], best[(c, rival)][index]
        ax.plot([h, r], [y, y + offset[rival]], color="#BBBBBB", linewidth=1.5, zorder=1,
                solid_capstyle="round")
        ratios.append(r / h)
        ax.text(top * 1.04, y, f"{r / h:.2f}×", va="center", ha="left", fontsize=6.5, color=INK)
    for k in compilers:
        ax.scatter([best[(c, k)][index] for c in circuits], [y + offset[k] for y in ys], s=26,
                   color=COLORS[k], marker=MARKERS.get(k, "o"), edgecolor="white",
                   linewidth=0.6, zorder=4 if k == "HARVEST" else 3, label=k)
    ax.set_yticks(ys)
    ax.set_yticklabels([c.split("_")[-1] for c in circuits])
    ax.set_xlim(0, top * 1.18)
    ax.set_xlabel(xlabel)
    ax.set_title(f"{title}\n{rival} / HARVEST: {geomean(ratios):.2f}× (geomean)",
                 fontsize=7.5, color=INK)
    ax.xaxis.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def relative_figure(args, circuits, best, compilers) -> plt.Figure:
    """Tiles and cycles of every compiler divided by HARVEST's on the same circuit."""
    fig, ax = plt.subplots(figsize=(args.width, args.height))
    rel = {k: [(best[(c, k)][1] / best[(c, "HARVEST")][1],
                best[(c, k)][0] / best[(c, "HARVEST")][0]) for c in circuits]
           for k in compilers}
    xs = [x for pts in rel.values() for x, _ in pts]
    ys = [y for pts in rel.values() for _, y in pts]
    pad_x, pad_y = 0.12 * (max(xs) - min(xs)) + 0.02, 0.12 * (max(ys) - min(ys)) + 0.02
    ax.axvline(1.0, color="#999999", linewidth=0.7, linestyle="--", zorder=1)
    ax.axhline(1.0, color="#999999", linewidth=0.7, linestyle="--", zorder=1)
    # One equal-size dot per circuit and compiler; HARVEST is drawn last, at 1x / 1x.
    for k in [c for c in compilers if c != "HARVEST"] + ["HARVEST"]:
        ax.scatter([x for x, _ in rel[k]], [y for _, y in rel[k]], s=28, color=COLORS[k],
                   marker=MARKERS.get(k, "o"), edgecolor="white", linewidth=0.6, zorder=3,
                   label=k)
    ax.set_xlim(min(xs) - pad_x, max(xs) + pad_x)
    ax.set_ylim(min(ys) - pad_y, max(ys) + pad_y)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:g}×")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:g}×")
    ax.set_xlabel(args.xlabel)
    ax.set_ylabel(args.ylabel)
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index(k) for k in compilers]  # legend in --compilers order
    ax.legend([handles[i] for i in order], [labels[i] for i in order], frameon=False,
              loc="upper right")
    ax.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("csv", type=Path)
    parser.add_argument("--compilers", default="HARVEST,Silva EAF,PureMagic")
    parser.add_argument("--time-rival", default="Silva EAF")
    parser.add_argument("--space-rival", default="PureMagic")
    parser.add_argument("--output", required=True, help="Output stem (no suffix).")
    parser.add_argument("--width", type=float, default=7.0)
    parser.add_argument("--height", type=float, default=2.6)
    parser.add_argument("--exclude", default="",
                        help="Comma-separated circuits to leave out (state why in the caption).")
    parser.add_argument("--xlabel", default="Layout size",
                        help="x-axis label of --relative (values are relative to HARVEST).")
    parser.add_argument("--ylabel", default="Timesteps",
                        help="y-axis label of --relative (values are relative to HARVEST).")
    parser.add_argument("--relative", action="store_true",
                        help="Only the tiles-vs-cycles view, both relative to HARVEST per circuit.")
    args = parser.parse_args(argv)
    compilers = [c.strip() for c in args.compilers.split(",")]
    excluded = {c.strip() for c in args.exclude.split(",") if c.strip()}
    circuits, best = load(args.csv, compilers, excluded)
    if excluded:
        print(f"excluded: {', '.join(sorted(excluded))}")

    _style()
    if args.relative:
        fig = relative_figure(args, circuits, best, compilers)
        stem = Path(args.output)
        stem.parent.mkdir(parents=True, exist_ok=True)
        for suffix in (".pdf", ".png"):
            fig.savefig(stem.with_suffix(suffix), bbox_inches="tight")
            print(f"wrote {stem.with_suffix(suffix)}")
        return 0
    fig, axes = plt.subplots(1, 3, figsize=(args.width, args.height),
                             gridspec_kw={"width_ratios": [1, 1, 1.1]})
    dot_panel(axes[0], circuits, best, compilers, 0, args.time_rival,
              "Schedule length (cycles)", "(a) Time")
    dot_panel(axes[1], circuits, best, compilers, 1, args.space_rival,
              "Footprint (tiles)", "(b) Space")
    ax = axes[2]
    for k in compilers:
        ax.scatter([best[(c, k)][1] for c in circuits], [best[(c, k)][0] for c in circuits],
                   s=22, color=COLORS[k], marker=MARKERS.get(k, "o"), edgecolor="white",
                   linewidth=0.6, zorder=3)
    # A scatter needs no zero baseline: zoom to the data so the clusters separate.
    for axis, i in ((ax.set_xlim, 1), (ax.set_ylim, 0)):
        values = [best[(c, k)][i] for c in circuits for k in compilers]
        pad = 0.08 * (max(values) - min(values))
        axis(min(values) - pad, max(values) + pad)
    ax.set_xlabel("Footprint (tiles)")
    ax.set_ylabel("Schedule length (cycles)")
    ax.set_title("(c) Both (zoomed); lower left is better", fontsize=7.5, color=INK)
    ax.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Circuit (qubits)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=len(compilers),
               frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout(w_pad=1.2)
    stem = Path(args.output)
    stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        fig.savefig(stem.with_suffix(suffix), bbox_inches="tight")
        print(f"wrote {stem.with_suffix(suffix)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
