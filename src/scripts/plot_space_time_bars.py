#!/usr/bin/env python3
"""Space and time as separate bar charts (instead of one space-time product).

Two figures:

``layouts``  HARVEST vs Silva EAF on a few layouts (one column each) for a set
    of circuits: top row schedule length / critical path of the matched DAG,
    bottom row logical patches / minimum (``space_lb``, data patches + 1).
    Both are linear bars from zero; the Silva / HARVEST ratio is printed above
    each pair.  Raw cycles and patches are written to ``<stem>.csv``.

``families``  one group per benchmark family, one bar per compiler: geometric
    mean of normalized space (x minimum) and time (x FLASQ bound) over the
    family, each compiler at its lowest-space-time configuration per circuit --
    the large labelled markers of ``plot_sota_space_time_tradeoff.py``.  Reads
    the per-family CSVs that script wrote.

Examples::

    PYTHONPATH=src python src/scripts/plot_space_time_bars.py layouts \\
        --layout "Top edge only=results/sota_families/sweep_blocks4_one_side_magic" \\
        --layout "Spread over 4 edges=results/sota_families/sweep_blocks4_sparse_ring" \\
        --layout "Full ring=results/sota_families/sweep_blocks4" \\
        --circuits ising_n10,ising_n26,ising_n34,ising_n42,ising_n66,ising_n98 \\
        --bounds results/sota_families/matched/bounds_cache.json \\
        --output results/sota_families/bars/ising_blocks4_magic_placement
    PYTHONPATH=src python src/scripts/plot_space_time_bars.py families \\
        --family "Feynman=results/sota_families/figures/space_time_tradeoff_feynman.csv" ... \\
        --output results/sota_families/bars/families
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
from typing import Dict, List, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/harvest-matplotlib")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.compare_sota_baselines import COLORS
from scripts.plot_sota_space_time_tradeoff import PLOT_ORDER, _style

INK = "#222222"
MUTED = "#666666"
GRID = "#E3E3E3"
BAR_EDGE = "white"  # surface gap between adjacent bars
PAIR = (("harvest", "HARVEST"), ("silva", "Silva EAF"))


def geomean(values: Sequence[float]) -> float:
    return math.exp(sum(math.log(v) for v in values) / len(values))


def _axes_style(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0)


def _save(fig, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        fig.savefig(stem.with_suffix(suffix), bbox_inches="tight")
        print(f"wrote {stem.with_suffix(suffix)}")


def _pairs(items: Sequence[str], flag: str) -> List[Tuple[str, str]]:
    out = []
    for item in items:
        if "=" not in item:
            raise SystemExit(f"{flag} expects LABEL=PATH, got {item!r}")
        label, path = item.split("=", 1)
        out.append((label, path))
    return out


def _short(circuit: str) -> str:
    return circuit.split("_")[-1] if "_n" in circuit or "_N" in circuit else circuit


# ---------------------------------------------------------------- layouts


def load_layout(result_dir: Path, circuits: Sequence[str]) -> Dict[Tuple[str, str], Dict]:
    found: Dict[Tuple[str, str], Dict] = {}
    for path in (result_dir / "raw").glob("*.json"):
        result = json.loads(path.read_text())["result"]
        key = (result.get("circuit"), result.get("baseline"))
        if key[0] in circuits and key[1] in dict(PAIR) and result.get("completed"):
            found[key] = result
    missing = [(c, b) for c in circuits for b, _ in PAIR if (c, b) not in found]
    if missing:
        raise SystemExit(f"{result_dir}: no completed result for {missing}")
    return found


def load_bounds(path: Path) -> Dict[str, Dict]:
    bounds = {}
    for key, value in json.loads(path.read_text()).items():
        if "|ppr|" in key:
            bounds[value["circuit"]] = value
    return bounds


def plot_layouts(args: argparse.Namespace) -> None:
    layouts = _pairs(args.layout, "--layout")
    circuits = [c.strip() for c in args.circuits.split(",") if c.strip()]
    bounds = load_bounds(Path(args.bounds))
    data = {label: load_layout(Path(path), circuits) for label, path in layouts}
    rows = []
    for label, found in data.items():
        for c in circuits:
            for baseline, name in PAIR:
                r = found[(c, baseline)]
                rows.append({
                    "layout": label, "circuit": c, "compiler": name,
                    "schedule_length": r["schedule_length"],
                    "logical_patches": r["logical_patches"],
                    "time_lb": bounds[c]["time_lb"], "space_lb": bounds[c]["space_lb"],
                    "time_x": r["schedule_length"] / bounds[c]["time_lb"],
                    "space_x": r["logical_patches"] / bounds[c]["space_lb"],
                })
    stem = Path(args.output)
    stem.parent.mkdir(parents=True, exist_ok=True)
    with stem.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    _style()
    metrics = (("time_x", "Time (× critical path)"), ("space_x", "Space (× minimum)"))
    fig, axes = plt.subplots(2, len(layouts), figsize=(args.width, args.height),
                             sharey="row", squeeze=False)
    width = 0.38
    x = list(range(len(circuits)))
    value = {(r["layout"], r["circuit"], r["compiler"]): r for r in rows}
    for row, (metric, ylabel) in enumerate(metrics):
        top = max(r[metric] for r in rows)
        for col, (label, _) in enumerate(layouts):
            ax = axes[row][col]
            _axes_style(ax)
            for k, (_, name) in enumerate(PAIR):
                heights = [value[(label, c, name)][metric] for c in circuits]
                ax.bar([i + (k - 0.5) * width for i in x], heights, width,
                       color=COLORS[name], edgecolor=BAR_EDGE, linewidth=0.6,
                       label=name if (row, col) == (0, 0) else None)
            for i, c in enumerate(circuits):
                h = value[(label, c, "HARVEST")][metric]
                s = value[(label, c, "Silva EAF")][metric]
                ax.text(i, max(h, s) + 0.02 * top, f"{s / h:.2f}×", ha="center",
                        va="bottom", fontsize=5.5, color=MUTED)
            ratios = [value[(label, c, "Silva EAF")][metric] / value[(label, c, "HARVEST")][metric]
                      for c in circuits]
            if row == 0:
                ax.set_title(label, fontsize=8, color=INK, pad=10)
            ax.text(0.98, 0.97, f"Silva/HARVEST: {geomean(ratios):.2f}× (geomean)",
                    transform=ax.transAxes, ha="right", va="top", fontsize=6, color=INK)
            ax.set_xticks(x)
            ax.set_xticklabels([_short(c) for c in circuits] if row == 1 else [])
            ax.set_ylim(0, top * 1.22)
            if col == 0:
                ax.set_ylabel(ylabel)
    fig.legend(*axes[0][0].get_legend_handles_labels(), loc="upper center", ncol=2,
               frameon=False, bbox_to_anchor=(0.5, 1.04))
    if args.xlabel:
        fig.supxlabel(args.xlabel, fontsize=8, color=INK, y=0.0)
    fig.tight_layout(h_pad=1.0, w_pad=0.6)
    _save(fig, stem)


# ---------------------------------------------------------------- families


def family_geomeans(csv_path: Path) -> Dict[str, Tuple[float, float, int]]:
    best: Dict[Tuple[str, str], Tuple[float, float, float]] = {}
    for r in csv.DictReader(csv_path.open()):
        if r["compiler"] == "FLASQ model" or r.get("in_shared_subset") != "True":
            continue
        key = (r["compiler"], r["circuit"])
        point = (float(r["normalized_space"]), float(r["normalized_time"]),
                 float(r["normalized_space_time"]))
        if key not in best or point[2] < best[key][2]:
            best[key] = point
    by_compiler: Dict[str, List[Tuple[float, float, float]]] = defaultdict(list)
    for (compiler, _), point in best.items():
        by_compiler[compiler].append(point)
    return {c: (geomean([p[0] for p in pts]), geomean([p[1] for p in pts]), len(pts))
            for c, pts in by_compiler.items()}


def plot_families(args: argparse.Namespace) -> None:
    families = _pairs(args.family, "--family")
    stats = {label: family_geomeans(Path(path)) for label, path in families}
    excluded = {c.strip() for c in args.exclude.split(",") if c.strip()}
    compilers = [c for c in PLOT_ORDER
                 if c not in excluded and all(c in s for s in stats.values())]
    stem = Path(args.output)
    stem.parent.mkdir(parents=True, exist_ok=True)
    with stem.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["family", "compiler", "circuits", "normalized_space", "normalized_time"])
        for label, s in stats.items():
            for c in compilers:
                writer.writerow([label, c, s[c][2], f"{s[c][0]:.4f}", f"{s[c][1]:.4f}"])

    _style()
    fig, axes = plt.subplots(2, 1, figsize=(args.width, args.height), sharex=True)
    n = len(compilers)
    width = 0.8 / n
    x = list(range(len(families)))
    metrics = ((1, "Time (× FLASQ bound)", args.time_cap), (0, "Space (× minimum)", None))
    for ax, (index, ylabel, cap) in zip(axes, metrics):
        _axes_style(ax)
        for k, c in enumerate(compilers):
            heights = [stats[label][c][index] for label, _ in families]
            shown = [min(h, cap) if cap else h for h in heights]
            xs = [i + (k - (n - 1) / 2) * width for i in x]
            ax.bar(xs, shown, width, color=COLORS[c], edgecolor=BAR_EDGE, linewidth=0.5,
                   label=c if ax is axes[0] else None)
            if cap:
                for xi, h in zip(xs, heights):
                    if h > cap:  # clipped bar: print its value instead of hiding it
                        ax.text(xi, cap, f"{h:.0f}", ha="center", va="bottom",
                                fontsize=5, color=MUTED, rotation=90)
        if "HARVEST" in compilers:
            for i, (label, _) in enumerate(families):
                h = stats[label]["HARVEST"][index]
                xi = i + (compilers.index("HARVEST") - (n - 1) / 2) * width
                ax.text(xi, min(h, cap) if cap else h, f"{h:.2f}", ha="center",
                        va="bottom", fontsize=5, color=INK, rotation=90)
        ax.set_ylabel(ylabel)
        if cap:
            ax.set_ylim(0, cap * 1.15)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"{label}\n(n={next(iter(stats[label].values()))[2]})"
                             for label, _ in families])
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=min(n, 6),
               frameon=False, bbox_to_anchor=(0.5, 1.05))
    fig.tight_layout(h_pad=0.8)
    _save(fig, stem)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="figure", required=True)
    lay = sub.add_parser("layouts", help="HARVEST vs Silva per circuit on several layouts")
    lay.add_argument("--layout", action="append", required=True, metavar="LABEL=RESULT_DIR")
    lay.add_argument("--circuits", required=True, help="Comma-separated, in plot order.")
    lay.add_argument("--bounds", required=True, help="bounds_cache.json with PPR bounds.")
    lay.add_argument("--xlabel", default="")
    lay.add_argument("--output", required=True, help="Output stem (no suffix).")
    lay.add_argument("--width", type=float, default=7.0)
    lay.add_argument("--height", type=float, default=3.6)
    fam = sub.add_parser("families", help="All compilers per benchmark family")
    fam.add_argument("--family", action="append", required=True, metavar="LABEL=TRADEOFF_CSV")
    fam.add_argument("--exclude", default="", help="Comma-separated compiler names to omit.")
    fam.add_argument("--time-cap", type=float, default=None,
                     help="Clip time bars at this value and print the clipped values.")
    fam.add_argument("--output", required=True)
    fam.add_argument("--width", type=float, default=7.0)
    fam.add_argument("--height", type=float, default=4.2)
    args = parser.parse_args(argv)
    plot_layouts(args) if args.figure == "layouts" else plot_families(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
