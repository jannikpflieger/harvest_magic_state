#!/usr/bin/env python3
"""Box plot of execution-time overhead (x FLASQ) per compiler.

Reads the per-point CSV written by ``plot_sota_space_time_tradeoff.py`` (time
already normalized by FLASQ's per-circuit floor, so FLASQ = 1x) and draws, for
every compiler, the distribution over the shared circuits of
``normalized_time`` for one configuration:

* ``best_space_time`` (default): the configuration with the lowest geometric
  mean space-time product -- the one labelled in the trade-off figures;
* ``fastest``: the configuration with the lowest geometric-mean time;
* ``default``: the configuration from the first (default) result directory.

A time-only view hides space, so every tick label states the chosen
configuration and its geometric-mean space overhead.

Example::

    PYTHONPATH=src python src/scripts/plot_sota_time_boxplot.py \\
        --panel "Instant magic states=results/sota_sweep_feynman/space_time_tradeoff_sweep.csv" \\
        --panel "Cultivated magic states=results/sota_cultivation_matched_ir/space_time_tradeoff_cultivation.csv" \\
        --output-dir results/sota_time_boxplot
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/harvest-matplotlib")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

from harvest.baselines.metrics import geometric_mean
from scripts.compare_sota_baselines import COLORS
from scripts.plot_sota_space_time_tradeoff import FLASQ_COLOR, INK, MUTED, PLOT_ORDER, _style, _ticks

FLASQ = "FLASQ model"
_DIR_NAMES = {
    "layout_dense2": "dense, 2 ports",
    "layout_dense4": "dense, 4 ports",
    "layout_dense8": "dense, 8 ports",
    "layout_blocks4": "blocks4",
    "layout_one_side_magic": "one-side magic",
    "layout_fixed_magic2": "2 magic ports",
    "layout_large_spacing": "wide spacing",
    "puremagic_ancilla2": "2 ancilla rows",
    "puremagic_ancilla3": "3 ancilla rows",
}
_VARIANT_NAMES = {
    "litinski_compact": "compact block",
    "litinski_intermediate": "intermediate block",
    "litinski_fast": "fast block",
    "o3ls_min": "smallest board",
    "o3ls_medium": "medium board",
    "o3ls_large": "large board",
}


def config_name(source_dir: str, config_id: str) -> str:
    variant = config_id.split("@", 1)[0]
    if variant in _VARIANT_NAMES:
        return _VARIANT_NAMES[variant]
    return _DIR_NAMES.get(Path(source_dir).name, "default")


def load(path: Path) -> Dict[Tuple[str, str, str], Dict[str, Tuple[float, float]]]:
    """{(compiler, source_dir, config_id): {circuit: (space, time)}} for shared circuits."""
    configs: Dict[Tuple[str, str, str], Dict[str, Tuple[float, float]]] = defaultdict(dict)
    with path.open() as handle:
        for row in csv.DictReader(handle):
            if row["compiler"] == FLASQ or row.get("in_shared_subset") != "True":
                continue
            key = (row["compiler"], row["source_dir"], row["config_id"])
            configs[key][row["circuit"]] = (float(row["normalized_space"]), float(row["normalized_time"]))
    return configs


def choose(configs, choice: str, first_dir: Optional[str]):
    """One configuration per compiler, over the circuits every configuration covers."""
    circuits = set.intersection(*(set(c) for c in configs.values()))
    chosen = {}
    for (compiler, source_dir, config_id), data in configs.items():
        vals = [data[c] for c in circuits]
        space = geometric_mean(s for s, _ in vals)
        time = geometric_mean(t for _, t in vals)
        score = {"best_space_time": space * time, "fastest": time}.get(choice, 0.0)
        if choice == "default" and source_dir != first_dir and not config_id.startswith(
            ("litinski_fast", "o3ls_min")
        ):
            continue
        entry = (score, compiler, source_dir, config_id, space, [data[c][1] for c in sorted(circuits)])
        if compiler not in chosen or entry[0] < chosen[compiler][0]:
            chosen[compiler] = entry
    return chosen, sorted(circuits)


_SHORT_CONFIG = {"dense, 2 ports": "dense-2", "dense, 4 ports": "dense-4",
                 "dense, 8 ports": "dense-8", "smallest board": "smallest",
                 "fast block": "fast", "compact block": "compact",
                 "intermediate block": "interm.", "default": "default"}
_SHORT_NAME = {"PureMagic Bus": "PM Bus", "PureMagic": "PureMagic", "Silva EAF": "Silva"}


def draw_box_panel(ax, title, chosen, circuits, rng, config_choice="best_space_time",
                   compact=False, show_flasq_label=True, show_points=True):
    """One box per compiler (normalized time over circuits); returns CSV rows."""
    rows = []
    order = [c for c in PLOT_ORDER if c in chosen] + sorted(set(chosen) - set(PLOT_ORDER))
    data = [chosen[c][5] for c in order]
    # Outliers are not drawn (they stay in the CSV); the caption must say so when
    # show_points is False.
    boxes = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=False,
                       medianprops={"color": INK, "linewidth": 1.1},
                       whiskerprops={"color": MUTED, "linewidth": 0.7},
                       capprops={"color": MUTED, "linewidth": 0.7})
    print(f"== {title} ({len(circuits)} circuits, configuration = {config_choice})")
    for i, (compiler, box) in enumerate(zip(order, boxes["boxes"]), start=1):
        color = COLORS.get(compiler, MUTED)
        emphasized = compiler == "HARVEST"
        box.set(facecolor=color, alpha=0.45 if not emphasized else 0.6,
                edgecolor=INK if emphasized else color, linewidth=1.2 if emphasized else 0.8)
        times = chosen[compiler][5]
        if show_points:
            ax.scatter([i + rng.uniform(-0.16, 0.16) for _ in times], times, s=5,
                       color=color, alpha=0.5, edgecolor="none", zorder=3)
        _, _, source_dir, config_id, space, _ = chosen[compiler]
        print(f"  {compiler:14s} [{config_name(source_dir, config_id)}] median time "
              f"{median(times):.2f}x, geomean {geometric_mean(times):.2f}x, space {space:.2f}x")
        for circuit, t in zip(circuits, times):
            rows.append({"panel": title, "compiler": compiler,
                         "configuration": config_name(source_dir, config_id),
                         "geomean_space": space, "circuit": circuit, "normalized_time": t})
    ax.axhline(1.0, color=FLASQ_COLOR, linewidth=1.0, zorder=1)
    if show_flasq_label:
        ax.text(len(order) + 0.45, 1.0, "FLASQ", ha="right", va="bottom", fontsize=6.5, color=INK)
    ax.set_xticks(range(1, len(order) + 1))
    if compact:
        # Names only: the chosen configuration is stated in the caption / CSV.
        ax.set_xticklabels(order, fontsize=6.5, rotation=30, ha="right", rotation_mode="anchor")
    else:
        ax.set_xticklabels(
            [f"{c}\n{config_name(chosen[c][2], chosen[c][3])}\n{chosen[c][4]:.1f}× space"
             for c in order], fontsize=6, rotation=0)
    for label, compiler in zip(ax.get_xticklabels(), order):
        if compiler == "HARVEST":
            label.set_fontweight("bold")
    ax.set_title(f"{title} ({len(circuits)} circuits)", fontsize=8, color=INK)
    ax.set_yscale("log")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, axis="y", color="#E6E6E6", linewidth=0.5)
    ax.set_axisbelow(True)
    return rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--panel", action="append", required=True, metavar="TITLE=CSV")
    parser.add_argument("--config-choice", choices=["best_space_time", "fastest", "default"],
                        default="best_space_time")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-stem", default="time_boxplot")
    parser.add_argument("--width", type=float, default=6.9)
    parser.add_argument("--height", type=float, default=2.7)
    args = parser.parse_args(argv)

    _style()
    panels = []
    for spec in args.panel:
        title, path = spec.split("=", 1)
        configs = load(Path(path))
        if not configs:
            raise SystemExit(f"{path}: no shared points")
        first_dir = sorted({k[1] for k in configs}, key=len)[0]
        chosen, circuits = choose(configs, args.config_choice, first_dir)
        panels.append((title, chosen, circuits))

    fig, axes = plt.subplots(1, len(panels), figsize=(args.width, args.height), sharey=True,
                             squeeze=False)
    rows = []
    rng = random.Random(0)
    all_times = [t for _, chosen, _ in panels for e in chosen.values() for t in e[5]]
    for ax, (title, chosen, circuits) in zip(axes[0], panels):
        rows += draw_box_panel(ax, title, chosen, circuits, rng, args.config_choice)
    lo = min(0.9, min(all_times) / 1.15)
    hi = max(all_times) * 1.15
    axes[0][0].set_ylim(lo, hi)
    axes[0][0].yaxis.set_major_locator(FixedLocator(_ticks(lo, hi)))
    axes[0][0].yaxis.set_minor_locator(NullLocator())
    axes[0][0].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}×"))
    axes[0][0].set_ylabel("Time overhead (× FLASQ bound)")
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
