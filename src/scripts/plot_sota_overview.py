#!/usr/bin/env python3
"""Overview figure: the resource-sweep space-time trade-off (free magic states).

A single boxed panel with each compiler's Pareto front through the geomeans of
its configurations and the FLASQ staircase, computed with the trade-off
script's own pipeline (FLASQ = 1x time).

Example::

    PYTHONPATH=src python src/scripts/plot_sota_overview.py \\
        --free "results/sota_comparison_feynman_matched_ir results/sota_sweep_feynman/layout_dense4 ..." \\
        --output-dir results/sota_overview
"""

from __future__ import annotations

import argparse
import os
import shlex
import sys
from pathlib import Path
from typing import Optional, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/harvest-matplotlib")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts import plot_sota_space_time_tradeoff as tradeoff


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--free", required=True,
                        help="Arguments for the free-magic trade-off (result dirs + options).")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-stem", default="sota_overview")
    parser.add_argument("--width", type=float, default=3.4)
    parser.add_argument("--height", type=float, default=2.4)
    args = parser.parse_args(argv)

    free_args = tradeoff.parse_args(shlex.split(args.free))
    free_args.per_benchmark = "none"  # frontiers only
    free = tradeoff.prepare(free_args)

    tradeoff._style()
    fig, ax = plt.subplots(figsize=(args.width, args.height))
    handles = tradeoff.plot(
        free["data"], free["geomeans"], Path("unused"), free_args.log_scale, 0, 0,
        free["flasq"], free["configs"], free["time_bound"], free_args.per_benchmark,
        free_args.iso_contours, tradeoff._parse_labels(free_args.label),
        ax=ax, legend=False, xlabel="Space overhead", ylabel="Time overhead",
        direct_labels=False, x_ticks=(1, 2, 4, 8, 16),
    )
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.8)
    ax.set_title("Space–time trade-off", loc="left", fontsize=10, pad=2, fontweight="bold")
    ax.set_title("lower is better ↓", loc="right", fontsize=9, pad=2, fontweight="bold",
                 color=tradeoff.COLORS["HARVEST"])
    ax.xaxis.label.set_size(10)
    ax.yaxis.label.set_size(10)
    ax.tick_params(labelsize=9)

    handles = [h for h in handles if h.get_label() != "dominated config."]
    ax.legend(handles=handles, loc="upper right", ncol=1, frameon=False,
              handletextpad=0.3, labelspacing=0.25, handlelength=1.2,
              borderaxespad=0.4, fontsize=8)

    fig.tight_layout(pad=0.2)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / args.output_stem
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    for suffix in (".pdf", ".png"):
        print(f"wrote {stem.with_suffix(suffix)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
