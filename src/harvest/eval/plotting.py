"""Shared figure style for the paper PDFs (EVAL_PLAN B.1).

Every figure: PDF only, TrueType fonts (``pdf.fonttype = 42``), ACM column
widths, 8 pt text, and one fixed colorblind-safe color per compiler that is
the same in every plot.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/harvest-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SINGLE = 3.33  # ACM single-column width (in)
DOUBLE = 7.0   # ACM double-column width (in)

# Okabe-Ito (colorblind-safe) plus two neutrals; one entry per compiler.
COMPILER_COLORS = {
    "HARVEST": "#0072B2",
    "Silva EAF": "#E69F00",
    "Silva EAF + pruning": "#D55E00",
    "PureMagic": "#009E73",
    "PureMagic Bus": "#56B4E9",
    "Litinski": "#CC79A7",
    "O3LS": "#F0E442",
    "DASCOT": "#999999",
    "FLASQ": "#000000",
}
COMPILER_MARKERS = {
    "HARVEST": "o", "Silva EAF": "s", "Silva EAF + pruning": "D", "PureMagic": "^",
    "PureMagic Bus": "v", "Litinski": "P", "O3LS": "X", "DASCOT": "*", "FLASQ": "_",
}
# Scheduler ablation (E3): sequential greys to blue = full HARVEST.
ABLATION_ORDER = ("seq", "eaf", "pack", "nego", "+prio", "+match", "full")
ABLATION_COLORS = {
    "seq": "#BBBBBB", "eaf": "#E69F00", "pack": "#9ecae1", "nego": "#6baed6",
    "+prio": "#4292c6", "+match": "#2171b5", "full": "#0072B2",
}
LAYOUT_COLORS = {"ring": "#0072B2", "blocks4": "#009E73", "dense2": "#D55E00",
                 "dense4": "#CC79A7", "dense": "#D55E00"}
SEQUENTIAL = "viridis"


def compiler_family(label: str) -> str:
    """Compiler of a label like ``"Litinski (fast)"`` or ``"Silva EAF [ring]"``."""
    return label.split(" (")[0].split(" [")[0]


def plain_log_ticks(ax, axis: str = "both") -> None:
    """Log axes labelled 1, 2, 5, 10 instead of 2x10^0."""
    from matplotlib.ticker import FuncFormatter, LogLocator

    fmt = FuncFormatter(lambda v, _: f"{v:g}")
    for name in ("x", "y") if axis == "both" else (axis,):
        target = ax.xaxis if name == "x" else ax.yaxis
        target.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        target.set_major_formatter(fmt)
        target.set_minor_formatter(FuncFormatter(lambda v, _: ""))


def color(label: str) -> str:
    return COMPILER_COLORS.get(compiler_family(label), "#666666")


def marker(label: str) -> str:
    return COMPILER_MARKERS.get(compiler_family(label), "o")


def setup() -> None:
    plt.rcParams.update({
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "sans-serif",
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.0,
        "lines.markersize": 4,
        "grid.linewidth": 0.4,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def figure(width: float = SINGLE, height: float = 2.2, **kwargs):
    setup()
    return plt.subplots(figsize=(width, height), **kwargs)


def save(fig, path: Path) -> Path:
    path = Path(path)
    if path.suffix != ".pdf":
        raise ValueError(f"figures are PDF only, got {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")
    return path


def empty(ax, message: str = "no data yet") -> None:
    """Placeholder for a panel whose data has not been produced."""
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes,
            fontsize=7, color="#888888")
    ax.set_xticks([])
    ax.set_yticks([])
