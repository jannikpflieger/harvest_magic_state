"""Shared figure style of the RQ plots (reference palette, fixed colour per entity)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SURFACE = "#fcfcfb"
# colour follows the entity, never its rank
COMPILER_COLORS = {"Harvest": SERIES[0], "EAF": SERIES[1], "PureMagic": SERIES[2],
                   "Litinski": SERIES[3], "O3LS": SERIES[4]}
MODEL_COLORS = {"instant": SERIES[0], "cultivation": SERIES[1], "dist4": SERIES[2],
                "dist10": SERIES[3], "multiplexed": SERIES[4], "lognormal": SERIES[6]}
TEMPLATE_COLORS = {"ring": SERIES[0], "blocks4": SERIES[1], "dense4": SERIES[2]}


def apply():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
        "font.size": 9, "axes.titlesize": 10, "legend.frameon": False,
        "lines.linewidth": 2, "lines.markersize": 5, "pdf.fonttype": 42,
    })


def plain_log(ax, axis="y"):
    """Plain-number tick labels (0.9, 1, 1.2) on a log axis instead of 1.2x10^0."""
    from matplotlib.ticker import LogLocator, NullFormatter

    def fmt(v, _):
        return f"{v:,.0f}" if v >= 1000 else f"{v:g}"

    target = ax.yaxis if axis == "y" else ax.xaxis
    lo, hi = ax.get_ylim() if axis == "y" else ax.get_xlim()
    target.set_major_formatter(fmt)
    # label minor ticks only on narrow axes (ratios); wide axes get crowded
    target.set_minor_formatter(fmt if hi < 10 * lo else NullFormatter())
    target.set_minor_locator(LogLocator(subs=(2, 3, 5)))


def save(fig, path):
    for ax in fig.axes:
        for axis in ("x", "y"):
            if (ax.get_yscale() if axis == "y" else ax.get_xscale()) == "log":
                plain_log(ax, axis)
    # leave room for a figure-level legend above the panels
    fig.tight_layout(rect=(0, 0, 1, 0.9) if fig.legends else (0, 0, 1, 1))
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"), dpi=160)
    plt.close(fig)
