#!/usr/bin/env python3
"""E6 figures: placement speedup per layout, placement-cost change vs. schedule-length change."""

from __future__ import annotations

from plot_common import kind, load
from harvest.eval import plotting as P

PLACEMENT_COLORS = {"circuit_aware": "#0072B2", "random": "#BBBBBB"}


def speedup_box(exp_dir, summary) -> None:
    pairs = kind(summary, "pair")
    fig, ax = P.figure(P.SINGLE, 2.2)
    if pairs.empty:
        P.empty(ax)
    else:
        layouts = sorted(pairs["layout"].unique())
        data, positions, colours = [], [], []
        for i, layout in enumerate(layouts):
            for j, placement in enumerate(("random", "circuit_aware")):
                vals = pairs[(pairs["layout"] == layout) & (pairs["placement"] == placement)]["speedup"]
                data.append(vals.dropna())
                positions.append(i * 3 + j)
                colours.append(PLACEMENT_COLORS[placement])
        parts = ax.boxplot(data, positions=positions, patch_artist=True, widths=0.8,
                           flierprops={"markersize": 1.5})
        for patch, c in zip(parts["boxes"], colours):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
        ax.axhline(1.0, color="black", lw=0.6)
        ax.set_yscale("log")
        ax.set_xticks([i * 3 + 0.5 for i in range(len(layouts))])
        ax.set_xticklabels(layouts)
        ax.set_ylabel("Speedup T_index / T_placement")
        handles = [P.plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.7) for c in PLACEMENT_COLORS.values()]
        ax.legend(handles, ["circuit-aware", "random (5 seeds)"], frameon=False, fontsize=6)
    P.save(fig, exp_dir / "e6_speedup_box.pdf")


def cost_vs_dt(exp_dir, summary) -> None:
    pairs = kind(summary, "pair")
    fig, ax = P.figure(P.SINGLE, 2.2)
    if pairs.empty:
        P.empty(ax)
    else:
        for placement, sub in pairs.groupby("placement"):
            for layout, s in sub.groupby("layout"):
                ax.scatter(s["d_cost_rel"], s["d_T_rel"], s=6, alpha=0.7, linewidths=0,
                           color=P.LAYOUT_COLORS.get(layout, "#666666"),
                           marker="o" if placement == "circuit_aware" else "x",
                           label=f"{layout}, {placement}")
        ax.axhline(0, color="black", lw=0.5)
        ax.axvline(0, color="black", lw=0.5)
        ax.set_xlabel("Relative change in placement cost C(pi)")
        ax.set_ylabel("Relative change in T")
        ax.legend(frameon=False, fontsize=5)
    P.save(fig, exp_dir / "e6_cost_vs_dT.pdf")


def main() -> None:
    exp_dir, summary = load(__doc__)
    speedup_box(exp_dir, summary)
    cost_vs_dt(exp_dir, summary)


if __name__ == "__main__":
    main()
