#!/usr/bin/env python3
"""E2 figures: volume ratio vs. EAF against parallelism, and win/tie/loss per layout."""

from __future__ import annotations

import numpy as np

from plot_common import kind, load
from harvest.eval import plotting as P


def ratio_vs_parallelism(exp_dir, summary) -> None:
    per = kind(summary, "circuit")
    per = per[per["baseline"] == "Silva EAF"] if not per.empty else per
    fig, ax = P.figure(P.SINGLE, 2.3)
    if per.empty:
        P.empty(ax)
    else:
        for layout, sub in per.groupby("layout"):
            sub = sub[sub["feat_parallelism"].notna() & (sub["volume_ratio"] > 0)]
            c = P.LAYOUT_COLORS.get(layout, "#666666")
            ax.scatter(sub["feat_parallelism"], sub["volume_ratio"], s=8, color=c, alpha=0.75,
                       linewidths=0, label=layout)
            if len(sub) > 2:
                x, y = np.log(sub["feat_parallelism"]), np.log(sub["volume_ratio"])
                slope, icpt = np.polyfit(x, y, 1)
                xs = np.linspace(x.min(), x.max(), 20)
                ax.plot(np.exp(xs), np.exp(icpt + slope * xs), color=c, lw=0.8)
        ax.axhline(1.0, color="black", lw=0.6)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Parallelism (#ops / critical path)")
        ax.set_ylabel("Volume ratio (EAF / HARVEST)")
        ax.legend(frameon=False, fontsize=6)
    P.save(fig, exp_dir / "e2_ratio_vs_parallelism.pdf")


def win_loss(exp_dir, summary) -> None:
    wl = kind(summary, "winloss")
    wl = wl[wl["baseline"] == "Silva EAF"] if not wl.empty else wl
    fig, axes = P.figure(P.SINGLE, 2.0, ncols=2, sharey=True, squeeze=False)
    for ax, metric, title in zip(axes[0], ("time_ratio", "volume_ratio"), ("Time", "Volume")):
        sub = wl[wl["metric"] == metric] if not wl.empty else wl
        if sub.empty:
            P.empty(ax)
            continue
        x = np.arange(len(sub))
        bottom = np.zeros(len(sub))
        for col, colour, label in (("wins", "#0072B2", "HARVEST better"), ("ties", "#BBBBBB", "tie"),
                                   ("losses", "#E69F00", "EAF better")):
            ax.bar(x, sub[col], bottom=bottom, color=colour, label=label, width=0.7)
            bottom += sub[col].to_numpy()
        ax.set_xticks(x)
        ax.set_xticklabels(sub["layout"], rotation=30, ha="right")
        ax.set_title(title)
    axes[0][0].set_ylabel("Circuits")
    axes[0][1].legend(frameon=False, fontsize=6, loc="upper right")
    P.save(fig, exp_dir / "e2_win_loss.pdf")


def main() -> None:
    exp_dir, summary = load(__doc__)
    ratio_vs_parallelism(exp_dir, summary)
    win_loss(exp_dir, summary)


if __name__ == "__main__":
    main()
