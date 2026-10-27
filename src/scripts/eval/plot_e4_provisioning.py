#!/usr/bin/env python3
"""E4 figures: volume vs. working-layout area before/after pruning; retained vs. removed tiles."""

from __future__ import annotations

import numpy as np

from plot_common import kind, load
from harvest.eval import plotting as P

CLASSES = ("low", "medium", "high")


def volume_vs_area(exp_dir, summary) -> None:
    lay = kind(summary, "layout")
    models = [m for m in ("free", "cultivation") if not lay.empty and m in set(lay["magic_model"])]
    fig, axes = P.figure(P.DOUBLE, 1.6 * max(1, len(models)), nrows=max(1, len(models)),
                         ncols=len(CLASSES), sharex=True, squeeze=False)
    if not models:
        P.empty(axes[0][0])
    for row, magic in enumerate(models):
        for col, cls in enumerate(CLASSES):
            ax = axes[row][col]
            sub = lay[(lay["magic_model"] == magic) & (lay["circuit_class"] == cls)]
            if sub.empty:
                P.empty(ax)
                continue
            for preset, s in sub.groupby("layout_preset"):
                s = s.sort_values("area_initial_rel")
                c = P.LAYOUT_COLORS.get(preset, None)
                ax.plot(s["area_initial_rel"], s["volume_initial_rel"], ls="--", marker=".", color=c,
                        lw=0.7)
                ax.plot(s["area_initial_rel"], s["volume_pruned_rel"], ls="-", marker="o", color=c,
                        ms=2.5, label=preset)
            ax.set_xscale("log")
            ax.set_yscale("log")
            P.plain_log_ticks(ax)
            ax.set_title(f"{cls} parallelism, {magic}", fontsize=7)
            if col == 0:
                ax.set_ylabel("Volume / lower bound")
            if row == len(models) - 1:
                ax.set_xlabel("Working-layout area / area bound")
    handles, labels = {}, []
    for ax in axes.flat:
        for h, l in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(l, h)
    if handles:
        fig.legend(list(handles.values()), list(handles), frameon=False, fontsize=6,
                   ncol=len(handles), loc="upper center", bbox_to_anchor=(0.5, 1.06),
                   title="solid: after pruning, dashed: before", title_fontsize=6)
    P.save(fig, exp_dir / "e4_volume_vs_area.pdf")


def retained_removed(exp_dir, summary) -> None:
    lay = kind(summary, "layout")
    lay = lay[lay["magic_model"] == "free"] if not lay.empty else lay
    fig, ax = P.figure(P.DOUBLE, 2.0)
    if lay.empty:
        P.empty(ax)
    else:
        agg = lay.groupby("layout").sum(numeric_only=True)
        agg = agg.loc[sorted(agg.index, key=lambda l: (l.split("_m")[0], int(l.split("_m")[1])
                                                       if "_m" in l else 10 ** 6))]
        x = np.arange(len(agg))
        bottom = np.zeros(len(agg))
        colours = {"data": "#555555", "routing": "#0072B2", "magic": "#009E73"}
        for kind_ in ("data", "routing", "magic"):
            kept = agg[f"{kind_}_tiles_pruned_sum"].to_numpy()
            removed = agg[f"{kind_}_tiles_initial_sum"].to_numpy() - kept
            total = agg[[f"{k}_tiles_initial_sum" for k in ("data", "routing", "magic")]].sum(axis=1).to_numpy()
            ax.bar(x, kept / total, bottom=bottom, color=colours[kind_], label=f"{kind_} kept")
            bottom += kept / total
            ax.bar(x, removed / total, bottom=bottom, color=colours[kind_], alpha=0.3,
                   hatch="///", label=f"{kind_} removed")
            bottom += removed / total
        ax.set_xticks(x)
        ax.set_xticklabels(agg.index, rotation=60, ha="right", fontsize=5)
        ax.set_ylabel("Share of working-layout tiles")
        ax.legend(frameon=False, ncol=6, fontsize=5, loc="upper center", bbox_to_anchor=(0.5, 1.15))
    P.save(fig, exp_dir / "e4_retained_removed.pdf")


def main() -> None:
    exp_dir, summary = load(__doc__)
    volume_vs_area(exp_dir, summary)
    retained_removed(exp_dir, summary)


if __name__ == "__main__":
    main()
