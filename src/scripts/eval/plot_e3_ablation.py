#!/usr/bin/env python3
"""E3 figures: cumulative ablation bars, alpha/beta and iteration heatmaps, iteration trade-off."""

from __future__ import annotations

import numpy as np

from plot_common import kind, load
from harvest.eval import plotting as P

ORDER = ("critical_path", "seq", "eaf", "pack", "nego", "+prio", "+match", "full")
LABELS = {"critical_path": "CP bound", "full": "+fill (full)"}


def bars(exp_dir, summary) -> None:
    abl = kind(summary, "ablation")
    fig, ax = P.figure(P.DOUBLE, 2.0)
    if abl.empty:
        P.empty(ax)
    else:
        layouts = sorted(abl["layout"].unique())
        variants = [v for v in ORDER if v in set(abl["ablation"])]
        width = 0.8 / max(1, len(variants))
        x = np.arange(len(layouts))
        for i, v in enumerate(variants):
            vals = [abl[(abl["layout"] == l) & (abl["ablation"] == v)]["T_rel_eaf_geomean"]
                    .squeeze() if len(abl[(abl["layout"] == l) & (abl["ablation"] == v)]) else np.nan
                    for l in layouts]
            colour = "#000000" if v == "critical_path" else P.ABLATION_COLORS.get(v, "#888888")
            ax.bar(x + (i - (len(variants) - 1) / 2) * width, vals, width, color=colour,
                   label=LABELS.get(v, v), alpha=0.35 if v == "critical_path" else 1.0)
        ax.axhline(1.0, color="black", lw=0.6)
        n = abl.groupby("layout")["n_shared"].min()
        ax.set_xticks(x)
        ax.set_xticklabels([f"{l} (n={int(n[l])})" for l in layouts])
        ax.set_ylabel("Geomean T relative to EAF")
        ax.legend(frameon=False, ncol=len(variants), fontsize=6, loc="upper center",
                  bbox_to_anchor=(0.5, 1.18))
    P.save(fig, exp_dir / "e3_ablation_bars.pdf")


def heatmap(exp_dir, summary, stage, xk, yk, xl, yl, stem) -> None:
    sens = kind(summary, f"sensitivity_{stage}")
    layouts = sorted(sens["layout"].unique()) if not sens.empty else []
    fig, axes = P.figure(P.DOUBLE, 2.0, ncols=max(1, len(layouts)), squeeze=False)
    if not layouts:
        P.empty(axes[0][0])
    for ax, layout in zip(axes[0], layouts):
        sub = sens[sens["layout"] == layout]
        grid = sub.pivot_table(index=yk, columns=xk, values="T_geomean")
        ref = grid.min().min()
        im = ax.imshow(grid.to_numpy() / ref, origin="lower", cmap=P.SEQUENTIAL, aspect="auto")
        ax.set_xticks(range(len(grid.columns)))
        ax.set_xticklabels([f"{c:g}" for c in grid.columns])
        ax.set_yticks(range(len(grid.index)))
        ax.set_yticklabels([f"{r:g}" for r in grid.index])
        for (i, j), v in np.ndenumerate(grid.to_numpy() / ref):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5,
                    color="white" if v > np.nanmedian(grid.to_numpy() / ref) else "black")
        ax.set_xlabel(xl)
        ax.set_title(f"{layout} (n={int(sub['n_shared'].min())})")
    if layouts:
        fig.colorbar(im, ax=list(axes[0]), fraction=0.02, pad=0.02).set_label(
            "geomean T / best cell of its panel", fontsize=6)
    axes[0][0].set_ylabel(yl)
    P.save(fig, exp_dir / stem)


def iters_tradeoff(exp_dir, summary) -> None:
    sens = kind(summary, "sensitivity_iters")
    fig, ax = P.figure(P.SINGLE, 2.0)
    if sens.empty:
        P.empty(ax)
    else:
        sub = sens[sens["cfg_stagnation_limit"] == 2] if (sens["cfg_stagnation_limit"] == 2).any() else sens
        ax2 = ax.twinx()
        for layout, s in sub.groupby("layout"):
            s = s.sort_values("cfg_max_iters")
            c = P.LAYOUT_COLORS.get(layout, "#666666")
            ax.plot(s["cfg_max_iters"], s["T_geomean"] / s["T_geomean"].iloc[0], marker="o", color=c,
                    label=layout)
            ax2.plot(s["cfg_max_iters"], s["compile_time_s_geomean"], ls="--", color=c, lw=0.7)
        ax.axvline(4, color="#999999", lw=0.5, ls=":")
        ax.set_xscale("symlog", base=2, linthresh=1)
        ax.set_xlabel("max_iters (stagnation 2)")
        ax.set_ylabel("Geomean T / T(max_iters=0)")
        ax2.set_ylabel("Compile time (s, dashed)")
        ax2.spines["right"].set_visible(True)
        ax.legend(frameon=False, fontsize=6, loc="lower left")
    P.save(fig, exp_dir / "e3_iters_tradeoff.pdf")


def main() -> None:
    exp_dir, summary = load(__doc__)
    bars(exp_dir, summary)
    heatmap(exp_dir, summary, "alpha_beta", "cfg_alpha", "cfg_beta", "alpha (present congestion)",
            "beta (history)", "e3_alpha_beta_heatmap.pdf")
    heatmap(exp_dir, summary, "iters", "cfg_max_iters", "cfg_stagnation_limit", "max_iters",
            "stagnation limit", "e3_iters_heatmap.pdf")
    iters_tradeoff(exp_dir, summary)


if __name__ == "__main__":
    main()
