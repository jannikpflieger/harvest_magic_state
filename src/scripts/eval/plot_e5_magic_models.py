#!/usr/bin/env python3
"""E5 figures: T/T_lb vs. magic capacity per supply model, seed spread, utilization heatmap."""

from __future__ import annotations

import numpy as np

from plot_common import kind, load
from harvest.eval import plotting as P

FAMILY_COLORS = {"free": "#000000", "fixed": "#0072B2", "geometric": "#009E73",
                 "cultivation": "#D55E00"}


def _ordered_models(frame):
    def key(model):
        family = model.split()[0]
        value = float(model.split("=")[1]) if "=" in model else 0.0
        return ({"free": 0, "fixed": 1, "geometric": 2, "exponential": 3}.get(family, 9), value)
    return sorted(frame["model"].unique(), key=key)


def t_vs_capacity(exp_dir, summary) -> None:
    mc = kind(summary, "model_capacity")
    fig, ax = P.figure(P.SINGLE, 2.4)
    if mc.empty:
        P.empty(ax)
    else:
        models = _ordered_models(mc)
        for model in models:
            s = mc[mc["model"] == model].sort_values("capacity_geomean")
            family = s["magic_model"].iloc[0]
            same = [m for m in models if mc[mc["model"] == m]["magic_model"].iloc[0] == family]
            alpha = 0.35 + 0.65 * (same.index(model) + 1) / len(same)
            ax.plot(s["capacity_geomean"], s["T_over_lb_geomean"], marker="o", ms=2,
                    color=FAMILY_COLORS.get(family, "#666666"), alpha=alpha, label=model,
                    ls={"fixed": "-", "geometric": "--", "cultivation": ":"}.get(family, "-"))
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("Magic patches")
        ax.set_ylabel("Geomean T / critical path")
        ax.legend(frameon=False, fontsize=4.5, ncol=2)
    P.save(fig, exp_dir / "e5_T_vs_capacity.pdf")


def seed_spread(exp_dir, summary) -> None:
    seeds = kind(summary, "seed")
    seeds = seeds[seeds["model"].str.startswith("exponential")] if not seeds.empty else seeds
    fig, ax = P.figure(P.SINGLE, 2.2)
    if seeds.empty:
        P.empty(ax)
    else:
        caps = sorted(seeds["capacity_req"].unique())
        caps = [caps[0], caps[len(caps) // 2], caps[-1]] if len(caps) >= 3 else caps
        groups, labels, colours = [], [], []
        for model in _ordered_models(seeds):
            for cap in caps:
                vals = seeds[(seeds["model"] == model) & (seeds["capacity_req"] == cap)]["T_rel_mean"]
                if len(vals):
                    groups.append(vals)
                    labels.append(f"{model.split('=')[1]}/{'full' if cap == 0 else int(cap)}")
                    colours.append(FAMILY_COLORS["cultivation"])
        parts = ax.boxplot(groups, patch_artist=True, widths=0.6, flierprops={"markersize": 1.5})
        for patch in parts["boxes"]:
            patch.set_facecolor("#E69F00")
            patch.set_alpha(0.6)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=5)
        ax.set_xlabel("Exponential mean / magic patches")
        ax.set_ylabel("T / mean T over seeds")
        ax.axhline(1.0, color="black", lw=0.5)
    P.save(fig, exp_dir / "e5_seed_spread.pdf")


def utilization(exp_dir, summary) -> None:
    mc = kind(summary, "model_capacity")
    fig, ax = P.figure(P.DOUBLE, 2.6)
    if mc.empty:
        P.empty(ax)
    else:
        models = _ordered_models(mc)
        grid = mc.pivot_table(index="model", columns="capacity_req", values="utilization_mean")
        grid = grid.loc[[m for m in models if m in grid.index]]
        im = ax.imshow(grid.to_numpy(), cmap=P.SEQUENTIAL, aspect="auto", vmin=0)
        ax.set_yticks(range(len(grid.index)))
        ax.set_yticklabels(grid.index, fontsize=6)
        ax.set_xticks(range(len(grid.columns)))
        ax.set_xticklabels(["full" if c == 0 else int(c) for c in grid.columns])
        ax.set_xlabel("Magic patches requested")
        for (i, j), v in np.ndenumerate(grid.to_numpy()):
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5,
                        color="white" if v < np.nanmax(grid.to_numpy()) / 2 else "black")
        fig.colorbar(im, ax=ax, fraction=0.03).set_label("Consumptions / (patches x T)", fontsize=6)
    P.save(fig, exp_dir / "e5_utilization.pdf")


def main() -> None:
    exp_dir, summary = load(__doc__)
    t_vs_capacity(exp_dir, summary)
    seed_spread(exp_dir, summary)
    utilization(exp_dir, summary)


if __name__ == "__main__":
    main()
