#!/usr/bin/env python3
"""E1 figures: space-time trade-off scatter (one per magic model) and volume-ratio boxes."""

from __future__ import annotations

import numpy as np

from plot_common import kind, load
from harvest.eval import plotting as P

MAGIC_TITLES = {"free": "Free magic states", "cultivation": "Cultivated magic states (mean 3)"}


def tradeoff(exp_dir, summary, magic: str, stem: str) -> None:
    per = kind(summary, "circuit")
    per = per[(per["magic_model"] == magic) & per["T_cycles"].notna()]
    geo = kind(summary, "system_geomean")
    geo = geo[geo["magic_model"] == magic]
    use_flasq = geo["T_over_flasq"].notna().any() and len(geo) and geo["T_over_flasq"].notna().all()
    ycol = "T_over_flasq" if use_flasq else "T_over_lb"
    fig, ax = P.figure(P.SINGLE, 2.5)
    if geo.empty:
        P.empty(ax)
    else:
        for system, sub in per.groupby("system"):
            ax.scatter(sub["area_over_lb"], sub[ycol], s=4, alpha=0.18, color=P.color(system),
                       marker=P.marker(system), linewidths=0)
        seen = set()
        for _, g in geo.sort_values("system").iterrows():
            family = P.compiler_family(g["config"])
            ax.scatter(g["area_over_lb"], g[ycol], s=22, color=P.color(g["config"]),
                       marker=P.marker(g["config"]), edgecolors="black", linewidths=0.4, zorder=3,
                       label=family if family not in seen else None)
            seen.add(family)
            tag = g["layout"] if isinstance(g["layout"], str) else g["config"].partition(" (")[2][:-1]
            if tag:
                ax.annotate(tag, (g["area_over_lb"], g[ycol]), xytext=(3, 2),
                            textcoords="offset points", fontsize=4.5, color="#444444")
        xs = np.geomspace(max(1.0, per["area_over_lb"].min() * 0.9), per["area_over_lb"].max() * 1.1, 50)
        best = (geo["area_over_lb"] * geo[ycol]).min()
        for mult in (1, 2, 4, 8):
            ax.plot(xs, best * mult / xs, color="#cccccc", lw=0.5, ls="--", zorder=0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        P.plain_log_ticks(ax)
        ax.set_xlabel("Area / area lower bound")
        ax.set_ylabel("Time / FLASQ floor" if use_flasq else "Time / critical path")
        n = int(geo["n_shared"].min())
        ax.set_title(f"{MAGIC_TITLES.get(magic, magic)} (geomean, n={n})")
        ax.legend(frameon=False, ncol=2, loc="upper right")
        ax.grid(True, which="major", alpha=0.3)
    P.save(fig, exp_dir / f"{stem}.pdf")


def ratio_box(exp_dir, summary) -> None:
    ratios = kind(summary, "ratio")
    models = [m for m in ("free", "cultivation") if m in set(ratios.get("magic_model", []))]
    fig, axes = P.figure(P.DOUBLE, 3.0, ncols=max(1, len(models)), sharey=True, squeeze=False)
    if not models:
        P.empty(axes[0][0])
    for ax, magic in zip(axes[0], models):
        sub = ratios[ratios["magic_model"] == magic]
        systems = sorted(sub["system"].unique())
        data = [sub[sub["system"] == s]["volume_ratio"].dropna() for s in systems]
        parts = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=True,
                           flierprops={"markersize": 2})
        for patch, s in zip(parts["boxes"], systems):
            patch.set_facecolor(P.color(s))
            patch.set_alpha(0.7)
        for median in parts["medians"]:
            median.set_color("black")
        ax.axhline(1.0, color="black", lw=0.6)
        ax.set_yscale("log")
        P.plain_log_ticks(ax, "y")
        ax.set_xticks(range(1, len(systems) + 1))
        ax.set_xticklabels([f"{s.replace('Silva EAF', 'EAF')} n={len(d)}" for s, d in zip(systems, data)],
                           rotation=90, fontsize=5)
        ax.set_title(MAGIC_TITLES.get(magic, magic))
    axes[0][0].set_ylabel("Volume ratio (baseline / HARVEST)")
    P.save(fig, exp_dir / "e1_volume_ratio_box.pdf")


def family_heatmap(exp_dir, summary) -> None:
    fam = kind(summary, "family")
    models = [m for m in ("free", "cultivation") if not fam.empty and m in set(fam["magic_model"])]
    fig, axes = P.figure(P.DOUBLE, 3.2, ncols=max(1, len(models)), squeeze=False)
    if not models:
        P.empty(axes[0][0])
    for ax, magic in zip(axes[0], models):
        sub = fam[fam["magic_model"] == magic]
        grid = sub.pivot_table(index="family", columns="system", values="volume_ratio_geomean")
        counts = sub.pivot_table(index="family", columns="system", values="n_shared")
        vals = np.log2(grid.to_numpy(dtype=float))
        lim = max(0.5, np.nanmax(np.abs(vals))) if np.isfinite(vals).any() else 1
        ax.imshow(vals, cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
        ax.set_xticks(range(len(grid.columns)))
        ax.set_xticklabels([c.replace("Silva EAF", "EAF") for c in grid.columns], rotation=90,
                           fontsize=5)
        ax.set_yticks(range(len(grid.index)))
        ax.set_yticklabels([f"{f} (n={int(counts.loc[f].max())})" for f in grid.index], fontsize=6)
        for (i, j), v in np.ndenumerate(grid.to_numpy(dtype=float)):
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=4.5)
        ax.set_title(f"{MAGIC_TITLES.get(magic, magic)}: volume baseline / HARVEST (blue: HARVEST better)",
                     fontsize=6)
    P.save(fig, exp_dir / "e1_family_volume.pdf")


def main() -> None:
    exp_dir, summary = load(__doc__)
    tradeoff(exp_dir, summary, "free", "e1_tradeoff_free")
    tradeoff(exp_dir, summary, "cultivation", "e1_tradeoff_cult")
    ratio_box(exp_dir, summary)
    family_heatmap(exp_dir, summary)


if __name__ == "__main__":
    main()
