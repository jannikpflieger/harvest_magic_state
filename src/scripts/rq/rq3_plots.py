#!/usr/bin/env python3
"""RQ3 tables and figures from ``results/rq/rq3/rq3.csv``.

Per circuit/template/model/level the seeds are averaged.  The reference of a
circuit/template/model is the smallest layout (fewest magic terminals) with
its pruned schedule.  Lines of Figure 4, at each level (x: initial area of
the level's layout relative to the smallest layout, geomean over circuits):

* schedule large + prune   = ``large`` row, ``volume_pruned``
* schedule large, no prune = ``large`` row, ``volume_initial``
* direct at pruned size    = ``direct`` row, ``volume_initial`` (scheduled from
  scratch on a layout whose footprint equals the large schedule's pruned one)
* direct + prune           = ``direct`` row, ``volume_pruned``

Outputs (``results/rq/rq3/``): rq3_levels.csv, rq3_summary.csv,
rq3_removed_tiles.csv, fig4_provisioning.pdf, fig4_removed_tiles.pdf
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import rq_plotstyle as style
from rq_common import RESULTS, geomean, read_rows

OUT = RESULTS / "rq3"
TEMPLATES = ["ring", "blocks4", "dense4"]
MODELS = ["instant", "cultivation"]
LINES = [("large_pruned", "schedule large + prune", "-"),
         ("large_initial", "schedule large, no prune", ":"),
         ("direct_initial", "schedule directly at pruned size", "--"),
         ("direct_pruned", "direct + prune", "-.")]


def main() -> int:
    style.apply()
    import matplotlib.pyplot as plt

    df = read_rows(OUT / "rq3.csv")
    if len(df) == 0:
        print("rq3: no data yet")
        return 0
    num = ["T_cycles", "volume_initial", "volume_pruned", "area_initial_tiles", "area_pruned_tiles",
           "routing_tiles_initial", "routing_tiles_pruned", "magic_tiles_initial",
           "magic_tiles_pruned", "level", "magic_terminals"]
    for col in num:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    keys = ["circuit", "template", "model", "level"]
    bad = df[df.status != "ok"].groupby(keys).size()
    df = df[df.status == "ok"]
    agg = df.groupby([*keys, "role"])[num[:-2]].mean().unstack("role")
    agg.columns = [f"{c}_{r}" for c, r in agg.columns]
    agg = agg.reset_index()
    agg = agg[~agg.set_index(keys).index.isin(bad.index)]
    agg = agg.dropna(subset=["volume_initial_large", "volume_initial_direct"])
    agg["magic_terminals"] = df[df.role == "large"].groupby(keys).magic_terminals.first().reindex(
        list(zip(*[agg[k] for k in keys]))).values
    first = agg.sort_values("magic_terminals").groupby(["circuit", "template", "model"]).first()
    ref = first.set_index(first.index)  # minimal layout per circuit/template/model
    idx = list(zip(agg.circuit, agg.template, agg.model))
    ref_vol = ref["volume_pruned_large"].reindex(idx).values
    agg["area_rel"] = agg.area_initial_tiles_large.values / ref["area_initial_tiles_large"].reindex(idx).values
    agg["large_pruned"] = agg.volume_pruned_large / ref_vol
    agg["large_initial"] = agg.volume_initial_large / ref_vol
    agg["direct_initial"] = agg.volume_initial_direct / ref_vol
    agg["direct_pruned"] = agg.volume_pruned_direct / ref_vol
    agg["direct_over_large_pruned"] = agg.volume_initial_direct / agg.volume_pruned_large
    agg["T_direct_over_large"] = agg.T_cycles_direct / agg.T_cycles_large
    agg["level_rank"] = agg.groupby(["circuit", "template", "model"]).magic_terminals.rank(method="dense")
    agg.to_csv(OUT / "rq3_levels.csv", index=False)

    rows = []
    for (template, model, rank), g in agg.groupby(["template", "model", "level_rank"]):
        row = {"template": template, "model": model, "level_rank": int(rank), "n": len(g),
               "magic_terminals_median": g.magic_terminals.median(), "area_rel_geomean": geomean(g.area_rel),
               "direct_over_large_pruned_geomean": geomean(g.direct_over_large_pruned),
               "share_large_pruned_better": (g.direct_over_large_pruned > 1.001).mean(),
               "T_direct_over_large_geomean": geomean(g.T_direct_over_large)}
        for key, _, _ in LINES:
            row[f"{key}_geomean"] = geomean(g[key])
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "rq3_summary.csv", index=False)

    fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharey="row")
    for i, model in enumerate(MODELS):
        for j, template in enumerate(TEMPLATES):
            ax = axes[i, j]
            s = summary[(summary.template == template) & (summary.model == model)].sort_values("level_rank")
            # only levels every circuit reaches would be cleaner; keep levels with >= half the circuits
            s = s[s.n >= s.n.max() / 2] if len(s) else s
            for (key, label, ls), color in zip(LINES, style.SERIES[:4]):
                ax.plot(s.area_rel_geomean, s[f"{key}_geomean"], ls, marker="o", color=color, label=label)
            ax.axhline(1, color=style.INK2, linewidth=1)
            ax.set_title(f"{template}, {model}")
            ax.set_xlabel("initial layout area / minimal layout area")
            if j == 0:
                ax.set_ylabel("geomean volume / minimal-layout schedule")
    axes[0, 0].legend(fontsize=7)
    style.save(fig, OUT / "fig4_provisioning")

    # removed routing tiles and magic terminals (fractions of the initial area) per template
    large = df[df.role == "large"].copy()
    large["removed_routing_frac"] = (large.routing_tiles_initial - large.routing_tiles_pruned) / large.area_initial_tiles
    large["removed_magic_frac"] = (large.magic_tiles_initial - large.magic_tiles_pruned) / large.area_initial_tiles
    large["removed_total_frac"] = 1 - large.area_pruned_tiles / large.area_initial_tiles
    removed = large.groupby(["template", "model", "magic_terminals"])[
        ["removed_routing_frac", "removed_magic_frac", "removed_total_frac"]].mean().reset_index()
    removed.to_csv(OUT / "rq3_removed_tiles.csv", index=False)
    full = large.sort_values("magic_terminals").groupby(["circuit", "template", "model"]).last()
    by_t = full.groupby(["template", "model"])[["removed_routing_frac", "removed_magic_frac"]].mean()
    fig, ax = plt.subplots(figsize=(6, 3))
    labels = [f"{t}\n{m}" for t, m in by_t.index]
    x = np.arange(len(labels))
    ax.bar(x, by_t.removed_routing_frac * 100, 0.6, color=style.SERIES[0], label="routing tiles")
    ax.bar(x, by_t.removed_magic_frac * 100, 0.6, bottom=by_t.removed_routing_frac * 100,
           color=style.SERIES[1], label="magic terminals", edgecolor=style.SURFACE, linewidth=2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("area removed by pruning [% of initial]")
    ax.set_title("largest layout per circuit")
    ax.legend(fontsize=8)
    ax.grid(axis="x", visible=False)
    style.save(fig, OUT / "fig4_removed_tiles")
    print(summary.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
