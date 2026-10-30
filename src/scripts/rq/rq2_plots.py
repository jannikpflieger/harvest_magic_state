#!/usr/bin/env python3
"""RQ2 tables and figures from ``results/rq/rq2/rq2.csv``.

Per circuit/template/model the seeds are averaged; a variant's ratio is its
mean volume (or T) over the full configuration's on the same circuit,
template and model.  ``-pruning`` is the full schedule priced on its unpruned
footprint.

Outputs (``results/rq/rq2/``):
  rq2_ratios.csv            per circuit/template/model/variant ratios
  rq2_ablation_summary.csv  geomean + 10th/90th percentile per variant/model(/template)
  rq2_placement_by_template.csv, rq2_sensitivity.csv, rq2_negotiation_cdf.csv
  fig3_ablation.pdf, fig3_sensitivity.pdf, fig3_negotiation_cdf.pdf
  rq2_scheduler_ratios.csv, rq2_scheduler_summary.csv, fig3_scheduler_steps.pdf
                            scheduler versions 2 and 3 step by step (study ``scheduler``)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import rq_plotstyle as style
from rq_common import RESULTS, geomean, read_rows

OUT = RESULTS / "rq2"
ORDER = ["-negotiation", "-retry", "-admission", "-matching", "-priority", "-placement",
         "-yfactor", "-pruning"]
MODELS = ["instant", "cultivation"]


def ratios(df: pd.DataFrame, study: str) -> pd.DataFrame:
    df = df[(df.study == study)].copy()
    ok = df.groupby(["circuit", "template", "model", "ablation"])["status"].apply(
        lambda s: (s == "ok").all())
    for col in ("T_cycles", "volume_tile_cycles", "volume_initial", "area_tiles", "compile_time_s"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    agg = df.groupby(["circuit", "family", "template", "model", "ablation"]).agg(
        T=("T_cycles", "mean"), volume=("volume_tile_cycles", "mean"),
        volume_unpruned=("volume_initial", "mean"), area=("area_tiles", "mean"),
        compile_time_s=("compile_time_s", "mean"), n=("status", "size")).reset_index()
    agg = agg[ok.reindex(list(zip(agg.circuit, agg.template, agg.model, agg.ablation))).values]
    return agg


def main() -> int:
    style.apply()
    import matplotlib.pyplot as plt

    df = read_rows(OUT / "rq2.csv")
    if len(df) == 0:
        print("rq2: no data yet")
        return 0
    abl = ratios(df, "ablation")
    full = abl[abl.ablation == "full"].set_index(["circuit", "template", "model"])
    prune = full.reset_index().assign(ablation="-pruning", volume=lambda d: d.volume_unpruned)
    abl = pd.concat([abl, prune], ignore_index=True)
    idx = list(zip(abl.circuit, abl.template, abl.model))
    abl["volume_ratio"] = abl.volume.values / full.volume.reindex(idx).values
    abl["T_ratio"] = abl["T"].values / full["T"].reindex(idx).values
    abl["compile_time_ratio"] = abl.compile_time_s.values / full.compile_time_s.reindex(idx).values
    abl.to_csv(OUT / "rq2_ratios.csv", index=False)

    rows = []
    for (model, ablation), g in abl[abl.ablation != "full"].groupby(["model", "ablation"]):
        r = g.volume_ratio.dropna()
        rows.append({"template": "all", "model": model, "ablation": ablation, "n": len(r),
                     "volume_ratio_geomean": geomean(r), "p10": r.quantile(0.1), "p90": r.quantile(0.9),
                     "T_ratio_geomean": geomean(g.T_ratio), "share_worse": (r > 1.001).mean()})
    for (template, model, ablation), g in abl[abl.ablation != "full"].groupby(["template", "model", "ablation"]):
        r = g.volume_ratio.dropna()
        rows.append({"template": template, "model": model, "ablation": ablation, "n": len(r),
                     "volume_ratio_geomean": geomean(r), "p10": r.quantile(0.1), "p90": r.quantile(0.9),
                     "T_ratio_geomean": geomean(g.T_ratio), "share_worse": (r > 1.001).mean()})
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "rq2_ablation_summary.csv", index=False)
    summary[summary.ablation.isin(["-placement", "-yfactor"])].to_csv(
        OUT / "rq2_placement_by_template.csv", index=False)

    # Figure 3: horizontal bars per variant, instant vs cultivation, 10-90 % whiskers
    allt = summary[summary.template == "all"]
    variants = [v for v in ORDER if v in set(allt.ablation)]
    fig, ax = plt.subplots(figsize=(6, 0.45 * len(variants) + 1))
    for i, model in enumerate(MODELS):
        s = allt[allt.model == model].set_index("ablation").reindex(variants)
        y = np.arange(len(variants)) + (i - 0.5) * 0.38
        g = s.volume_ratio_geomean.values
        ax.barh(y, g - 1, 0.34, left=1, color=style.MODEL_COLORS[model], label=model)
        ax.errorbar(g, y, xerr=[np.clip(g - s.p10.values, 0, None), np.clip(s.p90.values - g, 0, None)], fmt="none",
                    ecolor=style.INK2, elinewidth=1, capsize=2)
    ax.axvline(1, color=style.INK2, linewidth=1)
    ax.set_yticks(np.arange(len(variants)))
    ax.set_yticklabels(variants)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("geomean volume ratio vs full Harvest (whiskers: 10th-90th pct.)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="y", visible=False)
    style.save(fig, OUT / "fig3_ablation")

    # sensitivity: I_max (T and compile time, two panels: no dual axis), alpha/beta
    sens = ratios(df, "sensitivity")
    base = abl[(abl.ablation == "full") & (abl.template == "blocks4")].copy()
    sens = pd.concat([sens, base.assign(ablation="imax4")], ignore_index=True)
    sens = pd.concat([sens, base.assign(ablation="alpha2"), base.assign(ablation="beta1.5")],
                     ignore_index=True)
    ref = base.set_index(["circuit", "model"])
    idx = list(zip(sens.circuit, sens.model))
    sens["T_ratio"] = sens["T"].values / ref["T"].reindex(idx).values
    sens["compile_time_ratio"] = sens.compile_time_s.values / ref.compile_time_s.reindex(idx).values
    rows = []
    for (model, name), g in sens.groupby(["model", "ablation"]):
        rows.append({"model": model, "setting": name, "n": len(g),
                     "T_ratio_geomean": geomean(g.T_ratio),
                     "compile_time_ratio_geomean": geomean(g.compile_time_ratio),
                     "compile_time_geomean_s": geomean(g.compile_time_s)})
    sens_summary = pd.DataFrame(rows)
    sens_summary.to_csv(OUT / "rq2_sensitivity.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.0))
    imax = [0, 1, 2, 4, 8]
    for model in MODELS:
        s = sens_summary[sens_summary.model == model].set_index("setting")
        t = [s.T_ratio_geomean.get(f"imax{i}", np.nan) for i in imax]
        c = [s.compile_time_geomean_s.get(f"imax{i}", np.nan) for i in imax]
        axes[0].plot(imax, t, marker="o", color=style.MODEL_COLORS[model], label=model)
        axes[1].plot(imax, c, marker="o", color=style.MODEL_COLORS[model], label=model)
        for name, vals, ls in (("alpha", (1, 2, 4), "-"), ("beta", (1, 1.5, 3), "--")):
            key = [f"{name}{v:g}" for v in vals]
            axes[2].plot(range(3), [s.T_ratio_geomean.get(k, np.nan) for k in key], ls, marker="o",
                         color=style.MODEL_COLORS[model], label=f"{name}, {model}")
    axes[0].set(xlabel="I_max (negotiation iterations)", ylabel="geomean makespan vs I_max=4")
    axes[1].set(xlabel="I_max", ylabel="geomean compile time [s]")
    axes[2].set(xlabel="setting (low / default / high)", ylabel="geomean makespan vs default")
    axes[2].set_xticks(range(3))
    axes[2].set_xticklabels(["α=1 / β=1", "α=2 / β=1.5", "α=4 / β=3"])
    axes[0].legend(fontsize=8)
    axes[2].legend(fontsize=7)
    style.save(fig, OUT / "fig3_sensitivity")

    # negotiation: per-circuit makespan reduction, T(-negotiation) / T(full)
    neg = abl[abl.ablation == "-negotiation"][["circuit", "template", "model", "T_ratio"]].dropna()
    neg["makespan_reduction"] = 1 - 1 / neg["T_ratio"]
    neg.to_csv(OUT / "rq2_negotiation_cdf.csv", index=False)
    fig, ax = plt.subplots(figsize=(5, 3))
    for model in MODELS:
        r = np.sort(neg[neg.model == model].makespan_reduction.values)
        if len(r):
            ax.plot(r * 100, np.arange(1, len(r) + 1) / len(r), color=style.MODEL_COLORS[model],
                    drawstyle="steps-post",
                    label=f"{model}: median {np.median(r) * 100:.1f} %, improves {(r > 0.001).mean() * 100:.0f} %")
    ax.axvline(0, color=style.INK2, linewidth=1)
    ax.set(xlabel="makespan reduction from negotiation [%]", ylabel="fraction of runs (CDF)")
    ax.legend(fontsize=7)
    style.save(fig, OUT / "fig3_negotiation_cdf")
    print(summary[summary.template == "all"].round(3).to_string(index=False))
    scheduler_steps(df, full)
    return 0


STEPS = ["+reuse", "+resched", "+reuse+resched"]
STEP_LABELS = {"+reuse": "+ reuse cost\n(HARVEST-2)", "+resched": "+ prune &\nreschedule",
               "+reuse+resched": "both\n(HARVEST-3)"}


def scheduler_steps(df: pd.DataFrame, full: pd.DataFrame) -> None:
    """Scheduler versions 2 and 3, step by step: volume, T and area vs full HARVEST, per template."""
    import matplotlib.pyplot as plt

    steps = ratios(df, "scheduler")
    if steps.empty:
        return
    idx = list(zip(steps.circuit, steps.template, steps.model))
    for col, name in (("volume", "volume_ratio"), ("T", "T_ratio"), ("area", "area_ratio"),
                      ("compile_time_s", "compile_time_ratio")):
        steps[name] = steps[col].values / full[col].reindex(idx).values
    steps.to_csv(OUT / "rq2_scheduler_ratios.csv", index=False)
    rows = []
    for template in ["all", *sorted(steps.template.unique())]:
        for (model, step), g in steps.groupby(["model", "ablation"]):
            if template != "all":
                g = g[g.template == template]
            r = g.volume_ratio.dropna()
            if r.empty:
                continue
            rows.append({"template": template, "model": model, "step": step, "n": len(r),
                         "volume_ratio_geomean": geomean(r), "p10": r.quantile(0.1),
                         "p90": r.quantile(0.9), "T_ratio_geomean": geomean(g.T_ratio),
                         "area_ratio_geomean": geomean(g.area_ratio),
                         "compile_time_ratio_geomean": geomean(g.compile_time_ratio),
                         "share_better": (r < 0.999).mean(), "share_worse": (r > 1.001).mean()})
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "rq2_scheduler_summary.csv", index=False)
    print(summary[summary.template == "all"].round(3).to_string(index=False))

    templates = ["all", *sorted(steps.template.unique())]
    metrics = [("volume_ratio_geomean", "volume"), ("T_ratio_geomean", "makespan T"),
               ("area_ratio_geomean", "area")]
    fig, axes = plt.subplots(len(metrics), len(templates), figsize=(3.2 * len(templates), 7.5),
                             sharey="row", squeeze=False)
    x = np.arange(len(STEPS))
    for j, template in enumerate(templates):
        for i, (col, label) in enumerate(metrics):
            ax = axes[i][j]
            for k, model in enumerate(MODELS):
                s = summary[(summary.template == template) & (summary.model == model)].set_index("step")
                vals = s[col].reindex(STEPS).values
                ax.bar(x + (k - 0.5) * 0.38, vals - 1, 0.36, bottom=1, color=style.MODEL_COLORS[model],
                       label=model, edgecolor=style.SURFACE, linewidth=1)
                for xi, v in zip(x + (k - 0.5) * 0.38, vals):
                    if v == v:
                        ax.text(xi, v, f"{v:.3f}", ha="center", va="bottom" if v >= 1 else "top",
                                fontsize=6.5, color=style.INK2)
            ax.axhline(1, color=style.INK2, linewidth=1)
            ax.set_xticks(x)
            ax.set_xticklabels([STEP_LABELS[s] for s in STEPS] if i == len(metrics) - 1 else [],
                               fontsize=7)
            ax.grid(axis="x", visible=False)
            if i == 0:
                ax.set_title(f"template: {template}", fontsize=9)
            if j == 0:
                ax.set_ylabel(f"geomean {label}\n÷ full HARVEST")
    axes[0][0].legend(fontsize=7)
    style.save(fig, OUT / "fig3_scheduler_steps")


if __name__ == "__main__":
    raise SystemExit(main())
