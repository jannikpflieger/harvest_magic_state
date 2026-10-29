#!/usr/bin/env python3
"""RQ5 tables and figures: validity census, fault injection, scalability.

* Census: every Harvest/EAF schedule of rq1-rq4 (``validated_pre`` before
  pruning, ``validated_post`` after) with its violation count; the validator
  time share is validate / (compile + validate).  PureMagic, Litinski and O3LS
  report no routes in Harvest's schedule format, so they cannot be replayed
  through the validator (listed with ``validatable = False``).
* Fault injection (rq5.csv): share of activated scheduler faults and of
  corrupted schedules the validator flags, with the rules that fired.
* Scalability (rq1.csv, instant model): compile time vs number of Pauli
  rotations, log-log, one line per compiler, fitted slope; timeouts marked.
  Progress bound (rq4.csv aware rows): T <= (Lambda + 1) |O| for the instant
  (Lambda = 0) and fixed-latency (Lambda = mu) models.

Outputs (``results/rq/rq5/``): rq5_census.csv, rq5_faults.csv,
rq5_scaling_fit.csv, rq5_progress_bound.csv, fig_rq5_scaling.pdf,
fig_rq5_progress_bound.pdf, fig_rq5_faults.pdf
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import rq_plotstyle as style
from rq_common import RESULTS, geomean, read_rows

OUT = RESULTS / "rq5"
COMPILERS = ["Harvest", "EAF", "PureMagic", "Litinski", "O3LS"]


def truthy(s: pd.Series) -> pd.Series:
    return s.astype(str).isin(["True", "true", "1", "1.0"])


def census() -> pd.DataFrame:
    rows = []
    for exp in ("rq1", "rq2", "rq3", "rq4", "rq5"):
        df = read_rows(RESULTS / exp / f"{exp}.csv")
        if len(df) == 0:
            continue
        if exp == "rq5":  # only the unfaulted base schedules of the corruption study
            df = df[df.get("study") == "corrupt_base"]
        df = df[df.status == "ok"]
        for compiler, g in df.groupby("compiler"):
            validatable = compiler in ("Harvest", "EAF")
            pre, post = truthy(g.get("validated_pre", pd.Series(dtype=str))), truthy(
                g.get("validated_post", pd.Series(dtype=str)))
            vt = pd.to_numeric(g.get("validate_time_s"), errors="coerce")
            ct = pd.to_numeric(g.get("compile_time_s"), errors="coerce")
            rows.append({
                "experiment": exp, "compiler": compiler, "validatable": validatable,
                "schedules": len(g),
                "checked_pre": int(pre.sum()) if validatable else 0,
                "checked_post": int(post.sum()) if validatable else 0,
                "violations": int(pd.to_numeric(g.get("n_violations"), errors="coerce").fillna(0).sum())
                if validatable else None,
                "V1": int(pd.to_numeric(g.get("violations_V1"), errors="coerce").fillna(0).sum()) if validatable else None,
                "V2": int(pd.to_numeric(g.get("violations_V2"), errors="coerce").fillna(0).sum()) if validatable else None,
                "V3": int(pd.to_numeric(g.get("violations_V3"), errors="coerce").fillna(0).sum()) if validatable else None,
                "V4": int(pd.to_numeric(g.get("violations_V4"), errors="coerce").fillna(0).sum()) if validatable else None,
                "validator_share_of_time": (vt.sum() / (vt.sum() + ct.sum())) if validatable and vt.notna().any() else None,
                "validator_share_geomean": geomean(vt / (vt + ct)) if validatable and vt.notna().any() else None,
            })
    out = pd.DataFrame(rows)
    if len(out):
        tot = out[out.validatable].sum(numeric_only=True)
        total = {"experiment": "all", "compiler": "Harvest+EAF", "validatable": True,
                 **{k: tot[k] for k in ("schedules", "checked_pre", "checked_post", "violations",
                                        "V1", "V2", "V3", "V4")}}
        out = pd.concat([out, pd.DataFrame([total])], ignore_index=True)
    return out


def faults(plt) -> pd.DataFrame:
    df = read_rows(RESULTS / "rq5" / "rq5.csv")
    if len(df) == 0:
        return pd.DataFrame()
    rows = []
    sched = df[df.study == "scheduler"].copy()
    sched["act"] = pd.to_numeric(sched.fault_activations, errors="coerce").fillna(0) > 0
    for (fault, model), g in sched.groupby(["fault", "model"]):
        act = g[g.act]
        rows.append({"study": "scheduler", "kind": fault, "model": model, "runs": len(g),
                     "applicable": len(act), "caught": int(truthy(act.caught).sum()),
                     "caught_share": truthy(act.caught).mean() if len(act) else np.nan,
                     "rules": ",".join(sorted(set(act.first_violation_rule.dropna().astype(str))))})
    cor = df[df.study == "corrupt"]
    for (kind, model), g in cor.groupby(["mutation", "model"]):
        app = g[g.status == "ok"]
        rows.append({"study": "corrupt", "kind": kind, "model": model, "runs": len(g),
                     "applicable": len(app), "caught": int(truthy(app.caught).sum()),
                     "caught_share": truthy(app.caught).mean() if len(app) else np.nan,
                     "V4_fired_share": (pd.to_numeric(app.violations_V4, errors="coerce") > 0).mean()
                     if len(app) else np.nan,
                     "rules": ",".join(sorted(set(app.first_violation_rule.dropna().astype(str))))})
    out = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    labels = [f"{r.kind}\n({r.model})" for r in out.itertuples()]
    colors = [style.SERIES[0] if s == "scheduler" else style.SERIES[1] for s in out.study]
    ax.bar(np.arange(len(out)), out.caught_share * 100, 0.7, color=colors)
    for i, r in enumerate(out.itertuples()):
        ax.text(i, (0 if np.isnan(r.caught_share) else r.caught_share * 100) + 1,
                f"{r.caught}/{r.applicable}", ha="center", fontsize=6, color=style.INK2)
    ax.set_xticks(np.arange(len(out)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=6)
    ax.set_ylabel("caught by validator [%]")
    ax.set_ylim(0, 110)
    ax.grid(axis="x", visible=False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=style.SERIES[0], label="injected scheduler fault"),
                       Patch(color=style.SERIES[1], label="corrupted schedule")], fontsize=7,
              loc="lower right")
    style.save(fig, OUT / "fig_rq5_faults")
    return out


def scaling(plt) -> pd.DataFrame:
    df = read_rows(RESULTS / "rq1" / "rq1.csv")
    if len(df) == 0:
        return pd.DataFrame()
    df = df[df.model == "instant"].copy()
    df["compile_time_s"] = pd.to_numeric(df.compile_time_s, errors="coerce")
    df["wall_time_s"] = pd.to_numeric(df.wall_time_s, errors="coerce")
    fig, ax = plt.subplots(figsize=(6, 4))
    rows = []
    for compiler in COMPILERS:
        g = df[df.compiler == compiler]
        ok = g[g.status == "ok"].groupby("circuit").agg(ops=("ops", "first"),
                                                        t=("compile_time_s", "median"))
        ok = ok[(ok.t > 0) & (ok.ops > 0)].sort_values("ops")
        to = g[g.status == "timeout"].groupby("circuit").ops.first()
        color = style.COMPILER_COLORS[compiler]
        if len(ok) >= 2:
            slope, icpt = np.polyfit(np.log10(ok.ops), np.log10(ok.t), 1)
            big = ok[ok.ops >= 1000]
            slope_big = np.polyfit(np.log10(big.ops), np.log10(big.t), 1)[0] if len(big) >= 2 else np.nan
            rows.append({"compiler": compiler, "circuits": len(ok), "timeouts": len(to),
                         "slope_loglog": slope, "slope_loglog_ops_ge_1000": slope_big,
                         "intercept": icpt})
            ax.plot(ok.ops, ok.t, "o", ms=3, alpha=0.5, color=color)
            xs = np.array([ok.ops.min(), ok.ops.max()])
            ax.plot(xs, 10 ** icpt * xs ** slope, "-", color=color, label=f"{compiler} (slope {slope:.2f})")
        if len(to):
            ax.plot(to.values, [df.compile_time_s.max() or 3600] * len(to), "x", color=color, ms=6)
    ax.set(xscale="log", yscale="log", xlabel="Pauli rotations |O|",
           ylabel="compile time [s] (instant model)")
    ax.legend(fontsize=7)
    style.save(fig, OUT / "fig_rq5_scaling")
    return pd.DataFrame(rows)


def progress_bound(plt) -> pd.DataFrame:
    df = read_rows(RESULTS / "rq4" / "rq4.csv")
    if len(df) == 0:
        return pd.DataFrame()
    lam = {"instant": 0, "dist4": 4, "dist10": 10}
    df = df[(df["mode"] == "aware") & df.model.isin(lam) & (df.status == "ok")].copy()
    df["T"] = pd.to_numeric(df.T_cycles, errors="coerce")
    df["bound"] = (df.model.map(lam) + 1) * pd.to_numeric(df.ops, errors="coerce")
    df["T_over_bound"] = df["T"] / df.bound
    fig, ax = plt.subplots(figsize=(5, 4))
    for model in lam:
        g = df[df.model == model]
        ax.plot(g.bound, g["T"], "o", ms=3, color=style.MODEL_COLORS[model],
                label=f"{model} (max T/bound {g.T_over_bound.max():.2f})")
    lim = [df.bound.min() * 0.8, df.bound.max() * 1.2]
    ax.plot(lim, lim, ":", color=style.INK2, label="T = (Λ+1)|O|")
    ax.set(xscale="log", yscale="log", xlabel="progress bound (Λ+1)|O|", ylabel="makespan T")
    ax.legend(fontsize=7)
    style.save(fig, OUT / "fig_rq5_progress_bound")
    return df.groupby("model").agg(schedules=("T", "size"), violations=("T_over_bound", lambda x: int((x > 1).sum())),
                                   T_over_bound_max=("T_over_bound", "max"),
                                   T_over_bound_median=("T_over_bound", "median")).reset_index()


def main() -> int:
    style.apply()
    import matplotlib.pyplot as plt

    OUT.mkdir(parents=True, exist_ok=True)
    c = census()
    c.to_csv(OUT / "rq5_census.csv", index=False)
    f = faults(plt)
    f.to_csv(OUT / "rq5_faults.csv", index=False)
    s = scaling(plt)
    s.to_csv(OUT / "rq5_scaling_fit.csv", index=False)
    p = progress_bound(plt)
    p.to_csv(OUT / "rq5_progress_bound.csv", index=False)
    for name, table in (("census", c), ("faults", f), ("scaling", s), ("progress bound", p)):
        print(f"== {name}")
        print(table.round(3).to_string(index=False) if len(table) else "(no data yet)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
