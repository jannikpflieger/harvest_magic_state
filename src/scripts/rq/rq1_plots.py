#!/usr/bin/env python3
"""RQ1 tables and figures from ``results/rq/rq1/rq1.csv``.

Aggregation: per (circuit, compiler, variant, model) the mean T, area and
volume over seeds; each baseline is represented by its best variant per
circuit (lowest mean volume: PureMagic bus/magic routing, Litinski
compact/intermediate/fast, O3LS min/medium/large).  A circuit counts for a
compiler/model only if every trial finished (``status == ok``); ratios use the
circuits both compilers finished.  Ratios above 1: the baseline is worse.
Ties: |ratio - 1| <= 1 %.

Outputs (``results/rq/rq1/``):
  rq1_per_circuit.csv   one row per circuit/compiler/model (best variant), with ratios
  rq1_table.csv/.tex    Table 1
  rq1_losses.csv        circuits where a baseline beats Harvest, with the cause
  fig1_family_volume.pdf      all five compilers, volume / lower bound per family
  fig1_per_circuit_volume.pdf volume / best compiler per circuit, sorted
  fig1_family_raw_{volume,timesteps,area}.pdf  geomean raw value per family
  rq1_family_advantage.csv    Harvest volume / each baseline per family
  fig1_tiles_vs_size.pdf, fig1_timesteps_vs_size.pdf  tiles / timesteps vs qubits,
                              one line per compiler (+ _by_family_<model>)
  fig2a_volume_decomposition.pdf  area / bound and makespan / critical path per compiler
  fig1_geomean_bars.pdf       geomean time steps, space and volume per compiler
  families/fig1_bars_<family>.pdf  the same per family (Ising split at 100 qubits)
  fig1_by_size_<model>.pdf    time steps, space and volume / Harvest per circuit, by size

Figures use the circuits every compiler finished, so each bar averages the same set.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import rq_plotstyle as style
from rq_common import RESULTS, geomean, read_rows

OUT = RESULTS / "rq1"
BASELINES = ["EAF", "PureMagic", "Litinski", "O3LS"]
COMPILERS = ["Harvest", *BASELINES]
MODELS = ["instant", "cultivation"]
TIE = 0.01


def per_circuit(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ok"] = df["status"] == "ok"
    for col in ("T_cycles", "area_tiles", "volume_tile_cycles", "compile_time_s",
                "critical_path", "area_initial_tiles"):
        df[col] = pd.to_numeric(df.get(col), errors="coerce")
    group = df.groupby(["circuit", "compiler", "variant", "model"])
    agg = group.agg(family=("family", "first"), num_qubits=("num_qubits", "first"),
                    ops=("ops", "first"), critical_path=("critical_path", "first"),
                    n_runs=("ok", "size"), n_ok=("ok", "sum"),
                    T=("T_cycles", "mean"), area=("area_tiles", "mean"),
                    volume=("volume_tile_cycles", "mean"), T_std=("T_cycles", "std"),
                    volume_std=("volume_tile_cycles", "std"),
                    compile_time_s=("compile_time_s", "mean"),
                    area_initial=("area_initial_tiles", "mean")).reset_index()
    agg["completed"] = agg["n_ok"] == agg["n_runs"]
    agg.loc[~agg["completed"], ["T", "area", "volume"]] = np.nan
    # best variant per circuit for each compiler/model (completed variants only)
    agg = agg.sort_values("volume", na_position="last")
    best = agg.groupby(["circuit", "compiler", "model"], as_index=False).first()
    best["completed"] = best["volume"].notna()
    ref = best[best.compiler == "Harvest"].set_index(["circuit", "model"])
    for col, name in (("volume", "volume_ratio"), ("T", "T_ratio"), ("area", "area_ratio"),
                      ("compile_time_s", "compile_time_ratio")):
        best[name] = best[col].values / ref[col].reindex(
            list(zip(best.circuit, best.model))).values
    best["T_over_cp"] = best["T"] / best["critical_path"]
    # neutral references, so Harvest can be shown next to the baselines:
    # lower bound = critical path x (data patches + one magic patch)
    best["area_lb"] = best["num_qubits"] + 1
    best["area_over_lb"] = best["area"] / best["area_lb"]
    best["volume_over_lb"] = best["volume"] / (best["critical_path"] * best["area_lb"])
    best["volume_over_best"] = best["volume"] / best.groupby(["circuit", "model"]).volume.transform("min")
    return best


def common(best: pd.DataFrame, model: str) -> pd.DataFrame:
    """Rows of ``model`` on the circuits every compiler finished (same set per bar)."""
    sub = best[(best.model == model) & best.volume.notna()]
    n = sub.groupby("circuit").compiler.nunique()
    return sub[sub.circuit.isin(n[n == len(COMPILERS)].index)]


def family_group(fam: pd.Series, counts: pd.Series, min_count: int = 3) -> pd.Series:
    return fam.where(fam.map(counts) >= min_count, "other")


def table(best: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n_circuits = best.circuit.nunique()
    for compiler in ["Harvest", *BASELINES]:
        row = {"compiler": compiler}
        for model in MODELS:
            sub = best[(best.compiler == compiler) & (best.model == model)]
            both = sub[sub["volume_ratio"].notna()]
            r = both["volume_ratio"]
            row.update({
                f"{model}_volume_ratio_geomean": geomean(r),
                f"{model}_T_ratio_geomean": geomean(both["T_ratio"]),
                f"{model}_area_ratio_geomean": geomean(both["area_ratio"]),
                f"{model}_T_over_cp_geomean": geomean(sub["T_over_cp"]),
                f"{model}_patches_median": sub["area"].median(),
                f"{model}_wins": int((r < 1 - TIE).sum()),   # baseline better than Harvest
                f"{model}_ties": int(((r - 1).abs() <= TIE).sum()),
                f"{model}_losses": int((r > 1 + TIE).sum()),  # baseline worse than Harvest
                f"{model}_faster_than_harvest": int((both["T_ratio"] < 1 - TIE).sum()),
                f"{model}_compile_time_geomean_s": geomean(sub["compile_time_s"]),
                f"{model}_completed": int(sub["completed"].sum()),
                f"{model}_circuits": int(sub.circuit.nunique()),
                "suite_circuits": n_circuits,
            })
        rows.append(row)
    return pd.DataFrame(rows)


def write_tex(tab: pd.DataFrame, path) -> None:
    lines = [r"\begin{tabular}{l" + "rrrrrr" * 2 + "}", r"\toprule",
             r" & \multicolumn{6}{c}{instant} & \multicolumn{6}{c}{cultivation} \\",
             r"Compiler" + r" & vol.\ ratio & $T/T_{cp}$ & patches & W/T/L & compile [s] & done" * 2
             + r" \\", r"\midrule"]
    for _, r in tab.iterrows():
        cells = [r.compiler]
        for m in MODELS:
            wtl = "--" if r.compiler == "Harvest" else f"{r[m + '_losses']}/{r[m + '_ties']}/{r[m + '_wins']}"
            cells += [f"{r[m + '_volume_ratio_geomean']:.3f}", f"{r[m + '_T_over_cp_geomean']:.2f}",
                      f"{r[m + '_patches_median']:.0f}", wtl,
                      f"{r[m + '_compile_time_geomean_s']:.2f}",
                      f"{r[m + '_completed']}/{r['suite_circuits']}"]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              "% W/T/L from Harvest's view: circuits where Harvest has lower / equal (1%) / higher volume."]
    path.write_text("\n".join(lines) + "\n")


def losses(best: pd.DataFrame) -> pd.DataFrame:
    """Circuits where a baseline reaches lower volume than Harvest, with the dominant cause."""
    lost = best[(best.compiler != "Harvest") & (best.volume_ratio < 1 - TIE)].copy()
    lost["cause"] = np.where(np.log(lost.area_ratio) < np.log(lost.T_ratio), "area", "makespan")
    cols = ["model", "compiler", "variant", "circuit", "family", "num_qubits", "ops",
            "volume_ratio", "T_ratio", "area_ratio", "cause"]
    return lost[cols].sort_values(["model", "compiler", "volume_ratio"])


def fig_family(best: pd.DataFrame) -> None:
    """Grouped bars per circuit family: geomean volume / lower bound, all five compilers."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    width = 0.8 / len(COMPILERS)
    for ax, model in zip(axes, MODELS):
        sub = common(best, model)
        counts = sub[sub.compiler == "Harvest"].groupby("family").circuit.nunique()
        sub = sub.assign(group=family_group(sub.family, counts))
        groups = sorted(sub.group.unique(), key=lambda g: (g == "other", g)) + ["All"]
        for i, compiler in enumerate(COMPILERS):
            vals = [geomean(sub[(sub.compiler == compiler) & ((sub.group == g) | (g == "All"))].volume_over_lb)
                    for g in groups]
            x = np.arange(len(groups)) + (i - (len(COMPILERS) - 1) / 2) * width
            ax.bar(x, [v - 1 if v == v else 0 for v in vals], width * 0.9, bottom=1,
                   color=style.COMPILER_COLORS[compiler], label=compiler)
        ax.set_yscale("log")
        ax.set_xticks(np.arange(len(groups)))
        n_fam = sub[sub.compiler == "Harvest"].groupby("group").circuit.nunique()
        ax.set_xticklabels([f"{g}\n({n_fam.get(g, sub.circuit.nunique())})" if g != "All"
                            else f"All\n({sub.circuit.nunique()})" for g in groups],
                           rotation=45, ha="right", fontsize=7)
        ax.set_title(f"{model} magic states ({sub.circuit.nunique()} circuits all compilers finished)")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("geomean volume / lower bound\n(critical path × (qubits+1)), lower is better")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=5, loc="upper center", fontsize=8)
    style.save(fig, OUT / "fig1_family_volume")


def fig_per_circuit(best: pd.DataFrame) -> None:
    """Sorted per-circuit volume relative to the best compiler on that circuit."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, model in zip(axes, MODELS):
        sub = common(best, model)
        for compiler in COMPILERS:
            r = np.sort(sub[sub.compiler == compiler].volume_over_best.dropna())
            if len(r):
                ax.plot(np.arange(1, len(r) + 1) / len(r), r, color=style.COMPILER_COLORS[compiler],
                        label=f"{compiler} (best on {(r <= 1 + TIE).sum()})")
        ax.axhline(1, color=style.INK2, linewidth=1)
        ax.set_yscale("log")
        ax.set_xlabel("fraction of circuits (each line sorted)")
        ax.set_title(f"{model} magic states ({sub.circuit.nunique()} circuits)")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("volume / best compiler on the circuit")
    style.save(fig, OUT / "fig1_per_circuit_volume")


def fig_decomposition(best: pd.DataFrame) -> None:
    """Volume / bound = (area / area bound) x (makespan / critical path), per compiler."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    x = np.arange(len(COMPILERS))
    for ax, model in zip(axes, MODELS):
        sub = common(best, model)
        area = [geomean(sub[sub.compiler == c].area_over_lb) for c in COMPILERS]
        time = [geomean(sub[sub.compiler == c].T_over_cp) for c in COMPILERS]
        vol = [geomean(sub[sub.compiler == c].volume_over_lb) for c in COMPILERS]
        ax.bar(x - 0.2, np.subtract(area, 1), 0.38, bottom=1, color=style.SERIES[6],
               label="area / (qubits+1)")
        ax.bar(x + 0.2, np.subtract(time, 1), 0.38, bottom=1, color=style.SERIES[3],
               label="makespan / critical path")
        ax.scatter(x, vol, color=style.INK, zorder=3, s=26, label="volume / bound (= product)")
        for xi, v in zip(x, vol):
            ax.annotate(f"{v:.2f}", (xi, v), textcoords="offset points", xytext=(0, 5),
                        ha="center", fontsize=7, color=style.INK2)
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(COMPILERS)
        ax.set_title(f"{model} magic states ({sub.circuit.nunique()} circuits)")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("geomean over circuits (lower is better)")
    axes[0].legend(fontsize=7)
    style.save(fig, OUT / "fig2a_volume_decomposition")


def family_advantage(best: pd.DataFrame) -> pd.DataFrame:
    """Per family and model: Harvest's volume against each baseline and the best of them."""
    rows = []
    for model in MODELS:
        sub = common(best, model)
        piv = sub.pivot_table(index=["family", "circuit"], columns="compiler", values="volume")
        piv["best_baseline"] = piv[BASELINES].min(axis=1)
        for family, g in piv.groupby(level="family"):
            row = {"model": model, "family": family, "circuits": len(g)}
            for c in [*BASELINES, "best_baseline"]:
                r = g["Harvest"] / g[c]
                row[f"harvest_over_{c}"] = geomean(r)  # < 1: Harvest needs less volume
            row["harvest_best_count"] = int((g["Harvest"] <= g["best_baseline"] * (1 + TIE)).sum())
            row["harvest_strictly_best"] = int((g["Harvest"] < g["best_baseline"] * (1 - TIE)).sum())
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["model", "harvest_over_best_baseline"])


RAW = [("volume", "volume [tile × timestep]", "fig1_family_raw_volume"),
       ("T", "makespan [timesteps]", "fig1_family_raw_timesteps"),
       ("area", "space [tiles]", "fig1_family_raw_area")]


def fig_family_raw(best: pd.DataFrame) -> None:
    """One chart per metric: geomean raw value per circuit family, all five compilers."""
    import matplotlib.pyplot as plt

    for metric, label, name in RAW:
        fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
        width = 0.8 / len(COMPILERS)
        for ax, model in zip(axes, MODELS):
            sub = common(best, model)
            counts = sub[sub.compiler == "Harvest"].groupby("family").circuit.nunique()
            sub = sub.assign(group=family_group(sub.family, counts))
            groups = sorted(sub.group.unique(), key=lambda g: (g == "other", g)) + ["All"]
            for i, compiler in enumerate(COMPILERS):
                vals = [geomean(sub[(sub.compiler == compiler) & ((sub.group == g) | (g == "All"))][metric])
                        for g in groups]
                x = np.arange(len(groups)) + (i - (len(COMPILERS) - 1) / 2) * width
                ax.bar(x, vals, width * 0.9, color=style.COMPILER_COLORS[compiler], label=compiler)
            n_fam = sub[sub.compiler == "Harvest"].groupby("group").circuit.nunique()
            ax.set_yscale("log")
            ax.set_xticks(np.arange(len(groups)))
            ax.set_xticklabels([f"{g}\n({n_fam.get(g, sub.circuit.nunique())})" if g != "All"
                                else f"All\n({sub.circuit.nunique()})" for g in groups],
                               rotation=45, ha="right", fontsize=7)
            ax.set_title(f"{model} magic states ({sub.circuit.nunique()} circuits)")
            ax.grid(axis="x", visible=False)
        axes[0].set_ylabel(f"geomean {label}\nper family (lower is better)")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, ncol=5, loc="upper center", fontsize=8)
        style.save(fig, OUT / name)


def fig_ising_large(best: pd.DataFrame) -> None:
    """Ising n >= 400, per circuit: volume, makespan and space of every finished compiler."""
    import matplotlib.pyplot as plt

    sub = best[(best.family == "ising") & (best.num_qubits >= 400) & best.volume.notna()]
    if sub.empty:
        return
    circuits = sorted(sub.circuit.unique(), key=lambda c: int(c.split("_n")[1]))
    fig, axes = plt.subplots(3, 2, figsize=(12, 8.5), sharex=True)
    width = 0.8 / len(COMPILERS)
    for j, model in enumerate(MODELS):
        m = sub[sub.model == model]
        missing = [c for c in COMPILERS if c not in set(m.compiler)]
        for i, (metric, label, _) in enumerate(RAW):
            ax = axes[i, j]
            for k, compiler in enumerate(COMPILERS):
                vals = m[m.compiler == compiler].set_index("circuit")[metric].reindex(circuits)
                x = np.arange(len(circuits)) + (k - (len(COMPILERS) - 1) / 2) * width
                ax.bar(x, vals.values, width * 0.9, color=style.COMPILER_COLORS[compiler], label=compiler)
            ax.set_yscale("log")
            ax.set_ylabel(label)
            ax.grid(axis="x", visible=False)
            if i == 0:
                note = f" — not finished yet: {', '.join(missing)}" if missing else ""
                ax.set_title(f"{model} magic states{note}", fontsize=9)
        axes[2, j].set_xticks(np.arange(len(circuits)))
        axes[2, j].set_xticklabels([c.replace("ising_", "") for c in circuits], rotation=45)
        axes[2, j].set_xlabel("Ising circuit (qubits)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=5, loc="upper center", fontsize=8)
    style.save(fig, OUT / "fig1_ising_large_raw")
    sub[["model", "circuit", "num_qubits", "ops", "compiler", "variant", "n_runs", "T", "area",
         "volume"]].sort_values(["model", "num_qubits", "compiler"]).to_csv(
        OUT / "rq1_ising_large.csv", index=False)


def fig_ising_large_model(best: pd.DataFrame, model: str = "instant", name: str = "fig1_ising_large_instant",
                          title: str = "instant magic states", compilers=None) -> None:
    """Ising n >= 400 under one magic model: volume, makespan and space against qubits.

    The makespan panel also shows the critical path of the Pauli-product DAG,
    the lower bound every compiler is measured against.
    """
    import matplotlib.pyplot as plt

    sub = best[(best.family == "ising") & (best.num_qubits >= 400) & (best.model == model)
               & best.volume.notna()]
    if sub.empty:
        return
    compilers = compilers or COMPILERS
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (metric, label, _) in zip(axes, RAW):
        for compiler in compilers:
            c = sub[sub.compiler == compiler].sort_values("num_qubits")
            if c.empty:
                continue
            ax.plot(c.num_qubits, c[metric], "-", marker="o", ms=3.5, lw=1.5,
                    color=style.COMPILER_COLORS[compiler], label=compiler)
        if metric == "T":
            cp = sub.groupby("num_qubits").critical_path.first().sort_index()
            ax.plot(cp.index, cp.values, "--", lw=1, color="0.4", label="critical path")
        ax.set_yscale("log")
        ax.set_ylabel(label)
        ax.set_xlabel("Ising circuit size [qubits]")
    missing = [c for c in compilers if c not in set(sub.compiler)]
    partial = [c for c in compilers
               if c not in missing and sub[sub.compiler == c].circuit.nunique() < sub.circuit.nunique()]
    note = []
    if partial:
        note.append(f"partly finished: {', '.join(partial)}")
    if missing:
        note.append(f"not finished yet: {', '.join(missing)}")
    fig.suptitle(f"Ising n400–n440, {title}" + (f" ({'; '.join(note)})" if note else ""),
                 fontsize=10, y=0.99)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=6, loc="upper center", bbox_to_anchor=(0.5, 0.94), fontsize=8)
    style.save(fig, OUT / name)


def fig_metric_vs_size(best: pd.DataFrame, metric: str = "area", label: str = "space [tiles]",
                       name: str = "fig1_tiles_vs_size", short: str = "tiles") -> None:
    """``metric`` (tiles or timesteps) against circuit size (qubits), one line per compiler.

    Each line passes through the mean value of the circuits with the same
    number of qubits; the faint dots are the individual circuits.  Uses the
    circuits every compiler finished.  A second figure repeats it per family.
    """
    import matplotlib.pyplot as plt

    def draw(ax, sub):
        for compiler in COMPILERS:
            c = sub[sub.compiler == compiler]
            if c.empty:
                continue
            color = style.COMPILER_COLORS[compiler]
            ax.scatter(c.num_qubits, c[metric], s=9, color=color, alpha=0.25, linewidths=0)
            line = c.groupby("num_qubits")[metric].mean().sort_index()
            ax.plot(line.index, line.values, "-", marker="o", ms=3, lw=1.5, color=color,
                    label=compiler)
        ax.set_xscale("log")
        ax.set_yscale("log")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, model in zip(axes, MODELS):
        sub = common(best, model)
        draw(ax, sub)
        ax.set_title(f"{model} magic states ({sub.circuit.nunique()} circuits)")
        ax.set_xlabel("circuit size [qubits]")
    axes[0].set_ylabel(f"{label} (lower is better)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=5, loc="upper center", fontsize=8)
    style.save(fig, OUT / name)

    for model in MODELS:
        sub = common(best, model)
        counts = sub[sub.compiler == "Harvest"].groupby("family").circuit.nunique()
        families = sorted(counts[counts >= 3].index)
        if not families:
            continue
        cols = 4
        rows = -(-len(families) // cols)
        fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.7 * rows), squeeze=False)
        for ax, family in zip(axes.flat, families):
            draw(ax, sub[sub.family == family])
            ax.set_title(f"{family} ({counts[family]})", fontsize=9)
            ax.set_xlabel("qubits", fontsize=8)
            ax.set_ylabel(short, fontsize=8)
        for ax in list(axes.flat)[len(families):]:
            ax.set_visible(False)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, ncol=5, loc="upper center", fontsize=8,
                   title=f"{label.split(' [')[0]} vs circuit size per family, {model} magic states",
                   title_fontsize=9)
        style.save(fig, OUT / f"{name}_by_family_{model}")


METRICS = [("T", "time steps"), ("area", "space [tiles]"), ("volume", "volume [tiles × steps]")]
MARKERS = {"Harvest": "o", "EAF": "s", "PureMagic": "^", "Litinski": "D", "O3LS": "v"}


def bar_group(fam: str, qubits: int, counts: pd.Series) -> str:
    """Family for the per-family bars: Ising split by size, single circuits as "other"."""
    if fam == "ising":
        return "ising (< 100 qubits)" if qubits < 100 else "ising (>= 400 qubits)"
    return fam if counts.get(fam, 0) >= 2 else "other"


def bar_rows(best: pd.DataFrame, model: str, circuits) -> tuple[pd.DataFrame, dict]:
    """Rows for one bar chart: the circuits every shown compiler finished.

    A compiler is shown if it finished at least half of ``circuits`` (those
    Harvest finished); the others get an empty bar labelled with their count,
    so one missing compiler does not shrink the set the others are averaged on.
    """
    sub = best[(best.model == model) & best.circuit.isin(circuits) & best.volume.notna()]
    n = sub[sub.compiler == "Harvest"].circuit.nunique()
    done = sub.groupby("compiler").circuit.nunique().reindex(COMPILERS, fill_value=0)
    shown = [c for c in COMPILERS if c == "Harvest" or done[c] * 2 >= n]
    sub = sub[sub.compiler.isin(shown)]
    k = sub.groupby("circuit").compiler.nunique()
    sub = sub[sub.circuit.isin(k[k == len(shown)].index)]
    return sub, {c: (int(done[c]), n) for c in COMPILERS if c not in shown}


def bar_figure(best: pd.DataFrame, circuits, title: str, path) -> None:
    """Geomean time steps, space and volume per compiler: one row per magic-state model."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(MODELS), len(METRICS), figsize=(11, 3.4 * len(MODELS)), squeeze=False)
    for row, model in zip(axes, MODELS):
        sub, missing = bar_rows(best, model, circuits)
        n = sub.circuit.nunique()
        ref = {col: geomean(sub[sub.compiler == "Harvest"][col]) for col, _ in METRICS}
        for ax, (col, label) in zip(row, METRICS):
            vals = [geomean(sub[sub.compiler == c][col]) if c not in missing else np.nan for c in COMPILERS]
            x = np.arange(len(COMPILERS))
            ax.bar(x, [v if v == v else 0 for v in vals], 0.7, color=[style.COMPILER_COLORS[c] for c in COMPILERS],
                   edgecolor=style.SURFACE, linewidth=2)
            shown = [v for v in vals if v == v] or [1]
            top, low = max(shown), min(shown)
            log = top > 10 * low  # e.g. large Ising: Litinski/O3LS 20-90x Harvest
            if log:
                ax.set_yscale("log")
            above = (lambda v: v * 1.15) if log else (lambda v: v + 0.02 * top)
            for xi, c, v in zip(x, COMPILERS, vals):
                if v == v:
                    ax.text(xi, above(v), f"{v:,.0f}\n{v / ref[col]:.2f}×",
                            ha="center", va="bottom", fontsize=7.5, color=style.INK2)
                else:
                    done, total = missing.get(c, (0, 0))
                    ax.text(xi, low * 0.6 if log else 0.02 * top, f"{done}/{total}\ndone",
                            ha="center", va="bottom", fontsize=7, color=style.MUTED)
            ax.set_ylim((low * 0.5, top * 6) if log else (0, top * 1.22))
            if not log:
                ax.yaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
            ax.set_xticks(x)
            ax.set_xticklabels(COMPILERS, rotation=30, ha="right", fontsize=8)
            ax.set_title(f"{label}\n{model} magic states, {n} circuits", fontsize=9)
            ax.grid(axis="x", visible=False)
        row[0].set_ylabel("geomean over circuits\n(label: value, ratio to Harvest)")
    fig.suptitle(title, fontsize=10)
    style.save(fig, path)


def fig_geomean_bars(best: pd.DataFrame) -> None:
    """Geomean time steps, space and volume per compiler over all circuits."""
    bar_figure(best, best.circuit.unique(), "All circuits", OUT / "fig1_geomean_bars")


def fig_family_bars(best: pd.DataFrame) -> None:
    """``fig1_geomean_bars`` per circuit family (``families/``); Ising split at 100 qubits."""
    h = best[best.compiler == "Harvest"].drop_duplicates("circuit")
    counts = h.family.value_counts()
    groups = {c: bar_group(f, int(q), counts) for c, f, q in zip(h.circuit, h.family, h.num_qubits)}
    (OUT / "families").mkdir(exist_ok=True)
    for g in sorted(set(groups.values())):
        circuits = [c for c, gg in groups.items() if gg == g]
        name = g.replace(" ", "").replace("(", "_").replace(")", "").replace("<", "lt").replace(">=", "ge")
        bar_figure(best, circuits, f"{g}: {len(circuits)} circuits", OUT / "families" / f"fig1_bars_{name}")


def fig_by_size(best: pd.DataFrame, model: str, window: int = 9) -> None:
    """Per-circuit time steps, space and volume relative to Harvest, circuits sorted by size.

    One point per circuit and compiler (circuits every compiler finished),
    ordered by qubits then operations; the line is a sliding geomean over
    ``window`` neighbouring circuits.
    """
    import matplotlib.pyplot as plt

    sub = common(best, model)
    if sub.empty:
        return
    size = (sub[sub.compiler == "Harvest"].set_index("circuit")[["num_qubits", "ops"]]
            .sort_values(["num_qubits", "ops"]))
    order = list(size.index)
    x = np.arange(len(order))
    fig, axes = plt.subplots(len(METRICS), 1, figsize=(11, 8.2), sharex=True)
    for ax, (col, label) in zip(axes, METRICS):
        ref = sub[sub.compiler == "Harvest"].set_index("circuit")[col].reindex(order)
        for compiler in COMPILERS:
            color = style.COMPILER_COLORS[compiler]
            if compiler == "Harvest":
                ax.axhline(1, color=color, linewidth=2, label="Harvest")
                continue
            s = sub[sub.compiler == compiler].set_index("circuit")[col].reindex(order)
            r = (s / ref).astype(float)
            ax.scatter(x, r.values, s=14, marker=MARKERS[compiler], color=color, alpha=0.45,
                       linewidths=0)
            trend = np.exp(np.log(r).rolling(window, center=True, min_periods=window // 2 + 1).mean())
            ax.plot(x, trend.values, color=color, linewidth=2, marker=MARKERS[compiler],
                    markevery=max(1, len(order) // 8), markersize=6, label=compiler)
        ax.set_yscale("log")
        ax.set_ylabel(f"{label}\n÷ Harvest")
        ax.grid(axis="x", visible=False)
    ticks = list(range(0, len(order), max(1, len(order) // 15)))
    axes[-1].set_xticks(ticks)
    axes[-1].set_xticklabels([str(int(size.num_qubits.iloc[i])) for i in ticks], fontsize=8)
    axes[-1].set_xlabel(f"circuits sorted by size ({len(order)} circuits; tick = qubits); "
                        f"above 1 = worse than Harvest; line = sliding geomean of {window}")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=5, loc="upper center", fontsize=8,
               title=f"{model} magic states: each compiler relative to Harvest, by circuit size",
               title_fontsize=9)
    style.save(fig, OUT / f"fig1_by_size_{model}")


def main() -> int:
    style.apply()
    df = read_rows(OUT / "rq1.csv")
    if len(df) == 0:
        print("rq1: no data yet")
        return 0
    best = per_circuit(df)
    best.to_csv(OUT / "rq1_per_circuit.csv", index=False)
    tab = table(best)
    tab.to_csv(OUT / "rq1_table.csv", index=False)
    write_tex(tab, OUT / "rq1_table.tex")
    losses(best).to_csv(OUT / "rq1_losses.csv", index=False)
    fig_geomean_bars(best)
    fig_family_bars(best)
    family_advantage(best).to_csv(OUT / "rq1_family_advantage.csv", index=False)
    print(tab.round(3).T.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
