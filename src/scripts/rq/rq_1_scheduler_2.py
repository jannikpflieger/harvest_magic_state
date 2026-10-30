#!/usr/bin/env python3
"""RQ1 with HARVEST-2: HARVEST plus a reuse cost on unused tiles.

HARVEST-2 (``harvest.routing.scheduler_reuse``) adds a cost for entering a
routing cell or magic patch that no earlier time step has used, so routes
prefer tiles pruning keeps anyway.  Every cost setting in ``COSTS`` is run
and reported (no setting is picked after the fact).

Setup: the 47-circuit ``small`` tier, template ``auto`` (as RQ1), instant
(1 run) and cultivation (mean 3) with seeds 42-44.  The comparison rows are
the existing RQ1 rows of the same circuits and seeds: HARVEST from this
checkout, PureMagic from ``harvest_magic_state_perf``, EAF / Litinski / O3LS
from ``harvest_magic_state`` (as ``rq1_view.py``).  Nothing else is rerun.

Output: ``results/rq/rq_1_scheduler_2/``: ``rq_1_scheduler_2.csv`` (HARVEST-2
rows), ``per_circuit.csv``, ``table.csv/.md`` (geomean ratios to HARVEST,
W/T/L), ``fig_bars.pdf`` (geomean time steps, space, volume per compiler).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from rq_common import (RESULTS, Task, cli, cost_estimate, geomean, load_suite, read_rows, run_key,
                       run_tasks, setup_logging)

EXP = "rq_1_scheduler_2"
SEEDS = list(range(42, 45))
# reuse cost (unused routing cell, unused magic patch); one cell step costs 1
COSTS = {"c0.5m1": {"cell": 0.5, "magic": 1.0}, "c1m2": {"cell": 1.0, "magic": 2.0},
         "c2m4": {"cell": 2.0, "magic": 4.0}, "c4m8": {"cell": 4.0, "magic": 8.0}}
COLUMNS = ["reuse", "reuse_cell", "reuse_magic", "template_chosen", "auto_volumes", "validated_pre",
           "validated_post", "n_violations", "magic_wait_cycles"]
CODE = Path(__file__).resolve().parents[3].parent
SOURCES = {"Harvest": RESULTS / "rq1" / "rq1.csv",
           "PureMagic": CODE / "harvest_magic_state_perf/results/rq/rq1/rq1.csv",
           "EAF": CODE / "harvest_magic_state/results/rq/rq1/rq1.csv",
           "Litinski": CODE / "harvest_magic_state/results/rq/rq1/rq1.csv",
           "O3LS": CODE / "harvest_magic_state/results/rq/rq1/rq1.csv"}
BASELINES = ["EAF", "PureMagic", "Litinski", "O3LS"]
TIE = 0.01


def build_tasks(circuits, timeout):
    tasks = []
    for circ in circuits:
        for model, seeds in (("instant", [None]), ("cultivation", SEEDS)):
            for seed in seeds:
                for name, cost in COSTS.items():
                    key = run_key(EXP, circ["circuit"], "harvest2", name, "auto", model, seed)
                    extra = {"variant": f"harvest2_{name}", "reuse": name, "reuse_cell": cost["cell"],
                             "reuse_magic": cost["magic"]}
                    tasks.append(Task(EXP, key, "rq_common:run_in_tree", dict(
                        circ=circ, compiler="harvest2", template="auto", model=model, seed=seed,
                        reuse_cost=cost, timeout=timeout, extra=extra), cost_estimate(circ) * 4))
    return tasks


def new_rows() -> pd.DataFrame:
    """This experiment's rows, one compiler per reuse cost."""
    new = read_rows(RESULTS / EXP / f"{EXP}.csv")
    return new.assign(compiler="Harvest-2 " + new.reuse.astype(str)) if len(new) else new


def per_circuit(circuits, new: pd.DataFrame) -> pd.DataFrame:
    """Mean T, area, volume per (circuit, compiler, model) over the seeds; best variant per baseline.

    ``new``: the rows of the new scheduler(s), ``compiler`` set per configuration."""
    frames, cache = [], {}
    for comp, path in SOURCES.items():
        if path not in cache:
            cache[path] = read_rows(path)
        d = cache[path]
        frames.append(d[d.compiler == comp])
    if len(new):
        frames.append(new)
    df = pd.concat(frames, ignore_index=True)
    df = df[df.circuit.isin(circuits) & df.model.isin(["instant", "cultivation"])].copy()
    df["seed"] = pd.to_numeric(df.seed, errors="coerce")
    df = df[(df.model == "instant") | df.seed.isin(SEEDS)]
    if "template" in df:
        df = df[~df.compiler.str.startswith("Harvest") | (df.template == "auto")]
    for c in ("T_cycles", "area_tiles", "volume_tile_cycles"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["ok"] = df.status == "ok"
    g = df.groupby(["circuit", "compiler", "variant", "model"]).agg(
        n=("ok", "size"), n_ok=("ok", "sum"), T=("T_cycles", "mean"), area=("area_tiles", "mean"),
        volume=("volume_tile_cycles", "mean")).reset_index()
    expected = g.model.map({"instant": 1, "cultivation": len(SEEDS)})
    g = g[(g.n_ok == g.n) & (g.n == expected)]
    best = g.sort_values("volume").groupby(["circuit", "compiler", "model"], as_index=False).first()
    ref = best[best.compiler == "Harvest"].set_index(["circuit", "model"])
    idx = list(zip(best.circuit, best.model))
    for c in ("volume", "T", "area"):
        best[f"{c}_ratio"] = best[c].values / ref[c].reindex(idx).values
    return best


def table(best: pd.DataFrame, compilers) -> pd.DataFrame:
    rows = []
    for model in ("instant", "cultivation"):
        sub = best[best.model == model]
        for comp in compilers:
            s = sub[(sub.compiler == comp) & sub.volume_ratio.notna()]
            r = s.volume_ratio
            rows.append({"model": model, "compiler": comp, "circuits": len(s),
                         "volume_ratio": geomean(r), "T_ratio": geomean(s.T_ratio),
                         "area_ratio": geomean(s.area_ratio),
                         "wins": int((r < 1 - TIE).sum()), "ties": int(((r - 1).abs() <= TIE).sum()),
                         "losses": int((r > 1 + TIE).sum()),
                         "p10": r.quantile(0.1), "p90": r.quantile(0.9)})
    return pd.DataFrame(rows)


def fig_bars(best: pd.DataFrame, compilers, out: Path,
             title: str = "HARVEST-2 (reuse cost cell/magic) vs HARVEST and baselines, 47-circuit subset") -> None:
    """Geomean time steps, space and volume per compiler on the circuits all finished."""
    import matplotlib.pyplot as plt

    import rq_plotstyle as style

    style.apply()
    colors = [style.COMPILER_COLORS.get(c, style.SERIES[6]) for c in compilers]
    metrics = [("T", "time steps"), ("area", "space [tiles]"), ("volume", "volume [tiles × steps]")]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.4))
    for row, model in zip(axes, ("instant", "cultivation")):
        sub = best[(best.model == model) & best.compiler.isin(compilers)]
        k = sub.groupby("circuit").compiler.nunique()
        sub = sub[sub.circuit.isin(k[k == len(compilers)].index)]
        n = sub.circuit.nunique()
        for ax, (col, label) in zip(row, metrics):
            vals = [geomean(sub[sub.compiler == c][col]) for c in compilers]
            ref = vals[0]
            x = np.arange(len(compilers))
            ax.bar(x, vals, 0.7, color=colors, edgecolor=style.SURFACE, linewidth=2)
            top, low = max(vals), min(vals)
            log = top > 10 * low
            if log:
                ax.set_yscale("log")
            for xi, v in zip(x, vals):
                ax.text(xi, v * 1.15 if log else v + 0.02 * top, f"{v:,.0f}\n{v / ref:.2f}×",
                        ha="center", va="bottom", fontsize=7, color=style.INK2)
            ax.set_ylim((low * 0.5, top * 6) if log else (0, top * 1.25))
            if not log:
                ax.yaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
            ax.set_xticks(x)
            ax.set_xticklabels(compilers, rotation=35, ha="right", fontsize=7.5)
            ax.set_title(f"{label}\n{model} magic states, {n} circuits", fontsize=9)
            ax.grid(axis="x", visible=False)
        row[0].set_ylabel("geomean over circuits\n(label: value, ratio to Harvest)")
    fig.suptitle(title, fontsize=10)
    style.save(fig, out / "fig_bars")


def analyze() -> int:
    out = RESULTS / EXP
    circuits = [c["circuit"] for c in load_suite(["core"], small_only=True)]
    raw = read_rows(out / f"{EXP}.csv")
    if len(raw):
        bad = pd.to_numeric(raw.get("n_violations"), errors="coerce").fillna(0)
        print(f"{EXP}: {len(raw)} rows, status {raw.status.value_counts().to_dict()}, "
              f"rows with validator violations: {int((bad > 0).sum())}")
    best = per_circuit(circuits, new_rows())
    best.to_csv(out / "per_circuit.csv", index=False)
    compilers = ["Harvest", *[f"Harvest-2 {c}" for c in COSTS], *BASELINES]
    compilers = [c for c in compilers if (best.compiler == c).any()]
    tab = table(best, compilers)
    tab.to_csv(out / "table.csv", index=False)
    (out / "table.md").write_text(tab.round(3).to_markdown(index=False) + "\n")
    print(tab.round(3).to_string(index=False))
    fig_bars(best, compilers, out)
    return 0


def main(argv=None) -> int:
    parser = cli(__doc__.split("\n\n")[0], default_jobs=2)
    args = parser.parse_args(argv)
    setup_logging(EXP)
    if not args.plot_only:
        circuits = load_suite(["core"], small_only=True,
                              circuits=[c for c in args.circuits.split(",") if c])
        tasks = build_tasks(circuits, args.timeout)
        if args.dry_run:
            print(f"{len(circuits)} circuits, {len(tasks)} runs")
            return 0
        run_tasks(tasks, {EXP: COLUMNS}, jobs=args.jobs)
    if not args.no_plot:
        analyze()
    return 0


if __name__ == "__main__":
    sys.exit(main())
