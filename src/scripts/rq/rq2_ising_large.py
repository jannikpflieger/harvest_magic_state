#!/usr/bin/env python3
"""RQ2 on the large Ising circuits (400+ qubits): ablation and scheduler steps.

The ``rq2`` study runs on the ``small`` tier only.  This repeats its
leave-one-out ablation (``rq2_ablation.VARIANTS``; ``-pruning`` is derived
from ``full``) and its scheduler steps (``rq2_ablation.SCHEDULER_STEPS``:
``+reuse`` = HARVEST-2, ``+resched``, ``+reuse+resched`` = HARVEST-3) on every
core Ising circuit with at least 400 qubits, with the same settings.

Template: blocks4 only (HARVEST ``auto`` picks it for every cultivation trial
and 6 of 11 instant trials of these circuits; dense4 is ~16x worse in volume).
Trials: instant (1 run) and cultivation seed 42.  Reference: the ``full`` rows
of this experiment.

Output: ``results/rq/rq2_ising_large/``: ``rq2_ising_large.csv``,
``ratios.csv`` (per circuit, model and variant: ratios to ``full``),
``summary.csv`` / ``summary.md`` (geomeans, share better / worse).
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from rq2_ablation import COLUMNS, SCHEDULER_STEPS, VARIANTS
from rq_common import (RESULTS, Task, cli, cost_estimate, geomean, load_suite, read_rows, run_key, run_tasks,
                       setup_logging)

EXP = "rq2_ising_large"
MIN_QUBITS = 400
TEMPLATE = "blocks4"
TRIALS = (("instant", None), ("cultivation", 42))
TIE = 0.01


def large_ising(names=()):
    return [c for c in load_suite(["core"], circuits=[n for n in names if n])
            if c["family"] == "ising" and c["num_qubits"] >= MIN_QUBITS]


def build_tasks(circuits, timeout):
    tasks = []
    for circ in circuits:
        cost = cost_estimate(circ)
        for model, seed in TRIALS:
            for name, (hconfig, layout_kw) in VARIANTS.items():
                key = run_key(EXP, "ablation", circ["circuit"], name, TEMPLATE, model, seed)
                extra = {"study": "ablation", "ablation": name, "variant": name}
                tasks.append(Task(EXP, key, "rq_common:run_in_tree", dict(
                    circ=circ, compiler="harvest", template=TEMPLATE, model=model, seed=seed,
                    hconfig=hconfig, timeout=timeout, extra=extra, **layout_kw), cost))
            for name, compiler in SCHEDULER_STEPS.items():
                key = run_key(EXP, "scheduler", circ["circuit"], name, TEMPLATE, model, seed)
                extra = {"study": "scheduler", "ablation": name, "variant": name}
                tasks.append(Task(EXP, key, "rq_common:run_in_tree", dict(
                    circ=circ, compiler=compiler, template=TEMPLATE, model=model, seed=seed,
                    timeout=timeout, extra=extra), cost * (5 if "resched" in name else 1)))
    return tasks


def analyze() -> int:
    out = RESULTS / EXP
    raw = read_rows(out / f"{EXP}.csv")
    if len(raw) == 0:
        print(f"{EXP}: no data yet")
        return 0
    bad = pd.to_numeric(raw.get("n_violations"), errors="coerce").fillna(0)
    print(f"{EXP}: {len(raw)} rows, status {raw.status.value_counts().to_dict()}, "
          f"rows with validator violations: {int((bad > 0).sum())}")
    d = raw[raw.status == "ok"].copy()
    for c in ("T_cycles", "area_tiles", "volume_tile_cycles", "volume_initial", "compile_time_s"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.rename(columns={"T_cycles": "T", "area_tiles": "area", "volume_tile_cycles": "volume",
                          "compile_time_s": "compile"})
    keys = ["circuit", "model"]
    full = d[d.ablation == "full"].set_index(keys)
    prune = full.reset_index().assign(ablation="-pruning", study="ablation", volume=lambda x: x.volume_initial,
                                      area=lambda x: x.volume_initial / x["T"])
    d = pd.concat([d[d.ablation != "full"], prune], ignore_index=True)
    idx = list(zip(d.circuit, d.model))
    for c in ("volume", "T", "area", "compile"):
        d[f"{c}_ratio"] = d[c].values / full[c].reindex(idx).values
    d.loc[d.ablation == "-pruning", "compile_ratio"] = np.nan
    ratios = d[["study", "ablation", "circuit", "model", "volume_ratio", "T_ratio", "area_ratio", "compile_ratio"]]
    ratios = ratios.dropna(subset=["volume_ratio"]).sort_values(["study", "ablation", "model", "circuit"])
    ratios.to_csv(out / "ratios.csv", index=False)
    order = [*[v for v in VARIANTS if v != "full"], "-pruning", *SCHEDULER_STEPS]
    rows = []
    for model in ("instant", "cultivation"):
        for name in order:
            s = ratios[(ratios.model == model) & (ratios.ablation == name)]
            if not len(s):
                continue
            r = s.volume_ratio
            rows.append({"model": model, "variant": name, "n": len(s), "volume": geomean(r),
                         "min": r.min(), "max": r.max(), "T": geomean(s.T_ratio), "area": geomean(s.area_ratio),
                         "compile": geomean(s.compile_ratio.dropna()) if s.compile_ratio.notna().any() else np.nan,
                         "better": int((r < 1 - TIE).sum()), "worse": int((r > 1 + TIE).sum())})
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "summary.csv", index=False)
    (out / "summary.md").write_text(summary.round(3).to_markdown(index=False) + "\n")
    print(summary.round(3).to_string(index=False))
    return 0


def main(argv=None) -> int:
    parser = cli(__doc__.split("\n\n")[0], default_jobs=3)
    args = parser.parse_args(argv)
    setup_logging(EXP)
    if not args.plot_only:
        circuits = large_ising(args.circuits.split(","))
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
