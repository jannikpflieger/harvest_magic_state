#!/usr/bin/env python3
"""HARVEST-3 on the large Ising circuits (400+ qubits).

``rq_1_scheduler_3`` ran on the 47-circuit ``small`` tier only.  This runs
HARVEST-3 (HARVEST-2 reuse cost cell 1 / magic 2, then up to four prune and
reschedule rounds, as ``rq_1_scheduler_3``) on every core Ising circuit with
at least 400 qubits.  ``auto`` would schedule all four templates five times
each, so every trial uses the template HARVEST's ``auto`` chose for that
circuit, model and seed in RQ1 (read from ``results/rq/rq1/rq1.csv``).

Trials: instant (1 run) and cultivation seed 42.  Compared against the RQ1
HARVEST row of the same circuit, model and seed; ``volume_round0`` etc. hold
HARVEST-2 on that template (the first schedule, before any pruning round).

Output: ``results/rq/rq_1_scheduler_3_ising/``: ``rq_1_scheduler_3_ising.csv``,
``per_circuit.csv`` (ratios to HARVEST), ``rounds.csv``, ``table.md``.
"""

from __future__ import annotations

import sys

import pandas as pd

import rq_1_scheduler_3 as s3
from rq_common import RESULTS, Task, cli, cost_estimate, geomean, load_suite, read_rows, run_key, run_tasks, setup_logging

EXP = "rq_1_scheduler_3_ising"
MIN_QUBITS = 400
TRIALS = (("instant", None), ("cultivation", 42))
COLUMNS = s3.COLUMNS + ["harvest_template"]


def large_ising(names=()):
    return [c for c in load_suite(["core"], circuits=[n for n in names if n])
            if c["family"] == "ising" and c["num_qubits"] >= MIN_QUBITS]


def harvest_rows() -> pd.DataFrame:
    d = read_rows(RESULTS / "rq1" / "rq1.csv")
    d = d[(d.compiler == "Harvest") & (d.template == "auto") & (d.status == "ok")].copy()
    d["seed"] = pd.to_numeric(d.seed, errors="coerce")
    return d


def chosen_template(harvest: pd.DataFrame, circuit: str, model: str, seed):
    sub = harvest[(harvest.circuit == circuit) & (harvest.model == model)]
    sub = sub[sub.seed.isna()] if seed is None else sub[sub.seed == seed]
    return sub.template_chosen.iloc[0] if len(sub) else None


def build_tasks(circuits, timeout):
    harvest, tasks = harvest_rows(), []
    for circ in circuits:
        for model, seed in TRIALS:
            template = chosen_template(harvest, circ["circuit"], model, seed)
            if template is None:
                continue  # no HARVEST reference row
            key = run_key(EXP, circ["circuit"], "harvest3", template, model, seed)
            tasks.append(Task(EXP, key, "rq_common:run_in_tree", dict(
                circ=circ, compiler="harvest3", template=template, model=model, seed=seed,
                timeout=timeout, extra={"variant": "harvest3", "harvest_template": template}),
                cost_estimate(circ) * 5))
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
    s3.rounds_table(raw).to_csv(out / "rounds.csv", index=False)
    new = raw[raw.status == "ok"].copy()
    new["seed"] = pd.to_numeric(new.seed, errors="coerce")
    harvest = harvest_rows()
    rows = []
    for r in new.itertuples():
        h = harvest[(harvest.circuit == r.circuit) & (harvest.model == r.model)]
        h = h[h.seed.isna()] if pd.isna(r.seed) else h[h.seed == r.seed]
        if not len(h):
            continue
        h = h.iloc[0]
        row = {"circuit": r.circuit, "model": r.model, "seed": r.seed, "template": r.template,
               "best_round": int(float(r.reschedule_best_round))}
        for col, name in (("volume_tile_cycles", "volume"), ("T_cycles", "T"), ("area_tiles", "area")):
            row[f"{name}_ratio"] = float(getattr(r, col)) / float(h[col])
        row["volume_round0_ratio"] = float(r.volume_round0) / float(h.volume_tile_cycles)
        row["compile_ratio"] = float(r.compile_time_s) / float(h.compile_time_s)
        rows.append(row)
    per = pd.DataFrame(rows).sort_values(["model", "circuit"])
    per.to_csv(out / "per_circuit.csv", index=False)
    summary = per.groupby("model").agg(
        n=("circuit", "size"), volume=("volume_ratio", geomean), round0=("volume_round0_ratio", geomean),
        T=("T_ratio", geomean), area=("area_ratio", geomean), best=("volume_ratio", "min"),
        improved=("best_round", lambda s: int((s > 0).sum()))).reset_index()
    text = (per.round(3).to_markdown(index=False) + "\n\n" + summary.round(3).to_markdown(index=False) + "\n")
    (out / "table.md").write_text(text)
    print(text)
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
            for t in tasks:
                print(t.key)
            return 0
        run_tasks(tasks, {EXP: COLUMNS}, jobs=args.jobs)
    if not args.no_plot:
        analyze()
    return 0


if __name__ == "__main__":
    sys.exit(main())
