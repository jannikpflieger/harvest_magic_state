#!/usr/bin/env python3
"""RQ1 with HARVEST-3: prune the least-used tiles and schedule again.

HARVEST-3 (``harvest.routing.prune_reschedule``) runs HARVEST-2 (reuse cost
cell 1 / magic 2, the setting fixed before ``rq_1_scheduler_2`` ran), then up
to four rounds of: drop the least-used 10 % of the retained routing cells and
magic patches (never a cell whose removal disconnects a used port), schedule
again on the smaller layout, keep the round only if the volume drops (else
retry from the best layout with half the fraction).  Template ``auto`` picks
the best template after the rounds.

Setup and comparison as ``rq_1_scheduler_2``: 47-circuit ``small`` tier,
instant (1 run) and cultivation (mean 3, seeds 42-44), against the existing
RQ1 rows and the ``rq_1_scheduler_2`` rows of the same circuits and seeds.

Output: ``results/rq/rq_1_scheduler_3/``: ``rq_1_scheduler_3.csv`` (one row
per trial with every round in ``reschedule_rounds``), ``per_circuit.csv``,
``table.csv/.md``, ``rounds.csv`` (per round accepted / rejected, T and
volume), ``fig_bars.pdf``.
"""

from __future__ import annotations

import json
import sys

import pandas as pd

import rq_1_scheduler_2 as s2
from rq_common import RESULTS, Task, cli, cost_estimate, load_suite, read_rows, run_key, run_tasks, setup_logging

EXP = "rq_1_scheduler_3"
SEEDS = s2.SEEDS
COLUMNS = ["template_chosen", "auto_volumes", "reschedule_best_round", "volume_round0", "T_round0",
           "area_round0", "reschedule_rounds", "validated_pre", "validated_post", "n_violations",
           "magic_wait_cycles"]


def build_tasks(circuits, timeout):
    tasks = []
    for circ in circuits:
        for model, seeds in (("instant", [None]), ("cultivation", SEEDS)):
            for seed in seeds:
                key = run_key(EXP, circ["circuit"], "harvest3", "auto", model, seed)
                tasks.append(Task(EXP, key, "rq_common:run_in_tree", dict(
                    circ=circ, compiler="harvest3", template="auto", model=model, seed=seed,
                    timeout=timeout, extra={"variant": "harvest3"}), cost_estimate(circ) * 12))
    return tasks


def rounds_table(raw: pd.DataFrame) -> pd.DataFrame:
    """Per trial and round (of the chosen template): dropped tiles, T, volume, accepted."""
    rows = []
    for r in raw[raw.status == "ok"].itertuples():
        for x in json.loads(r.reschedule_rounds):
            rows.append({"circuit": r.circuit, "model": r.model, "seed": r.seed, "round": x["round"],
                         "fraction": x.get("fraction"), "dropped_cells": x.get("dropped_cells"),
                         "dropped_patches": x.get("dropped_patches"), "status": x.get("status"),
                         "T": x.get("T"), "area": x.get("area"), "volume": x.get("volume"),
                         "best": x["round"] == r.reschedule_best_round})
    return pd.DataFrame(rows)


def analyze() -> int:
    out = RESULTS / EXP
    circuits = [c["circuit"] for c in load_suite(["core"], small_only=True)]
    raw = read_rows(out / f"{EXP}.csv")
    if len(raw) == 0:
        print(f"{EXP}: no data yet")
        return 0
    bad = pd.to_numeric(raw.get("n_violations"), errors="coerce").fillna(0)
    print(f"{EXP}: {len(raw)} rows, status {raw.status.value_counts().to_dict()}, "
          f"rows with validator violations: {int((bad > 0).sum())}")
    rounds = rounds_table(raw)
    rounds.to_csv(out / "rounds.csv", index=False)
    if len(rounds):
        later = rounds[rounds["round"] > 0]
        print(f"rounds after the first schedule: {len(later)}, accepted {int(later.best.sum())} "
              f"(trials improved: {int((raw.reschedule_best_round.astype(float) > 0).sum())}/{len(raw)})")
    new = pd.concat([s2.new_rows(), raw.assign(compiler="Harvest-3")], ignore_index=True)
    best = s2.per_circuit(circuits, new)
    best.to_csv(out / "per_circuit.csv", index=False)
    compilers = ["Harvest", "Harvest-2 c1m2", "Harvest-3", *s2.BASELINES]
    compilers = [c for c in compilers if (best.compiler == c).any()]
    tab = s2.table(best, compilers)
    tab.to_csv(out / "table.csv", index=False)
    (out / "table.md").write_text(tab.round(3).to_markdown(index=False) + "\n")
    print(tab.round(3).to_string(index=False))
    s2.fig_bars(best, compilers, out, "HARVEST-3 (HARVEST-2 + prune & reschedule) vs HARVEST and baselines, "
                                      "47-circuit subset")
    return 0


def main(argv=None) -> int:
    parser = cli(__doc__.split("\n\n")[0], default_jobs=3)
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
