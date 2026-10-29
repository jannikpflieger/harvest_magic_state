#!/usr/bin/env python3
"""RQ1: Harvest vs. state-of-the-art lattice-surgery compilers.

Five compilers on the identical Pauli-rotation input (matched IR): Harvest,
Silva EAF (on Harvest's layout, priced on the same pruned footprint),
PureMagic (bus and magic-routing variants), Litinski (compact / intermediate /
fast blocks) and O3LS (min / medium / large).  Two magic-state models:
``instant`` (one deterministic run; PureMagic: lambda = 1e6 and no T failures)
and ``cultivation`` (Exp(mean 3) cycles per state per port, 10 seeds; PureMagic
gets the same distribution natively; for n >= 400 EAF and O3LS run 3 seeds
because of their per-run cost).  Every in-tree schedule (Harvest, EAF)
is replayed through Harvest's validator before and after pruning.

Output: ``results/rq/rq1/rq1.csv`` (one row per trial); figures and tables
by ``rq1_plots.py`` (called at the end unless ``--no-plot``).
"""

from __future__ import annotations

import logging

from rq_common import (Task, cli, cost_estimate, load_suite, run_key, run_tasks,
                       setup_logging)

EXP = "rq1"
SEEDS = list(range(42, 52))
EXTERNAL = {
    "puremagic": ("puremagic_bus", "puremagic"),
    "litinski": ("litinski_compact", "litinski_intermediate", "litinski_fast"),
    "o3ls": ("o3ls_min", "o3ls_medium", "o3ls_large"),
}
COLUMNS = ["T_over_lb", "magic_terminals", "validated_pre", "validated_post", "n_violations",
           "violations_V1", "violations_V2", "violations_V3", "violations_V4",
           "t_place_s", "t_schedule_s", "t_prune_s", "t_validate_s", "validate_time_s",
           "magic_wait_cycles", "mean_ops_per_step", "steps_idle", "routing_tiles_initial",
           "routing_tiles_pruned", "magic_tiles_initial", "magic_tiles_pruned",
           "data_tiles_initial", "data_tiles_pruned", "volume_removed_frac", "timed_out",
           "wall_time_s", "process_peak_rss_mb"]


def build_tasks(circuits, templates, models, timeout):
    tasks = []
    for circ in circuits:
        cost = cost_estimate(circ)
        for model in models:
            seeds = SEEDS if model == "cultivation" else [None]
            for seed in seeds:
                # n >= 400 (large Ising): EAF (~5000 s/run) and O3LS (~300 s/run) get
                # 3 cultivation seeds; every other compiler keeps all 10.
                reduced = circ["num_qubits"] >= 400 and seed is not None and seed >= SEEDS[3]
                for template in templates:
                    for compiler, factor in (("harvest", 1.0), ("eaf", 2.5)):
                        if compiler == "eaf" and reduced:
                            continue
                        key = run_key(EXP, circ["circuit"], compiler, template, model, seed)
                        extra = {"variant": compiler, "template": template, "model": model,
                                 "compiler": "Harvest" if compiler == "harvest" else "EAF"}
                        tasks.append(Task(EXP, key, "rq_common:run_in_tree", dict(
                            circ=circ, compiler=compiler, template=template, model=model,
                            seed=seed, timeout=timeout, extra=extra), cost * factor))
                for compiler, variants in EXTERNAL.items():
                    if compiler == "o3ls" and reduced:
                        continue
                    for variant in variants:
                        ext_seed = seed if seed is not None else 42
                        key = run_key(EXP, circ["circuit"], compiler, variant, model, seed)
                        extra = {"variant": variant, "template": variant, "model": model,
                                 "compiler": {"puremagic": "PureMagic", "litinski": "Litinski",
                                              "o3ls": "O3LS"}[compiler]}
                        tasks.append(Task(EXP, key, "rq_common:run_external", dict(
                            circ=circ, compiler=compiler, variant=variant, model=model,
                            seed=ext_seed, timeout=timeout, extra=extra),
                            cost * (0.3 if compiler == "o3ls" else 0.01)))
    return tasks


# Large Ising and every wstate size under PureMagic's published cultivation
# rate (mean ~25.8 cycles per state): Harvest and EAF on blocks4 with one seed
# (a large run takes up to hours), PureMagic with all seeds.
PM_MODEL = "cultivation_pm"
PM_TIMEOUT = 4 * 3600


def build_pm_cultivation_tasks(circuits, timeout):
    tasks = []
    for circ in circuits:
        if not ((circ["family"] == "ising" and circ["num_qubits"] >= 400)
                or circ["family"] == "wstate"):
            continue
        cost = cost_estimate(circ) * 3
        for compiler, factor in (("harvest", 1.0), ("eaf", 1.5)):
            key = run_key(EXP, circ["circuit"], compiler, "blocks4", PM_MODEL, SEEDS[0])
            extra = {"variant": compiler, "template": "blocks4", "model": PM_MODEL,
                     "compiler": "Harvest" if compiler == "harvest" else "EAF"}
            tasks.append(Task(EXP, key, "rq_common:run_in_tree", dict(
                circ=circ, compiler=compiler, template="blocks4", model=PM_MODEL,
                seed=SEEDS[0], timeout=max(timeout, PM_TIMEOUT), extra=extra), cost * factor))
        for seed in SEEDS:
            for variant in EXTERNAL["puremagic"]:
                key = run_key(EXP, circ["circuit"], "puremagic", variant, PM_MODEL, seed)
                extra = {"variant": variant, "template": variant, "model": PM_MODEL,
                         "compiler": "PureMagic"}
                tasks.append(Task(EXP, key, "rq_common:run_external", dict(
                    circ=circ, compiler="puremagic", variant=variant, model=PM_MODEL,
                    seed=seed, timeout=timeout, extra=extra), cost * 0.01))
    return tasks


def main(argv=None) -> int:
    parser = cli(__doc__.split("\n\n")[0], default_jobs=4)
    parser.add_argument("--templates", default="blocks4",
                        help="Harvest/EAF layout templates (comma list of ring,blocks4,dense4)")
    parser.add_argument("--models", default="instant,cultivation")
    args = parser.parse_args(argv)
    out = setup_logging(EXP)
    if not args.plot_only:
        circuits = load_suite(args.tiers.split(","), circuits=[c for c in args.circuits.split(",") if c],
                              max_ops=args.max_ops)
        tasks = build_tasks(circuits, args.templates.split(","), args.models.split(","), args.timeout)
        if args.dry_run:
            print(f"{len(circuits)} circuits, {len(tasks)} runs")
            return 0
        run_tasks(tasks, {EXP: COLUMNS}, jobs=args.jobs)
    if not args.no_plot:
        import rq1_plots
        rq1_plots.main()
    return 0


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    raise SystemExit(main())
