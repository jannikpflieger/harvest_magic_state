#!/usr/bin/env python3
"""RQ2: what does each Harvest component contribute? (leave-one-out ablation)

Variants, each on the identical layout (same template, same qubit placement
unless placement is the ablated component):

* ``full``         the paper configuration;
* ``-negotiation`` greedy first-fit Steiner packing (EAF-style commit),
                   priorities unchanged;
* ``-retry``       negotiate, then defer: no greedy fill and no retry through
                   another terminal in the space left free;
* ``-admission``   magic products admitted regardless of ready terminals (a
                   route to an empty terminal stalls);
* ``-matching``    nearest-terminal assignment instead of min-cost matching;
* ``-priority``    DAG order instead of critical-path priority;
* ``-placement``   index / row-major placement instead of circuit-aware;
* ``-yfactor``     circuit-aware placement without the Y centrality term;
* ``-pruning``     derived from ``full``: the unpruned footprint (no rerun).

Sensitivity (blocks4 only): I_max (``max_iters``) in {0,1,2,4,8} and the
congestion weights alpha in {1,2,4}, beta in {1,1.5,3}.

Models: instant (1 run) and cultivation (3 seeds).  Suite: ``small`` tier.
Output: ``results/rq/rq2/rq2.csv``.
"""

from __future__ import annotations

from rq_common import Task, cli, cost_estimate, load_suite, run_key, run_tasks, setup_logging

EXP = "rq2"
SEEDS = list(range(42, 45))
SENS_SEEDS = list(range(42, 45))
VARIANTS = {
    "full": ({}, {}),
    "-negotiation": ({"negotiate": False}, {}),
    "-retry": ({"greedy_fill": False, "magic_retry": False}, {}),
    "-admission": ({"admission": False}, {}),
    "-matching": ({"terminals": "nearest"}, {}),
    "-priority": ({"priority": "dag_order", "router_priority": False}, {}),
    "-placement": ({}, {"placement": "index"}),
    "-yfactor": ({}, {"y_factor": False}),
}
SENSITIVITY = {
    **{f"imax{i}": {"max_iters": i} for i in (0, 1, 2, 8)},
    "alpha1": {"alpha": 1.0}, "alpha4": {"alpha": 4.0},
    "beta1": {"beta": 1.0}, "beta3": {"beta": 3.0},
}
COLUMNS = ["study", "ablation", "max_iters", "alpha", "beta", "stalled_total", "magic_wait_cycles",
           "nego_iterations_total", "reroutes_total", "dropped_total", "filled_total",
           "magic_retries_total", "deferrals_total", "mean_ops_per_step", "t_schedule_s",
           "t_place_s", "placement_cost", "validated_pre", "validated_post", "n_violations"]


def build_tasks(circuits, templates, timeout):
    tasks = []
    for circ in circuits:
        cost = cost_estimate(circ)
        for model, seeds in (("instant", [None]), ("cultivation", SEEDS)):
            for seed in seeds:
                for template in templates:
                    for name, (hconfig, layout_kw) in VARIANTS.items():
                        key = run_key(EXP, "ablation", circ["circuit"], name, template, model, seed)
                        extra = {"study": "ablation", "ablation": name, "variant": name}
                        tasks.append(Task(EXP, key, "rq_common:run_in_tree", dict(
                            circ=circ, compiler="harvest", template=template, model=model,
                            seed=seed, hconfig=hconfig, timeout=timeout, extra=extra,
                            **layout_kw), cost))
        for model, seeds in (("instant", [None]), ("cultivation", SENS_SEEDS)):
            for seed in seeds:
                for name, hconfig in SENSITIVITY.items():
                    key = run_key(EXP, "sensitivity", circ["circuit"], name, "blocks4", model, seed)
                    extra = {"study": "sensitivity", "ablation": name, "variant": name,
                             "max_iters": hconfig.get("max_iters", 4),
                             "alpha": hconfig.get("alpha", 2.0), "beta": hconfig.get("beta", 1.5)}
                    tasks.append(Task(EXP, key, "rq_common:run_in_tree", dict(
                        circ=circ, compiler="harvest", template="blocks4", model=model,
                        seed=seed, hconfig=hconfig, timeout=timeout, extra=extra), cost))
    return tasks


def main(argv=None) -> int:
    parser = cli(__doc__.split("\n\n")[0])
    parser.add_argument("--templates", default="ring,blocks4,dense4")
    args = parser.parse_args(argv)
    out = setup_logging(EXP)
    if not args.plot_only:
        circuits = load_suite(args.tiers.split(","), small_only=True,
                              circuits=[c for c in args.circuits.split(",") if c],
                              max_ops=args.max_ops)
        tasks = build_tasks(circuits, args.templates.split(","), args.timeout)
        if args.dry_run:
            print(f"{len(circuits)} circuits, {len(tasks)} runs")
            return 0
        run_tasks(tasks, {EXP: COLUMNS}, jobs=args.jobs)
    if not args.no_plot:
        import rq2_plots
        rq2_plots.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
