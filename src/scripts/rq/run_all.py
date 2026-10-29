#!/usr/bin/env python3
"""Run RQ1-RQ5 through one shared worker pool, cheapest trials first.

Every trial appends its row to its experiment's CSV the moment it finishes
(``results/rq/<exp>/<exp>.csv``), so the run can be interrupted at any time;
rerunning this command continues with the missing trials.  Each experiment
can also be run on its own (``rq1_sota.py`` ... ``rq5_validity.py``).

    python src/scripts/rq/run_all.py --jobs 5            # core + small tiers
    python src/scripts/rq/run_all.py --rq1-tiers core,extra   # also every qaoa/qft size
    python src/scripts/rq/run_all.py --priority 'ising_n4'   # these circuits first
    python src/scripts/rq/run_all.py --models instant    # only instant-magic trials
    python src/scripts/rq/run_all.py --skip-compilers litinski,o3ls   # without these baselines
    python src/scripts/rq/run_all.py --plot-only         # redraw every figure
"""

from __future__ import annotations

import argparse
import importlib

from rq_common import load_suite, run_tasks, setup_logging

MODULES = {"rq1": "rq1_sota", "rq2": "rq2_ablation", "rq3": "rq3_provisioning",
           "rq4": "rq4_magic", "rq5": "rq5_validity"}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--jobs", type=int, default=5)
    parser.add_argument("--experiments", default="rq1,rq2,rq3,rq4,rq5")
    parser.add_argument("--rq1-tiers", default="core")
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--priority", default="",
                        help="regex on circuit names to run first, e.g. 'ising_n4'")
    parser.add_argument("--priority-compilers", default="",
                        help="with --priority: these compilers first, in order, e.g. 'harvest,eaf'")
    parser.add_argument("--models", default="",
                        help="only run trials of these magic models (comma list, e.g. 'instant'); "
                             "the others are skipped for now and picked up by a later run without it")
    parser.add_argument("--skip-compilers", default="",
                        help="leave out the trials of these compilers for now (comma list, e.g. 'litinski,o3ls')")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    experiments = args.experiments.split(",")
    setup_logging("run_all")
    if not args.plot_only:
        tasks, columns = [], {}
        small = load_suite(["core"], small_only=True)
        for exp in experiments:
            module = importlib.import_module(MODULES[exp])
            columns[exp] = module.COLUMNS
            if exp == "rq1":
                circuits = load_suite(args.rq1_tiers.split(","))
                exp_tasks = module.build_tasks(circuits, ["blocks4"], ["instant", "cultivation"],
                                               args.timeout)
                exp_tasks += module.build_pm_cultivation_tasks(circuits, args.timeout)
            elif exp in ("rq2", "rq3"):
                exp_tasks = module.build_tasks(small, ["ring", "blocks4", "dense4"], args.timeout)
            else:
                exp_tasks = module.build_tasks(small, args.timeout)
            print(f"{exp}: {len(exp_tasks)} task(s), estimated {sum(t.cost for t in exp_tasks) / 3600:.1f} CPU-h")
            tasks += exp_tasks
        if args.models:
            keep = set(args.models.split(","))
            skipped = [t for t in tasks if t.kwargs.get("model") not in keep]
            tasks = [t for t in tasks if t.kwargs.get("model") in keep]
            print(f"--models {args.models}: {len(tasks)} task(s) kept, {len(skipped)} paused")
        if args.skip_compilers:
            drop = set(args.skip_compilers.split(","))
            n = len(tasks)
            tasks = [t for t in tasks if t.kwargs.get("compiler") not in drop]
            print(f"--skip-compilers {args.skip_compilers}: {n - len(tasks)} task(s) paused")
        if args.dry_run:
            return 0
        priority = None
        if args.priority:
            import re

            pattern = re.compile(args.priority)
            first = [c for c in args.priority_compilers.split(",") if c]

            def priority(t):
                # matching circuits first; among them the listed compilers, in order
                if not pattern.search(t.kwargs["circ"]["circuit"]):
                    return len(first) + 1
                compiler = t.kwargs.get("compiler")
                return first.index(compiler) if compiler in first else len(first)
        run_tasks(tasks, columns, jobs=args.jobs, priority=priority)
    for exp in experiments:
        try:
            importlib.import_module(MODULES[exp].split("_")[0] + "_plots").main()
        except Exception as exc:  # one broken figure must not hide the others
            print(f"{exp} plots failed: {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
