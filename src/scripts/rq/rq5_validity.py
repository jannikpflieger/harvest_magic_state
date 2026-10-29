#!/usr/bin/env python3
"""RQ5: are all schedules valid, does the validator catch bugs, does Harvest scale?

This script runs the fault-injection study; the validity census (every
schedule of rq1-rq4, before and after pruning), validator time share,
compile-time scaling and the progress bound are aggregated from the other
CSVs by ``rq5_plots.py``.

Fault injection on the ``small`` suite, blocks4 template, instant and
cultivation (seed 42):

* ``scheduler`` faults - the bugs of ``harvest.faults`` injected into the
  scheduler/pruner (``no_defer``, ``reuse_terminal``, ``skip_dep``,
  ``early_consume``, ``prune_used``, ``drop_y_port``); a fault counts as
  caught when it activated and the validator reported a violation;
* ``corrupt`` mutations - a valid schedule is corrupted after the fact,
  ``TRIALS`` random mutations per kind: ``drop_dependency`` (an operation
  moved into its predecessor's step), ``overlap_route`` (a routing tile of
  one route added to another route of the same step), ``disconnect_route``
  (an interior tile removed from a route), ``unavailable_state``
  (cultivation only: an operation re-pointed to a terminal holding no ready
  state).  Every mutation is invalid by construction.

Output: ``results/rq/rq5/rq5.csv``.
"""

from __future__ import annotations

import random

from rq_common import (Task, circuit_columns, cli, cost_estimate, load_circuit, load_suite,
                       run_in_tree, run_key, run_tasks, setup_logging)

EXP = "rq5"
SCHEDULER_FAULTS = ("no_defer", "reuse_terminal", "skip_dep", "early_consume", "prune_used",
                    "drop_y_port")
MUTATIONS = ("drop_dependency", "overlap_route", "disconnect_route", "unavailable_state")
TRIALS = 20
COLUMNS = ["study", "fault", "fault_activations", "caught", "mutation", "trial_index",
           "n_violations", "violations_V1", "violations_V2", "violations_V3", "violations_V4",
           "first_violation_rule", "validated_pre", "validated_post"]


def run_fault(circ, model, seed, fault, timeout, extra):
    row = run_in_tree(circ, "harvest", "blocks4", model, seed, fault=fault, timeout=timeout,
                      extra={**extra, "study": "scheduler", "fault": fault, "variant": fault})
    activated = (row.get("fault_activations") or 0) > 0
    row["caught"] = bool(activated and (row.get("n_violations") or 0) > 0) if activated else None
    return row


def _mutate(kind, rng, dag, processor, routed, source_factory):
    """A corrupted copy of ``routed`` (list of result dicts), or None if inapplicable."""
    from harvest.eval.validate import needs_magic

    rs = [dict(r, steiner_nodes=set(r.get("steiner_nodes", ())),
               steiner_edges=set(map(tuple, r.get("steiner_edges", ())))) for r in routed]
    by_id = {r["node"]._node_id: r for r in rs}
    if kind == "drop_dependency":
        cand = [r for r in rs if any(hasattr(p, "op") for p in dag.predecessors(r["node"]))]
        if not cand:
            return None
        r = rng.choice(cand)
        pred = rng.choice([p for p in dag.predecessors(r["node"]) if hasattr(p, "op")])
        r["time_step"] = by_id[pred._node_id]["time_step"]
        return rs
    if kind == "overlap_route":
        steps = {}
        for r in rs:
            steps.setdefault(r["time_step"], []).append(r)
        multi = [ops for ops in steps.values() if len(ops) >= 2
                 and any(isinstance(n, tuple) for o in ops for n in o["steiner_nodes"])]
        if not multi:
            return None
        ops = rng.choice(multi)
        a = rng.choice([o for o in ops if any(isinstance(n, tuple) for n in o["steiner_nodes"])])
        b = rng.choice([o for o in ops if o is not a])
        tile = rng.choice(sorted(n for n in a["steiner_nodes"] if isinstance(n, tuple)))
        b["steiner_nodes"].add(tile)
        return rs
    if kind == "disconnect_route":
        cand = [r for r in rs if sum(isinstance(n, tuple) for n in r["steiner_nodes"]) >= 1]
        if not cand:
            return None
        r = rng.choice(cand)
        tile = rng.choice(sorted(n for n in r["steiner_nodes"] if isinstance(n, tuple)))
        r["steiner_nodes"].discard(tile)
        r["steiner_edges"] = {e for e in r["steiner_edges"] if tile not in e}
        return rs
    if kind == "unavailable_state":
        if source_factory is None:
            return None
        # replay the supply; find a (step, terminal) pair holding no ready state
        source = source_factory(list(processor.magic_terminals))
        steps = {}
        for r in rs:
            steps.setdefault(r["time_step"], []).append(r)
        options = []
        for t in range(1 + max(steps, default=-1)):
            source.tick()
            ready = set(source.get_ready_terminals() or ())
            ops = steps.get(t, [])
            used = {o.get("magic_terminal") for o in ops}
            empty = [m for m in processor.magic_terminals if m not in ready and m not in used]
            for o in ops:
                if needs_magic(o["node"]) and empty:
                    options.append((o, empty))
                if needs_magic(o["node"]):
                    source.consume(o["magic_terminal"])
        if not options:
            return None
        o, empty = rng.choice(options)
        new = rng.choice(sorted(empty))
        o["steiner_nodes"].discard(o["magic_terminal"])
        o["steiner_nodes"].add(new)
        o["magic_terminal"] = new
        return rs
    raise ValueError(kind)


def run_corrupt(circ, model, seed, timeout, extra):
    from harvest.baselines.cultivation import make_magic_source
    from harvest.eval.validate import validate
    from rq_common import MODELS

    sink = {}
    base_row = run_in_tree(circ, "harvest", "blocks4", model, seed, timeout=timeout, sink=sink,
                           extra={**extra, "study": "corrupt_base", "variant": "valid"})
    rows = [base_row]
    if base_row.get("status") != "ok":
        return rows
    _, dag = load_circuit(circ)
    processor, results = sink["processor"], sink["results"]
    routed = [r for r in results if r.get("success", True)]
    total = processor._scheduling_metadata["total_elapsed_steps"]
    options = MODELS[model]
    factory = (lambda terms: make_magic_source(options, terms, seed)) if options else None
    for kind in MUTATIONS:
        rng = random.Random(f"{circ['circuit']}:{model}:{kind}")
        for i in range(TRIALS):
            mutated = _mutate(kind, rng, dag, processor, routed, factory)
            out = {**circuit_columns(circ), **extra, "status": "ok" if mutated else "n/a",
                   "study": "corrupt", "mutation": kind, "trial_index": i, "compiler": "Harvest",
                   "variant": kind, "template": "blocks4"}
            if mutated is not None:
                report = validate(processor, dag, mutated, factory, total_steps=total)
                by_rule = report["violations_by_rule"]
                out.update(n_violations=report["n_violations"], caught=report["n_violations"] > 0,
                           first_violation_rule=report["first_violation_rule"],
                           validate_time_s=report["validate_time_s"],
                           **{f"violations_{k}": by_rule.get(k, 0) for k in ("V1", "V2", "V3", "V4")})
            rows.append(out)
    return rows


def build_tasks(circuits, timeout):
    tasks = []
    for circ in circuits:
        cost = cost_estimate(circ)
        for model, seed in (("instant", None), ("cultivation", 42)):
            extra = {"model": model}
            for fault in SCHEDULER_FAULTS:
                key = run_key(EXP, "scheduler", circ["circuit"], fault, model)
                tasks.append(Task(EXP, key, "rq5_validity:run_fault", dict(
                    circ=circ, model=model, seed=seed, fault=fault, timeout=timeout,
                    extra=extra), cost))
            base = run_key(EXP, "corrupt", circ["circuit"], model)
            keys = [base + "|base"] + [f"{base}|{k}|{i}" for k in MUTATIONS for i in range(TRIALS)]
            tasks.append(Task(EXP, base, "rq5_validity:run_corrupt", dict(
                circ=circ, model=model, seed=seed, timeout=timeout, extra=extra), cost, keys=keys))
    return tasks


def main(argv=None) -> int:
    parser = cli(__doc__.split("\n\n")[0])
    args = parser.parse_args(argv)
    out = setup_logging(EXP)
    if not args.plot_only:
        circuits = load_suite(args.tiers.split(","), small_only=True,
                              circuits=[c for c in args.circuits.split(",") if c],
                              max_ops=args.max_ops)
        tasks = build_tasks(circuits, args.timeout)
        if args.dry_run:
            print(f"{len(circuits)} circuits, {len(tasks)} tasks")
            return 0
        run_tasks(tasks, {EXP: COLUMNS}, jobs=args.jobs)
    if not args.no_plot:
        import rq5_plots
        rq5_plots.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
