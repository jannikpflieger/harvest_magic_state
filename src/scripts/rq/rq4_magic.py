#!/usr/bin/env python3
"""RQ4: does resource-awareness matter, and is the magic-state interface plug-and-play?

A (resource-awareness).  Ring template, magic ring thinned to
M in {1, 2, 4, 8, 16, 32, 64} terminals (levels that realize the same layout
run once).  Per level:

* ``aware``     - Harvest scheduled with the model: instant, distillation
  (fixed latency mu = 4, 10), cultivation (Exp mean 3, 3 seeds) and the two
  models added for this paper, ``multiplexed`` (2 cultivation units per port,
  mean 6 each, 1-state buffer) and ``lognormal`` (log-normal latency, mean 3,
  std 3), both 3 seeds;
* ``oblivious`` - Harvest scheduled as if magic were instant, then replayed
  under each finite-supply model: (a) the unmodified schedule is checked with
  the validator (rule V4 violations = consumptions from empty terminals), and
  (b) stalls are inserted (a step waits until every terminal it consumes from
  holds a state; order and routes unchanged) and the stretched schedule is
  validated again.

B (plug-ability) is tabulated by ``rq4_plots.py`` from the ``aware`` rows
(validator passes before/after pruning) plus a restriction-stability check
and the interface line counts of every model.

Suite: ``small`` tier.  Output: ``results/rq/rq4/rq4.csv``.
"""

from __future__ import annotations

from rq_common import (MODELS, Task, base_layout, cli, cost_estimate, load_circuit, load_suite,
                       run_in_tree, run_key, run_tasks, setup_logging)

EXP = "rq4"
LEVELS = (1, 2, 4, 8, 16, 32, 64)
SEEDS = list(range(42, 45))
AWARE = {"instant": [None], "dist4": [None], "dist10": [None], "cultivation": SEEDS,
         "multiplexed": SEEDS, "lognormal": SEEDS}
REPLAY = {"dist4": [None], "dist10": [None], "cultivation": SEEDS}
COLUMNS = ["mode", "level", "magic_terminals", "replay_model", "replay_seed", "T_oblivious",
           "replay_T", "replay_stall_steps", "v4_violations_unmodified",
           "steps_needing_delay", "replay_validated", "replay_n_violations", "magic_ops",
           "magic_wait_cycles", "stalled_total", "validated_pre", "validated_post",
           "n_violations", "supply_bound", "t_schedule_s"]


def _latency(model: str) -> float:
    spec = MODELS[model]
    if not spec:
        return 1.0
    return float(spec.get("cultivation_mean_cycles", spec.get("magic_mean_cycles", 1.0)))


def realized_levels(circ):
    from harvest.layout.presets import thin_magic_ring

    seen, out = set(), []
    for level in LEVELS:
        layout = thin_magic_ring(base_layout(circ["num_qubits"], None, "ring"), level)
        m = sum(1 for p in layout.patches.values() if any(x.port_type == "M" for x in p.ports))
        if m not in seen:
            seen.add(m)
            out.append((level, m))
    return out


def replay(dag, processor, results, model: str, seed):
    """Validate the unmodified oblivious schedule under ``model``, then insert stalls."""
    from harvest.baselines.cultivation import make_magic_source
    from harvest.eval.validate import needs_magic, validate

    options = MODELS[model]

    def make_source(terminals):
        return make_magic_source(options, terminals, seed)

    total = processor._scheduling_metadata["total_elapsed_steps"]
    unmodified = validate(processor, dag, results, make_source, total_steps=total,
                          rules=("V4",))
    by_step = {}
    for r in results:
        if r.get("success", True):
            by_step.setdefault(r["time_step"], []).append(r)
    source = make_source(list(processor.magic_terminals))
    new_step, t, stalls = {}, 0, 0
    for s in range(total):
        ops = by_step.get(s, [])
        need = [r["magic_terminal"] for r in ops if needs_magic(r["node"])]
        while True:
            source.tick()
            ready = set(source.get_ready_terminals() or ())
            if all(m in ready for m in need):
                for m in need:
                    source.consume(m)
                for r in ops:
                    new_step[id(r)] = t
                t += 1
                break
            t += 1
            stalls += 1
            if stalls > 10 ** 7:
                raise RuntimeError("replay does not terminate")
    stretched = [{**r, "time_step": new_step[id(r)]} for r in results if r.get("success", True)]
    check = validate(processor, dag, stretched, make_source, total_steps=t)
    return {
        "v4_violations_unmodified": unmodified["violations_by_rule"].get("V4", 0),
        "steps_needing_delay": unmodified["steps_needing_delay"],
        "replay_T": t, "replay_stall_steps": stalls,
        "replay_validated": check["ok"], "replay_n_violations": check["n_violations"],
        "magic_ops": unmodified["magic_ops"],
    }


def run_oblivious(circ, level, magic_terminals, timeout, extra):
    """One instant-magic schedule plus its replays under every finite-supply model."""
    from rq_common import circuit_columns

    sink = {}
    base = {**extra, "level": level, "magic_terminals": magic_terminals}
    row = run_in_tree(circ, "harvest", "ring", "instant", None, magic_ring=level,
                      timeout=timeout, sink=sink, extra={**base, "mode": "oblivious_schedule"})
    rows = [row]
    if row.get("status") != "ok":
        return rows
    _, dag = load_circuit(circ)
    for model, seeds in REPLAY.items():
        for seed in seeds:
            out = {**circuit_columns(circ), **base, "status": "ok", "compiler": "Harvest",
                   "variant": "oblivious", "mode": "oblivious", "template": "ring",
                   "model": model, "replay_model": model, "replay_seed": seed, "seed": seed,
                   "T_oblivious": row["T_cycles"], "area_tiles": row["area_tiles"],
                   "supply_bound": (row.get("magic_ops") or circ["ops"]) * _latency(model)
                   / max(1, magic_terminals)}
            out.update(replay(dag, sink["processor"], sink["results"], model, seed))
            out["T_cycles"] = out["replay_T"]
            out["volume_tile_cycles"] = out["replay_T"] * row["area_tiles"]
            rows.append(out)
    return rows


def run_aware(circ, level, magic_terminals, model, seed, timeout, extra):
    row = run_in_tree(circ, "harvest", "ring", model, seed, magic_ring=level, timeout=timeout,
                      extra={**extra, "level": level, "magic_terminals": magic_terminals,
                             "mode": "aware", "variant": "aware"})
    row["supply_bound"] = (row.get("magic_ops") or circ["ops"]) * _latency(model) / max(1, magic_terminals)
    return row


def build_tasks(circuits, timeout):
    tasks = []
    for circ in circuits:
        cost = cost_estimate(circ)
        for level, m in realized_levels(circ):
            extra = {"template": "ring"}
            for model, seeds in AWARE.items():
                for seed in seeds:
                    key = run_key(EXP, "aware", circ["circuit"], level, model, seed)
                    tasks.append(Task(EXP, key, "rq4_magic:run_aware", dict(
                        circ=circ, level=level, magic_terminals=m, model=model, seed=seed,
                        timeout=timeout, extra={**extra, "model": model}), cost))
            base = run_key(EXP, "oblivious", circ["circuit"], level)
            keys = [base + "|schedule"] + [f"{base}|{model}|{seed}" for model, seeds in REPLAY.items()
                                           for seed in seeds]
            tasks.append(Task(EXP, base, "rq4_magic:run_oblivious", dict(
                circ=circ, level=level, magic_terminals=m, timeout=timeout, extra=extra),
                1.5 * cost, keys=keys))
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
        import rq4_plots
        rq4_plots.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
