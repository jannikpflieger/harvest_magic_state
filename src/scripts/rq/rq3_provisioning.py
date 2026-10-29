#!/usr/bin/env python3
"""RQ3: does "schedule large, report small" beat right-sizing up front?

For every circuit, template (ring, blocks4, dense4) and magic model, the
initial layout size k is swept through the number of magic terminals
M in {1, 2, 4, 8, 16, 32, all} (ring/blocks4: the magic ring thinned evenly;
dense: M ports in the magic row(s); levels that realize the same layout are
run once).  For each k one task records two rows:

* ``large``  - Harvest on the k-layout: ``volume_initial`` (schedule large,
  no prune) and ``volume_pruned`` (schedule large + prune);
* ``direct`` - Harvest scheduled from scratch on the right-sized layout whose
  footprint is exactly the pruned footprint of ``large`` (every routing tile
  and magic patch the large schedule never touched is BLOCKED; data patches
  stay), i.e. scheduling directly at the pruned size.

Models: instant (1 run) and cultivation (3 seeds, same seed for both rows).
Suite: ``small`` tier.  Output: ``results/rq/rq3/rq3.csv``.
"""

from __future__ import annotations

from rq_common import (Task, base_layout, cli, cost_estimate, load_suite, run_in_tree,
                       run_key, run_tasks, setup_logging)

EXP = "rq3"
LEVELS = (1, 2, 4, 8, 16, 32, 0)  # 0 = every magic patch of the template
SEEDS = list(range(42, 45))
COLUMNS = ["role", "level", "magic_terminals", "magic_tiles_pruned", "routing_tiles_initial",
           "routing_tiles_pruned", "data_tiles_initial", "volume_removed_frac",
           "large_area_initial", "large_area_pruned", "large_T", "validated_pre",
           "validated_post", "n_violations", "magic_wait_cycles", "t_schedule_s"]


def magic_count_of(layout) -> int:
    return sum(1 for p in layout.patches.values() if any(port.port_type == "M" for port in p.ports))


def level_layout_args(template: str, level: int):
    """(magic_ring, magic_count) arguments of ``placed_layout`` for a level."""
    if template == "dense4":
        return None, (level or 64)
    return (level or None), None


def realized_levels(circ, template: str):
    """Levels with distinct realized magic counts (cheapest layout first)."""
    from harvest.layout.presets import thin_magic_ring

    seen, out = set(), []
    for level in LEVELS:
        ring, count = level_layout_args(template, level)
        layout = base_layout(circ["num_qubits"], None, template, count)
        if ring:
            layout = thin_magic_ring(layout, ring)
        m = magic_count_of(layout)
        if m not in seen:
            seen.add(m)
            out.append((level, m))
    return out


def right_sized_layout(processor, num_qubits: int):
    """The k-layout restricted to the pruned footprint of its schedule.

    Routing cells absent from the pruned routing graph and magic patches
    without a retained port become BLOCKED (they count neither as routing nor
    as area); every register qubit's data patch is kept, as in the pruned area
    accounting.
    """
    from harvest.layout.engine import LayoutEngine

    eng = processor.eng
    kept_cells = {n for n in processor.graph if isinstance(n, tuple)}
    kept_patches = set(processor.ports_by_patch)
    out = LayoutEngine(eng.W, eng.H)
    for cell, value in eng.occ.items():
        if value == "BLOCKED":
            out.occ[cell] = "BLOCKED"
    for x in range(eng.W):
        for y in range(eng.H):
            if eng.is_free((x, y)) and (x, y) not in kept_cells:
                out.occ[(x, y)] = "BLOCKED"
    register = {f"q_{i}" for i in range(num_qubits)}
    for patch in eng.patches.values():
        magic = any(port.port_type == "M" for port in patch.ports)
        # unbound data slots (no logical qubit) are not part of the pruned footprint either
        if (magic and patch.name not in kept_patches) or (not magic and patch.name not in register):
            for cell in patch.cells:
                out.occ[cell] = "BLOCKED"
            continue
        out.add_patch(patch)
    return out


def run_pair(circ, template, model, seed, level, magic_terminals, timeout, extra):
    ring, count = level_layout_args(template, level)
    base = {**extra, "level": level, "magic_terminals": magic_terminals}
    sink = {}
    large = run_in_tree(circ, "harvest", template, model, seed, magic_ring=ring,
                        magic_count=count, timeout=timeout, sink=sink,
                        extra={**base, "role": "large", "variant": "large"})
    rows = [large]
    if large.get("status") != "ok" or "processor" not in sink:
        return rows
    layout = right_sized_layout(sink["processor"], circ["num_qubits"])
    direct = run_in_tree(circ, "harvest", template, model, seed, magic_ring=None,
                         magic_count=count, layout=layout, timeout=timeout,
                         extra={**base, "role": "direct", "variant": "direct",
                                "large_area_initial": large.get("area_initial_tiles"),
                                "large_area_pruned": large.get("area_pruned_tiles"),
                                "large_T": large.get("T_cycles")})
    direct["magic_terminals"] = direct.get("magic_tiles_initial")
    rows.append(direct)
    return rows


def build_tasks(circuits, templates, timeout):
    tasks = []
    for circ in circuits:
        cost = cost_estimate(circ)
        for template in templates:
            for level, m in realized_levels(circ, template):
                for model, seeds in (("instant", [None]), ("cultivation", SEEDS)):
                    for seed in seeds:
                        base = run_key(EXP, circ["circuit"], template, level, model, seed)
                        tasks.append(Task(EXP, base, "rq3_provisioning:run_pair", dict(
                            circ=circ, template=template, model=model, seed=seed, level=level,
                            magic_terminals=m, timeout=timeout,
                            extra={"template": template, "model": model}),
                            2 * cost, keys=[base + "|large", base + "|direct"]))
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
            print(f"{len(circuits)} circuits, {len(tasks)} tasks")
            return 0
        run_tasks(tasks, {EXP: COLUMNS}, jobs=args.jobs)
    if not args.no_plot:
        import rq3_plots
        rq3_plots.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
