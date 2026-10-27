#!/usr/bin/env python3
"""E7d: robustness of a cultivation schedule to the supply trace (EVAL_PLAN B.8).

HARVEST schedules against one sampled cultivation trace.  This script
compiles each ``sweep`` circuit with seed ``s`` and replays fresh cultivators
with seeds ``s+1 .. s+R`` against the fixed schedule (validator rule V4
only), recording the fraction of magic operations whose terminal holds no
ready state and the fraction of steps that would have to be delayed.
Output: ``<exp-dir>/e7_trace_robustness.csv``.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
for path in (PROJECT_ROOT / "src", PROJECT_ROOT / "src" / "scripts"):
    sys.path.insert(0, str(path))

FIELDS = ("circuit", "circuit_class", "layout", "mean_cycles", "compile_seed", "replay_seed",
          "T_cycles", "magic_ops", "magic_not_ready", "frac_magic_not_ready",
          "steps_needing_delay", "frac_steps_delayed")
LAYOUTS = (("ring", 2), ("blocks4", 2), ("dense", 2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--exp-dir", type=Path, required=True)
    parser.add_argument("--suite", default="sweep")
    parser.add_argument("--mean-cycles", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument("--quick", type=int)
    args = parser.parse_args()
    logging.disable(logging.WARNING)

    from compare_sota_baselines import build_layout, load_benchmark

    from harvest.baselines.cultivation import make_magic_source
    from harvest.eval.validate import validate
    from harvest.routing.processor import DAGProcessor

    suites = json.loads((PROJECT_ROOT / "experiments" / "eval" / "suites.json").read_text())
    files = suites[args.suite]["files"]
    if args.quick:
        files = sorted(files, key=lambda f: f["ops"])[: args.quick]
    options = {"magic_model": "cultivation", "cultivation_mean_cycles": args.mean_cycles}
    out = args.exp_dir / "e7_trace_robustness.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for f in files:
            circuit, dag = load_benchmark(PROJECT_ROOT / f["path"])
            par = f["parallelism"]
            cls = "low" if par < 2 else ("medium" if par < 4 else "high")
            for preset, magic in LAYOUTS:
                layout = build_layout(circuit.num_qubits, None, None, dag, "row_major", args.seed,
                                      preset, magic)
                processor = DAGProcessor(layout_engine=layout)
                processor.magic_source = make_magic_source(options, processor.magic_terminals,
                                                           args.seed)
                results = processor.process_entire_dag(dag, mode="harvest")
                meta = processor._scheduling_metadata
                if not meta.get("completed"):
                    print(f"{f['circuit']}/{preset}: schedule incomplete, skipped")
                    continue
                steps = meta["total_elapsed_steps"]
                for replay in range(args.seed + 1, args.seed + 1 + args.replays):
                    report = validate(
                        processor, dag, results,
                        make_source=lambda t, s=replay: make_magic_source(options, t, s),
                        total_steps=steps, rules=("V4",),
                    )
                    writer.writerow({
                        "circuit": f["circuit"], "circuit_class": cls,
                        "layout": f"{preset}{magic}" if preset == "dense" else preset,
                        "mean_cycles": args.mean_cycles, "compile_seed": args.seed,
                        "replay_seed": replay, "T_cycles": steps,
                        "magic_ops": report["magic_ops"],
                        "magic_not_ready": report["magic_not_ready"],
                        "frac_magic_not_ready": report["magic_not_ready"] / max(1, report["magic_ops"]),
                        "steps_needing_delay": report["steps_needing_delay"],
                        "frac_steps_delayed": report["steps_needing_delay"] / max(1, steps),
                    })
                handle.flush()
                print(f"{f['circuit']}/{preset}: T={steps}, {args.replays} replays", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
