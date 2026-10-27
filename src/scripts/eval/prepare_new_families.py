#!/usr/bin/env python3
"""Lower the not-yet-used benchmark classes to Clifford+T, grouped by circuit family.

Sources: QASMBench (small/medium/large), chemistry Trotter circuits, DTC, QV,
random circuits and the Feynman circuits missing from the core suite.  Each
family goes through ``prepare_family_benchmarks.py`` (exact lowering when
possible, otherwise PureMagic ``compile_cliffordt -e EPS``) into
``results/eval_inputs/families/<family>/``.

A cheap text pre-filter runs first so hopeless inputs are never synthesized:
Clifford-only circuits (no magic state to schedule) and circuits whose
estimated T count exceeds the budget are recorded in ``prefilter_skipped.json``
with the reason.  Already-prepared circuits (same stem in an existing input
directory) are skipped; reruns only process what is new.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from harvest.eval.families import circuit_family  # noqa: E402

QASM = PROJECT_ROOT / "benchmark_circuits" / "qasm"
OUT = PROJECT_ROOT / "results" / "eval_inputs" / "families"
EXISTING = ("results/sota_feynman_cliffordt_inputs", "results/sota_extended_cliffordt_inputs",
            "results/sota_parallel_cliffordt_inputs", "results/sota_family_inputs",
            "results/eval_inputs/scale_cliffordt")
ROTATIONS = re.compile(r"^\s*(rz|rx|ry|u|u1|u2|u3|p|cp|crz|crx|cry|cu1|cu3|cu|rzz|rxx|ryy|cphase)\s*\(",
                       re.M)
T_GATES = re.compile(r"^\s*(t|tdg)\s", re.M)
TOFFOLI = re.compile(r"^\s*(ccx|cswap|ccz)\s", re.M)
T_PER_ROTATION = 13  # observed for compile_cliffordt -e 1e-3 on the Ising family


def sources():
    out = []
    for size in ("qasmbench-small", "qasmbench-medium", "qasmbench-large"):
        for d in sorted((QASM / size).iterdir()):
            f = d / f"{d.name}.qasm"
            if d.is_dir() and f.exists():
                out.append(f)
    out += sorted((QASM / "chemical").glob("Transpiled_*.qasm"))
    for name in ("dtc", "qv", "random_circuits", "feynman"):
        out += sorted((QASM / name).glob("*.qasm"))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--synth-bin", type=Path,
                        default=Path.home() / ".cache/harvest-tools/PureMagic/target/release/compile_cliffordt")
    parser.add_argument("--epsilon", type=float, default=1e-3)
    parser.add_argument("--max-t", type=int, default=40000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    have = {p.stem for d in EXISTING for p in (PROJECT_ROOT / d).rglob("*.qasm")}
    have |= {p.stem for p in OUT.rglob("*.qasm")} if OUT.exists() else set()
    skipped, todo = {}, defaultdict(list)
    for src in sources():
        if src.stem in have:
            continue
        text = src.read_text(errors="ignore")
        rotations = len(ROTATIONS.findall(text))
        t_est = rotations * T_PER_ROTATION + len(T_GATES.findall(text)) + 7 * len(TOFFOLI.findall(text))
        if t_est == 0:
            skipped[src.stem] = {"source": str(src.relative_to(PROJECT_ROOT)),
                                 "reason": "Clifford-only: no magic-state operation to schedule"}
        elif t_est > args.max_t:
            skipped[src.stem] = {"source": str(src.relative_to(PROJECT_ROOT)),
                                 "reason": f"estimated T count {t_est} > --max-t {args.max_t}",
                                 "rotations": rotations}
        else:
            todo[circuit_family(src.stem)].append(src)
    OUT.mkdir(parents=True, exist_ok=True)
    previous = OUT / "prefilter_skipped.json"
    merged = {**(json.loads(previous.read_text()) if previous.exists() else {}), **skipped}
    previous.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n")
    print(f"pre-filter: {len(skipped)} skipped, {sum(map(len, todo.values()))} to prepare "
          f"in {len(todo)} families")
    for family, files in sorted(todo.items()):
        print(f"== {family}: {', '.join(f.stem for f in files)}", flush=True)
        if args.dry_run:
            continue
        subprocess.run([sys.executable, str(PROJECT_ROOT / "src/scripts/prepare_family_benchmarks.py"),
                        *map(str, files), "--family", family, "--output-dir", str(OUT),
                        "--synth-bin", str(args.synth_bin), "--epsilon", str(args.epsilon),
                        "--max-t", str(args.max_t)], check=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
