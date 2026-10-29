#!/usr/bin/env python3
"""Select, lower and describe the benchmark suite of the RQ1-RQ5 experiments.

Selection rule: every circuit in ``benchmark_circuits/qasm`` with fewer than
100 qubits, except the ``qv``, ``chemical`` and ``feynman`` folders, plus every
Ising and wstate circuit regardless of size.  QASMBench circuits are taken from their
``<name>/<name>.qasm`` file (not the ``_transpiled`` copy).

Every compiler receives the same Clifford+T input.  Inputs that already exist
under ``results/`` (earlier evaluations, same front end) are reused; the rest
are lowered with ``prepare_family_benchmarks.py`` (exact when possible, else
PureMagic ``compile_cliffordt -e 1e-3``) into ``results/rq_inputs/<family>/``.

Output: ``results/rq/suite.csv``, one row per selected circuit with its
features, its status (``ok``, ``clifford_only``, ``over_budget``, ``failed``)
and its tiers:

* ``core``  - RQ1/RQ5 suite: everything usable, but the dense qaoa/qft size
  scans (one circuit per qubit count) are sampled on a size grid and the
  random circuits keep seeds 1-2;
* ``extra`` - the remaining qaoa/qft sizes and random seeds (run after core);
* ``small`` - RQ2-RQ4 suite: all Ising circuits below 100 qubits plus up to
  three sizes per family with 100 <= ops <= 6000.
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
sys.path[:0] = [str(PROJECT_ROOT / "src"), str(PROJECT_ROOT / "src" / "scripts")]

from harvest.eval.families import circuit_family, synthesis_kind  # noqa: E402

QASM = PROJECT_ROOT / "benchmark_circuits" / "qasm"
EXCLUDED_FOLDERS = {"qv", "chemical", "feynman"}
ALL_SIZES = {"ising", "wstate"}  # families kept at every size, not only < 100 qubits
OUT_INPUTS = PROJECT_ROOT / "results" / "rq_inputs"
SUITE_CSV = PROJECT_ROOT / "results" / "rq" / "suite.csv"
EXISTING = ("results/sota_family_inputs", "results/eval_inputs/families",
            "results/sota_extended_cliffordt_inputs", "results/sota_parallel_cliffordt_inputs",
            "results/sota_feynman_cliffordt_inputs", "results/rq_inputs")
SYNTH_BIN = Path.home() / ".cache/harvest-tools/PureMagic/target/release/compile_cliffordt"
SIZE_GRID = {4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 56, 64, 72, 80, 90, 99}
ROTATIONS = re.compile(r"^\s*(rz|rx|ry|u|u1|u2|u3|p|cp|crz|crx|cry|cu1|cu3|cu|rzz|rxx|ryy|cphase)\s*\(",
                       re.M)
MAGIC_GATES = re.compile(r"^\s*(t|tdg|ccx|cswap|ccz|mcx|c3x|c4x|rccx|rc3x|cu1|cp|crz)\b", re.M)
T_PER_ROTATION = 13
MAX_T = 40000
SMALL_MAX_OPS = 6000

COLUMNS = ["circuit", "family", "synthesis", "folder", "source", "input", "status", "reason",
           "num_qubits", "ops", "critical_path", "parallelism", "t_fraction",
           "mean_pauli_weight", "y_fraction", "source_gates", "source_t_gates",
           "tier", "small"]


def num_qubits(text: str) -> int:
    return sum(int(m) for m in re.findall(r"^\s*qreg\s+\w+\s*\[\s*(\d+)\s*\]", text, re.M))


def sources():
    out = []
    for folder in sorted(QASM.iterdir()):
        if not folder.is_dir() or folder.name in EXCLUDED_FOLDERS:
            continue
        if folder.name.startswith("qasmbench"):
            out += [d / f"{d.name}.qasm" for d in sorted(folder.iterdir())
                    if (d / f"{d.name}.qasm").exists()]
        else:
            out += sorted(folder.glob("*.qasm"))
    selected, seen = [], set()
    for path in out:
        text = path.read_text(errors="ignore")
        n, family = num_qubits(text), circuit_family(path.stem)
        if (n < 100 or family in ALL_SIZES) and path.stem not in seen:
            seen.add(path.stem)
            selected.append((path, n, family, text))
    return selected


def prepared_inputs():
    have = {}
    for d in EXISTING:
        for p in sorted((PROJECT_ROOT / d).rglob("*.qasm")):
            have.setdefault(p.stem, p)
    return have


def size_of(stem: str):
    m = re.search(r"_[nN]0*(\d+)", stem)
    return int(m.group(1)) if m else None


def tier(path: Path, family: str) -> str:
    stem = path.stem
    if path.parent.name in ("qaoa", "qft"):
        return "core" if size_of(stem) in SIZE_GRID else "extra"
    if family == "random":
        return "core" if re.search(r"_seed[12]$", stem) else "extra"
    return "core"


def synthesize(todo, dry_run: bool) -> None:
    by_family = defaultdict(list)
    for path, family in todo:
        by_family[family].append(path)
    for family, paths in sorted(by_family.items()):
        cmd = [sys.executable, str(PROJECT_ROOT / "src/scripts/prepare_family_benchmarks.py"),
               *map(str, paths), "--family", family, "--output-dir", str(OUT_INPUTS),
               "--synth-bin", str(SYNTH_BIN), "--max-t", str(MAX_T)]
        print(f"[prepare] {family}: {len(paths)} circuit(s)", flush=True)
        if not dry_run:
            subprocess.run(cmd, check=False)


def _skip_reason(path: Path):
    """Why the front end produced no input (its skipped.json), else a generic failure."""
    import json

    skipped = OUT_INPUTS / circuit_family(path.stem) / "skipped.json"
    entry = json.loads(skipped.read_text()).get(path.stem) if skipped.exists() else None
    if entry is None:
        return "failed", "front end produced no input"
    return ("over_budget" if "budget" in entry.get("reason", "") else "failed"), entry["reason"]


def describe(input_path: Path):
    from compare_sota_baselines import load_benchmark
    from harvest.eval.features import circuit_features

    circuit, dag = load_benchmark(input_path)
    return circuit_features(circuit, dag)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dry-run", action="store_true", help="list what would be lowered")
    parser.add_argument("--no-synthesis", action="store_true")
    args = parser.parse_args(argv)

    selected = sources()
    have = prepared_inputs()
    todo, status = [], {}
    for path, n, family, text in selected:
        t_est = len(ROTATIONS.findall(text)) * T_PER_ROTATION + len(MAGIC_GATES.findall(text))
        if path.stem in have:
            continue
        if t_est == 0:
            status[path.stem] = ("clifford_only", "no non-Clifford operation to schedule")
        elif t_est > 10 * MAX_T:
            # hopeless even for the rough estimate; the real budget check on the
            # synthesized T count is done by prepare_family_benchmarks (--max-t)
            status[path.stem] = ("over_budget", f"estimated T count {t_est} > {10 * MAX_T}")
        else:
            todo.append((path, family))
    print(f"{len(selected)} selected, {len(selected) - len(todo) - len(status)} already lowered, "
          f"{len(todo)} to lower, {len(status)} unusable", flush=True)
    if todo and not args.no_synthesis:
        synthesize(todo, args.dry_run)
        have = prepared_inputs()
    if args.dry_run:
        return 0

    rows = []
    for path, n, family, _ in selected:
        row = {"circuit": path.stem, "family": family, "synthesis": synthesis_kind(family),
               "folder": path.parent.name if not path.parent.parent.name.startswith("qasmbench")
               else path.parent.parent.name,
               "source": str(path.relative_to(PROJECT_ROOT)), "num_qubits": n,
               "tier": tier(path, family)}
        if path.stem in have:
            row["input"] = str(have[path.stem].relative_to(PROJECT_ROOT))
            try:
                features = describe(have[path.stem])
                row.update({k: features.get(k) for k in COLUMNS if k in features})
                row["status"] = "ok" if row["ops"] else "clifford_only"
            except Exception as exc:  # recorded, never dropped
                row.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
        else:
            st, reason = status.get(path.stem) or _skip_reason(path)
            row.update(status=st, reason=reason)
        rows.append(row)
        print(f"  {row['circuit']}: {row['status']} ops={row.get('ops')}", flush=True)

    # small tier: all Ising < 100 qubits + up to three sizes per family (min/median/max)
    usable = [r for r in rows if r["status"] == "ok" and r["tier"] == "core"]
    by_family = defaultdict(list)
    for r in usable:
        if r["family"] == "ising" and r["num_qubits"] < 100:
            r["small"] = True
        elif 100 <= r["ops"] <= SMALL_MAX_OPS:
            by_family[r["family"]].append(r)
    for family, members in by_family.items():
        members.sort(key=lambda r: (r["ops"], r["circuit"]))
        for i in sorted({0, len(members) // 2, len(members) - 1}):
            members[i]["small"] = True
    for r in rows:
        r["small"] = bool(r.get("small"))

    SUITE_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows.sort(key=lambda r: (r["family"], r["num_qubits"], r["circuit"]))
    with SUITE_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    ok = [r for r in rows if r["status"] == "ok"]
    print(f"wrote {SUITE_CSV}: {len(ok)} usable ({sum(r['tier'] == 'core' for r in ok)} core, "
          f"{sum(r['tier'] == 'extra' for r in ok)} extra, {sum(r['small'] for r in ok)} small)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
