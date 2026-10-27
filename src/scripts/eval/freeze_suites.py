#!/usr/bin/env python3
"""Freeze the evaluation benchmark suites (EVAL_PLAN A.3) into ``experiments/eval/suites.json``.

Each suite lists its files with SHA-256, qubit and Pauli-product counts, and
a suite hash recorded in every experiment manifest.  Re-freezing never
silently changes a suite: if a file's hash differs from the frozen one the
script stops unless ``--refreeze`` is given.

``core``   every exactly-lowered Clifford+T circuit (Feynman, extended
           QASMBench, parallel families) except the documented exclusions.
``sweep``  nine representative ``core`` circuits for parameter sweeps.
``scale``  the large ``core`` circuits plus a generated ``mcx_tree`` series up
           to ~10^5 Pauli products (``--generate-scale`` lowers it first).

Example::

    python src/scripts/eval/freeze_suites.py --generate-scale
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
for path in (PROJECT_ROOT / "src", PROJECT_ROOT / "src" / "scripts"):
    sys.path.insert(0, str(path))

SUITES_FILE = PROJECT_ROOT / "experiments" / "eval" / "suites.json"
FEATURES_CACHE = PROJECT_ROOT / "results" / "eval" / "features_cache.json"
CORE_DIRS = (
    "results/sota_feynman_cliffordt_inputs",
    "results/sota_extended_cliffordt_inputs",
    "results/sota_parallel_cliffordt_inputs",
)
# Kept out of `core` exactly as in the earlier matched-IR runs
# (experiments/sota_suite.json, --skip-circuits); they stay in `scale`.
CORE_EXCLUDE = {
    "gf2^64_mult": "28.7k products; external matched-IR baselines exceed the 3 h budget",
    "hwb11": "84k products; external matched-IR baselines exceed the 3 h budget",
}
SWEEP = (
    # low parallelism (#ops / critical path < 2)
    "tof_4", "rc_adder_6", "adder_n118",
    # medium (2-4)
    "adder_8", "qcla_mod_7",
    # high (>= 4), small and large registers
    "csum_mux_9", "mcx_tree_n16", "mcx_tree_n32", "ks_adder_n12",
)
FAMILY_DIRS = ("results/sota_family_inputs", "results/eval_inputs/families")
ISING_LARGE = ("ising_n400", "ising_n420", "ising_n440")
FAMILY_MIN_OPS = 100
SCALE_MIN_OPS = 1000
SCALE_DIR = "results/eval_inputs/scale_cliffordt"
SCALE_MCX = (64, 256, 1024, 4096, 8192)


def generate_scale() -> None:
    sources = PROJECT_ROOT / "results" / "eval_inputs" / "scale_sources"
    subprocess.run([sys.executable, str(PROJECT_ROOT / "src/scripts/generate_parallel_benchmarks.py"),
                    "--output-dir", str(sources), "--ks-sizes", "--mcx-sizes",
                    *map(str, SCALE_MCX), "--trials", "200"], check=True)
    subprocess.run([sys.executable, str(PROJECT_ROOT / "src/scripts/prepare_cliffordt_benchmarks.py"),
                    *sorted(str(p) for p in sources.glob("*.qasm")),
                    "--output-dir", str(PROJECT_ROOT / SCALE_DIR)], check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--generate-scale", action="store_true",
                        help="Generate and lower the synthetic mcx_tree scaling series first.")
    parser.add_argument("--refreeze", action="store_true",
                        help="Accept input files whose SHA-256 changed since the last freeze.")
    args = parser.parse_args()

    import logging

    logging.disable(logging.WARNING)
    from compare_sota_baselines import load_benchmark

    from harvest.eval.features import load_features
    from harvest.eval.manifest import suite_hash

    if args.generate_scale:
        generate_scale()
    core_files = sorted(p for d in CORE_DIRS for p in (PROJECT_ROOT / d).glob("*.qasm"))
    scale_files = sorted((PROJECT_ROOT / SCALE_DIR).glob("*.qasm"))
    # Family inputs (approximate synthesis allowed); one file per circuit name.
    family_files = {}
    for d in FAMILY_DIRS:
        for p in sorted((PROJECT_ROOT / d).glob("*/*.qasm")):
            family_files.setdefault(p.stem, p)
    # Skip outputs the synthesizer left empty (e.g. wstate_n3 -> 0 qubits).
    family_files = sorted((p for p in family_files.values() if "qreg" in p.read_text()
                           and "qreg q[0]" not in p.read_text()), key=lambda p: p.stem)
    features = load_features(core_files + scale_files + family_files, load_benchmark,
                             FEATURES_CACHE)

    from harvest.eval.families import circuit_family, synthesis_kind

    def entry(path: Path):
        f = features[path.stem]
        family = circuit_family(path.stem)
        return {"circuit": path.stem, "family": family, "synthesis": synthesis_kind(family),
                "path": str(path.relative_to(PROJECT_ROOT)),
                "sha256": f["input_sha256"], "num_qubits": f["num_qubits"], "ops": f["ops"],
                "parallelism": round(f["parallelism"], 3)}

    core = [p for p in core_files if p.stem not in CORE_EXCLUDE]
    by_name = {p.stem: p for p in core_files}
    missing = [name for name in SWEEP if name not in by_name]
    if missing:
        raise SystemExit(f"sweep circuits missing from core: {missing}")
    sweep = [by_name[name] for name in SWEEP]
    scale = [p for p in core_files if features[p.stem]["ops"] >= SCALE_MIN_OPS] + scale_files
    ising = [p for p in family_files if circuit_family(p.stem) == "ising"]
    ising_large = [p for p in ising if p.stem in ISING_LARGE]
    ising_small = [p for p in ising if features[p.stem]["num_qubits"] < 400]
    # Every family circuit with a magic-state operation, except the 400+ qubit
    # Ising series (its own suite: one run there costs up to hours).
    # Toy circuits (< FAMILY_MIN_OPS products, e.g. 1-op teleportation) only add noise
    # to per-family geomeans; they stay prepared but out of the suite.
    families = [p for p in family_files if features[p.stem]["ops"] >= FAMILY_MIN_OPS
                and features[p.stem]["num_qubits"] < 400]

    suites = {}
    for name, files, description in (
        ("core", core, "All exactly-lowered Clifford+T circuits (Feynman, extended "
                       "QASMBench, parallel families) minus documented exclusions."),
        ("sweep", sweep, "Representative core circuits: low/medium/high parallelism, "
                         "small/large registers."),
        ("scale", scale, f"Core circuits with >= {SCALE_MIN_OPS} Pauli products (exclusions "
                         f"included) plus mcx_tree_n{{{','.join(map(str, SCALE_MCX))}}}."),
        ("families", families, "Every prepared circuit family (most approximately synthesized, "
                               "compile_cliffordt -e 1e-3) below 400 qubits, >= 100 products."),
        ("ising", ising_small, "Ising family below 400 qubits (n10-n98)."),
        ("ising_large", ising_large, "Large Ising instances n400, n420, n440."),
    ):
        suites[name] = {"description": description, "n": len(files),
                        "hash": suite_hash(files), "files": [entry(p) for p in files]}
    suites["core"]["excluded"] = CORE_EXCLUDE

    if SUITES_FILE.exists() and not args.refreeze:
        frozen = json.loads(SUITES_FILE.read_text())
        changed = [
            f"{s}:{e['circuit']}"
            for s, suite in frozen.items()
            for e in suite["files"]
            if s in suites and {x["circuit"]: x["sha256"] for x in suites[s]["files"]}.get(
                e["circuit"], e["sha256"]) != e["sha256"]
        ]
        if changed:
            raise SystemExit(f"inputs changed since the freeze: {changed}; pass --refreeze")
    SUITES_FILE.parent.mkdir(parents=True, exist_ok=True)
    SUITES_FILE.write_text(json.dumps(suites, indent=2) + "\n")
    for name, suite in suites.items():
        print(f"{name}: {suite['n']} circuits (hash {suite['hash'][:12]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
