#!/usr/bin/env python3
"""Incrementally run the SOTA trade-off suite described by a JSON file.

Nothing is recomputed that is already stored:

* ``--add`` lowers new source circuits exactly to Clifford+T into the suite's
  ``added_inputs_dir`` (unchanged sources are skipped by SHA-256);
* every experiment runs ``compare_sota_baselines.py --resume --keep-failed``,
  so only trials missing from ``<output_dir>/raw`` are computed (a trial is
  identified by baseline, variant, circuit, mode, trial, seed, options and
  input SHA-256);
* figures reuse ``<output_dir>/bounds_cache.json`` and the per-circuit FLASQ
  cache in ``results/flasq_cache``.

Examples::

    # add circuits, run what is missing, redraw every figure
    python src/scripts/run_sota_suite.py experiments/sota_suite.json --add path/to/new.qasm
    # only redraw the figures from stored data
    python src/scripts/run_sota_suite.py experiments/sota_suite.json --figures-only
    # recompute one baseline after changing its code
    python src/scripts/run_sota_suite.py experiments/sota_suite.json --force-rerun harvest
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent


def _run(command: List[str], dry_run: bool, log: Optional[Path] = None) -> None:
    print("$ " + shlex.join(command), flush=True)
    if dry_run:
        return
    env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src"), "PYTHONHASHSEED": "0"}
    if log is None:
        proc = subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=False)
    else:
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a") as handle:
            proc = subprocess.run(
                command, cwd=PROJECT_ROOT, env=env, check=False, stdout=handle, stderr=subprocess.STDOUT
            )
    if proc.returncode != 0:
        raise SystemExit(f"command failed with exit code {proc.returncode}: {shlex.join(command)}")


def _qasm_sources(paths: Sequence[str]) -> List[Path]:
    sources: List[Path] = []
    for item in paths:
        path = Path(item)
        if path.is_dir():
            sources.extend(sorted(path.rglob("*.qasm")))
        elif path.is_file():
            sources.append(path)
        else:
            raise SystemExit(f"--add: {item} does not exist")
    return sources


def experiment_command(suite: Dict, experiment: Dict, force_rerun: str) -> List[str]:
    tools = suite.get("tools", {})
    command = [sys.executable, str(SCRIPT_DIR / "compare_sota_baselines.py")]
    # An experiment may restrict itself to a subset of the suite's inputs.
    for directory in experiment.get("inputs", suite["inputs"]):
        command += ["--benchmarks", directory]
    command += [
        "--baselines", experiment["baselines"],
        "--comparison-mode", experiment["mode"],
        "--trials", str(experiment.get("trials", 1)),
        "--seed", str(experiment.get("seed", 42)),
        "--timeout", str(experiment.get("timeout", 1800)),
        "--resume", "--keep-failed",
        "--output-dir", experiment["output_dir"],
    ]
    baselines = set(experiment["baselines"].split(","))
    if "puremagic" in baselines:
        command += ["--puremagic-bin", tools["puremagic_bin"],
                    "--puremagic-transpile-bin", tools["puremagic_transpile_bin"]]
    if "taco" in baselines:
        command += ["--taco-bin", tools["taco_bin"], "--nwqec-version", tools["nwqec_version"]]
    if force_rerun:
        command += ["--force-rerun", force_rerun]
    return command + list(experiment.get("args", []))


def figure_command(suite: Dict, figure: Dict) -> List[str]:
    experiments = {e["name"]: e for e in suite["experiments"]}
    if figure.get("kind") == "overview":
        def part(names, extra):
            return " ".join([experiments[n]["output_dir"] for n in names] + list(extra))
        command = [sys.executable, str(SCRIPT_DIR / "plot_sota_overview.py"),
                   "--free", part(figure["free_experiments"], figure.get("free_args", [])),
                   "--output-dir", figure["output_dir"], "--output-stem", figure["stem"]]
        return command + list(figure.get("args", []))
    if figure.get("kind") == "time_box":
        command = [sys.executable, str(SCRIPT_DIR / "plot_sota_time_boxplot.py")]
        for title, csv_path in figure["panels"]:
            command += ["--panel", f"{title}={csv_path}"]
        command += ["--output-dir", figure["output_dir"], "--output-stem", figure["stem"]]
        return command + list(figure.get("args", []))
    if figure.get("kind") == "compile_time":
        command = [sys.executable, str(SCRIPT_DIR / "plot_sota_compile_time.py")]
        for title, name in figure["panels"]:
            exp = experiments[name]
            command += ["--panel", f"{title}={exp['output_dir']}:{exp['mode']}"]
        command += ["--inputs", *suite["inputs"], "--output-dir", figure["output_dir"],
                    "--output-stem", figure["stem"]]
        return command + list(figure.get("args", []))
    command = [sys.executable, str(SCRIPT_DIR / "plot_sota_space_time_tradeoff.py")]
    command += [experiments[name]["output_dir"] for name in figure["experiments"]]
    command += ["--output-dir", figure["output_dir"], "--output-stem", figure["stem"]]
    flasq_python = suite.get("tools", {}).get("flasq_python")
    if flasq_python:
        command += ["--flasq-python", flasq_python]
    return command + list(figure.get("args", []))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("suite", type=Path)
    parser.add_argument("--add", nargs="+", default=[], metavar="QASM",
                        help="Source circuits (files or directories) to lower and add.")
    parser.add_argument("--only", nargs="+", metavar="EXPERIMENT",
                        help="Run only these experiments (figures are still all drawn).")
    parser.add_argument("--figures-only", action="store_true")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--force-rerun", default="",
                        help="Comma-separated baselines to recompute even if stored.")
    parser.add_argument("--dry-run", action="store_true", help="Print the commands only.")
    args = parser.parse_args(argv)

    suite = json.loads(args.suite.read_text())
    if args.add:
        sources = _qasm_sources(args.add)
        _run([sys.executable, str(SCRIPT_DIR / "prepare_cliffordt_benchmarks.py"),
              *map(str, sources), "--output-dir", suite["added_inputs_dir"]], args.dry_run)

    if not args.figures_only:
        names = {e["name"] for e in suite["experiments"]}
        unknown = set(args.only or []) - names
        if unknown:
            raise SystemExit(f"unknown experiment(s): {sorted(unknown)}")
        for experiment in suite["experiments"]:
            if args.only and experiment["name"] not in args.only:
                continue
            log = PROJECT_ROOT / experiment["output_dir"] / "suite.log"
            print(f"== experiment {experiment['name']} (log: {log.relative_to(PROJECT_ROOT)})")
            _run(experiment_command(suite, experiment, args.force_rerun), args.dry_run, log)

    if not args.no_figures:
        for figure in suite["figures"]:
            print(f"== figure {figure['name']}")
            _run(figure_command(suite, figure), args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
