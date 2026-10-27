#!/usr/bin/env python3
"""Run one evaluation experiment end to end (EVAL_PLAN B.6).

    python src/scripts/eval/run_eval.py e1_sota --jobs 4        # run what is missing, then derive
    python src/scripts/eval/run_eval.py e1_sota --figures-only  # redraw PDFs from the CSVs
    python src/scripts/eval/run_eval.py e3_ablation --quick 3   # smoke test: 3 smallest circuits

Steps: expand ``experiments/eval/<id>.json`` into sweep points; run each point
with ``compare_sota_baselines.py --resume --keep-failed`` (only missing trials
are computed; ``PYTHONHASHSEED=0`` because HARVEST's tie-breaking iterates
string-keyed sets); collect every raw record into ``data.csv``; aggregate
``summary.csv``; draw the PDFs with ``plot_<id>.py``; write ``runs.csv``,
``manifest.json``, ``README.md`` and refresh ``results/eval/INDEX.md``.

Experiment schema (see the files in ``experiments/eval``)::

    {"id", "section", "claim", "question", "caveats": [...],
     "defaults": {"mode", "trials", "seed", "timeout", "args"},
     "inputs_from": ["e1_sota"],            # aggregate experiments read other data.csv
     "stages": [{"name", "suite", "baselines", "args",
                 "axes": [[{"id", "args", "baselines"?}, ...], ...],   # cartesian product
                 "harvest_configs": {"name": {HarvestConfig overrides}},  # one more axis
                 "sweeps": {"alpha": [...], "beta": [...]},               # one more axis
                 "strict_validate": true,
                 "kind": "compare" | "script", "script", "script_args"}]}
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
for path in (PROJECT_ROOT / "src", PROJECT_ROOT / "src" / "scripts", SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

EXPERIMENTS = PROJECT_ROOT / "experiments" / "eval"
SUITES_FILE = EXPERIMENTS / "suites.json"
TOOLS_FILE = EXPERIMENTS / "tools.json"
FLASQ_CACHE = PROJECT_ROOT / "results" / "flasq_cache"
COMPARE = PROJECT_ROOT / "src" / "scripts" / "compare_sota_baselines.py"


def env() -> Dict[str, str]:
    out = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src"), "PYTHONHASHSEED": "0",
           "MPLCONFIGDIR": "/tmp/harvest-matplotlib"}
    out.pop("HARVEST_FAULT", None)  # faults are injected per point via --fault only
    return out


def load_experiment(exp_id: str) -> Dict[str, Any]:
    path = EXPERIMENTS / f"{exp_id}.json"
    if not path.exists():
        known = sorted(p.stem for p in EXPERIMENTS.glob("e*.json"))
        raise SystemExit(f"unknown experiment {exp_id!r}; known: {known}")
    return json.loads(path.read_text())


def suite_files(name: str, quick: Optional[int]) -> List[Dict[str, Any]]:
    suites = json.loads(SUITES_FILE.read_text())
    if name not in suites:
        raise SystemExit(f"unknown suite {name!r}; run freeze_suites.py")
    files = suites[name]["files"]
    if quick:
        files = sorted(files, key=lambda f: (f["ops"], f["circuit"]))[:quick]
    return files


# --------------------------------------------------------------------------
# Expansion
# --------------------------------------------------------------------------


def _config_axis(stage: Dict[str, Any]) -> List[List[Dict[str, Any]]]:
    axes = []
    if stage.get("harvest_configs"):
        axes.append([
            {"id": name, "harvest_variant": name, "harvest_config": overrides}
            for name, overrides in stage["harvest_configs"].items()
        ])
    if stage.get("sweeps"):
        knobs = list(stage["sweeps"])
        entries = []
        for values in itertools.product(*(stage["sweeps"][k] for k in knobs)):
            overrides = dict(zip(knobs, values))
            name = "_".join(f"{k}{v}" for k, v in overrides.items())
            entries.append({"id": name, "harvest_variant": name, "harvest_config": overrides})
        axes.append(entries)
    return axes


def expand(experiment: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Every sweep point: ``{"stage", "point", "baselines", "args", ...}``."""
    defaults = experiment.get("defaults", {})
    points = []
    for stage in experiment.get("stages", []):
        if stage.get("kind", "compare") != "compare":
            points.append({"stage": stage["name"], "point": stage["name"], "stage_def": stage})
            continue
        axes = list(stage.get("axes", [])) + _config_axis(stage)
        for combo in itertools.product(*axes) if axes else [()]:
            args = list(defaults.get("args", [])) + list(stage.get("args", []))
            baselines = stage.get("baselines", defaults.get("baselines", "harvest"))
            harvest_config: Dict[str, Any] = {}
            variant = None
            ids = []
            for entry in combo:
                ids.append(entry["id"])
                args += entry.get("args", [])
                baselines = entry.get("baselines", baselines)
                harvest_config.update(entry.get("harvest_config", {}))
                variant = entry.get("harvest_variant", variant)
            if harvest_config:
                args += ["--harvest-config", json.dumps(harvest_config, sort_keys=True)]
            if variant:
                args += ["--harvest-variant", variant]
            points.append({
                "stage": stage["name"],
                "point": "__".join([stage["name"], *ids]),
                "suite": stage.get("suite", defaults.get("suite", "core")),
                "baselines": baselines,
                "args": args,
                "strict_validate": stage.get("strict_validate", True),
                "mode": stage.get("mode", defaults.get("mode", "matched_ir")),
                "trials": stage.get("trials", defaults.get("trials", 1)),
                "seed": stage.get("seed", defaults.get("seed", 42)),
                "timeout": stage.get("timeout", defaults.get("timeout", 1800)),
            })
    return points


def tool_args(baselines: str) -> List[str]:
    tools = json.loads(TOOLS_FILE.read_text()) if TOOLS_FILE.exists() else {}
    selected = set(baselines.split(","))
    args: List[str] = []

    def add(flag: str, key: str) -> None:
        value = tools.get(key)
        if value and Path(value).exists():
            args.extend([flag, value])
        elif value:
            print(f"warning: {key} {value} does not exist; affected runs will be recorded as failures")

    if "puremagic" in selected:
        add("--puremagic-bin", "puremagic_bin")
        add("--puremagic-transpile-bin", "puremagic_transpile_bin")
        if tools.get("puremagic_version"):
            args += ["--puremagic-version", tools["puremagic_version"]]
    if "dascot" in selected:
        add("--wisq-bin", "wisq_bin")
    return args


def point_command(point: Dict[str, Any], exp_dir: Path, exp_id: str, quick: Optional[int],
                  force_rerun: str) -> List[str]:
    command = [sys.executable, str(COMPARE)]
    for f in suite_files(point["suite"], quick):
        command += ["--benchmarks", str(PROJECT_ROOT / f["path"])]
    command += [
        "--baselines", point["baselines"],
        "--comparison-mode", point["mode"],
        "--trials", str(point["trials"]),
        "--seed", str(point["seed"]),
        "--timeout", str(point["timeout"]),
        "--in-process-timeout", str(point["timeout"]),
        "--resume", "--keep-failed", "--no-derive",
        "--output-dir", str(exp_dir / "runs" / point["point"]),
        "--runs-csv", str(exp_dir / "runs.csv"),
        "--exp-id", exp_id, "--point", point["point"],
    ]
    if point["strict_validate"]:
        command.append("--strict-validate")
    if force_rerun:
        command += ["--force-rerun", force_rerun]
    return command + tool_args(point["baselines"]) + list(point["args"])


def run_point(command: List[str], log: Path, dry_run: bool) -> int:
    print("$ " + shlex.join(command[:2] + ["..."] + command[-12:]), flush=True)
    if dry_run:
        return 0
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as handle:
        handle.write("$ " + shlex.join(command) + "\n")
        handle.flush()
        proc = subprocess.run(command, cwd=PROJECT_ROOT, env=env(), stdout=handle,
                              stderr=subprocess.STDOUT, check=False)
    if proc.returncode != 0:
        print(f"!! point failed (exit {proc.returncode}); see {log}", flush=True)
    return proc.returncode


# --------------------------------------------------------------------------
# Collection
# --------------------------------------------------------------------------


def collect(experiment: Dict[str, Any], exp_dir: Path, out_root: Path, quick: Optional[int],
            write_runs: bool = True):
    import pandas as pd

    from compare_sota_baselines import load_benchmark
    from harvest.eval.collect import add_bounds, flasq_floors, flatten, load_raw, ordered_columns
    from harvest.eval.features import load_features
    from harvest.eval.manifest import run_row, write_runs_csv

    exp_id = experiment["id"]
    rows, run_rows = [], []
    for point_dir in sorted((exp_dir / "runs").glob("*")):
        raw_dir = point_dir / "raw"
        stage = point_dir.name.split("__")[0]
        for payload in load_raw(raw_dir):
            rows.append(flatten(payload, exp_id=exp_id, stage=stage, point=point_dir.name))
            run_rows.append(run_row(exp_id, point_dir.name, payload))
    for source in experiment.get("inputs_from", []):
        path = out_root / source / "data.csv"
        if path.exists():
            frame = pd.read_csv(path, low_memory=False)
            rows.extend(frame.to_dict("records"))
        else:
            print(f"note: {path} does not exist yet; run {source} first")
    suites = json.loads(SUITES_FILE.read_text())
    all_files = {f["circuit"]: PROJECT_ROOT / f["path"] for s in suites.values() for f in s["files"]}
    needed = sorted({r["circuit"] for r in rows if r.get("circuit") in all_files})
    features = load_features([all_files[c] for c in needed], load_benchmark,
                             out_root / "features_cache.json")
    floors = {"free": flasq_floors(FLASQ_CACHE, "matched"),
              "cultivation": flasq_floors(FLASQ_CACHE, "cultivation")}
    for row in rows:
        floor = floors.get(row.get("magic_model") or "free", {}).get(row["circuit"])
        add_bounds(row, features.get(row["circuit"], {}), floor)
    frame = pd.DataFrame(rows, columns=ordered_columns(rows)) if rows else pd.DataFrame()
    tmp = exp_dir / "data.csv.tmp"  # atomic: readers never see a half-written file
    frame.to_csv(tmp, index=False)
    tmp.replace(exp_dir / "data.csv")
    if run_rows and write_runs:
        run_rows.sort(key=lambda r: (r["point"], r["circuit"] or "", r["compiler"] or "",
                                     r["variant"] or "", r["trial"] or 0))
        write_runs_csv(exp_dir / "runs.csv", run_rows)
    print(f"data.csv: {len(frame)} rows")
    return frame


# --------------------------------------------------------------------------
# README / INDEX
# --------------------------------------------------------------------------


def write_readme(experiment: Dict[str, Any], exp_dir: Path, quick: Optional[int],
                 snapshot: bool = False) -> None:
    pdfs = sorted(p.name for p in exp_dir.glob("*.pdf"))
    extra = sorted(p.name for p in exp_dir.glob("*.tex")) + sorted(
        p.name for p in exp_dir.glob("*.csv") if p.name not in ("data.csv", "summary.csv", "runs.csv")
    )
    command = f"python src/scripts/eval/run_eval.py {experiment['id']} --jobs 4"
    lines = [
        f"# {experiment['id']} — {experiment.get('title', '')}".rstrip(" —"),
        "",
        f"**Question.** {experiment.get('question', '')}",
        f"**Paper.** §{experiment.get('section', '?')} (claim {experiment.get('claim', '-')}).",
        f"**Reproduce.** `{command}`; redraw only: `... --figures-only`.",
        f"**Files.** `data.csv` (one row per trial), `summary.csv` (plotted aggregates), "
        f"`runs.csv` (how each tool ran), `manifest.json`, `runs/<point>/raw/*.json`; "
        f"figures: {', '.join(f'`{p}`' for p in pdfs) or '(none yet)'}"
        + (f"; also {', '.join(f'`{p}`' for p in extra)}" if extra else "") + ".",
    ]
    caveats = list(experiment.get("caveats", []))
    if quick:
        caveats.insert(0, f"QUICK RUN: only the {quick} smallest circuits per suite.")
    if snapshot:
        from harvest.eval.manifest import now

        caveats.insert(0, f"INTERIM SNAPSHOT ({now()}): runs still in progress; numbers cover "
                          "only the trials stored so far.")
    if caveats:
        lines.append("**Caveats.** " + " ".join(caveats))
    (exp_dir / "README.md").write_text("\n".join(lines) + "\n")


def write_index(out_root: Path) -> None:
    lines = ["# HARVEST evaluation results", "",
             "| Experiment | Paper section | Claim | Figures | Status |",
             "|---|---|---|---|---|"]
    for path in sorted(EXPERIMENTS.glob("e*.json")):
        exp = json.loads(path.read_text())
        exp_dir = out_root / exp["id"]
        pdfs = sorted(p.name for p in exp_dir.glob("*.pdf")) if exp_dir.exists() else []
        status = "done" if (exp_dir / "summary.csv").exists() else "not run"
        figures = "<br>".join(f"[{p}]({exp['id']}/{p})" for p in pdfs) or "—"
        lines.append(f"| [{exp['id']}]({exp['id']}/README.md) | §{exp.get('section', '?')} | "
                     f"{exp.get('claim', '-')} | {figures} | {status} |")
    lines += ["", "Paper macros: [`paper_numbers.tex`](paper_numbers.tex) "
              "(`python src/scripts/eval/make_paper_numbers.py`)."]
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "INDEX.md").write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("exp_id")
    parser.add_argument("--figures-only", action="store_true",
                        help="Only redraw the PDFs from data.csv / summary.csv.")
    parser.add_argument("--snapshot", action="store_true",
                        help="Read-only interim view while runs are still going: build "
                             "data.csv / summary.csv / PDFs from the stored records without "
                             "running anything and without touching runs.csv.")
    parser.add_argument("--no-run", action="store_true",
                        help="Skip running; re-collect, summarize and plot stored records.")
    parser.add_argument("--resume", action="store_true",
                        help="Accepted for symmetry; every run already resumes stored trials.")
    parser.add_argument("--jobs", type=int, default=1, help="Sweep points run in parallel.")
    parser.add_argument("--only-stage", nargs="+", help="Run only these stages.")
    parser.add_argument("--quick", type=int, metavar="N",
                        help="Smoke test on the N smallest circuits of each suite "
                             "(written under results/eval_quick).")
    parser.add_argument("--force-rerun", default="", help="Baselines to recompute.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    from harvest.eval.manifest import now, suite_hash, write_manifest
    from harvest.routing.config import HarvestConfig

    experiment = load_experiment(args.exp_id)
    out_root = PROJECT_ROOT / ("results/eval_quick" if args.quick else "results/eval")
    exp_dir = out_root / args.exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    started = now()
    # An experiment may reuse another one's summary and plots ("template").
    template = experiment.get("template", args.exp_id)
    plot_script = SCRIPT_DIR / f"plot_{template}.py"

    if not args.figures_only:
        points = expand(experiment)
        if args.only_stage:
            points = [p for p in points if p["stage"] in args.only_stage]
        if args.snapshot:
            args.no_run = True
        if not args.no_run:
            jobs = []
            for point in points:
                log = exp_dir / "logs" / f"{point['point']}.log"
                if "stage_def" in point:
                    stage = point["stage_def"]
                    command = [sys.executable, str(SCRIPT_DIR / stage["script"]),
                               "--exp-dir", str(exp_dir), *stage.get("script_args", [])]
                    if args.quick:
                        command += ["--quick", str(args.quick)]
                else:
                    command = point_command(point, exp_dir, args.exp_id, args.quick,
                                            args.force_rerun)
                jobs.append((command, log))
            workers = max(1, min(args.jobs, experiment.get("max_jobs", args.jobs)))
            print(f"{args.exp_id}: {len(jobs)} point(s), {workers} in parallel")
            with ThreadPoolExecutor(max_workers=workers) as pool:
                codes = list(pool.map(lambda job: run_point(job[0], job[1], args.dry_run), jobs))
            failed = sum(1 for c in codes if c)
            if failed:
                print(f"{failed} point(s) exited non-zero; their recorded trials are still collected")
        if args.dry_run:
            return 0
        frame = collect(experiment, exp_dir, out_root, args.quick, write_runs=not args.snapshot)
        from summaries import summarize

        summarize(template, frame, exp_dir)
        suites = json.loads(SUITES_FILE.read_text())
        used = sorted({p["suite"] for p in points if "suite" in p})
        write_manifest(
            exp_dir / "manifest.json", exp_id=args.exp_id, argv=sys.argv,
            experiment=experiment, started=started,
            suite={name: {"hash": suites[name]["hash"], "n": suites[name]["n"],
                          "quick": args.quick} for name in used},
            harvest_config=HarvestConfig().to_dict(),
            extra={"points": [{k: v for k, v in p.items() if k != "stage_def"} for p in points],
                   "pythonhashseed": "0"},
        )
    if plot_script.exists():
        subprocess.run([sys.executable, str(plot_script), "--exp-dir", str(exp_dir)],
                       cwd=PROJECT_ROOT, env=env(), check=True)
    stray = sorted(p.name for p in exp_dir.glob("*.png"))
    if stray:
        raise SystemExit(f"PNG files are not allowed in {exp_dir}: {stray}")
    write_readme(experiment, exp_dir, args.quick, snapshot=args.snapshot)
    write_index(out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
