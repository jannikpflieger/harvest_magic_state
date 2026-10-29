"""Shared machinery of the RQ1-RQ5 experiment scripts.

* the suite (``results/rq/suite.csv``, written by ``prepare_suite.py``);
* layout templates, magic-state models and cached placements;
* one function per compiler that runs a single trial and returns one flat row
  (``harvest.eval.collect.flatten`` plus circuit features and RQ columns);
* ``CsvLog``: every row is appended to ``results/rq/<exp>/<exp>.csv`` and
  flushed to disk (fsync) the moment its run finishes, so an interrupted run
  loses at most the trials in flight.  Rerunning a script skips every row key
  already in the CSV;
* ``run_tasks``: a process pool that survives crashed workers (OOM kills) and
  stops each in-process run after a timeout (recorded as ``status=timeout``).
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import logging
import math
import os
import signal
import sys
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("MPLBACKEND", "Agg")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
for _p in (PROJECT_ROOT / "src", PROJECT_ROOT / "src" / "scripts", SCRIPT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

RESULTS = PROJECT_ROOT / "results" / "rq"
SUITE_CSV = RESULTS / "suite.csv"
CACHE = RESULTS / "cache"
TOOLS = json.loads((PROJECT_ROOT / "experiments" / "eval" / "tools.json").read_text())

# Layout templates (all one-tile patches): ring = data grid with 1-tile spacing
# and a magic ring; blocks4 = 2x2 data blocks; dense4 = ~2.25 tiles/qubit comb
# with 4 magic ports; dense4full = the same comb with a magic port on every
# routing column, top and bottom (pruning drops the ports a schedule leaves
# unused).
TEMPLATES = {
    "ring": {"preset": "ring"},
    "blocks4": {"preset": "blocks4"},
    "dense4": {"preset": "dense", "magic_count": 4},
    "dense4full": {"preset": "dense", "magic_count": "all"},
}
# ``auto``: schedule on every candidate template and keep the smallest volume
# (one fixed rule for every circuit; the compile time is the sum over all).
AUTO_TEMPLATES = ("ring", "blocks4", "dense4", "dense4full")
CULT_MEAN = 3.0  # logical cycles; FLASQ cultivation volume at d=13, p=1e-3
PM_LAMBDA = 0.0387396  # PureMagic's published cultivation rate (states per lcycle per patch)
MODELS = {
    "instant": {},
    "cultivation": {"magic_model": "cultivation", "cultivation_mean_cycles": CULT_MEAN},
    # PureMagic's default (mean ~25.8 cycles); RQ1 large Ising only
    "cultivation_pm": {"magic_model": "cultivation", "cultivation_mean_cycles": 1.0 / PM_LAMBDA},
    "dist4": {"magic_model": "fixed", "magic_mean_cycles": 4},
    "dist10": {"magic_model": "fixed", "magic_mean_cycles": 10},
    "multiplexed": {"magic_model": "multiplexed", "magic_mean_cycles": 2 * CULT_MEAN,
                    "magic_units": 2, "magic_capacity": 1},
    "lognormal": {"magic_model": "lognormal", "magic_mean_cycles": CULT_MEAN},
}
STOCHASTIC = {"cultivation", "cultivation_pm", "multiplexed", "lognormal"}
DROP_COLUMNS = {"input_sha256", "upstream_version", "harvest_config", "exp_id", "stage", "point"}

logger = logging.getLogger("rq")


# ---------------------------------------------------------------------------
# suite
# ---------------------------------------------------------------------------

def load_suite(tiers: Sequence[str] = ("core",), small_only: bool = False,
               circuits: Optional[Sequence[str]] = None,
               max_ops: Optional[int] = None) -> List[Dict[str, Any]]:
    """Usable circuits of the given tiers, cheapest first."""
    if not SUITE_CSV.exists():
        raise SystemExit(f"{SUITE_CSV} missing: run src/scripts/rq/prepare_suite.py first")
    rows = []
    with SUITE_CSV.open() as handle:
        for row in csv.DictReader(handle):
            if row["status"] != "ok" or row["tier"] not in tiers:
                continue
            if small_only and row["small"] != "True":
                continue
            if circuits and row["circuit"] not in circuits:
                continue
            for key in ("num_qubits", "ops", "critical_path", "source_gates", "source_t_gates"):
                row[key] = int(float(row[key])) if row.get(key) not in (None, "") else None
            for key in ("parallelism", "t_fraction", "mean_pauli_weight", "y_fraction"):
                row[key] = float(row[key]) if row.get(key) not in (None, "") else None
            if max_ops is not None and row["ops"] > max_ops:
                continue
            rows.append(row)
    rows.sort(key=lambda r: (r["ops"] * max(1, r["num_qubits"]), r["circuit"]))
    return rows


def circuit_columns(circ: Dict[str, Any]) -> Dict[str, Any]:
    return {"circuit": circ["circuit"], "family": circ["family"], "synthesis": circ["synthesis"],
            "folder": circ["folder"], "num_qubits": circ["num_qubits"], "ops": circ["ops"],
            "critical_path": circ["critical_path"], "parallelism": circ["parallelism"],
            "t_fraction": circ["t_fraction"], "mean_pauli_weight": circ["mean_pauli_weight"],
            "y_fraction": circ["y_fraction"]}


def cost_estimate(circ: Dict[str, Any]) -> float:
    """Approximate seconds of one HARVEST run (fitted on earlier runs; ordering only)."""
    return circ["ops"] * (1.0 + circ["num_qubits"] / 10.0) / 1000.0


_DAG_CACHE: Dict[str, Any] = {}


def load_circuit(circ: Dict[str, Any]):
    """(qiskit circuit, Pauli-product DAG); memoized per worker process."""
    key = circ["input"]
    if key not in _DAG_CACHE:
        from compare_sota_baselines import load_benchmark

        _DAG_CACHE.clear()
        _DAG_CACHE[key] = load_benchmark(PROJECT_ROOT / key)
    return _DAG_CACHE[key]


# ---------------------------------------------------------------------------
# layouts
# ---------------------------------------------------------------------------

def base_layout(num_qubits: int, dag, template: str, magic_count: Optional[int] = None):
    from compare_sota_baselines import build_layout

    spec = TEMPLATES[template]
    count = magic_count if magic_count is not None else spec.get("magic_count", 2)
    if count == "all":  # more than the comb's 2 * (routing columns) ports
        count = 2 * num_qubits + 2
    return build_layout(num_qubits, None, None, dag, "row_major", 0, spec["preset"], count)


def placed_layout(circ, dag, template: str, placement: str = "circuit_aware",
                  y_factor: bool = True, magic_ring: Optional[int] = None,
                  magic_count: Optional[int] = None):
    """Template layout with the qubit->site map of ``placement``.

    ``circuit_aware`` (interaction distance + Y-factor centrality term) or
    ``index`` (row-major).  ``y_factor=False`` drops the centrality term
    (beta = 0).  The assignment is cached on disk (it is deterministic), so
    seeds and compilers share it and pay the placement cost once.
    Returns (layout, placement_cost, t_place_s).
    """
    from harvest.eval.placement import layout_template, relabel
    from harvest.synthesis.placement import PlacementConfig

    layout = base_layout(circ["num_qubits"], dag, template, magic_count)
    if magic_ring:
        from harvest.layout.presets import thin_magic_ring

        layout = thin_magic_ring(layout, magic_ring)
    mode = placement if placement == "index" else ("circuit_aware" if y_factor else "circuit_aware_noy")
    # The placement only depends on the data sites, which the magic count does
    # not change for ring/blocks4 (dense: magic sits in an extra row).
    tag = f"{template}_m{magic_count}" if template == "dense4" and magic_count else template
    path = CACHE / "placement" / f"{circ['circuit']}__{tag}__{mode}.json"
    config = PlacementConfig(alpha=1.0, beta=0.5 if y_factor else 0.0,
                             max_swap_iterations=100, seed=0)
    started = perf_counter()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Missing placements are computed one at a time, whatever the template
    # (wstate_n380 needs ~3 GB to place; several at once exceed the memory
    # cap); cached ones are read without waiting.
    if path.exists():
        assignment, cost, t_place = _placement(path, layout, dag, placement, config, started)
    else:
        with open(path.parent / "compute.lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            assignment, cost, t_place = _placement(path, layout, dag, placement, config, started)
    # Register qubits no product touches are absent from the interaction
    # summary; they still need a data patch, so they fill the free sites in order.
    n_sites = len(layout_template(layout).data_sites)
    free_sites = iter(s for s in range(n_sites) if s not in set(assignment.values()))
    for q in range(circ["num_qubits"]):
        if q not in assignment:
            assignment[q] = next(free_sites)
    return relabel(layout, assignment), cost, t_place


def _placement(path: Path, layout, dag, placement: str, config, started: float):
    """The cached qubit->site assignment at ``path``, computed and stored if missing."""
    from harvest.eval.placement import layout_template
    from harvest.synthesis.circuit_summary import extract_circuit_summary
    from harvest.synthesis.placement import _total_cost, baseline_placement, circuit_aware_placement

    if path.exists():
        cached = json.loads(path.read_text())
        assignment = {int(k): v for k, v in cached["assignment"].items()}
        cost, t_place = cached["cost"], cached["t_place_s"]
    else:
        summary = extract_circuit_summary(dag)
        template_obj = layout_template(layout)
        if placement == "index":
            result = baseline_placement(template_obj, summary.num_qubits, mode="row_major", seed=0)
        else:
            result = circuit_aware_placement(summary, template_obj, config)
        cost, _ = _total_cost(result.assignment, summary, template_obj, config)
        assignment = dict(result.assignment)
        t_place = perf_counter() - started
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps({"assignment": assignment, "cost": cost, "t_place_s": t_place}))
        tmp.replace(path)
    return assignment, cost, t_place


# ---------------------------------------------------------------------------
# rows
# ---------------------------------------------------------------------------

def _run_config(circ, baseline: str, variant: str, options: Dict[str, Any], seed, trial: int,
                input_path: Optional[Path] = None):
    from harvest.baselines.base import BaselineRunConfig

    return BaselineRunConfig(
        baseline=baseline, variant=variant, circuit=circ["circuit"],
        input_path=input_path or (PROJECT_ROOT / circ["input"]),
        output_dir=CACHE / "work", comparison_mode="matched_ir",
        num_qubits=circ["num_qubits"], trial=trial, seed=seed,
        timeout_s=float(options.pop("_timeout", 3600)), options=options,
    )


def result_row(config, result, circ, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    from harvest.eval.collect import flatten

    payload = {"config": {"options": config.options}, "result": result.to_dict()}
    row = flatten(payload)
    for key in DROP_COLUMNS:
        row.pop(key, None)
    row.update(circuit_columns(circ))
    row["timed_out"] = result.timed_out
    meta = result.metadata or {}
    pruning = meta.get("pruning") or {}
    for key in ("routing_cells_removed", "port_nodes_removed", "patches_before", "patches_after"):
        if key in pruning:
            row[f"prune_{key}"] = pruning[key]
    row["magic_terminals"] = row.get("magic_tiles_initial")
    row.update(extra or {})
    return row


def failed_row(circ, extra: Dict[str, Any], status: str, error: str) -> Dict[str, Any]:
    row = circuit_columns(circ)
    row.update(extra)
    row.update(status=status, error=error[:2000])
    return row


class RunTimeout(Exception):
    pass


class time_limit:
    """SIGALRM-based limit for in-process compilers (main thread of a worker)."""

    def __init__(self, seconds: Optional[float]):
        self.seconds = seconds

    def __enter__(self):
        if self.seconds:
            def on_alarm(signum, frame):
                raise RunTimeout(f"timeout after {self.seconds:.0f} s")
            self.previous = signal.signal(signal.SIGALRM, on_alarm)
            signal.setitimer(signal.ITIMER_REAL, self.seconds)
        return self

    def __exit__(self, *exc):
        if self.seconds:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, self.previous)
        return False


def model_options(model: str) -> Dict[str, Any]:
    return dict(MODELS[model])


def layout_options(template: str, magic_ring=None, magic_count=None) -> Dict[str, Any]:
    spec = TEMPLATES[template]
    out: Dict[str, Any] = {"layout_preset": spec["preset"]}
    if spec["preset"] == "dense":
        out["layout_magic_count"] = magic_count or spec["magic_count"]
    if magic_ring:
        out["layout_magic_ring"] = magic_ring
    return out


def run_in_tree(circ, compiler: str, template: str, model: str, seed: Optional[int],
                hconfig: Optional[Dict[str, Any]] = None, placement: str = "circuit_aware",
                y_factor: bool = True, magic_ring=None, magic_count=None, layout=None,
                timeout: float = 3600, extra: Optional[Dict[str, Any]] = None,
                sink: Optional[dict] = None, strict: bool = False,
                fault: Optional[str] = None) -> Dict[str, Any]:
    """One HARVEST (``compiler='harvest'``) or Silva EAF (``'eaf'``) trial.

    ``template='auto'`` runs every template of ``AUTO_TEMPLATES`` (see
    :func:`run_auto`)."""
    if template == "auto":
        return run_auto(circ, compiler, model, seed, hconfig=hconfig, placement=placement,
                        y_factor=y_factor, timeout=timeout, extra=extra, sink=sink,
                        strict=strict, fault=fault)
    from harvest import faults
    from harvest.baselines.harvest import HarvestAdapter
    from harvest.baselines.silva import SilvaEAFAdapter

    circuit, dag = load_circuit(circ)
    t_place, cost = 0.0, None
    if layout is None:
        layout, cost, t_place = placed_layout(circ, dag, template, placement, y_factor,
                                              magic_ring, magic_count)
    options = {**layout_options(template, magic_ring, magic_count), **model_options(model),
               "placement": placement if y_factor else f"{placement}_noy", "_timeout": timeout}
    if hconfig and compiler == "harvest":
        options["harvest_config"] = dict(hconfig)
    if fault:
        options["fault"] = fault
    baseline, variant = (("harvest", "harvest") if compiler == "harvest"
                         else ("silva", "silva_eaf_pruned"))
    if compiler == "eaf":
        options["prune"] = True  # EAF's schedule priced on HARVEST's pruned footprint
    config = _run_config(circ, baseline, variant, options, seed, 0)
    adapter = HarvestAdapter() if compiler == "harvest" else SilvaEAFAdapter()
    if fault:
        os.environ[faults.ENV_VAR] = fault
    try:
        with time_limit(timeout):
            result = adapter.run(config, dag=dag, layout_engine=layout,
                                 phase_times={"t_place_s": t_place}, placement_cost=cost,
                                 validate=True, strict_validate=strict, sink=sink)
    finally:
        if fault:
            os.environ.pop(faults.ENV_VAR, None)
    row = result_row(config, result, circ, extra)
    row["compiler"] = "Harvest" if compiler == "harvest" else "EAF"
    row["template"] = template
    row["model"] = model
    row["compile_time_s"] = (row.get("compile_time_s") or 0.0) + t_place
    return row


def run_auto(circ, compiler: str, model: str, seed: Optional[int], *, timeout: float = 3600,
             sink: Optional[dict] = None, **kwargs) -> Dict[str, Any]:
    """The smallest-volume trial over ``AUTO_TEMPLATES`` (each with the full timeout).

    The row is the winner's, with ``template='auto'``, ``template_chosen``,
    every candidate's volume in ``auto_volumes`` and the compile time summed
    over all candidates.  A candidate that times out or fails is skipped;
    if all do, the first failure is raised / returned.
    """
    rows, sinks, failure = {}, {}, None
    for template in AUTO_TEMPLATES:
        sinks[template] = {} if sink is not None else None
        try:
            row = run_in_tree(circ, compiler, template, model, seed, timeout=timeout,
                              sink=sinks[template], **kwargs)
        except RunTimeout as exc:
            failure = failure or exc
            continue
        rows[template] = row
    ok = {t: r for t, r in rows.items() if r.get("status") == "ok"}
    if not ok:
        if rows:
            return next(iter(rows.values()))
        raise failure
    chosen = min(ok, key=lambda t: (ok[t]["volume_tile_cycles"], AUTO_TEMPLATES.index(t)))
    row = dict(ok[chosen])
    row["template"] = "auto"
    row["template_chosen"] = chosen
    row["auto_volumes"] = {t: r.get("volume_tile_cycles") for t, r in rows.items()}
    row["compile_time_s"] = sum(r.get("compile_time_s") or 0.0 for r in rows.values())
    if sink is not None:
        sink.update(sinks[chosen])
    return row


def run_external(circ, compiler: str, variant: str, model: str, seed: Optional[int],
                 timeout: float = 3600, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """One Litinski / O3LS / PureMagic trial on the identical Pauli-product stream."""
    import importlib

    circuit, dag = load_circuit(circ)
    options = {**model_options(model), "_timeout": timeout}
    input_path = None
    if compiler == "puremagic":
        from harvest.baselines.puremagic import PureMagicAdapter, write_trans_file

        trans = CACHE / "trans" / f"{circ['circuit']}.trans"
        if not trans.exists():
            trans.parent.mkdir(parents=True, exist_ok=True)
            tmp = trans.with_suffix(f".{os.getpid()}.tmp")
            write_trans_file(dag, tmp, circuit.num_qubits)
            tmp.replace(trans)
        input_path = trans
        options.update({
            "input_format": "trans", "ancilla_rows": 1,
            # instant: states (almost) never wait, and T gates never fail
            "magic_state_lambda": (1.0 / MODELS[model]["cultivation_mean_cycles"]
                                   if "cultivation_mean_cycles" in MODELS[model] else 1e6),
            "no_t_failures": model == "instant",
            "transpile_bin": TOOLS.get("puremagic_transpile_bin"),
        })
        adapter = PureMagicAdapter()
        baseline = "puremagic"
    else:
        module = importlib.import_module(f"harvest.baselines.{compiler}")
        adapter = getattr(module, {"litinski": "LitinskiAdapter", "o3ls": "O3LSAdapter"}[compiler])()
        baseline = compiler
    config = _run_config(circ, baseline, variant, options, seed, 0, input_path)
    if compiler == "puremagic":
        from dataclasses import replace

        # The work directory is per circuit/variant/seed; instant, cultivation
        # and cultivation_pm all use seed 42, so without the model in the path
        # concurrent runs read each other's ``*.schedule`` (newest file wins).
        config = replace(config, executable=Path(TOOLS["puremagic_bin"]),
                         output_dir=CACHE / "work" / model)
    with time_limit(timeout + 60 if compiler == "puremagic" else timeout):
        result = adapter.run(config, dag=dag)
    row = result_row(config, result, circ, extra)
    row["compiler"] = {"litinski": "Litinski", "o3ls": "O3LS", "puremagic": "PureMagic"}[compiler]
    row["template"] = variant
    row["model"] = model
    return row


# ---------------------------------------------------------------------------
# crash-safe CSV
# ---------------------------------------------------------------------------

LEAD = ["run_key", "exp", "status", "error", "circuit", "family", "synthesis", "folder",
        "num_qubits", "ops", "critical_path", "parallelism", "compiler", "variant", "template",
        "model", "seed", "T_cycles", "area_tiles", "volume_tile_cycles", "area_initial_tiles",
        "area_pruned_tiles", "volume_initial", "volume_pruned", "compile_time_s"]


class CsvLog:
    """Append-only CSV with a fixed header; new columns go to ``extra_json``.

    The header is the union of ``LEAD``, the experiment's ``columns`` and the
    columns of the first row; later rows with unknown keys keep them in the
    ``extra_json`` column, so nothing is dropped.  Writes are serialized with
    ``flock`` (several workers append) and fsync'ed.
    """

    def __init__(self, path: Path, columns: Sequence[str] = ()):
        self.path = path
        self.columns = list(dict.fromkeys([*LEAD, *columns]))
        path.parent.mkdir(parents=True, exist_ok=True)

    def header(self) -> Optional[List[str]]:
        if not self.path.exists() or self.path.stat().st_size == 0:
            return None
        with self.path.open(newline="") as handle:
            return next(csv.reader(handle), None)

    def done_keys(self) -> set:
        if self.header() is None:
            return set()
        keys = set()
        with self.path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                # an ``error`` row (exception, usually a bug since fixed) is retried;
                # the newer row with the same key supersedes it in ``read_rows``
                if row.get("status") != "error":
                    keys.add(row["run_key"])
        return keys

    def append(self, row: Dict[str, Any]) -> None:
        with open(self.path.with_suffix(".lock"), "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            header = self.header()
            if header is None:
                rest = sorted(k for k in row if k not in self.columns)
                header = [*self.columns, *rest, "extra_json"]
                with self.path.open("w", newline="") as handle:
                    csv.writer(handle).writerow(header)
            extra = {k: v for k, v in row.items() if k not in header}
            out = {k: _cell(row.get(k)) for k in header if k != "extra_json"}
            out["extra_json"] = json.dumps(extra, default=str) if extra else ""
            with self.path.open("a", newline="") as handle:
                csv.DictWriter(handle, fieldnames=header).writerow(out)
                handle.flush()
                os.fsync(handle.fileno())


def _cell(value):
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(value, default=str)
    if isinstance(value, float) and math.isnan(value):
        return ""
    return value


def read_rows(path: Path) -> List[Dict[str, Any]]:
    """All rows of an experiment CSV with ``extra_json`` merged back in."""
    import pandas as pd

    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    if "extra_json" in df and df["extra_json"].notna().any():
        extra = df["extra_json"].fillna("").map(lambda s: json.loads(s) if s else {})
        df = pd.concat([df.drop(columns="extra_json"),
                        pd.DataFrame(list(extra), index=df.index)], axis=1)
        df = df.loc[:, ~df.columns.duplicated()]
    return df.drop_duplicates("run_key", keep="last").reset_index(drop=True)


# ---------------------------------------------------------------------------
# task pool
# ---------------------------------------------------------------------------

@dataclass
class Task:
    exp: str
    key: str
    func: str            # "module:function", called as func(**kwargs) -> row or [rows]
    kwargs: Dict[str, Any]
    cost: float = 1.0
    keys: List[str] = field(default_factory=list)  # all run_keys the task writes

    def __post_init__(self):
        if not self.keys:
            self.keys = [self.key]


def csv_path(exp: str) -> Path:
    return RESULTS / exp / f"{exp}.csv"


def _execute(task: Task, columns: Sequence[str]) -> str:
    import importlib

    logging.disable(logging.WARNING)
    module_name, func_name = task.func.split(":")
    func = getattr(importlib.import_module(module_name), func_name)
    log = CsvLog(csv_path(task.exp), columns)
    started = perf_counter()
    try:
        out = func(**task.kwargs)
        rows = out if isinstance(out, list) else [out]
    except RunTimeout as exc:
        rows = [failed_row(task.kwargs["circ"], task.kwargs.get("extra") or {}, "timeout", str(exc))]
    except Exception as exc:  # recorded as a row, never dropped
        rows = [failed_row(task.kwargs["circ"], task.kwargs.get("extra") or {}, "error",
                           f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-1500:]}")]
    for i, row in enumerate(rows):
        row.setdefault("run_key", task.keys[i] if i < len(task.keys) else f"{task.key}#{i}")
        row["exp"] = task.exp
        row.setdefault("wall_time_s", perf_counter() - started)
        log.append(row)
    return task.key


def run_tasks(tasks: Sequence[Task], columns: Dict[str, Sequence[str]], jobs: int = 4,
              max_attempts: int = 2, priority: Optional[Callable[[Task], bool]] = None) -> None:
    """Run every task whose rows are not yet in its experiment's CSV, cheapest first.

    ``columns`` maps experiment id -> the leading CSV columns of that experiment.
    ``priority(task)`` returns a rank (lower first; True = 0, False/None = 1);
    tasks of equal rank run cheapest first.  The CSVs are re-read before tasks are
    submitted, so a task another runner finished in the meantime is skipped.
    """
    logs = {exp: CsvLog(csv_path(exp), cols) for exp, cols in columns.items()}

    def read_done():
        return set().union(*(log.done_keys() for log in logs.values())) if logs else set()

    def order(t: Task):
        # ``priority`` returns a rank (lower runs first) or a bool (True first)
        rank = priority(t) if priority else None
        rank = 1 if rank is None or rank is False else (0 if rank is True else rank)
        return (rank, t.cost)

    done = read_done()
    done_read_at = time.time()
    todo = sorted((t for t in tasks if not set(t.keys) <= done), key=order)
    logger.info("%s: %d task(s), %d already done, %d to run with %d worker(s)",
                ",".join(columns), len(tasks), len(tasks) - len(todo), len(todo), jobs)
    attempts: Dict[str, int] = {}
    started = time.time()
    finished = 0
    while todo:
        pending = list(todo)
        todo = []
        in_flight = {}
        try:
            with ProcessPoolExecutor(max_workers=jobs, max_tasks_per_child=25) as pool:
                while pending or in_flight:
                    while pending and len(in_flight) < jobs:
                        if time.time() - done_read_at > 120:
                            done, done_read_at = read_done(), time.time()
                        task = pending.pop(0)
                        if set(task.keys) <= done:
                            continue
                        attempts[task.key] = attempts.get(task.key, 0) + 1
                        future = pool.submit(_execute, task, list(columns[task.exp]))
                        in_flight[future] = task
                    finished_futures, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                    for future in finished_futures:
                        task = in_flight.pop(future)
                        future.result()
                        finished += 1
                        logger.info("done %d (%d left, %.0f s): %s", finished,
                                    len(pending) + len(in_flight), time.time() - started, task.key)
        except BrokenProcessPool:
            # A worker died (e.g. OOM kill): retry what was in flight, then give up on it.
            for task in list(in_flight.values()):
                if attempts[task.key] >= max_attempts:
                    for key in task.keys:
                        row = failed_row(task.kwargs["circ"], task.kwargs.get("extra") or {},
                                         "crashed", "worker process died (out of memory?)")
                        row.update(run_key=key, exp=task.exp)
                        logs[task.exp].append(row)
                    logger.error("%s crashed twice; recorded as status=crashed", task.key)
                else:
                    todo.append(task)
            todo += pending
            todo.sort(key=order)
            logger.warning("worker pool broke; restarting with %d task(s)", len(todo))


def run_key(*parts) -> str:
    return "|".join("" if p is None else str(p) for p in parts)


def cli(description: str, default_jobs: int = 4) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--jobs", type=int, default=default_jobs)
    parser.add_argument("--tiers", default="core", help="comma list: core,extra")
    parser.add_argument("--circuits", default="", help="comma list of circuit names")
    parser.add_argument("--max-ops", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=3600.0, help="seconds per run")
    parser.add_argument("--plot-only", action="store_true", help="only redraw the figures")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="list the tasks and exit")
    return parser


def setup_logging(exp: str) -> Path:
    out = RESULTS / exp
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(out / "run.log"), logging.StreamHandler(sys.stdout)],
    )
    return out


def geomean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if v is not None and v == v and v > 0]
    return math.exp(sum(math.log(v) for v in vals) / len(vals)) if vals else float("nan")
