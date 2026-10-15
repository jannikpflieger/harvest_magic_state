#!/usr/bin/env python3
"""Matched-IR space/time trade-off figure, normalized by per-benchmark lower bounds.

Only ``comparison_mode == "matched_ir"`` records are used; DASCOT never
appears (it has no matched-IR path).  For every benchmark:

* ``time_lb``  = dependency critical path of the matched Pauli-product DAG,
  one logical cycle per schedulable Pauli-product operation;
* ``space_lb`` = ``num_qubits + has_magic_state_operation`` (one data patch per
  logical qubit, plus one magic-state patch if the stream consumes a magic
  state; routing space is not counted).

See ``harvest.baselines.tradeoff`` for the exact definitions.  Each point is
``(logical_patches / space_lb, schedule_length / time_lb)``; stochastic
systems (PureMagic) contribute the arithmetic mean schedule length over their
successful trials of one configuration, matching ``summary.csv``.

Optionally (``--flasq-python``), the FLASQ cost model (Huggins et al.,
arXiv:2511.08508; Qualtran ``qualtran.surface_code.flasq``) prices the same
``.trans`` streams in its own isolated environment via
``flasq_price_trans.py``.  FLASQ is an idealized model, not a scheduler: it
assumes perfectly fluid ancilla and charges cultivation and reaction time,
which the matched-IR schedulers' cost model does not, so it is drawn as a
reference staircase (geometric mean over the shared circuits of the best
FLASQ depth attainable within each space budget), never as a competitor.
Results are cached under ``<result_dir>/flasq/``.

With several result directories (a resource sweep), every successful
configuration becomes a point, the non-dominated set in
``(logical_patches, schedule_length)`` is computed per compiler and circuit,
and only Pareto-optimal measured points are connected.

Example::

    PYTHONPATH=src python src/scripts/plot_sota_space_time_tradeoff.py \\
        results/sota_comparison_matched_ir
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shlex
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/harvest-matplotlib")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

from harvest.baselines.base import safe_name, stable_payload_hash
from harvest.baselines.tradeoff import (
    FLASQ_LABEL,
    TradeoffDataError,
    TradeoffPoint,
    aggregate_points,
    compiler_geomeans,
    config_geomeans,
    compute_bounds,
    flasq_points,
    flasq_time_floors,
    geomean_attainable_curve,
    rebase_time,
    load_raw_payloads,
    mark_pareto,
    normalize_point,
    shared_circuits,
)

# Colors are imported from the comparison driver so a compiler has the same
# color in every SOTA figure.  The existing palette has a weak red/green and
# red/orange pair, so identity is also carried by marker shape and labels.
from scripts.compare_sota_baselines import COLORS, DISPLAY

CSV_FIELDS = [
    "circuit",
    "compiler",
    "variant",
    "logical_patches",
    "schedule_length",
    "space_lb",
    "time_lb",
    "normalized_space",
    "normalized_time",
    "normalized_space_time",
    # provenance beyond the requested columns
    "config_id",
    "source_dir",
    "num_qubits",
    "logical_patches_min",
    "logical_patches_max",
    "schedule_length_min",
    "schedule_length_max",
    "trials_successful",
    "trials_total",
    "pareto_optimal",
    "in_shared_subset",
]
PLOT_ORDER = ["HARVEST", "Silva EAF", "PureMagic Bus", "PureMagic", "Litinski", "O3LS",
              "LSQCA", "TACO", "DASCOT Square Sparse", "DASCOT Compact"]
MARKERS = {"HARVEST": "o", "Silva EAF": "s", "PureMagic Bus": "^", "PureMagic": "D",
           "Litinski": "p", "O3LS": "h", "LSQCA": "*", "TACO": "<",
           "DASCOT Square Sparse": "v", "DASCOT Compact": "P"}
TIME_LABELS = {
    "ppr": "Time overhead (× critical path)",
    "tdepth": "Time overhead (× T-depth)",
    "flasq": "Time overhead (× FLASQ bound)",
}
ISO_PRODUCTS = (2, 4, 8, 16)
# Geomean label offsets (points, ha, va), chosen to clear per-benchmark marks.
# Sweep figures: labels sit at each compiler's lowest-space-time configuration.
LABEL_PLACEMENT_SWEEP = {
    "HARVEST": (-1, -8, "right", "top"),
    "Silva EAF": (0, -8, "center", "top"),
    "PureMagic Bus": (7, 0, "left", "center"),
    "PureMagic": (-7, 0, "right", "center"),
    "DASCOT Compact": (0, 6.5, "center", "bottom"),
    "DASCOT Square Sparse": (0, -8, "center", "top"),
    "Litinski": (-8, 0, "right", "center"),
    "O3LS": (0, 6.5, "center", "bottom"),
    "TACO": (7, 0, "left", "center"),
    "LSQCA": (0, 6.5, "center", "bottom"),
}
LABEL_PLACEMENT = {
    "HARVEST": (0, -8, "center", "top"),
    "Silva EAF": (0, -8, "center", "top"),
    "PureMagic Bus": (0, 6.5, "center", "bottom"),
    "PureMagic": (-7, 0, "right", "center"),
}
INK = "#222222"
MUTED = "#777777"
GUIDE = "#BBBBBB"
FLASQ_COLOR = "#3A3A3A"


def _display(baseline: str, variant: str) -> str:
    return DISPLAY.get((baseline, variant), variant)


def _load_source(path: Path):
    from harvest.compilation.circuit_analysis import pre_prep_circuit
    from harvest.compilation.qasm_loader import qasm_to_circuit

    return pre_prep_circuit(qasm_to_circuit(str(path)))


def load_points(
    result_dirs: Sequence[Path],
    include_unshared: bool,
    mode: str = "matched_ir",
    time_bound: str = "ppr",
    exclude_baselines: Sequence[str] = (),
    only_circuits: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    """Load, validate, bound, and normalize every point of one comparison mode."""
    from scripts.compare_sota_baselines import load_benchmark

    all_points: List[TradeoffPoint] = []
    failures = []
    notes: List[str] = []
    bounds_by_circuit = {}
    for result_dir in result_dirs:
        payloads = load_raw_payloads(result_dir)
        if exclude_baselines:
            kept = [p for p in payloads if p["result"].get("baseline") not in exclude_baselines]
            if len(kept) != len(payloads):
                notes.append(
                    f"{result_dir}: excluded {len(payloads) - len(kept)} record(s) of "
                    f"{', '.join(sorted(exclude_baselines))} (--exclude-baselines)"
                )
            payloads = kept
        if only_circuits is not None:
            payloads = [p for p in payloads if p["result"].get("circuit") in only_circuits]
        points, dir_failures, dir_notes = aggregate_points(
            payloads, str(result_dir), _display, mode=mode
        )
        if not points:
            raise TradeoffDataError(f"{result_dir}: no successful {mode} result")
        cache_path = Path(result_dir) / "bounds_cache.json"
        try:
            bounds_cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
        except json.JSONDecodeError:
            bounds_cache = {}
        before = len(bounds_cache)
        bounds = compute_bounds(
            payloads,
            result_dir,
            PROJECT_ROOT,
            load_benchmark,
            {p.circuit for p in points},
            mode=mode,
            time_bound=time_bound,
            load_source=_load_source,
            cache=bounds_cache,
        )
        if len(bounds_cache) != before:
            cache_path.write_text(json.dumps(bounds_cache, indent=1, sort_keys=True) + "\n")
        for circuit, bound in bounds.items():
            previous = bounds_by_circuit.setdefault(circuit, bound)
            if (previous.time_lb, previous.space_lb) != (bound.time_lb, bound.space_lb):
                raise TradeoffDataError(
                    f"{circuit}: lower bounds differ between result directories"
                )
        all_points.extend(normalize_point(p, bounds[p.circuit]) for p in points)
        failures.extend(dir_failures)
        notes.extend(dir_notes)

    compilers = [c for c in PLOT_ORDER if any(p.compiler == c for p in all_points)]
    compilers += sorted({p.compiler for p in all_points} - set(compilers))
    shared = shared_circuits(all_points, compilers)
    # In a resource sweep every configuration must be averaged over the same
    # circuits, so restrict to circuits that every configuration completed.
    configs: Dict[tuple, set] = defaultdict(set)
    for p in all_points:
        configs[(p.compiler, p.source_dir, p.config_id)].add(p.circuit)
    if len(configs) > len(compilers):
        core = sorted(set(shared).intersection(*configs.values()))
        dropped = sorted(set(shared) - set(core))
        if dropped:
            notes.append(
                f"resource sweep: {len(dropped)} circuit(s) not completed by every "
                f"configuration are excluded from the frontier: {', '.join(dropped)}"
            )
        shared = core
    for p in all_points:
        p.in_shared_subset = p.circuit in shared
    mark_pareto(all_points)
    plotted = all_points if include_unshared else [p for p in all_points if p.in_shared_subset]
    if not plotted:
        raise TradeoffDataError("no circuit is shared by all plotted systems")
    return {
        "all_points": all_points,
        "plotted": plotted,
        "compilers": compilers,
        "shared": shared,
        "failures": failures,
        "notes": notes,
        "bounds": bounds_by_circuit,
    }


FLASQ_CACHE_DIR = PROJECT_ROOT / "results" / "flasq_cache"


def load_flasq(args: argparse.Namespace, data: Dict[str, object]) -> Optional[Dict[str, object]]:
    """Price every shared circuit with FLASQ, reusing per-circuit cached results.

    FLASQ depends only on the circuit and the FLASQ settings, so results live in
    one project-wide cache (``results/flasq_cache``), one file per circuit keyed
    by its input SHA-256, the driver arguments and the driver source.  Adding
    circuits prices only the new ones; all figures share the cache.
    """
    if args.no_flasq:
        return None
    shared: List[str] = data["shared"]  # type: ignore[assignment]
    result_dir = args.result_dirs[0]
    if args.flasq_input == "source":
        sources = [data["bounds"][c].input_path for c in shared]  # type: ignore[index]
        if any(s is None for s in sources):
            raise TradeoffDataError("FLASQ source mode needs the recorded QASM inputs")
        inputs = [Path(s) for s in sources]
    else:
        inputs = [result_dir / "inputs" / f"{safe_name(c)}.trans" for c in shared]
    missing = [str(t) for t in inputs if not t.exists()]
    if missing:
        raise TradeoffDataError(f"FLASQ inputs are missing: {missing}")
    driver_args = [
        "--assumptions", args.flasq_assumptions,
        "--cost-model", args.flasq_cost_model,
        "--code-distance", str(args.flasq_code_distance),
        "--physical-error-rate", str(args.flasq_physical_error_rate),
        "--reaction-time-us", str(args.flasq_reaction_time_us),
        "--cycle-time-us", str(args.flasq_cycle_time_us),
    ]
    if args.flasq_assumptions == "cultivation":  # keeps other settings' cache keys unchanged
        driver_args += ["--v-cult", str(args.flasq_v_cult)]
    driver_sha = hashlib.sha256((SCRIPT_DIR / "flasq_price_trans.py").read_bytes()).hexdigest()

    def cache_file(path: Path) -> Path:
        key = stable_payload_hash(
            {
                "driver_args": driver_args,
                "driver": driver_sha,
                "input": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
        return FLASQ_CACHE_DIR / f"{safe_name(path.stem)}__{key}.json"

    todo = [path for path in inputs if not cache_file(path).exists()]
    if todo:
        if not args.flasq_python:
            print(f"note: FLASQ omitted ({len(todo)} circuit(s) not cached; pass --flasq-python)")
            return None
        command = [str(args.flasq_python), str(SCRIPT_DIR / "flasq_price_trans.py"),
                   *map(str, todo), *driver_args]
        print(f"FLASQ: pricing {len(todo)} uncached circuit(s), "
              f"{len(inputs) - len(todo)} reused from {FLASQ_CACHE_DIR}")
        proc = subprocess.run(command, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise TradeoffDataError(
                f"FLASQ driver failed ({proc.returncode}):\n{proc.stderr.strip()[-2000:]}"
            )
        fresh = json.loads(proc.stdout)
        FLASQ_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        for path in todo:
            record = fresh["circuits"][path.stem]
            entry = {
                "circuit_key": path.stem,
                "input": str(path),
                "command": shlex.join(command),
                "qualtran_commit": fresh.get("qualtran_commit"),
                "qualtran_path": fresh.get("qualtran_path"),
                "parameters": fresh["parameters"],
                "record": record,
            }
            cache_file(path).write_text(json.dumps(entry, indent=2, sort_keys=True) + "\n")
    entries = [json.loads(cache_file(path).read_text()) for path in inputs]
    commits = sorted({str(e.get("qualtran_commit")) for e in entries})
    payload = {
        "parameters": entries[0]["parameters"],
        "qualtran_commit": commits[0] if len(commits) == 1 else "mixed: " + ", ".join(commits),
        "circuits": {e["circuit_key"]: e["record"] for e in entries},
    }
    points = flasq_points(payload, data["bounds"], shared, str(FLASQ_CACHE_DIR))  # type: ignore[arg-type]
    return {
        "payload": payload,
        "points": points,
        "curve": geomean_attainable_curve(points, shared),
        "cache": FLASQ_CACHE_DIR,
    }


def write_csv(path: Path, points: Sequence[TradeoffPoint]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for p in sorted(points, key=lambda q: (q.circuit, q.compiler, q.config_id)):
            writer.writerow(p.to_row())


def args_input(payload: Dict[str, object]) -> str:
    kinds = {rec.get("input_kind", "ppr_lowered") for rec in payload["circuits"].values()}  # type: ignore[union-attr]
    return ",".join(sorted(kinds))


def print_flasq_summary(flasq: Dict[str, object], geomeans: Dict[str, Dict[str, float]]) -> None:
    payload = flasq["payload"]  # type: ignore[index]
    params = payload["parameters"]
    curve = flasq["curve"]  # type: ignore[index]
    print(
        f"FLASQ (input={args_input(payload)}, {params.get('assumptions', 'paper')} assumptions, "
        f"{params['cost_model']} costs, d={params['code_distance']}, "
        f"p={params['physical_error_rate']}, t_r={params['reaction_time_us']}us, "
        f"qualtran {payload.get('qualtran_commit', 'unknown')[:10]}) from {flasq['cache']}"
    )
    for circuit, rec in sorted(payload["circuits"].items()):
        print(
            f"  {circuit}: D={rec['measurement_depth']:g} Q={rec['algorithmic_qubits']} "
            f"V={rec['computational_volume']:.1f} "
            f"t_react={rec['t_react_logical_timesteps']:.3f} v_cult={rec['v_cult_blocks']:.2f} "
            f"steps/T-layer={rec.get('logical_timesteps_per_measurement', rec['t_react_logical_timesteps']):.3f} "
            f"frontier {len(rec['frontier'])} pts"
        )
    print(f"  geomean staircase: {curve[0][0]:.2f}x space -> {curve[0][1]:.2f}x time ... "
          f"{curve[-1][0]:.2f}x space -> {curve[-1][1]:.2f}x time (floor)")
    for compiler, g in geomeans.items():
        attainable = [y for x, y in curve if x <= g["normalized_space"] + 1e-12]
        if attainable:
            print(f"  at {compiler}'s geomean space ({g['normalized_space']:.2f}x) FLASQ "
                  f"attains {attainable[-1]:.2f}x time (compiler: {g['normalized_time']:.2f}x)")


def print_config_summary(configs: Dict[str, List[Dict[str, object]]]) -> None:
    if sum(len(v) for v in configs.values()) <= len(configs):
        return
    print("per-configuration geomeans over shared circuits (space x, time x, space-time x):")
    for compiler, entries in configs.items():
        for e in entries:
            if e.get("duplicate_of"):
                print(f"  {compiler:14s} {Path(str(e['source_dir'])).name:24s} identical results to "
                      f"{e['duplicate_of']} (setting has no effect) - not plotted twice")
                continue
            if not e["complete"]:
                print(f"  {compiler:14s} {e['source_dir']} [{e['config_id']}]: INCOMPLETE "
                      f"({e['n']} shared circuits) - not plotted")
                continue
            tag = "pareto" if e.get("pareto") else "dominated"
            print(f"  {compiler:14s} {Path(str(e['source_dir'])).name:24s} "
                  f"{e['normalized_space']:.2f}  {e['normalized_time']:.2f}  "
                  f"{e['normalized_space_time']:.2f}  {tag}")


def write_config_csv(path: Path, configs: Dict[str, List[Dict[str, object]]]) -> None:
    fields = ["compiler", "source_dir", "config_id", "n", "complete", "normalized_space",
              "normalized_time", "normalized_space_time", "pareto"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for compiler, entries in configs.items():
            for e in entries:
                writer.writerow({"compiler": compiler, **e})


def print_summary(data: Dict[str, object], geomeans: Dict[str, Dict[str, float]]) -> None:
    all_points: List[TradeoffPoint] = data["all_points"]  # type: ignore[assignment]
    compilers: List[str] = data["compilers"]  # type: ignore[assignment]
    shared: List[str] = data["shared"]  # type: ignore[assignment]
    print("== Matched-IR space/time trade-off ==")
    for note in data["notes"]:  # type: ignore[union-attr]
        print(f"note: {note}")
    print(f"systems: {', '.join(compilers)}")
    print(f"circuits shared by all systems ({len(shared)}): {', '.join(shared) or '-'}")
    by_circuit = defaultdict(set)
    for p in all_points:
        by_circuit[p.circuit].add(p.compiler)
    for circuit in sorted(set(by_circuit) - set(shared)):
        missing = sorted(set(compilers) - by_circuit[circuit])
        print(f"  not shared: {circuit} (no successful result for {', '.join(missing)})")
    for f in data["failures"]:  # type: ignore[union-attr]
        status = "FAILED" if f.trials_failed == f.trials_total else "partially failed"
        print(
            f"  {status}: {f.compiler} on {f.circuit} "
            f"({f.trials_failed}/{f.trials_total} trials): {f.errors[0][:110]}"
        )
    print("lower bounds:")
    for circuit, b in sorted(data["bounds"].items()):  # type: ignore[union-attr]
        print(
            f"  {circuit}: n={b.num_qubits} ops={b.num_operations} "
            f"time_lb={b.time_lb} space_lb={b.space_lb} [{b.source}]"
        )
    below = [p for p in all_points if p.normalized_time < 1.0]
    for p in below:
        print(f"  note: {p.compiler} on {p.circuit} beats time_lb ({p.normalized_time:.2f}x)")
    for compiler in compilers:
        pts = [p for p in all_points if p.compiler == compiler and p.in_shared_subset]
        tight = sum(math.isclose(p.normalized_time, 1.0) for p in pts)
        print(f"  {compiler}: meets time_lb exactly on {tight}/{len(pts)} shared circuits")
    print("geometric means over shared circuits (space x, time x, space-time x):")
    for compiler, g in geomeans.items():
        print(
            f"  {compiler:14s} {g['normalized_space']:.2f}  "
            f"{g['normalized_time']:.2f}  {g['normalized_space_time']:.2f}  (n={g['n']})"
        )


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Linux Libertine O",
                "Libertinus Serif",
                "Times New Roman",
                "STIXGeneral",
                "DejaVu Serif",
            ],
            "mathtext.fontset": "stix",
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 600,
        }
    )


def _limits(values: Sequence[float], log_scale: bool) -> tuple:
    lo, hi = min(min(values), 1.0), max(values)
    if log_scale:
        return lo / 1.25, hi * 1.35
    pad = 0.06 * (hi - lo or 1.0)
    return max(0.0, lo - pad), hi + pad


def _ticks(lo: float, hi: float) -> List[float]:
    candidates = [0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256]
    return [t for t in candidates if lo <= t <= hi]


def plot(
    data: Dict[str, object],
    geomeans: Dict[str, Dict[str, float]],
    stem: Path,
    log_scale: Optional[bool],
    width_in: float,
    height_in: float,
    flasq: Optional[Dict[str, object]] = None,
    configs: Optional[Dict[str, List[Dict[str, object]]]] = None,
    time_bound: str = "ppr",
    per_benchmark: str = "pareto",
    iso_contours: bool = False,
    label_overrides: Optional[Dict[str, tuple]] = None,
    ax=None,
    legend: bool = True,
    xlabel: str = "Space overhead (× minimum)",
    ylabel: Optional[str] = None,
    direct_labels: bool = True,
    x_ticks: Optional[Sequence[float]] = None,
):
    """Draw the trade-off figure.

    Standalone (``ax=None``) it writes ``stem``.pdf/.png and returns the paths;
    given an ``ax`` (e.g. one panel of a composite figure) it draws there and
    returns the legend handles instead of saving.
    """
    _style()
    configs = configs or {}
    # With a resource sweep, each compiler is one line through the geomeans of
    # its Pareto-optimal configurations; its label sits at the configuration
    # with the lowest space-time product (a measured configuration, not a blend).
    multi_config = any(
        sum(e["complete"] for e in entries) > 1 for entries in configs.values()
    )
    if multi_config:
        geomeans = {}
        for compiler, entries in configs.items():
            front = [e for e in entries if e["complete"] and e.get("pareto")]
            if front:
                geomeans[compiler] = min(front, key=lambda e: e["normalized_space_time"])
    points: List[TradeoffPoint] = data["plotted"]  # type: ignore[assignment]
    compilers: List[str] = data["compilers"]  # type: ignore[assignment]
    complete = [e for entries in configs.values() for e in entries if e["complete"]]
    # Per-benchmark observations: with a sweep, by default only those of each
    # compiler's Pareto-optimal configurations (the ones its line passes through).
    if not multi_config or per_benchmark == "all":
        scatter_points = list(points)
    elif per_benchmark == "pareto":
        front_keys = {
            (compiler, e["source_dir"], e["config_id"])
            for compiler, entries in configs.items()
            for e in entries
            if e["complete"] and e.get("pareto")
        }
        scatter_points = [p for p in points if (p.compiler, p.source_dir, p.config_id) in front_keys]
    else:
        scatter_points = []
    xs = [p.normalized_space for p in scatter_points] + [g["normalized_space"] for g in geomeans.values()]
    ys = [p.normalized_time for p in scatter_points] + [g["normalized_time"] for g in geomeans.values()]
    xs += [e["normalized_space"] for e in complete]
    ys += [e["normalized_time"] for e in complete]
    curve = list(flasq["curve"]) if flasq else []  # type: ignore[index]
    # The axes are sized to the compilers; the FLASQ staircase may enter from
    # above (its single-ancilla end is far slower) but its floor stays visible.
    y_top = max(ys)
    ys += [y for _, y in curve if y <= y_top]
    xs += [x for x, y in curve if y <= y_top]
    if log_scale is None:  # log only if the data span at least ~4x on some axis
        log_scale = max(max(xs) / min(xs), max(ys) / min(ys)) >= 4.0
    xlim, ylim = _limits(xs, log_scale), _limits(ys, log_scale)

    own_figure = ax is None
    if own_figure:
        fig, ax = plt.subplots(figsize=(width_in, height_in))
    if log_scale:
        ax.set_xscale("log")
        ax.set_yscale("log")

    # Guides: the ideal point lies at (1, 1); iso-space-time contours x*y=c.
    ax.axvline(1.0, color=GUIDE, linewidth=0.6, zorder=0)
    ax.axhline(1.0, color=GUIDE, linewidth=0.6, zorder=0)
    for product in ISO_PRODUCTS if iso_contours else ():
        x0, x1 = max(xlim[0], product / ylim[1]), min(xlim[1], product / ylim[0])
        if x0 >= x1:
            continue
        n = 200
        if log_scale:
            grid = [x0 * (x1 / x0) ** (i / (n - 1)) for i in range(n)]
        else:
            grid = [x0 + (x1 - x0) * i / (n - 1) for i in range(n)]
        ax.plot(grid, [product / x for x in grid], color=GUIDE, linewidth=0.5,
                linestyle=(0, (1, 2)), zorder=0)
        # Label each contour where it enters the plot (top edge, else left edge).
        if product / ylim[1] >= xlim[0]:
            anchor, offset = (product / ylim[1], ylim[1]), (2, -7)
        else:
            anchor, offset = (xlim[0], product / xlim[0]), (3, -2)
        ax.annotate(f"{product}×", anchor, xytext=offset, textcoords="offset points",
                    fontsize=6, color=MUTED, ha="left", va="center")

    for compiler in compilers:
        color = COLORS.get(compiler, MUTED)
        marker = MARKERS.get(compiler, "o")
        own = [p for p in scatter_points if p.compiler == compiler]
        emphasized = compiler == "HARVEST"
        ax.scatter(
            [p.normalized_space for p in own],
            [p.normalized_time for p in own],
            s=(9 if emphasized else 7) if multi_config else (14 if emphasized else 11),
            marker=marker,
            facecolor=color,
            edgecolor="none",
            alpha=0.18 if multi_config else 0.35,
            zorder=3 if emphasized else 2,
            label=None,
        )
        if multi_config:
            entries = [e for e in configs.get(compiler, []) if e["complete"]]
            front = sorted((e for e in entries if e.get("pareto")),
                           key=lambda e: e["normalized_space"])
            dominated = [e for e in entries if not e.get("pareto")]
            ax.plot([e["normalized_space"] for e in front], [e["normalized_time"] for e in front],
                    color=color, linewidth=1.3 if emphasized else 1.0, zorder=4,
                    marker=marker, markersize=3.6, markeredgecolor="white",
                    markeredgewidth=0.4)
            ax.scatter([e["normalized_space"] for e in dominated],
                       [e["normalized_time"] for e in dominated], s=12, marker=marker,
                       facecolor="none", edgecolor=color, linewidth=0.7, zorder=4)

    if flasq:
        # Per-circuit FLASQ staircases are in the CSV; only the geomean is drawn.
        cx = [x for x, _ in curve] + [xlim[1]]
        cy = [y for _, y in curve] + [curve[-1][1]]
        ax.step(cx, cy, where="post", color=FLASQ_COLOR, linewidth=1.2, zorder=4)
        # FLASQ is identified by the legend; a direct label collided with data.

    # Geometric-mean markers, directly labelled.
    for compiler in compilers:
        g = geomeans.get(compiler)
        if g is None:
            continue
        emphasized = compiler == "HARVEST"
        ax.scatter(
            [g["normalized_space"]],
            [g["normalized_time"]],
            s=58 if emphasized else 40,
            marker=MARKERS.get(compiler, "o"),
            facecolor=COLORS.get(compiler, MUTED),
            edgecolor="white" if not emphasized else INK,
            linewidth=0.9 if emphasized else 0.7,
            zorder=6 if emphasized else 5,
        )
        placement = {**(LABEL_PLACEMENT_SWEEP if multi_config else LABEL_PLACEMENT),
                     **(label_overrides or {})}
        if not direct_labels:
            continue
        dx, dy, ha, va = placement.get(compiler, (0, 6.5, "center", "bottom"))
        ax.annotate(
            compiler,
            (g["normalized_space"], g["normalized_time"]),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=7.5 if emphasized else 7,
            fontweight="bold" if emphasized else "normal",
            color=INK,
            zorder=7,
        )

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    if log_scale:
        fmt = FuncFormatter(lambda v, _: f"{v:g}×")
        for axis, lim in ((ax.xaxis, xlim), (ax.yaxis, ylim)):
            ticks = _ticks(*lim)
            if axis is ax.xaxis and x_ticks is not None:
                ticks = [t for t in x_ticks if lim[0] <= t <= lim[1]]
            axis.set_major_locator(FixedLocator(ticks))
            axis.set_minor_locator(NullLocator())
            axis.set_major_formatter(fmt)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel if ylabel is not None else TIME_LABELS[time_bound])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    # Compact legend for marker shapes (geomean markers are directly labelled).
    handles = [
        plt.Line2D([], [], linestyle="none", marker=MARKERS.get(c, "o"),
                   markerfacecolor=COLORS.get(c, MUTED), markeredgecolor="none",
                   markersize=4.5, label=c)
        for c in compilers
    ]
    if scatter_points:
        handles.append(plt.Line2D([], [], linestyle="none", marker="o", markerfacecolor=MUTED,
                                  markeredgecolor="none", alpha=0.35, markersize=3.5,
                                  label="per benchmark"))
    if multi_config and any(not e.get("pareto") for e in complete):
        handles.append(plt.Line2D([], [], linestyle="none", marker="o", markerfacecolor="none",
                                  markeredgecolor=MUTED, markersize=3.5,
                                  label="dominated config."))
    if flasq:
        handles.append(plt.Line2D([], [], color=FLASQ_COLOR, linewidth=1.2,
                                  label="FLASQ model"))
    if not own_figure:
        return handles
    if legend:
        ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.01),
                  frameon=False, handletextpad=0.2, borderaxespad=0.0, columnspacing=0.9,
                  labelspacing=0.25, ncol=3, handlelength=1.2)

    fig.tight_layout(pad=0.2)
    outputs = [stem.with_suffix(".pdf"), stem.with_suffix(".png")]
    fig.savefig(outputs[0], bbox_inches="tight")
    fig.savefig(outputs[1], dpi=600, bbox_inches="tight")
    plt.close(fig)
    return outputs


def _parse_labels(items: Sequence[str]) -> Dict[str, tuple]:
    """``NAME=dx,dy,ha,va`` -> {NAME: (dx, dy, ha, va)} (offsets in points)."""
    overrides = {}
    for item in items:
        try:
            name, spec = item.rsplit("=", 1)
            dx, dy, ha, va = spec.split(",")
            overrides[name] = (float(dx), float(dy), ha, va)
        except ValueError as exc:
            raise TradeoffDataError(f"bad --label {item!r}; expected NAME=dx,dy,ha,va") from exc
    return overrides


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "result_dirs",
        nargs="*",
        type=Path,
        default=[PROJECT_ROOT / "results" / "sota_comparison_matched_ir"],
        help="One or more matched_ir result directories (several = resource sweep).",
    )
    parser.add_argument("--output-dir", type=Path,
                        help="Defaults to the first result directory.")
    parser.add_argument("--output-stem", default="space_time_tradeoff")
    parser.add_argument("--mode", choices=["matched_ir", "native_end_to_end"], default="matched_ir",
                        help="Comparison regime; records of other regimes are never mixed in.")
    parser.add_argument("--time-bound", choices=["flasq", "ppr", "tdepth"], default="flasq",
                        help="flasq (default): FLASQ's best attainable depth per circuit, so "
                             "FLASQ is 1x; ppr: critical path of the matched Pauli-product "
                             "stream (matched_ir only); tdepth: T-depth of the Clifford+T source.")
    parser.add_argument("--exclude-baselines", default="",
                        help="Comma-separated baseline ids (e.g. dascot) left out of this "
                             "figure; the exclusion is reported, records stay on disk.")
    parser.add_argument("--per-benchmark", choices=["pareto", "all", "none"], default="pareto",
                        help="Per-benchmark points in sweep figures: those of Pareto-optimal "
                             "configurations (default), of all configurations, or none.")
    parser.add_argument("--label", action="append", default=[], metavar="NAME=dx,dy,ha,va",
                        help="Override a compiler's direct-label offset (points) and alignment "
                             "for this figure, e.g. 'HARVEST=-6,6,right,bottom'. Repeatable.")
    parser.add_argument("--iso-contours", action="store_true",
                        help="Draw dotted iso-space-time contours (off by default).")
    parser.add_argument("--circuits", default="",
                        help="Comma-separated circuits (or @file, one per line) to restrict "
                             "the figure to; the subset is reported, records stay on disk.")
    parser.add_argument("--include-unshared", action="store_true",
                        help="Also plot circuits that not every system completed.")
    scale = parser.add_mutually_exclusive_group()
    scale.add_argument("--log", dest="log_scale", action="store_true", default=None)
    scale.add_argument("--linear", dest="log_scale", action="store_false")
    flasq = parser.add_argument_group("FLASQ reference model")
    flasq.add_argument("--flasq-python", type=Path,
                       help="Python of an environment with qualtran.surface_code.flasq "
                            "(e.g. /tmp/harvest-flasq/venv/bin/python).")
    flasq.add_argument("--no-flasq", action="store_true", help="Omit the FLASQ reference.")
    flasq.add_argument("--flasq-input", choices=["source", "ppr"], default="source",
                       help="source: the Clifford+T circuit all systems started from "
                            "(FLASQ's native input); ppr: the matched stream lowered back "
                            "to CNOT ladders (overstates FLASQ's cost; diagnosis only).")
    flasq.add_argument("--flasq-v-cult", type=float, default=3.0,
                       help="Cultivation blocks per T for --flasq-assumptions cultivation.")
    flasq.add_argument("--flasq-assumptions", choices=["matched", "cultivation", "paper"],
                       default="matched",
                       help="matched: free magic states and one step per T layer, as the "
                            "matched-IR schedulers assume; paper: FLASQ/RushHour physics.")
    flasq.add_argument("--flasq-cost-model", choices=["optimistic", "conservative"],
                       default="optimistic")
    flasq.add_argument("--flasq-code-distance", type=int, default=13)
    flasq.add_argument("--flasq-physical-error-rate", type=float, default=1e-3)
    flasq.add_argument("--flasq-reaction-time-us", type=float, default=10.0)
    flasq.add_argument("--flasq-cycle-time-us", type=float, default=1.0)
    parser.add_argument("--width", type=float, default=3.33, help="inches (one column)")
    parser.add_argument("--height", type=float, default=2.5, help="inches")
    return parser.parse_args(argv)


def prepare(args: argparse.Namespace) -> Dict[str, object]:
    """Load, bound, FLASQ-normalize and summarize; shared by main() and composites."""
    time_bound = args.time_bound
    if args.mode != "matched_ir" and time_bound == "ppr":
        raise TradeoffDataError("the PPR critical path is only a common bound in matched_ir")
    # Points are first bounded by the regime's structural bound (validated against
    # the IR); with --time-bound flasq they are then rebased onto FLASQ's floor.
    structural = time_bound if time_bound != "flasq" else (
        "ppr" if args.mode == "matched_ir" else "tdepth"
    )
    excluded = [b.strip() for b in args.exclude_baselines.split(",") if b.strip()]
    only = None
    if args.circuits:
        spec = args.circuits
        items = Path(spec[1:]).read_text().split() if spec.startswith("@") else spec.split(",")
        only = {c.strip() for c in items if c.strip()}
        print(f"restricted to {len(only)} circuit(s) (--circuits)")
    data = load_points(args.result_dirs, args.include_unshared, args.mode, structural, excluded,
                       only)
    flasq = load_flasq(args, data)
    if time_bound == "flasq":
        if not flasq:
            raise TradeoffDataError(
                "--time-bound flasq needs FLASQ results (pass --flasq-python, or choose "
                "--time-bound ppr/tdepth with --no-flasq)"
            )
        if args.include_unshared:
            raise TradeoffDataError("FLASQ is only evaluated on shared circuits; drop --include-unshared")
        floors = flasq_time_floors(flasq["points"])  # type: ignore[arg-type]
        shared_set = set(data["shared"])  # type: ignore[arg-type]
        rebase_time([p for p in data["all_points"] if p.circuit in shared_set], floors)  # type: ignore[union-attr]
        rebase_time(flasq["points"], floors)  # type: ignore[arg-type]
        flasq["curve"] = geomean_attainable_curve(flasq["points"], data["shared"])  # type: ignore[arg-type,index]
        print(f"time normalized by FLASQ's per-circuit floor (FLASQ = 1x); {len(floors)} circuits")
    geomeans = compiler_geomeans(
        [p for p in data["all_points"] if p.circuit in set(data["shared"])], data["shared"]  # type: ignore
    )
    configs = config_geomeans(data["all_points"], data["shared"])  # type: ignore[arg-type]
    print_summary(data, geomeans)
    print_config_summary(configs)
    if flasq:
        print_flasq_summary(flasq, geomeans)
    return {"data": data, "geomeans": geomeans, "flasq": flasq, "configs": configs,
            "time_bound": time_bound}


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    prepared = prepare(args)
    data, geomeans, flasq, configs, time_bound = (
        prepared["data"], prepared["geomeans"], prepared["flasq"], prepared["configs"],
        prepared["time_bound"],
    )
    output_dir = args.output_dir or args.result_dirs[0]
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / args.output_stem
    csv_path = stem.with_suffix(".csv")
    rows = list(data["plotted"]) + (list(flasq["points"]) if flasq else [])  # type: ignore
    write_csv(csv_path, rows)
    write_config_csv(stem.parent / f"{stem.name}_configs.csv", configs)
    outputs = plot(data, geomeans, stem, args.log_scale, args.width, args.height, flasq, configs,
                   time_bound, args.per_benchmark, args.iso_contours,
                   _parse_labels(args.label))
    for path in [*outputs, csv_path]:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
