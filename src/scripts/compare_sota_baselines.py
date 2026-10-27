#!/usr/bin/env python3
"""Reproducible HARVEST/Silva/PureMagic/DASCOT comparison driver.

Raw trial records are the source of truth.  Summaries and plots are derived
from them and never silently remove failures or mix comparison regimes.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import importlib.metadata
import json
import logging
import math
import os
import platform
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional, Sequence

# Keep Matplotlib's cache out of read-only home directories in containers and
# CI.  This must be set before importing Matplotlib.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/harvest-matplotlib")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from harvest.baselines.base import (
    BaselineResult,
    BaselineRunConfig,
    deterministic_result_name,
    safe_name,
    write_result,
)
from harvest.baselines.dascot import DASCOTAdapter, UnsupportedDASCOTInput, validate_wisq_qasm
from harvest.baselines.harvest import HarvestAdapter
from harvest.baselines.metrics import aggregate_results, pairwise_geometric_means
from harvest.baselines.puremagic import (
    PureMagicAdapter,
    UnsupportedPureMagicIR,
    write_trans_file,
)
from harvest.baselines.silva import SilvaEAFAdapter
from harvest.baselines.cultivation import MAGIC_MODELS, STOCHASTIC_MODELS
from harvest.eval.manifest import append_run_row, run_row
from harvest.compilation.circuit_analysis import convert_rx_ry_to_rz, pre_prep_circuit
from harvest.compilation.pauli_block_conversion import convert_to_PCB, create_dag
from harvest.compilation.qasm_loader import find_qasm_files, qasm_to_circuit
from harvest.faults import FAULTS
from harvest.layout.presets import nxm_ring_layout_single_qubits

logger = logging.getLogger("SOTABaselineComparison")

DISPLAY = {
    ("harvest", "harvest"): "HARVEST",
    ("harvest", "harvest_circuit_aware"): "HARVEST (circuit-aware placement)",
    ("silva", "silva_eaf"): "Silva EAF",
    ("silva", "silva_eaf_matched_placement"): "Silva EAF (matched placement)",
    ("silva", "silva_eaf_pruned"): "Silva EAF + pruning",
    ("puremagic", "puremagic_bus"): "PureMagic Bus",
    ("puremagic", "puremagic"): "PureMagic",
    ("dascot", "dascot_square_sparse"): "DASCOT Square Sparse",
    ("dascot", "dascot_compact"): "DASCOT Compact",
    ("dascot", "matched_architecture"): "DASCOT Matched Architecture",
    # Multi-configuration baselines: every variant is one configuration of the
    # same compiler, so the trade-off plot draws them as one frontier.
    ("litinski", "litinski_compact"): "Litinski",
    ("litinski", "litinski_intermediate"): "Litinski",
    ("litinski", "litinski_fast"): "Litinski",
    ("o3ls", "o3ls_min"): "O3LS",
    ("o3ls", "o3ls_medium"): "O3LS",
    ("o3ls", "o3ls_large"): "O3LS",
    ("taco", "taco_k1"): "TACO",
    ("taco", "taco_k2"): "TACO",
    ("taco", "taco_k4"): "TACO",
    ("lsqca", "lsqca_point_b1"): "LSQCA",
    ("lsqca", "lsqca_point_b2"): "LSQCA",
    ("lsqca", "lsqca_line_b1"): "LSQCA",
    ("lsqca", "lsqca_line_b2"): "LSQCA",
    ("lsqca", "lsqca_line_b4"): "LSQCA",
}
COLORS = {
    "HARVEST": "#4C72B0",
    "HARVEST (circuit-aware placement)": "#4C72B0",
    "Silva EAF": "#DD8452",
    "Silva EAF (matched placement)": "#DD8452",
    "Silva EAF + pruning": "#E8A87C",
    "PureMagic Bus": "#55A868",
    "PureMagic": "#C44E52",
    "DASCOT Square Sparse": "#8172B3",
    "DASCOT Compact": "#937860",
    "DASCOT Matched Architecture": "#B07AA1",
    "Litinski": "#DA8BC3",
    "O3LS": "#CCB974",
    "LSQCA": "#64B5CD",
    "TACO": "#8C8C8C",
}


def discover_benchmarks(spec: str, max_circuits: Optional[int] = None) -> List[Path]:
    """Reuse the repository QASM discovery conventions for a file/dir/family."""
    path = Path(spec)
    if not path.exists():
        path = PROJECT_ROOT / "benchmark_circuits" / "qasm" / spec
    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = [Path(item) for item in find_qasm_files(str(path))]
    else:
        raise FileNotFoundError(f"benchmark path or family not found: {spec}")
    files = sorted(item.resolve() for item in files)
    return files[:max_circuits] if max_circuits else files


def load_benchmark(path: Path):
    """Exact Clifford+T QASM -> (circuit, Pauli-product DAG)."""
    circuit = qasm_to_circuit(str(path))
    circuit = pre_prep_circuit(circuit)
    if "rx" in circuit.count_ops() or "ry" in circuit.count_ops():
        circuit = convert_rx_ry_to_rz(circuit)
    pcb = convert_to_PCB(circuit, verbose=False)
    return circuit, create_dag(pcb)


LAYOUT_PRESETS = ("ring", "ring_large_spacing", "one_side_magic", "fixed_magic", "blocks4",
                  "blocks4_one_side_magic", "blocks4_sparse_ring", "dense")


def build_layout(
    num_qubits: int,
    rows: Optional[int],
    cols: Optional[int],
    dag,
    placement: str,
    seed: int,
    preset: str = "ring",
    magic_count: int = 2,
):
    """Build the HARVEST/Silva layout.

    ``preset`` is the resource knob for space/time sweeps; every preset uses
    one-tile patches (so DASCOT matched_architecture stays applicable):
    ``ring`` (default) data grid with 1-tile spacing and a magic ring;
    ``ring_large_spacing`` 2-tile spacing; ``one_side_magic`` magic on the top
    edge only; ``fixed_magic`` exactly ``magic_count`` magic patches on the top
    edge; ``blocks4`` dense 2x2 data blocks (``blocks4_one_side_magic``: magic on
    the top edge only; ``blocks4_sparse_ring``: as many magic patches, spread
    over all four edges); ``dense`` the ~2.25 tiles/qubit
    comb layout with ``magic_count`` magic ports (``presets.dense_comb_layout``).
    """
    from harvest.layout import presets

    if preset == "dense":
        return presets.dense_comb_layout(num_qubits, magic_count)
    if placement == "circuit_aware":
        from harvest.synthesis.placement import PlacementConfig
        from harvest.synthesis.synthesizer import StaticLayoutSynthesizer

        synthesizer = StaticLayoutSynthesizer(
            placement_config=PlacementConfig(
                alpha=1.0, beta=0.5, max_swap_iterations=100, seed=seed
            )
        )
        engine, _ = synthesizer.synthesize(dag)
        return engine

    from harvest.layout import presets

    slots_per_cell = 4 if preset.startswith("blocks4") else 1
    cells = math.ceil(num_qubits / slots_per_cell)
    resolved_cols = cols or max(1, math.ceil(math.sqrt(cells)))
    resolved_rows = rows or max(1, math.ceil(cells / resolved_cols))
    if resolved_cols * resolved_rows * slots_per_cell < num_qubits:
        raise ValueError(
            f"layout {resolved_rows}x{resolved_cols} ({preset}) has fewer data patches "
            f"than {num_qubits} qubits"
        )
    if preset == "ring":
        return nxm_ring_layout_single_qubits(resolved_cols, resolved_rows)
    if preset == "ring_large_spacing":
        return presets.nxm_ring_layout_single_qubits_large_spacing(resolved_cols, resolved_rows)
    if preset == "one_side_magic":
        return presets.nxm_one_side_magic_layout_single_qubits(resolved_cols, resolved_rows)
    if preset == "fixed_magic":
        return presets.nxm_fixed_magic_count_layout_single_qubits(
            resolved_cols, resolved_rows, magic_count
        )
    if preset == "blocks4":
        return presets.blocks_of_four_qubit_patches(resolved_cols, resolved_rows)
    if preset == "blocks4_one_side_magic":
        return presets.blocks_of_four_qubit_patches(resolved_cols, resolved_rows, magic_side="top")
    if preset == "blocks4_sparse_ring":
        # as many magic patches as blocks4_one_side_magic has on its top edge
        # (3 * cols + 1), spread evenly over all four edges
        return presets.blocks_of_four_qubit_patches(
            resolved_cols, resolved_rows, magic_ring_count=3 * resolved_cols + 1
        )
    raise ValueError(f"unknown layout preset {preset!r}")


def raw_result_path(raw_dir: Path, config: BaselineRunConfig) -> Path:
    return raw_dir / deterministic_result_name(config)


# Fields that define *what* a trial computed.  Paths, timeouts, executable
# locations and the recorded upstream version are provenance: they stay in every
# raw record but do not invalidate a stored result (use --force-rerun after
# changing a baseline's code).
RESUME_IDENTITY_FIELDS = (
    "baseline",
    "variant",
    "circuit",
    "comparison_mode",
    "num_qubits",
    "trial",
    "seed",
    "options",
    "input_sha256",
)


@dataclass
class ResumePolicy:
    retry_failed: bool = True
    force_rerun: frozenset = frozenset()


RESUME_POLICY = ResumePolicy()


def should_resume(
    path: Path,
    config: Optional[BaselineRunConfig] = None,
    *,
    retry_failed: Optional[bool] = None,
) -> bool:
    """Reuse a stored trial whose identity fields match ``config``.

    Completed trials are always reused.  Failed trials are reused only when
    ``retry_failed`` is false (``--keep-failed``); dry-run placeholders never.
    """
    retry = RESUME_POLICY.retry_failed if retry_failed is None else retry_failed
    if not path.exists():
        return False
    if config is not None and config.baseline in RESUME_POLICY.force_rerun:
        return False
    try:
        payload = json.loads(path.read_text())
        result = payload.get("result", {})
        if result.get("completed") is not True:
            if retry or str(result.get("error", "")).startswith("dry-run"):
                return False
        if config is None:
            return True
        stored, current = payload.get("config", {}), config.serializable()
        return all(stored.get(key) == current.get(key) for key in RESUME_IDENTITY_FIELDS)
    except (OSError, json.JSONDecodeError):
        return False


def _git_commit() -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.stdout.strip() if proc.returncode == 0 else "unknown"


def _git_dirty() -> Optional[bool]:
    proc = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return bool(proc.stdout.strip()) if proc.returncode == 0 else None


def _package_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _binary_version(executable: Optional[str]) -> Optional[str]:
    if not executable:
        return None
    try:
        proc = subprocess.run(
            [executable, "--version"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    text = (proc.stdout or proc.stderr).strip()
    return text.splitlines()[0] if text else None


def write_environment(output_dir: Path, args: argparse.Namespace) -> None:
    dependencies = {
        name: _package_version(name)
        for name in ("qiskit", "networkx", "numpy", "matplotlib", "pandas", "wisq")
    }
    payload = {
        "harvest_git_commit": _git_commit(),
        "harvest_git_dirty": _git_dirty(),
        "puremagic_version": _binary_version(args.puremagic_bin),
        "puremagic_expected_snapshot": args.puremagic_version,
        "dascot_wisq_version": _binary_version(args.wisq_bin)
        or dependencies.get("wisq"),
        "python_version": sys.version,
        "os": platform.platform(),
        "dependencies": dependencies,
        "argv": sys.argv,
        "arguments": vars(args),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "environment.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    )


def _parse_baselines(value: str) -> List[str]:
    aliases = {
        "harvest": "harvest",
        "silva": "silva",
        "silva_eaf": "silva",
        "silva_pruned": "silva_pruned",
        "puremagic": "puremagic",
        "dascot": "dascot",
        "litinski": "litinski",
        "o3ls": "o3ls",
        "lsqca": "lsqca",
        "taco": "taco",
    }
    selected = []
    for item in value.split(","):
        key = item.strip().lower()
        if key not in aliases:
            raise argparse.ArgumentTypeError(f"unknown baseline {item!r}")
        canonical = aliases[key]
        if canonical not in selected:
            selected.append(canonical)
    return selected


def _config(
    *,
    args,
    baseline: str,
    variant: str,
    circuit: Path,
    num_qubits: int,
    trial: int,
    seed: Optional[int],
    input_path: Optional[Path] = None,
    options: Optional[Dict[str, Any]] = None,
) -> BaselineRunConfig:
    versions = {
        "harvest": _git_commit(),
        "silva": "Silva et al., TQC 2024",
        "silva_pruned": "Silva et al., TQC 2024 + HARVEST pruning",
        "puremagic": args.puremagic_version,
        "dascot": args.wisq_version,
        "litinski": "Litinski, Quantum 3, 128 (2019)",
        "o3ls": "O3LS reimplementation (Zhu et al., ISCA 2026; no public artifact)",
        "lsqca": "LSQCA reimplementation (Kobori et al., HPCA 2025; no public artifact)",
        "taco": args.nwqec_version,
    }
    executable = None
    if baseline == "puremagic" and args.puremagic_bin:
        executable = Path(args.puremagic_bin)
    elif baseline == "dascot" and args.wisq_bin:
        executable = Path(args.wisq_bin)
    elif baseline == "taco" and args.taco_bin:
        executable = Path(args.taco_bin)
    return BaselineRunConfig(
        baseline=baseline,
        variant=variant,
        circuit=circuit.stem,
        input_path=(input_path or circuit),
        output_dir=Path(args.output_dir) / "work",
        comparison_mode=args.comparison_mode,
        num_qubits=num_qubits,
        trial=trial,
        seed=seed,
        timeout_s=args.timeout,
        executable=executable,
        upstream_version=versions[baseline],
        options=options or {},
    )


def _magic_options(args) -> Dict[str, Any]:
    """Magic-state model in a config; the free default keeps old configs unchanged."""
    if args.magic_model == "free":
        return {}
    if args.magic_model == "cultivation":
        return {"magic_model": "cultivation",
                "cultivation_mean_cycles": args.cultivation_mean_cycles}
    return {"magic_model": args.magic_model, "magic_mean_cycles": args.cultivation_mean_cycles}


def _model_trials(args) -> List[tuple]:
    """(trial, seed) pairs for in-tree models: one deterministic run, or seeded
    trials once a stochastic supply model makes them stochastic."""
    if args.magic_model not in STOCHASTIC_MODELS:
        return [(0, None)]
    return [(t, args.seed + t) for t in range(args.cultivation_trials)]


def _layout_options(args) -> Dict[str, Any]:
    """Layout knobs recorded in a config; the default keeps old configs unchanged."""
    options: Dict[str, Any] = {}
    if args.layout_preset != "ring":
        options["layout_preset"] = args.layout_preset
    if args.layout_preset in ("fixed_magic", "dense"):
        options["layout_magic_count"] = args.layout_magic_count
    if args.layout_magic_ring:
        options["layout_magic_ring"] = args.layout_magic_ring
    return options


def _harvest_options(args) -> Dict[str, Any]:
    """HARVEST scheduler knobs that differ from the defaults (none: old configs unchanged)."""
    options: Dict[str, Any] = {}
    if args.harvest_config:
        options["harvest_config"] = args.harvest_config
    if args.harvest_scheduler != "harvest":
        options["scheduler"] = args.harvest_scheduler
    if args.fault:
        options["fault"] = args.fault
    return options


def _dry_result(config: BaselineRunConfig) -> BaselineResult:
    return BaselineResult.failed(
        config,
        input_representation="not materialized (dry run)",
        error="dry-run: experiment planned but not executed",
        notes=["This record is incomplete and will not be treated as a resume hit."],
        metadata={"planned_config": config.serializable()},
    )


def synthesize_cliffordt_for_dascot(
    source: Path, compile_bin: str, inputs_dir: Path, timeout: float
) -> "tuple[Optional[Path], Optional[str]]":
    """Pre-synthesize *source* into Clifford+T for DASCOT's scmr-only mode.

    DASCOT's scmr mode has no synthesis pass of its own and GUOQ is
    deliberately excluded from the comparison (it would mix circuit
    optimization into a mapping/routing-only result), so a circuit with
    generic rotation angles cannot otherwise reach DASCOT at all. Reuses
    PureMagic's non-GUOQ `compile_cliffordt` (also used for its own native
    PureMagic runs via --puremagic-compile-bin) as a common, tool-neutral
    front end.  Returns (synthesized_path_or_None, command_str_or_None).
    """
    inputs_dir.mkdir(parents=True, exist_ok=True)
    target = inputs_dir / f"{safe_name(source.stem)}.cliffordt.qasm"
    command = [compile_bin, str(source), "-o", str(target)]
    if not target.exists():
        proc = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout, check=False
        )
        if proc.returncode != 0 or not target.exists():
            return None, None
    return target, shlex.join(command)


def execute_config(
    adapter,
    config: BaselineRunConfig,
    raw_dir: Path,
    *,
    resume: bool,
    dry_run: bool,
    **kwargs: Any,
) -> Optional[BaselineResult]:
    path = raw_result_path(raw_dir, config)
    if resume and should_resume(path, config):
        logger.info("resume: %s", path.name)
        return None
    logger.info(
        "run: %s / %s / %s / trial %d",
        config.circuit,
        config.baseline,
        config.variant,
        config.trial,
    )
    if dry_run:
        result = _dry_result(config)
    else:
        result = _run_adapter(adapter, config, **kwargs)
    write_result(path, result, config)
    _record_run(config, result)
    violations = (result.metadata.get("validation") or {}).get("n_violations")
    if RUN_CONTEXT.strict_validate and violations:
        raise SystemExit(
            f"strict validation: {config.circuit}/{config.variant} has {violations} "
            f"violation(s); record kept at {path}"
        )
    return result


@dataclass
class RunContext:
    """Driver-wide settings for the eval harness (set from the CLI in ``main``)."""

    runs_csv: Optional[Path] = None
    exp_id: str = ""
    point: str = ""
    strict_validate: bool = False
    in_process_timeout: Optional[float] = None


RUN_CONTEXT = RunContext()
IN_PROCESS = frozenset({"harvest", "silva", "litinski", "o3ls", "lsqca"})


def _record_run(config: BaselineRunConfig, result: BaselineResult) -> None:
    if RUN_CONTEXT.runs_csv is None:
        return
    payload = {"config": config.serializable(), "result": result.to_dict()}
    append_run_row(RUN_CONTEXT.runs_csv, run_row(RUN_CONTEXT.exp_id, RUN_CONTEXT.point, payload))


def _run_adapter(adapter, config: BaselineRunConfig, **kwargs: Any) -> BaselineResult:
    """Run one trial; an exception or in-process timeout becomes a failed record."""
    import signal

    timeout = RUN_CONTEXT.in_process_timeout if config.baseline in IN_PROCESS else None

    def on_alarm(signum, frame):
        raise TimeoutError(f"in-process timeout after {timeout} s")

    previous = None
    if timeout:
        previous = signal.signal(signal.SIGALRM, on_alarm)
        signal.setitimer(signal.ITIMER_REAL, timeout)
    started = perf_counter()
    try:
        return adapter.run(config, **kwargs)
    except TimeoutError as exc:
        return BaselineResult.failed(
            config, input_representation="in-process", error=str(exc), timed_out=True,
            runtime_s=perf_counter() - started,
        )
    except Exception as exc:  # recorded as a failed trial, never dropped
        logger.exception("%s/%s on %s raised", config.baseline, config.variant, config.circuit)
        return BaselineResult.failed(
            config, input_representation="in-process",
            error=f"exception: {type(exc).__name__}: {exc}",
            runtime_s=perf_counter() - started,
        )
    finally:
        if timeout:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)


def _load_raw_results(raw_dir: Path) -> List[Dict[str, Any]]:
    rows = []
    for path in sorted(raw_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text())
            result = payload["result"]
            result["raw_file"] = str(path)
            rows.append(result)
        except (OSError, json.JSONDecodeError, KeyError) as exc:
            logger.warning("ignoring malformed raw result %s: %s", path, exc)
    return rows


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fields = sorted({key for row in rows for key in row if key != "metadata"})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            cleaned = {
                key: json.dumps(value, sort_keys=True)
                if isinstance(value, (list, dict))
                else value
                for key, value in row.items()
            }
            writer.writerow(cleaned)


def _series_label(row: Dict[str, Any]) -> str:
    return DISPLAY.get(
        (row.get("baseline"), row.get("variant")), row.get("variant", "unknown")
    )


def _plot_metric(
    summaries: Sequence[Dict[str, Any]],
    metric: str,
    ylabel: str,
    path: Path,
    comparison_mode: str,
    log_scale: bool = False,
) -> None:
    circuits = sorted({row["circuit"] for row in summaries})
    systems = sorted({_series_label(row) for row in summaries})
    fig, ax = plt.subplots(figsize=(max(7.0, 1.15 * len(circuits)), 4.8))
    if not circuits or not systems:
        ax.text(0.5, 0.5, "No successful comparable results", ha="center", va="center")
        ax.set_axis_off()
    else:
        x = np.arange(len(circuits), dtype=float)
        width = 0.82 / max(1, len(systems))
        lookup = {(_series_label(row), row["circuit"]): row for row in summaries}
        for index, system in enumerate(systems):
            vals = []
            errs = []
            has_ci = False
            for circuit in circuits:
                row = lookup.get((system, circuit), {})
                value = row.get(f"{metric}_mean")
                error = row.get(f"{metric}_ci95")
                vals.append(np.nan if value is None else value)
                errs.append(0.0 if error is None else error)
                has_ci = has_ci or error is not None
            positions = x + (index - (len(systems) - 1) / 2) * width
            ax.bar(
                positions,
                vals,
                width=width,
                label=system,
                color=COLORS.get(system),
                alpha=0.9,
                yerr=errs if has_ci else None,
                capsize=2 if has_ci else 0,
                error_kw={"linewidth": 0.8},
            )
        ax.set_xticks(x)
        ax.set_xticklabels(circuits, rotation=35, ha="right")
        ax.set_ylabel(ylabel)
        if log_scale:
            ax.set_yscale("log")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.legend(fontsize=8, ncol=max(1, min(3, len(systems))))
    ax.set_title(f"{ylabel} — {comparison_mode.replace('_', ' ')}")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _plot_normalized(
    summaries: Sequence[Dict[str, Any]], path: Path, comparison_mode: str
) -> None:
    harvest = {
        row["circuit"]: row
        for row in summaries
        if row.get("baseline") == "harvest" and row.get("schedule_length_mean")
    }
    ratios = []
    for row in summaries:
        if row.get("baseline") == "harvest":
            continue
        reference = harvest.get(row["circuit"])
        value = row.get("schedule_length_mean")
        if reference is None or value is None:
            continue
        ratios.append(
            (
                row["circuit"],
                _series_label(row),
                value / reference["schedule_length_mean"],
            )
        )
    circuits = sorted({item[0] for item in ratios})
    systems = sorted({item[1] for item in ratios})
    lookup = {(circuit, system): ratio for circuit, system, ratio in ratios}
    fig, ax = plt.subplots(figsize=(max(7.0, 1.15 * len(circuits)), 4.8))
    if not ratios:
        ax.text(
            0.5, 0.5, "No shared successful HARVEST subset", ha="center", va="center"
        )
        ax.set_axis_off()
    else:
        x = np.arange(len(circuits), dtype=float)
        width = 0.82 / max(1, len(systems))
        for index, system in enumerate(systems):
            values = [lookup.get((circuit, system), np.nan) for circuit in circuits]
            positions = x + (index - (len(systems) - 1) / 2) * width
            ax.bar(
                positions, values, width=width, label=system, color=COLORS.get(system)
            )
        ax.axhline(1.0, color="black", linewidth=1, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(circuits, rotation=35, ha="right")
        ax.set_ylabel("Schedule length ratio (baseline / HARVEST)")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.legend(fontsize=8, ncol=max(1, min(3, len(systems))))
    ax.set_title(f"Normalized schedule length — {comparison_mode.replace('_', ' ')}")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _write_geomeans(output_dir: Path, rows: Sequence[Dict[str, Any]]) -> None:
    _write_csv(output_dir / "pairwise_geomeans.csv", rows)
    lines = [
        r"\begin{tabular}{lllrr}",
        r"\hline",
        r"Baseline & Variant & Metric & Shared $n$ & Geomean \\",
        r"\hline",
    ]

    def latex_escape(value: Any) -> str:
        text = str(value)
        for source, replacement in (
            ("\\", r"\textbackslash{}"),
            ("_", r"\_"),
            ("&", r"\&"),
            ("%", r"\%"),
            ("#", r"\#"),
        ):
            text = text.replace(source, replacement)
        return text

    for row in rows:
        value = row.get("geometric_mean_ratio")
        display = "--" if value is None else f"{value:.3f}"
        lines.append(
            f"{latex_escape(row['baseline'])} & {latex_escape(row['variant'])} & "
            f"{latex_escape(row['metric'])} & "
            f"{row['num_shared_benchmarks']} & {display} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}"])
    (output_dir / "pairwise_geomeans.tex").write_text("\n".join(lines) + "\n")


def derive_outputs(output_dir: Path, comparison_mode: str) -> None:
    raw_rows = _load_raw_results(output_dir / "raw")
    summaries = aggregate_results(raw_rows)
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "comparison_mode": comparison_mode,
                "ratio_convention": "baseline / HARVEST",
                "results": summaries,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    _write_csv(output_dir / "summary.csv", summaries)
    failures = [row for row in raw_rows if not row.get("completed")]
    _write_csv(output_dir / "failures.csv", failures)
    geomeans = pairwise_geometric_means(summaries)
    _write_geomeans(output_dir, geomeans)

    successful = [
        row
        for row in summaries
        if row.get("trials_successful", 0) > 0
        and row.get("comparison_mode") == comparison_mode
    ]
    _plot_metric(
        successful,
        "schedule_length",
        "Logical timesteps / native logical cycles",
        output_dir / "comparison_schedule_length.pdf",
        comparison_mode,
    )
    _plot_metric(
        successful,
        "space_time_volume",
        "Logical patches × logical cycles\n(HARVEST: post-pruning)",
        output_dir / "comparison_space_time.pdf",
        comparison_mode,
    )
    _plot_metric(
        successful,
        "compiler_runtime_s",
        "Compiler runtime (s)",
        output_dir / "comparison_runtime.pdf",
        comparison_mode,
        log_scale=True,
    )
    _plot_normalized(
        successful, output_dir / "comparison_normalized.pdf", comparison_mode
    )


def run_circuit(path: Path, args: argparse.Namespace, selected: Sequence[str]) -> None:
    output_dir = Path(args.output_dir)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    placement_cost = None
    try:
        frontend_started = perf_counter()
        circuit, dag = load_benchmark(path)
        compose_done = perf_counter()
        layout = build_layout(
            circuit.num_qubits,
            args.rows,
            args.cols,
            dag,
            args.placement,
            args.seed,
            args.layout_preset,
            args.layout_magic_count,
        )
        if args.layout_magic_ring:
            from harvest.layout.presets import thin_magic_ring

            layout = thin_magic_ring(layout, args.layout_magic_ring)
        layout_done = perf_counter()
        if args.placement in ("index", "random", "circuit_aware_preset"):
            from harvest.eval.placement import place

            layout, placement_cost, _ = place(
                layout, dag, args.placement.replace("_preset", ""), seed=args.seed
            )
        place_done = perf_counter()
        harvest_frontend_runtime = place_done - frontend_started
        phase_times = {
            "t_compose_s": compose_done - frontend_started,
            "t_layout_s": layout_done - compose_done,
            "t_place_s": place_done - layout_done,
        }
    except Exception as exc:
        logger.error("failed to load %s: %s", path, exc)
        return

    circuit_aware = args.placement == "circuit_aware"
    seeded_placement = args.placement in ("circuit_aware", "random", "circuit_aware_preset")
    deterministic = [
        (
            "harvest",
            "harvest",
            args.harvest_variant or ("harvest_circuit_aware" if circuit_aware else "harvest"),
            HarvestAdapter(),
            _harvest_options(args),
        ),
        (
            "silva",
            "silva",
            "silva_eaf_matched_placement" if circuit_aware else "silva_eaf",
            SilvaEAFAdapter(),
            {},
        ),
        (
            "silva_pruned",
            "silva",
            "silva_eaf_pruned",
            SilvaEAFAdapter(),
            {"prune": True},
        ),
    ]
    for (selector, baseline, variant, adapter, extra), (trial, seed) in (
        (entry, ts) for entry in deterministic for ts in _model_trials(args)
    ):
        if selector not in selected:
            continue
        config = _config(
            args=args,
            baseline=baseline,
            variant=variant,
            circuit=path,
            num_qubits=circuit.num_qubits,
            trial=trial,
            seed=seed,
            options={
                "placement": args.placement,
                "placement_seed": args.seed if seeded_placement else None,
                "requested_rows": args.rows,
                "requested_cols": args.cols,
                **_layout_options(args),
                **_magic_options(args),
                **extra,
            },
        )
        execute_config(
            adapter,
            config,
            raw_dir,
            resume=args.resume,
            dry_run=args.dry_run,
            dag=dag,
            layout_engine=layout,
            preprocessing_runtime_s=(
                harvest_frontend_runtime
                if baseline == "harvest" and args.comparison_mode == "native_end_to_end"
                else 0.0
            ),
            phase_times=phase_times,
            placement_cost=placement_cost,
            validate=not args.no_validate,
        )

    if args.magic_model not in ("free", "cultivation"):
        # The matched-IR baselines model only free and cultivated states.
        for baseline in ("litinski", "o3ls", "puremagic", "lsqca", "taco", "dascot"):
            if baseline in selected:
                config = _config(args=args, baseline=baseline, variant=baseline, circuit=path,
                                 num_qubits=circuit.num_qubits, trial=0, seed=None,
                                 options=_magic_options(args))
                result = BaselineResult.failed(
                    config, input_representation="not materialized",
                    error=f"unsupported magic model {args.magic_model!r} for {baseline}",
                )
                write_result(raw_result_path(raw_dir, config), result, config)
                _record_run(config, result)
        return

    # Matched-IR reimplementations/models that consume the same Pauli-product DAG.
    for baseline, module, cls in (
        ("litinski", "harvest.baselines.litinski", "LitinskiAdapter"),
        ("o3ls", "harvest.baselines.o3ls", "O3LSAdapter"),
    ):
        if baseline not in selected:
            continue
        mod = importlib.import_module(module)
        for variant in mod.VARIANTS:
            for trial, seed in _model_trials(args):
                config = _config(
                    args=args,
                    baseline=baseline,
                    variant=variant,
                    circuit=path,
                    num_qubits=circuit.num_qubits,
                    trial=trial,
                    seed=seed,
                    options=_magic_options(args),
                )
                execute_config(
                    getattr(mod, cls)(), config, raw_dir,
                    resume=args.resume, dry_run=args.dry_run, dag=dag,
                )

    # Gate-level architecture models consuming the Clifford+T source (native_end_to_end).
    for baseline, module, cls in (
        ("lsqca", "harvest.baselines.lsqca", "LSQCAAdapter"),
        ("taco", "harvest.baselines.taco", "TACOAdapter"),
    ):
        if baseline not in selected:
            continue
        mod = importlib.import_module(module)
        for variant in mod.VARIANTS:
            config = _config(
                args=args,
                baseline=baseline,
                variant=variant,
                circuit=path,
                num_qubits=circuit.num_qubits,
                trial=0,
                seed=None,
                options={"code_distance": args.code_distance},
            )
            execute_config(
                getattr(mod, cls)(), config, raw_dir,
                resume=args.resume, dry_run=args.dry_run,
            )

    if "puremagic" in selected:
        trans_path = output_dir / "inputs" / f"{safe_name(path.stem)}.trans"
        conversion_error = None
        if args.comparison_mode == "matched_ir" and not args.dry_run:
            try:
                write_trans_file(dag, trans_path, circuit.num_qubits)
            except UnsupportedPureMagicIR as exc:
                conversion_error = str(exc)
        for variant in ("puremagic_bus", "puremagic"):
            for trial in range(args.trials):
                seed = args.seed + trial
                native = args.comparison_mode == "native_end_to_end"
                options = {
                    "input_format": "qasm" if native else "trans",
                    # Common cultivation model: PureMagic's native Exp(1/lambda)
                    # cultivation with the shared mean.
                    "magic_state_lambda": (
                        1.0 / args.cultivation_mean_cycles
                        if args.magic_model == "cultivation"
                        else args.magic_state_lambda
                    ),
                    **_magic_options(args),
                    "ancilla_rows": args.ancilla_rows,
                    "no_t_failures": args.puremagic_no_t_failures,
                    "transpile_bin": args.puremagic_transpile_bin,
                    "compile_bin": args.puremagic_compile_bin,
                    "topology": args.puremagic_topology,
                }
                config = _config(
                    args=args,
                    baseline="puremagic",
                    variant=variant,
                    circuit=path,
                    num_qubits=circuit.num_qubits,
                    trial=trial,
                    seed=seed,
                    input_path=path if native else trans_path,
                    options=options,
                )
                result_path = raw_result_path(raw_dir, config)
                if args.resume and should_resume(result_path, config):
                    logger.info("resume: %s", result_path.name)
                    continue
                if args.comparison_mode == "matched_architecture":
                    result = BaselineResult.failed(
                        config,
                        input_representation="PureMagic .trans Pauli-product stream",
                        error="PureMagic matched_architecture conversion is not qualified",
                        notes=[
                            "HARVEST single-tile data patches do not match PureMagic's double data topology."
                        ],
                    )
                    write_result(result_path, result, config)
                    _record_run(config, result)
                elif conversion_error is not None:
                    result = BaselineResult.failed(
                        config,
                        input_representation="HARVEST Pauli-product DAG",
                        error=conversion_error,
                        notes=["No unsupported operation was discarded."],
                    )
                    write_result(result_path, result, config)
                    _record_run(config, result)
                else:
                    execute_config(
                        PureMagicAdapter(),
                        config,
                        raw_dir,
                        resume=False,
                        dry_run=args.dry_run,
                    )

    if "dascot" in selected:
        variants = (
            ("matched_architecture",)
            if args.comparison_mode == "matched_architecture"
            else ("dascot_square_sparse", "dascot_compact")
        )
        dascot_input_path = path
        dascot_pre_synthesis_note = None
        dascot_pre_synthesis_command = None
        if (
            args.comparison_mode == "native_end_to_end"
            and args.dascot_compile_bin
            and not args.dry_run
        ):
            try:
                validate_wisq_qasm(path)
            except UnsupportedDASCOTInput:
                synth_path, synth_command = synthesize_cliffordt_for_dascot(
                    path,
                    args.dascot_compile_bin,
                    output_dir / "inputs",
                    args.timeout,
                )
                if synth_path is not None:
                    dascot_input_path = synth_path
                    dascot_pre_synthesis_command = synth_command
                    dascot_pre_synthesis_note = (
                        "Input QASM was pre-synthesized to Clifford+T via PureMagic's "
                        "non-GUOQ compile_cliffordt before DASCOT scmr mapping/routing "
                        "because the original circuit used generic rotation angles; "
                        "HARVEST and Silva ran on the unsynthesized original circuit."
                    )
        for variant in variants:
            for trial in range(args.trials):
                config = _config(
                    args=args,
                    baseline="dascot",
                    variant=variant,
                    circuit=path,
                    num_qubits=circuit.num_qubits,
                    trial=trial,
                    seed=args.seed + trial,
                    input_path=dascot_input_path,
                    options=(
                        {
                            "harvest_layout_placement": args.placement,
                            "placement_seed": args.seed if circuit_aware else None,
                            "requested_rows": args.rows,
                            "requested_cols": args.cols,
                            **_layout_options(args),
                        }
                        if args.comparison_mode == "matched_architecture"
                        else {
                            "architecture_variant": variant,
                            "pre_synthesis_note": dascot_pre_synthesis_note,
                            "pre_synthesis_command": dascot_pre_synthesis_command,
                        }
                    ),
                )
                execute_config(
                    DASCOTAdapter(),
                    config,
                    raw_dir,
                    resume=args.resume,
                    dry_run=args.dry_run,
                    layout_engine=layout,
                )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare HARVEST with Silva EAF, PureMagic, and DASCOT.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--benchmarks",
        action="append",
        required=True,
        metavar="PATH_OR_FAMILY",
        help=(
            "QASM file, directory, or benchmark family. Repeat this option to "
            "run one recorded suite assembled from multiple inputs."
        ),
    )
    parser.add_argument(
        "--circuit", help="Only run a circuit whose filename/stem contains this text"
    )
    parser.add_argument(
        "--skip-circuits", default="",
        help="Comma-separated circuit names (exact file stems) to leave out, e.g. ones too "
             "expensive to run.",
    )
    parser.add_argument("--baselines", default="harvest,silva,puremagic,dascot")
    parser.add_argument(
        "--comparison-mode",
        choices=["matched_ir", "matched_architecture", "native_end_to_end"],
        default="matched_ir",
    )
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-circuits", type=int)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--resume", action="store_true",
                        help="Reuse stored trials whose identity (baseline, variant, circuit, "
                             "mode, trial, seed, options, input SHA-256) matches.")
    parser.add_argument("--keep-failed", action="store_true",
                        help="With --resume, also reuse recorded failures instead of retrying them.")
    parser.add_argument("--force-rerun", default="",
                        help="Comma-separated baselines to recompute even if stored "
                             "(e.g. after changing their code).")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", default="results/sota_comparison")
    parser.add_argument("--rows", type=int)
    parser.add_argument("--cols", type=int)
    parser.add_argument(
        "--placement",
        choices=["row_major", "circuit_aware", "index", "random", "circuit_aware_preset"],
        default="row_major",
        help="row_major: the preset's own order; circuit_aware: synthesized bus template "
             "(a separately identified matched-placement experiment); index / random / "
             "circuit_aware_preset: harvest.eval.placement on the chosen preset "
             "(same geometry, records the placement cost).",
    )
    parser.add_argument(
        "--layout-preset",
        choices=LAYOUT_PRESETS,
        default="ring",
        help="HARVEST/Silva (and DASCOT matched_architecture) layout; a resource-sweep knob.",
    )
    parser.add_argument("--layout-magic-count", type=int, default=2,
                        help="Magic patches for --layout-preset fixed_magic.")
    parser.add_argument("--layout-magic-ring", type=int, default=0,
                        help="Keep only this many ring magic patches, evenly spread "
                             "(presets.thin_magic_ring; 0 = the preset's full ring).")
    parser.add_argument(
        "--magic-model", choices=list(MAGIC_MODELS), default="free",
        help="free: magic states always ready (matched-IR default); cultivation: every "
             "compiler's magic ports cultivate states with Exp(mean) cycles "
             "(harvest.baselines.cultivation; PureMagic gets lambda = 1/mean); fixed / "
             "geometric (HARVEST and Silva only): fixed latency / geometric cultivation "
             "with the given mean.",
    )
    parser.add_argument("--cultivation-mean-cycles", type=float, default=3.0,
                        help="Mean preparation latency of the magic model (cycles).")
    harvest = parser.add_argument_group("HARVEST scheduler / evaluation harness")
    harvest.add_argument("--harvest-config", type=json.loads, default=None,
                         help='HarvestConfig overrides as JSON, e.g. \'{"alpha": 4}\'.')
    harvest.add_argument("--harvest-variant",
                         help="Variant name recorded for HARVEST runs (e.g. an ablation label).")
    harvest.add_argument("--harvest-scheduler", choices=["harvest", "steiner_tree"],
                         default="harvest",
                         help="steiner_tree: the sequential one-product-per-step ablation.")
    harvest.add_argument("--fault", choices=list(FAULTS),
                         help="Inject a HARVEST_FAULT mutation (validator study only).")
    harvest.add_argument("--no-validate", action="store_true",
                         help="Skip the schedule validator for HARVEST/Silva.")
    harvest.add_argument("--strict-validate", action="store_true",
                         help="Abort on the first schedule with a validator violation.")
    harvest.add_argument("--runs-csv", type=Path,
                         help="Append one metadata row per tool run to this CSV.")
    harvest.add_argument("--exp-id", default="", help="Experiment id written to --runs-csv.")
    harvest.add_argument("--point", default="", help="Sweep point id written to --runs-csv.")
    harvest.add_argument("--in-process-timeout", type=float,
                         help="Wall-clock limit for in-process compilers (default: none).")
    harvest.add_argument("--no-derive", action="store_true",
                         help="Skip summary/plot derivation (the eval harness collects raw/).")
    parser.add_argument("--cultivation-trials", type=int, default=5,
                        help="Seeded trials for HARVEST/Silva/Litinski/O3LS under cultivation.")
    parser.add_argument("--puremagic-bin")
    parser.add_argument("--puremagic-transpile-bin")
    parser.add_argument("--puremagic-compile-bin")
    parser.add_argument("--puremagic-topology", help="Optional PureMagic topology file")
    parser.add_argument("--puremagic-version", default="QCE (paper snapshot requested)")
    parser.add_argument("--magic-state-lambda", type=float, default=0.0387396)
    parser.add_argument("--ancilla-rows", type=int, default=1)
    parser.add_argument("--puremagic-no-t-failures", action="store_true")
    parser.add_argument("--taco-bin", help="nwqec-cli of an isolated nwqec install (TACO pass)")
    parser.add_argument("--nwqec-version", default="nwqec (TACO transpiler, pnnl/nwqec)")
    parser.add_argument("--code-distance", type=int, default=13,
                        help="Code distance used by models whose latencies are in QEC rounds (TACO).")
    parser.add_argument("--wisq-bin")
    parser.add_argument("--wisq-version", default="v0.2.7")
    parser.add_argument(
        "--dascot-compile-bin",
        help=(
            "Optional non-GUOQ Clifford+T synthesizer (e.g. PureMagic's "
            "compile_cliffordt) used only in native_end_to_end mode to pre-"
            "synthesize circuits whose original gate set DASCOT's scmr mode "
            "cannot accept directly. Never used for circuits that are already "
            "native Clifford+T/CX."
        ),
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    if args.trials < 1:
        parser.error("--trials must be at least 1")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.max_circuits is not None and args.max_circuits < 1:
        parser.error("--max-circuits must be at least 1")
    if args.rows is not None and args.rows < 1:
        parser.error("--rows must be at least 1")
    if args.cols is not None and args.cols < 1:
        parser.error("--cols must be at least 1")
    if args.ancilla_rows < 1:
        parser.error("--ancilla-rows must be at least 1")
    if args.magic_state_lambda <= 0:
        parser.error("--magic-state-lambda must be positive")
    try:
        args.baselines = _parse_baselines(args.baselines)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    if args.harvest_config is not None:
        from harvest.routing.config import HarvestConfig

        try:
            args.harvest_config = HarvestConfig.from_mapping(args.harvest_config).overrides()
        except (TypeError, ValueError) as exc:
            parser.error(f"--harvest-config: {exc}")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    RESUME_POLICY.retry_failed = not args.keep_failed
    RESUME_POLICY.force_rerun = frozenset(
        b.strip() for b in args.force_rerun.split(",") if b.strip()
    )
    RUN_CONTEXT.runs_csv = args.runs_csv
    RUN_CONTEXT.exp_id = args.exp_id
    RUN_CONTEXT.point = args.point
    RUN_CONTEXT.strict_validate = args.strict_validate
    RUN_CONTEXT.in_process_timeout = args.in_process_timeout
    if args.fault:
        os.environ["HARVEST_FAULT"] = args.fault
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s - %(message)s",
    )
    # Verbose experiment logging should not emit thousands of font-manager or
    # per-cell routing diagnostics.
    for noisy_logger in ("matplotlib", "PIL", "qiskit", "HarvestMagicState.Detailed"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)
    if not args.verbose:
        logging.getLogger("HarvestMagicState").setLevel(logging.WARNING)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_environment(output_dir, args)
    try:
        circuits = sorted(
            {
                path
                for benchmark_spec in args.benchmarks
                for path in discover_benchmarks(benchmark_spec)
            }
        )
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 2
    if args.circuit:
        circuits = [path for path in circuits if args.circuit in path.name]
    skip = {name.strip() for name in args.skip_circuits.split(",") if name.strip()}
    if skip:
        skipped = sorted(path.stem for path in circuits if path.stem in skip)
        circuits = [path for path in circuits if path.stem not in skip]
        logger.info("skipping %d circuit(s) by request: %s", len(skipped), ", ".join(skipped))
    if args.max_circuits:
        circuits = circuits[: args.max_circuits]
    if not circuits:
        logger.error("no circuits selected")
        return 2

    logger.info(
        "selected %d circuit(s), baselines=%s, mode=%s, trials=%d",
        len(circuits),
        ",".join(args.baselines),
        args.comparison_mode,
        args.trials,
    )
    for path in circuits:
        run_circuit(path, args, args.baselines)
    if args.no_derive:
        logger.info("raw results written to %s", output_dir / "raw")
        return 0
    derive_outputs(output_dir, args.comparison_mode)
    logger.info("outputs written to %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
