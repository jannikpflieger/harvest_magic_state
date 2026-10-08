#!/usr/bin/env python3
"""Reproducible HARVEST/Silva/PureMagic/DASCOT comparison driver.

Raw trial records are the source of truth.  Summaries and plots are derived
from them and never silently remove failures or mix comparison regimes.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import logging
import math
import os
import platform
import shlex
import subprocess
import sys
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
from harvest.compilation.circuit_analysis import convert_rx_ry_to_rz, pre_prep_circuit
from harvest.compilation.pauli_block_conversion import convert_to_PCB, create_dag
from harvest.compilation.qasm_loader import find_qasm_files, qasm_to_circuit
from harvest.layout.presets import nxm_ring_layout_single_qubits

logger = logging.getLogger("SOTABaselineComparison")

DISPLAY = {
    ("harvest", "harvest"): "HARVEST",
    ("harvest", "harvest_circuit_aware"): "HARVEST (circuit-aware placement)",
    ("silva", "silva_eaf"): "Silva EAF",
    ("silva", "silva_eaf_matched_placement"): "Silva EAF (matched placement)",
    ("puremagic", "puremagic_bus"): "PureMagic Bus",
    ("puremagic", "puremagic"): "PureMagic",
    ("dascot", "dascot_square_sparse"): "DASCOT Square Sparse",
    ("dascot", "dascot_compact"): "DASCOT Compact",
    ("dascot", "matched_architecture"): "DASCOT Matched Architecture",
}
COLORS = {
    "HARVEST": "#4C72B0",
    "HARVEST (circuit-aware placement)": "#4C72B0",
    "Silva EAF": "#DD8452",
    "Silva EAF (matched placement)": "#DD8452",
    "PureMagic Bus": "#55A868",
    "PureMagic": "#C44E52",
    "DASCOT Square Sparse": "#8172B3",
    "DASCOT Compact": "#937860",
    "DASCOT Matched Architecture": "#64B5CD",
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
    circuit = qasm_to_circuit(str(path))
    circuit = pre_prep_circuit(circuit)
    if "rx" in circuit.count_ops() or "ry" in circuit.count_ops():
        circuit = convert_rx_ry_to_rz(circuit)
    pcb = convert_to_PCB(circuit, verbose=False)
    return circuit, create_dag(pcb)


def build_layout(
    num_qubits: int,
    rows: Optional[int],
    cols: Optional[int],
    dag,
    placement: str,
    seed: int,
):
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

    resolved_cols = cols or max(1, math.ceil(math.sqrt(num_qubits)))
    resolved_rows = rows or max(1, math.ceil(num_qubits / resolved_cols))
    if resolved_cols * resolved_rows < num_qubits:
        raise ValueError(
            f"layout {resolved_rows}x{resolved_cols} has fewer data patches than {num_qubits} qubits"
        )
    return nxm_ring_layout_single_qubits(resolved_cols, resolved_rows)


def raw_result_path(raw_dir: Path, config: BaselineRunConfig) -> Path:
    return raw_dir / deterministic_result_name(config)


def should_resume(path: Path, config: Optional[BaselineRunConfig] = None) -> bool:
    """Resume only a completed trial with the same normalized configuration."""
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text())
        if payload.get("result", {}).get("completed") is not True:
            return False
        return config is None or payload.get("config") == config.serializable()
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
        "puremagic": "puremagic",
        "dascot": "dascot",
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
        "puremagic": args.puremagic_version,
        "dascot": args.wisq_version,
    }
    executable = None
    if baseline == "puremagic" and args.puremagic_bin:
        executable = Path(args.puremagic_bin)
    elif baseline == "dascot" and args.wisq_bin:
        executable = Path(args.wisq_bin)
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
    result = _dry_result(config) if dry_run else adapter.run(config, **kwargs)
    write_result(path, result, config)
    return result


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
    try:
        frontend_started = perf_counter()
        circuit, dag = load_benchmark(path)
        layout = build_layout(
            circuit.num_qubits,
            args.rows,
            args.cols,
            dag,
            args.placement,
            args.seed,
        )
        harvest_frontend_runtime = perf_counter() - frontend_started
    except Exception as exc:
        logger.error("failed to load %s: %s", path, exc)
        return

    circuit_aware = args.placement == "circuit_aware"
    deterministic = [
        (
            "harvest",
            "harvest_circuit_aware" if circuit_aware else "harvest",
            HarvestAdapter(),
        ),
        (
            "silva",
            "silva_eaf_matched_placement" if circuit_aware else "silva_eaf",
            SilvaEAFAdapter(),
        ),
    ]
    for baseline, variant, adapter in deterministic:
        if baseline not in selected:
            continue
        config = _config(
            args=args,
            baseline=baseline,
            variant=variant,
            circuit=path,
            num_qubits=circuit.num_qubits,
            trial=0,
            seed=None,
            options={
                "placement": args.placement,
                "placement_seed": args.seed if circuit_aware else None,
                "requested_rows": args.rows,
                "requested_cols": args.cols,
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
                    "magic_state_lambda": args.magic_state_lambda,
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
                elif conversion_error is not None:
                    result = BaselineResult.failed(
                        config,
                        input_representation="HARVEST Pauli-product DAG",
                        error=conversion_error,
                        notes=["No unsupported operation was discarded."],
                    )
                    write_result(result_path, result, config)
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
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", default="results/sota_comparison")
    parser.add_argument("--rows", type=int)
    parser.add_argument("--cols", type=int)
    parser.add_argument(
        "--placement",
        choices=["row_major", "circuit_aware"],
        default="row_major",
        help="Circuit-aware placement is a separately identified matched-placement experiment.",
    )
    parser.add_argument("--puremagic-bin")
    parser.add_argument("--puremagic-transpile-bin")
    parser.add_argument("--puremagic-compile-bin")
    parser.add_argument("--puremagic-topology", help="Optional PureMagic topology file")
    parser.add_argument("--puremagic-version", default="QCE (paper snapshot requested)")
    parser.add_argument("--magic-state-lambda", type=float, default=0.0387396)
    parser.add_argument("--ancilla-rows", type=int, default=1)
    parser.add_argument("--puremagic-no-t-failures", action="store_true")
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
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
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
    derive_outputs(output_dir, args.comparison_mode)
    logger.info("outputs written to %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
