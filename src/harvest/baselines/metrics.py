"""Metric normalization and aggregation for SOTA baseline comparisons."""

from __future__ import annotations

from collections import defaultdict
from math import exp, log, sqrt
from statistics import mean, stdev
from typing import Any, Dict, Iterable, List, Optional, Sequence


def space_time_volume(
    logical_patch_count: Optional[int], schedule_length: Optional[int]
) -> Optional[int]:
    """Return patches × logical cycles, or ``None`` if either is undefined."""
    if logical_patch_count is None or schedule_length is None:
        return None
    if logical_patch_count < 0 or schedule_length < 0:
        raise ValueError("space-time inputs must be non-negative")
    return logical_patch_count * schedule_length


def confidence_interval_95(values: Sequence[float]) -> Optional[float]:
    """Normal-approximation 95% half-width; undefined for fewer than 2 runs."""
    if len(values) < 2:
        return None
    return 1.96 * stdev(values) / sqrt(len(values))


def geometric_mean(values: Iterable[float]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and float(v) > 0]
    return exp(mean(log(v) for v in vals)) if vals else None


def aggregate_results(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Aggregate trials without counting failed/timeout values in statistics."""
    groups: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row.get("baseline"),
            row.get("variant"),
            row.get("circuit"),
            row.get("comparison_mode"),
        )
        groups[key].append(row)

    metrics = (
        "schedule_length",
        "logical_patches",
        "space_time_volume",
        "routing_wirelength",
        "compiler_runtime_s",
        "peak_memory_mb",
        "space_time_volume_initial",
        "space_time_volume_post_pruning",
    )
    summaries: List[Dict[str, Any]] = []
    for key in sorted(groups, key=lambda item: tuple(str(x) for x in item)):
        trials = groups[key]
        successful = [r for r in trials if r.get("completed") is True]
        summary: Dict[str, Any] = {
            "baseline": key[0],
            "variant": key[1],
            "circuit": key[2],
            "comparison_mode": key[3],
            "trials_total": len(trials),
            "trials_successful": len(successful),
            "trials_failed": len(trials) - len(successful),
            "trials_timed_out": sum(bool(r.get("timed_out")) for r in trials),
        }
        for metric in metrics:
            vals = [float(r[metric]) for r in successful if r.get(metric) is not None]
            summary[f"{metric}_mean"] = mean(vals) if vals else None
            summary[f"{metric}_std"] = stdev(vals) if len(vals) >= 2 else None
            summary[f"{metric}_ci95"] = confidence_interval_95(vals)
            summary[f"{metric}_n"] = len(vals)
        summaries.append(summary)
    return summaries


def pairwise_geometric_means(
    summaries: Sequence[Dict[str, Any]], reference: str = "harvest"
) -> List[Dict[str, Any]]:
    """Compute each system/reference geomean on that pair's shared subset."""
    by_system: Dict[tuple, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for row in summaries:
        if row.get("trials_successful", 0) <= 0:
            continue
        system = (row.get("baseline"), row.get("variant"), row.get("comparison_mode"))
        by_system[system][row["circuit"]] = row

    ref_systems = [key for key in by_system if key[0] == reference]
    output: List[Dict[str, Any]] = []
    for system, system_rows in sorted(by_system.items()):
        if system[0] == reference:
            continue
        candidates = [r for r in ref_systems if r[2] == system[2]]
        if not candidates:
            continue
        ref = sorted(candidates)[0]
        shared = sorted(set(system_rows) & set(by_system[ref]))
        for metric in ("schedule_length", "space_time_volume", "compiler_runtime_s"):
            ratios = []
            used = []
            field = f"{metric}_mean"
            for circuit in shared:
                numerator = system_rows[circuit].get(field)
                denominator = by_system[ref][circuit].get(field)
                if (
                    numerator is None
                    or denominator is None
                    or denominator <= 0
                    or numerator <= 0
                ):
                    continue
                ratios.append(numerator / denominator)
                used.append(circuit)
            output.append(
                {
                    "baseline": system[0],
                    "variant": system[1],
                    "reference": ref[0],
                    "reference_variant": ref[1],
                    "comparison_mode": system[2],
                    "metric": metric,
                    "ratio_convention": "baseline / HARVEST",
                    "geometric_mean_ratio": geometric_mean(ratios),
                    "num_shared_benchmarks": len(used),
                    "shared_benchmarks": used,
                }
            )
    return output
