"""Raw trial records -> tidy ``data.csv`` rows (EVAL_PLAN B.1 / B.5).

One row per (circuit, compiler, variant, sweep point, trial).  Units live in
column names; failures and timeouts are rows with ``status != "ok"``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from harvest.routing.config import HarvestConfig

from .families import circuit_family, synthesis_kind
from .manifest import run_status

COMPILER_LABELS = {
    "harvest": "HARVEST",
    "silva_eaf": "Silva EAF",
    "silva_eaf_matched_placement": "Silva EAF",
    "silva_eaf_pruned": "Silva EAF + pruning",
    "puremagic": "PureMagic",
    "puremagic_bus": "PureMagic Bus",
}

# Leading columns in a fixed order; everything else follows alphabetically.
LEAD_COLUMNS = [
    "exp_id", "stage", "point", "circuit", "family", "synthesis", "compiler", "config", "baseline", "variant",
    "comparison_mode", "layout", "layout_preset", "layout_magic_count", "layout_magic_ring",
    "placement", "magic_model", "magic_mean_cycles", "fault", "scheduler_mode", "trial", "seed",
    "status", "error", "num_qubits", "T_cycles", "area_tiles", "volume_tile_cycles",
    "area_initial_tiles", "area_pruned_tiles", "volume_initial", "volume_pruned",
    "compile_time_s", "T_lb_cycles", "area_lb_tiles", "T_flasq_cycles", "T_over_lb",
    "area_over_lb", "T_over_flasq", "volume_over_lb",
]


def compiler_label(baseline: str, variant: str) -> str:
    """``HARVEST``, ``Silva EAF + pruning``, ``Litinski (fast)``, ``O3LS (min)``, ..."""
    if variant in COMPILER_LABELS:
        return COMPILER_LABELS[variant]
    if baseline == "harvest":
        return "HARVEST"
    family = {"litinski": "Litinski", "o3ls": "O3LS", "dascot": "DASCOT", "lsqca": "LSQCA",
              "taco": "TACO"}.get(baseline, baseline)
    suffix = variant[len(baseline) + 1:] if variant.startswith(baseline + "_") else variant
    return f"{family} ({suffix})" if suffix and suffix != baseline else family


def _layout_name(options: Mapping[str, Any], baseline: str) -> Optional[str]:
    if baseline not in ("harvest", "silva"):
        return None
    preset = options.get("layout_preset", "ring")
    if preset in ("dense", "fixed_magic"):
        preset = f"{preset}{options.get('layout_magic_count', 2)}"
    ring = options.get("layout_magic_ring")
    return f"{preset}_m{ring}" if ring else preset


def flatten(payload: Mapping[str, Any], *, exp_id: str = "", stage: str = "",
            point: str = "") -> Dict[str, Any]:
    config, result = payload["config"], payload["result"]
    options = config.get("options") or {}
    meta = result.get("metadata") or {}
    baseline, variant = result.get("baseline"), result.get("variant")
    row: Dict[str, Any] = {
        "exp_id": exp_id, "stage": stage, "point": point,
        "circuit": result.get("circuit"),
        "family": circuit_family(result.get("circuit") or ""),
        "synthesis": synthesis_kind(circuit_family(result.get("circuit") or "")),
        "compiler": compiler_label(baseline, variant).split(" (")[0],
        "config": compiler_label(baseline, variant),
        "baseline": baseline, "variant": variant,
        "comparison_mode": result.get("comparison_mode"),
        "layout": _layout_name(options, baseline),
        "layout_preset": options.get("layout_preset", "ring" if baseline in ("harvest", "silva") else None),
        "layout_magic_count": options.get("layout_magic_count"),
        "layout_magic_ring": options.get("layout_magic_ring"),
        "placement": options.get("placement"),
        "magic_model": options.get("magic_model", "free"),
        "magic_mean_cycles": options.get("cultivation_mean_cycles", options.get("magic_mean_cycles")),
        "fault": options.get("fault"),
        "scheduler_mode": options.get("scheduler", "harvest" if baseline == "harvest" else None),
        "trial": result.get("trial"), "seed": result.get("seed"),
        "status": run_status(result),
        "error": result.get("error"),
        "num_qubits": result.get("num_qubits"),
        "T_cycles": result.get("schedule_length"),
        "area_tiles": result.get("logical_patches"),
        "volume_tile_cycles": result.get("space_time_volume"),
        "area_initial_tiles": result.get("initial_logical_patches"),
        "area_pruned_tiles": result.get("post_pruning_logical_patches"),
        "volume_initial": result.get("space_time_volume_initial"),
        "volume_pruned": result.get("space_time_volume_post_pruning"),
        "compile_time_s": result.get("compiler_runtime_s"),
        "peak_mem_mb": result.get("peak_memory_mb"),
        "process_peak_rss_mb": meta.get("process_peak_rss_mb"),
        "input_sha256": config.get("input_sha256"),
        "upstream_version": result.get("upstream_version"),
    }
    if baseline == "harvest":
        cfg = HarvestConfig.from_mapping(options.get("harvest_config"))
        row["harvest_config"] = cfg.to_json()
        for key, value in cfg.to_dict().items():
            row[f"cfg_{key}"] = value
    validation = meta.get("validation") or {}
    for key in ("validated_pre", "validated_post", "n_violations", "first_violation_rule",
                "validate_time_s"):
        row[key] = validation.get(key)
    for rule in ("V1", "V2", "V3", "V4"):
        row[f"violations_{rule}"] = sum(
            ((validation.get(stage_key) or {}).get("violations_by_rule") or {}).get(rule, 0)
            for stage_key in ("validation_pre", "validation_post")
        ) if validation else None
    if options.get("fault"):
        row["fault_activations"] = (meta.get("fault_activations") or {}).get(options["fault"], 0)
    for key, value in (meta.get("phase_times_s") or {}).items():
        row[key] = value
    instrumentation = meta.get("instrumentation") or {}
    for key, value in instrumentation.items():
        row[key] = value
    area = meta.get("area") or {}
    for stage_key, suffix in (("initial", "initial"), ("pruned", "pruned")):
        for kind in ("routing", "magic", "data"):
            row[f"{kind}_tiles_{suffix}"] = (area.get(stage_key) or {}).get(f"{kind}_tiles")
    if row["T_cycles"] is not None and row["volume_initial"] is not None and row["volume_pruned"]:
        row["volume_removed_frac"] = 1 - row["volume_pruned"] / row["volume_initial"]
    return row


def add_bounds(row: Dict[str, Any], features: Mapping[str, Any],
               flasq_floor: Optional[float]) -> None:
    """Normalize against the matched-IR bounds (``tradeoff``) and the FLASQ floor."""
    t_lb = features.get("critical_path") if features else None
    n = row.get("num_qubits") or (features or {}).get("num_qubits")
    row["T_lb_cycles"] = t_lb
    row["area_lb_tiles"] = (n + 1) if n is not None else None  # data patches + one magic
    row["T_flasq_cycles"] = flasq_floor
    T, area = row.get("T_cycles"), row.get("area_tiles")
    row["T_over_lb"] = T / t_lb if T and t_lb else None
    row["area_over_lb"] = area / row["area_lb_tiles"] if area and row["area_lb_tiles"] else None
    row["T_over_flasq"] = T / flasq_floor if T and flasq_floor else None
    row["volume_over_lb"] = (row["T_over_lb"] * row["area_over_lb"]
                             if row["T_over_lb"] and row["area_over_lb"] else None)
    for key in ("parallelism", "t_fraction", "mean_pauli_weight", "y_fraction", "ops",
                "critical_path"):
        row.setdefault(f"feat_{key}", (features or {}).get(key))


def load_raw(directory: Path) -> Iterable[Dict[str, Any]]:
    for path in sorted(Path(directory).glob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if "config" in payload and "result" in payload:
            yield payload


def flasq_floors(cache_dir: Path, assumptions: str) -> Dict[str, float]:
    """FLASQ's best depth per circuit from ``results/flasq_cache`` (``matched``/``cultivation``)."""
    floors: Dict[str, float] = {}
    for path in sorted(Path(cache_dir).glob("*.json")):
        try:
            entry = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if entry.get("parameters", {}).get("assumptions") != assumptions:
            continue
        frontier = entry.get("record", {}).get("frontier") or []
        if frontier:
            depth = min(float(f["total_depth"]) for f in frontier)
            key = entry["circuit_key"]
            floors[key] = min(floors.get(key, depth), depth)
    return floors


def ordered_columns(rows: List[Dict[str, Any]]) -> List[str]:
    present = {k for row in rows for k in row}
    return [c for c in LEAD_COLUMNS if c in present] + sorted(present - set(LEAD_COLUMNS))
