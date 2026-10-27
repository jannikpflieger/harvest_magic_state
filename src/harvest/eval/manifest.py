"""Run metadata: ``manifest.json`` per experiment, ``runs.csv`` per tool run (EVAL_PLAN B.2)."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]

RUNS_FIELDS = (
    "exp_id", "point", "circuit", "input_sha256", "compiler", "variant", "comparison_mode",
    "input_representation", "layout_preset", "magic_model", "magic_params", "seed", "trial",
    "timeout_s", "status", "upstream_version", "upstream_commit", "command", "executable",
    "executable_sha256", "wall_time_s", "peak_mem_mb", "notes",
)
PACKAGES = ("qiskit", "networkx", "scipy", "numpy", "matplotlib", "pandas", "bqskit")


def _git(*args: str) -> Optional[str]:
    proc = subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True,
                          check=False)
    return proc.stdout.strip() if proc.returncode == 0 else None


def git_state() -> Dict[str, Any]:
    status = _git("status", "--porcelain")
    return {
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": None if status is None else bool(status),
    }


def _cpu_model() -> Optional[str]:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def _ram_gb() -> Optional[float]:
    try:
        pages, size = os.sysconf("SC_PHYS_PAGES"), os.sysconf("SC_PAGE_SIZE")
        return round(pages * size / 1024 ** 3, 1)
    except (ValueError, OSError, AttributeError):
        return None


def package_versions() -> Dict[str, Optional[str]]:
    out = {}
    for name in PACKAGES:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


@lru_cache(maxsize=64)
def file_sha256(path: str) -> Optional[str]:
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def environment() -> Dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "packages": package_versions(),
        "os": platform.platform(),
        "cpu_model": _cpu_model(),
        "cpu_count": os.cpu_count(),
        "ram_gb": _ram_gb(),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
    }


def suite_hash(files: Iterable[Path]) -> str:
    """Hash of a benchmark suite: every file's name and SHA-256, sorted."""
    entries = sorted(f"{Path(f).name}:{file_sha256(str(Path(f).resolve()))}" for f in files)
    return hashlib.sha256("\n".join(entries).encode()).hexdigest()


def write_manifest(
    path: Path,
    *,
    exp_id: str,
    argv: Sequence[str],
    experiment: Mapping[str, Any],
    suite: Mapping[str, Any],
    started: str,
    harvest_config: Mapping[str, Any],
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    payload = {
        "exp_id": exp_id,
        "git": git_state(),
        "cli": list(argv),
        "experiment_config": dict(experiment),
        "environment": environment(),
        "started_utc": started,
        "finished_utc": now(),
        "suite": dict(suite),
        "harvest_config": dict(harvest_config),
        "magic_models": experiment.get("magic_models"),
        "layout_presets": experiment.get("layouts"),
        **dict(extra or {}),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    return payload


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------
# runs.csv
# --------------------------------------------------------------------------


def run_status(result: Mapping[str, Any]) -> str:
    if result.get("completed"):
        return "ok"
    if result.get("timed_out"):
        return "timeout"
    error = str(result.get("error") or "").lower()
    if error.startswith("unsupported") or any(
        marker in error for marker in ("is only qualified", "requires a post-transpilation",
                                       "not qualified", "unsupported")
    ):
        return "unsupported"
    return "error"


def magic_params(options: Mapping[str, Any]) -> Dict[str, Any]:
    keys = ("cultivation_mean_cycles", "magic_mean_cycles", "magic_state_lambda")
    return {k: options[k] for k in keys if k in options}


def run_row(exp_id: str, point: str, payload: Mapping[str, Any]) -> Dict[str, Any]:
    """One ``runs.csv`` row from a raw ``{"config", "result"}`` record."""
    config, result = payload["config"], payload["result"]
    options = config.get("options") or {}
    executable = config.get("executable")
    metadata = result.get("metadata") or {}
    upstream_commit = metadata.get("upstream_commit")
    if upstream_commit is None and result.get("baseline") in ("harvest", "silva"):
        upstream_commit = git_state()["commit"]
    return {
        "exp_id": exp_id,
        "point": point,
        "circuit": result.get("circuit"),
        "input_sha256": config.get("input_sha256"),
        "compiler": result.get("baseline"),
        "variant": result.get("variant"),
        "comparison_mode": result.get("comparison_mode"),
        "input_representation": result.get("input_representation"),
        "layout_preset": options.get("layout_preset", "ring" if result.get("baseline") in
                                     ("harvest", "silva") else None),
        "magic_model": options.get("magic_model", "free"),
        "magic_params": json.dumps(magic_params(options), sort_keys=True),
        "seed": result.get("seed"),
        "trial": result.get("trial"),
        "timeout_s": config.get("timeout_s"),
        "status": run_status(result),
        "upstream_version": result.get("upstream_version"),
        "upstream_commit": upstream_commit,
        "command": result.get("command"),
        "executable": executable,
        "executable_sha256": file_sha256(executable) if executable else None,
        "wall_time_s": result.get("compiler_runtime_s"),
        "peak_mem_mb": result.get("peak_memory_mb"),
        "notes": " | ".join(result.get("notes") or []),
    }


def append_run_row(path: Path, row: Mapping[str, Any]) -> None:
    """Append one row and flush, so a crash keeps everything written so far."""
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RUNS_FIELDS, extrasaction="ignore")
        if new:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def write_runs_csv(path: Path, rows: List[Mapping[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RUNS_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
