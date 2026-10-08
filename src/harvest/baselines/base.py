"""Common, JSON-serializable interface for external and in-tree baselines.

The normalized schema deliberately permits missing metrics.  A baseline must
leave a value as ``None`` when its native output does not define that value;
adapters must never manufacture a number merely to fill a plot column.
"""

from __future__ import annotations

import hashlib
import json
import re
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

COMPARISON_MODES = {"matched_ir", "matched_architecture", "native_end_to_end"}
_DEFAULT_SEED = object()


@lru_cache(maxsize=256)
def _file_sha256(path: str, mtime_ns: int, size: int) -> str:
    """Hash an input once per observed file revision."""
    del mtime_ns, size  # values deliberately participate in the cache key
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_name(value: str) -> str:
    """Return a deterministic, filesystem-safe identifier."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_.")
    return cleaned or "unnamed"


@dataclass(frozen=True)
class BaselineRunConfig:
    """Configuration shared by all baseline adapters."""

    baseline: str
    variant: str
    circuit: str
    input_path: Path
    output_dir: Path
    comparison_mode: str
    num_qubits: Optional[int] = None
    trial: int = 0
    seed: Optional[int] = None
    timeout_s: float = 1800.0
    executable: Optional[Path] = None
    upstream_version: str = "unknown"
    options: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.comparison_mode not in COMPARISON_MODES:
            raise ValueError(
                f"comparison_mode must be one of {sorted(COMPARISON_MODES)}, "
                f"got {self.comparison_mode!r}"
            )
        if self.trial < 0:
            raise ValueError("trial must be non-negative")
        if self.timeout_s <= 0:
            raise ValueError("timeout_s must be positive")

    def work_dir(self) -> Path:
        """Directory in which this one run stores every native artifact."""
        seed = "na" if self.seed is None else str(self.seed)
        return (
            self.output_dir
            / safe_name(self.circuit)
            / safe_name(self.baseline)
            / safe_name(self.variant)
            / f"trial_{self.trial:03d}_seed_{seed}"
        )

    def serializable(self) -> Dict[str, Any]:
        data = asdict(self)
        for key in ("input_path", "output_dir", "executable"):
            value = data.get(key)
            data[key] = str(value) if value is not None else None
        try:
            resolved_input = self.input_path.resolve()
            stat = resolved_input.stat()
            data["input_sha256"] = _file_sha256(
                str(resolved_input), stat.st_mtime_ns, stat.st_size
            )
        except OSError:
            data["input_sha256"] = None
        return data


@dataclass
class BaselineResult:
    """Normalized result for one trial of one compiler/baseline."""

    baseline: str
    variant: str
    circuit: str
    num_qubits: Optional[int]
    input_representation: str
    comparison_mode: str
    schedule_length: Optional[int] = None
    logical_patches: Optional[int] = None
    space_time_volume: Optional[int] = None
    routing_wirelength: Optional[int] = None
    compiler_runtime_s: Optional[float] = None
    peak_memory_mb: Optional[float] = None
    completed: bool = False
    trial: int = 0
    seed: Optional[int] = None
    upstream_version: str = "unknown"
    command: Optional[str] = None
    notes: List[str] = field(default_factory=list)
    timed_out: bool = False
    error: Optional[str] = None
    initial_logical_patches: Optional[int] = None
    post_pruning_logical_patches: Optional[int] = None
    space_time_volume_initial: Optional[int] = None
    space_time_volume_post_pruning: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.comparison_mode not in COMPARISON_MODES:
            raise ValueError(f"Unknown comparison mode: {self.comparison_mode!r}")
        for name in (
            "schedule_length",
            "logical_patches",
            "space_time_volume",
            "routing_wirelength",
            "initial_logical_patches",
            "post_pruning_logical_patches",
        ):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} cannot be negative")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def failed(
        cls,
        config: BaselineRunConfig,
        *,
        input_representation: str,
        error: str,
        command: Optional[str] = None,
        timed_out: bool = False,
        notes: Optional[List[str]] = None,
        runtime_s: Optional[float] = None,
        seed: Any = _DEFAULT_SEED,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "BaselineResult":
        return cls(
            baseline=config.baseline,
            variant=config.variant,
            circuit=config.circuit,
            num_qubits=config.num_qubits,
            input_representation=input_representation,
            comparison_mode=config.comparison_mode,
            compiler_runtime_s=runtime_s,
            completed=False,
            trial=config.trial,
            # Passing ``None`` explicitly is meaningful for tools such as
            # wisq/DASCOT that expose no seed control.  Omitting the argument
            # retains the requested seed for ordinary failures.
            seed=config.seed if seed is _DEFAULT_SEED else seed,
            upstream_version=config.upstream_version,
            command=command,
            notes=list(notes or []),
            timed_out=timed_out,
            error=error,
            metadata=dict(metadata or {}),
        )


class BaselineAdapter(ABC):
    """Minimal interface implemented by all baseline integrations."""

    name: str

    @abstractmethod
    def run(self, config: BaselineRunConfig, **kwargs: Any) -> BaselineResult:
        """Execute or evaluate one configured trial."""


def deterministic_result_name(config: BaselineRunConfig) -> str:
    """Return a stable raw-result filename for resume and reproducibility."""
    seed = "na" if config.seed is None else str(config.seed)
    parts = (
        safe_name(config.circuit),
        safe_name(config.baseline),
        safe_name(config.variant),
        safe_name(config.comparison_mode),
        f"trial-{config.trial:03d}",
        f"seed-{seed}",
    )
    return "__".join(parts) + ".json"


def write_result(path: Path, result: BaselineResult, config: BaselineRunConfig) -> None:
    """Atomically write a raw result with the exact run configuration."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"config": config.serializable(), "result": result.to_dict()}
    encoded = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(encoded)
    tmp.replace(path)


def stable_payload_hash(payload: Dict[str, Any]) -> str:
    """Short content hash used in environment/provenance records."""
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]
