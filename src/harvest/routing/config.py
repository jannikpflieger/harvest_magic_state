"""Every HARVEST scheduler knob in one place.

The defaults are the paper configuration (one fixed setting for every
circuit); the evaluation's ablation and sensitivity sweeps (EVAL_PLAN E3)
override individual fields.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Mapping, Optional


@dataclass(frozen=True)
class HarvestConfig:
    # Order in which ready products compete for magic terminals and routing
    # space: "dag_order" or "critical_path" (longest remaining dependency
    # chain first, ties in DAG order).
    priority: str = "critical_path"
    # Also route high-priority products first and, when the congestion loop
    # must drop products, drop the lowest-priority ones.
    router_priority: bool = True
    # Magic-terminal assignment: "nearest" (greedy per product, Euclidean
    # distance) or "matching" (one optimal assignment of free terminals to
    # products minimizing total routing distance).
    terminals: str = "matching"
    # False: plain greedy Steiner packing (no negotiated congestion).
    negotiate: bool = True
    max_iters: int = 4
    alpha: float = 2.0
    beta: float = 1.5
    stagnation_limit: int = 2
    # Retry dropped/unrouted products in the space left by committed routes.
    greedy_fill: bool = True
    # Re-route failed products through another free magic terminal.
    magic_retry: bool = True

    def __post_init__(self) -> None:
        if self.priority not in ("critical_path", "dag_order"):
            raise ValueError(f"unknown priority {self.priority!r}")
        if self.terminals not in ("matching", "nearest"):
            raise ValueError(f"unknown terminals policy {self.terminals!r}")
        if self.max_iters < 0 or self.stagnation_limit < 1:
            raise ValueError("max_iters must be >= 0 and stagnation_limit >= 1")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        """Compact, key-sorted JSON (the ``harvest_config`` data column)."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def overrides(self) -> Dict[str, Any]:
        """Only the fields that differ from the defaults."""
        default = HarvestConfig()
        return {f.name: getattr(self, f.name) for f in fields(self)
                if getattr(self, f.name) != getattr(default, f.name)}

    @classmethod
    def from_mapping(cls, values: Optional[Mapping[str, Any]]) -> "HarvestConfig":
        values = dict(values or {})
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown HarvestConfig field(s): {sorted(unknown)}")
        return cls(**values)


# Named configurations of the scheduler ablation (EVAL_PLAN E3), cumulative
# from plain greedy packing to the full scheduler.  ``seq`` (sequential
# Steiner trees) and ``eaf`` (Silva et al.) are separate schedulers, not
# HarvestConfig settings.
_PACK = dict(negotiate=False, priority="dag_order", terminals="nearest",
             greedy_fill=False, magic_retry=False)
ABLATION_VARIANTS: Dict[str, Dict[str, Any]] = {
    "pack": _PACK,
    "nego": {**_PACK, "negotiate": True},
    "+prio": {**_PACK, "negotiate": True, "priority": "critical_path"},
    "+match": {**_PACK, "negotiate": True, "priority": "critical_path", "terminals": "matching"},
    "+fill": {},  # greedy fill + terminal retry on top of +match == full HARVEST
    "full": {},
}
