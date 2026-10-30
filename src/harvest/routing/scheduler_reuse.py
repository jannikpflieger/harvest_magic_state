"""HARVEST-2: the HARVEST scheduler with a reuse cost on unused tiles.

Pruning keeps every routing cell and magic patch some route used, so a
schedule that spreads its routes over many tiles keeps a large footprint.
HARVEST-2 adds a cost for entering a tile no earlier step has used:

* ``cell``: every routing cell not yet in a committed route;
* ``magic``: every port of a magic patch none of whose ports has been used.

The cost enters every routing decision of the negotiated packing (initial
routes, the magic-terminal matching, rip-up and reroute, greedy fill) on top
of the congestion costs, so routes prefer tiles that are already kept.  Once
used, a tile is free.  Everything else (time-step loop, admission, priorities,
negotiation) is :func:`~harvest.routing.scheduler.process_dag_with_harvest`
unchanged; only the router is swapped.  Requires ``backend="rust"``.

The plain HARVEST scheduler is untouched: its router call takes no reuse cost
and runs the same code as before.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

from harvest import faults

from .config import HarvestConfig
from .rust_backend import RustRouter
from .scheduler import process_dag_with_harvest


@dataclass(frozen=True)
class ReuseCost:
    # Added to the cost of entering an unused routing cell (one cell step costs 1).
    cell: float = 1.0
    # Added to the cost of entering a port of an unused magic patch.
    magic: float = 2.0

    @classmethod
    def from_mapping(cls, values: Optional[Mapping[str, Any]]) -> "ReuseCost":
        return cls(**dict(values or {}))


class ReuseRouter(RustRouter):
    """``RustRouter`` whose negotiated packing adds the reuse cost."""

    def __init__(self, processor, cost: ReuseCost):
        super().__init__(processor)
        self.cost = cost
        self.calls = 0
        magic_ports = set(processor.magic_terminals)
        patch_of = processor.patch_used_by_port
        self.patch_of = [patch_of.get(n) if n in magic_ports else None for n in self.nodes]
        self.base = [cost.cell if isinstance(n, tuple) else (cost.magic if p is not None else 0.0)
                     for n, p in zip(self.nodes, self.patch_of)]
        self.ports_of: Dict[Any, list] = {}
        for i, p in enumerate(self.patch_of):
            if p is not None:
                self.ports_of.setdefault(p, []).append(i)

    def _mark_used(self, nodes) -> None:
        base, index = self.base, self.index
        for n in nodes:
            i = index.get(n)
            if i is None or base[i] == 0.0:
                continue
            base[i] = 0.0
            patch = self.patch_of[i]
            if patch is not None:
                for j in self.ports_of[patch]:
                    base[j] = 0.0

    def pack_harvest(self, terminal_sets, *, max_iters, alpha, beta, stagnation_limit, greedy_fill,
                     priorities=None, capacity=1, match_initial=False):
        self.calls += 1
        sets = [self._set_in(ts) for ts in terminal_sets]
        no_defer = faults.active("no_defer")
        outs, s = self._router.pack_harvest_reuse(
            sets, list(self.base), max_iters, float(alpha), float(beta), capacity, stagnation_limit,
            greedy_fill, [float(p) for p in priorities] if priorities is not None else None,
            no_defer, match_initial, "min_size",
        )
        if s[8]:
            faults.note("no_defer")
        stats = {
            "iterations": s[0], "initial_routes": s[1], "reroutes": s[2],
            "conflicted_reroutes": s[3], "dropped": s[4], "filled": s[5],
            "final_conflict_score": s[6], "stopped_reason": s[7],
        }
        results = [self._result(ts, o) for ts, o in zip(terminal_sets, outs)]
        # Admission control is on, so a successful route is committed this step.
        for r in results:
            if r["success"]:
                self._mark_used(r["sol_nodes"])
        return results, stats


def process_dag_with_harvest_reuse(processor, dag, visualize_each_step=False,
                                   config: Optional[HarvestConfig] = None,
                                   cost: Optional[ReuseCost] = None):
    """HARVEST-2: :func:`process_dag_with_harvest` routed by :class:`ReuseRouter`."""
    config = config or HarvestConfig()
    if config.backend != "rust" or not config.negotiate or not config.admission:
        raise ValueError("HARVEST-2 needs backend='rust', negotiate=True and admission=True")
    router = ReuseRouter(processor, cost or ReuseCost())
    processor._rust_router = router  # RustRouter.of returns it while the graph is unchanged
    results = process_dag_with_harvest(processor, dag, visualize_each_step, config=config)
    if processor.__dict__.get("_rust_router") is not router:
        raise RuntimeError("HARVEST-2: the routing graph changed during scheduling")
    processor._scheduling_metadata["reuse_cost"] = {"cell": router.cost.cell, "magic": router.cost.magic,
                                                    "router_calls": router.calls}
    return results
