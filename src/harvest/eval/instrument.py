"""Run instrumentation shared by the in-tree adapters (EVAL_PLAN B.5).

Everything here reads finished scheduler output; nothing feeds back into
scheduling.
"""

from __future__ import annotations

import resource
import sys
from collections import Counter
from typing import Any, Dict, Optional, Sequence

from harvest.compilation.utils import node_needs_magic_state


def _is_magic(patch) -> bool:
    return any(port.port_type == "M" for port in patch.ports)


def initial_area(layout_engine) -> Dict[str, int]:
    """Tiles of the working layout split by type (sum == ``layout_patch_count``)."""
    eng = layout_engine
    blocked = sum(1 for value in eng.occ.values() if value == "BLOCKED")
    magic = sum(len(p.cells) for p in eng.patches.values() if _is_magic(p))
    data = sum(len(p.cells) for p in eng.patches.values() if not _is_magic(p))
    routing = eng.W * eng.H - blocked - magic - data
    return {"routing_tiles": routing, "magic_tiles": magic, "data_tiles": data,
            "total_tiles": routing + magic + data}


def retained_area(processor, num_qubits: Optional[int]) -> Dict[str, int]:
    """Tiles retained after pruning, split as ``processor_patch_count`` counts them."""
    routing = sum(isinstance(node, tuple) for node in processor.graph)
    retained = set(processor.ports_by_patch)
    if num_qubits is not None:
        retained |= {f"q_{i}" for i in range(num_qubits)}
    magic = data = 0
    for name in retained:
        patch = processor.eng.patches.get(name)
        size = len(patch.cells) if patch is not None else 1
        if patch is not None and _is_magic(patch):
            magic += size
        else:
            data += size
    return {"routing_tiles": routing, "magic_tiles": magic, "data_tiles": data,
            "total_tiles": routing + magic + data}


def critical_path(dag) -> int:
    """Longest dependency chain of the DAG at one step per operation."""
    from harvest.routing.scheduler import ReadyTracker

    nodes = list(dag.op_nodes())
    if not nodes:
        return 0
    return max(ReadyTracker(dag, nodes).heights().values())


def schedule_stats(
    dag,
    results: Sequence[Dict[str, Any]],
    total_steps: Optional[int],
    num_magic_terminals: int,
    step_log: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    routed = [r for r in results if r.get("success", True)]
    per_step = Counter(r["time_step"] for r in routed)
    steps = total_steps or 0
    ops = len(list(dag.op_nodes()))
    magic_ops = sum(1 for r in routed if node_needs_magic_state(r["node"]))
    # Delay of an operation: steps between becoming dependency-ready and running.
    step_of = {r["node"]._node_id: r["time_step"] for r in routed}
    delays = []
    for r in routed:
        preds = [p for p in dag.predecessors(r["node"]) if hasattr(p, "op")]
        ready_at = max((step_of.get(p._node_id, -1) + 1 for p in preds), default=0)
        delays.append(r["time_step"] - ready_at)
    route_lengths = [
        sum(1 for n in r.get("steiner_nodes", ()) if isinstance(n, tuple)) for r in routed
    ]
    stats: Dict[str, Any] = {
        "steps_total": total_steps,
        "steps_idle": (steps - len(per_step)) if total_steps is not None else None,
        "ops": ops,
        "critical_path": critical_path(dag),
        "magic_ops": magic_ops,
        "mean_ops_per_step": (len(routed) / steps) if steps else None,
        "max_ops_per_step": max(per_step.values(), default=0),
        "mean_op_delay_steps": (sum(delays) / len(delays)) if delays else None,
        "mean_route_len": (sum(route_lengths) / len(route_lengths)) if route_lengths else None,
        "terminal_utilization": (
            magic_ops / (num_magic_terminals * steps) if num_magic_terminals and steps else None
        ),
    }
    if step_log is not None:
        attempted_steps = [s for s in step_log if s.get("executable", 0) > 0]
        deferrals = [s["executable"] - s["routed"] for s in attempted_steps]
        stats.update({
            "nego_iterations_total": sum(s.get("iterations", 0) for s in step_log),
            "reroutes_total": sum(s.get("reroutes", 0) for s in step_log),
            "dropped_total": sum(s.get("dropped", 0) for s in step_log),
            "filled_total": sum(s.get("filled", 0) for s in step_log),
            "magic_retries_total": sum(s.get("magic_retries", 0) for s in step_log),
            "stalled_total": sum(s.get("stalled", 0) for s in step_log),
            "deferrals_total": sum(deferrals),
            "mean_deferrals_per_step": (sum(deferrals) / len(deferrals)) if deferrals else 0.0,
        })
    return stats


def process_peak_rss_mb() -> float:
    """Peak resident set size of this process so far (MiB; monotone across runs)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024
