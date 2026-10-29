"""The Rust router (``harvest._router``) behind the Python scheduler.

``RustRouter`` mirrors one processor's routing graph as integer-indexed
arrays once and runs a time step's packing in Rust:
``LayoutEngine.steiner_packing_harvest`` (negotiated congestion) and
``LayoutEngine.steiner_packing`` (greedy), including the router's
magic-terminal choice (``TerminalChoice``), data-port orientation and the
min-cost matching that seeds it.  Results come back in the Python format
(node objects, sets), so the scheduler, validator and pruner are unchanged.

The algorithm is the same as the Python one; ties between equally short
paths or equally good terminals may break differently (Python's set
iteration order depends on its string hashing), so schedules are equivalent
rather than identical.  Build the module with ``rust/build.sh``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from harvest import faults
from harvest.layout.engine import TerminalChoice

_DIRS = {"N": 1, "S": 2, "E": 4, "W": 8}


def available() -> bool:
    try:
        from harvest import _router  # noqa: F401
    except ImportError:
        return False
    return True


def _port_dirs(node) -> int:
    """N/S/E/W bits of a port id, as ``DAGProcessor._port_has_direction``."""
    if isinstance(node, str) and ":" in node:
        parts = node.split(":")
        if len(parts) >= 3:
            return sum(bit for d, bit in _DIRS.items() if d in parts[2])
    return 0


class RustRouter:
    """The Rust router for ``processor.graph`` (rebuilt if the graph object changes)."""

    def __init__(self, processor):
        try:
            from harvest import _router
        except ImportError as exc:
            raise ImportError(
                "HarvestConfig(backend='rust') needs the compiled router: run rust/build.sh "
                "(or use backend='python')"
            ) from exc
        graph = processor.graph
        self.graph = graph
        self.nodes: List[Any] = list(graph)
        self.index: Dict[Any, int] = {n: i for i, n in enumerate(self.nodes)}
        index = self.index
        adj = [[(index[v], float(w)) for v, w in graph[u] if v in index] for u in self.nodes]
        cells = [n if isinstance(n, tuple) else None for n in self.nodes]
        pos = [tuple(processor.pos[n]) if n in processor.pos else None for n in self.nodes]
        dirs = [_port_dirs(n) for n in self.nodes]
        self.patches = list(processor.ports_by_patch)
        self.patch_index = {p: i for i, p in enumerate(self.patches)}
        patch_ports = [
            [(ord(t), [index[p] for p in ports if p in index])
             for t, ports in processor.ports_by_patch[name].items()]
            for name in self.patches
        ]
        patch_pos = []
        for name in self.patches:
            p = processor._get_patch_position_from_name(name)
            patch_pos.append(tuple(p) if p is not None else None)
        self._router = _router.Router(adj, cells, pos, dirs, patch_ports, patch_pos,
                                      bool(getattr(processor.eng, "_manhattan_bound", False)))

    @classmethod
    def of(cls, processor) -> "RustRouter":
        router = processor.__dict__.get("_rust_router")
        if router is None or router.graph is not processor.graph:
            router = cls(processor)
            processor._rust_router = router
        return router

    # -- conversion ---------------------------------------------------------

    def _set_in(self, terminals) -> Tuple:
        index = self.index
        if terminals and isinstance(terminals[0], TerminalChoice):
            choice = terminals[0]
            specs = [(self.patch_index[patch], [ord(t) for t in types])
                     for patch, types in choice.specs]
            initial = index.get(choice.initial) if choice.initial is not None else None
            return (1, [], [index[c] for c in choice.candidates], specs, initial)
        return (0, [index[t] for t in terminals], [], [], None)

    def _result(self, terminals, out) -> Dict[str, Any]:
        success, concrete, nodes, edges, error = out
        node = self.nodes
        result = {
            "terminal_set": [node[i] for i in concrete] if concrete is not None else terminals,
            "sol_nodes": {node[i] for i in nodes},
            "sol_edges": {(node[a], node[b]) for a, b in edges},
            "success": success,
        }
        if error is not None:
            result["error"] = error
        return result

    # -- packing --------------------------------------------------------------

    def pack_harvest(self, terminal_sets: Sequence, *, max_iters: int, alpha: float, beta: float,
                     stagnation_limit: int, greedy_fill: bool, priorities=None,
                     capacity: int = 1, match_initial: bool = False):
        """``(results, stats)`` as ``LayoutEngine.steiner_packing_harvest`` returns them."""
        sets = [self._set_in(ts) for ts in terminal_sets]
        no_defer = faults.active("no_defer")
        outs, s = self._router.pack_harvest(
            sets, max_iters, float(alpha), float(beta), capacity, stagnation_limit, greedy_fill,
            [float(p) for p in priorities] if priorities is not None else None,
            no_defer, match_initial, "min_size",
        )
        if s[8]:
            faults.note("no_defer")
        stats = {
            "iterations": s[0], "initial_routes": s[1], "reroutes": s[2],
            "conflicted_reroutes": s[3], "dropped": s[4], "filled": s[5],
            "final_conflict_score": s[6], "stopped_reason": s[7],
        }
        return [self._result(ts, o) for ts, o in zip(terminal_sets, outs)], stats

    def pack_greedy(self, terminal_sets: Sequence, *, match_initial: bool = False):
        """Results as ``LayoutEngine.steiner_packing`` returns them."""
        sets = [self._set_in(ts) for ts in terminal_sets]
        outs = self._router.pack_greedy(sets, "min_size", match_initial)
        return [self._result(ts, o) for ts, o in zip(terminal_sets, outs)]

    def steiner_tree(self, terminals, node_cost: Optional[Dict] = None, blocked=None):
        """``LayoutEngine.steiner_tree`` on this graph (raises ``ValueError``)."""
        index = self.index
        cost = {index[n]: float(c) for n, c in node_cost.items() if n in index} if node_cost is not None else None
        block = [index[n] for n in blocked if n in index] if blocked is not None else None
        nodes, edges = self._router.steiner_tree([index[t] for t in terminals], cost, block)
        node = self.nodes
        return {node[i] for i in nodes}, {(node[a], node[b]) for a, b in edges}

    def distances(self, source) -> Dict[Any, float]:
        """Uncongested distance from ``source`` to every reachable node."""
        inf = float("inf")
        return {self.nodes[i]: d for i, d in enumerate(self._router.distances(self.index[source]))
                if d != inf}
