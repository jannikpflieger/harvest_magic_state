"""O3LS (Zhu et al., ISCA 2026) reimplemented as a matched-IR baseline.

Paper: "O3LS: Optimizing Lattice Surgery via Automatic Layout Searching and
Loose Scheduling", arXiv:2604.15099.  No implementation was released, so this
module is a reimplementation from the paper text (in the spirit of RushHour,
arXiv:2608.18985, App. A).  It reads the exact matched ``±π/8`` Pauli-product
stream PureMagic receives (``pauli_dag_to_trans``) and runs the four O3LS
modules on it:

1. **Layout search (§III-A).**  Score ``S(B) = C(B)·(N_x + N_z − α_e·N_e)``
   with ``α_e = 0.1``.  The ancilla patch ``A`` sits in corner ``(0, 0)``;
   data patches are placed one by one at the best-scoring free tile, and after
   every placement the best strictly improving one-step relocation is adopted
   repeatedly.  ``C(B) = 1`` iff both of ``A``'s in-board edges (X side
   ``(0, 1)``, Z side ``(1, 0)``) lie in one routing component that touches at
   least one edge of every data patch.
2. **Y-synthesis (§III-B, Alg. 1).**  Every Y-containing π/8 rotation is
   rewritten with Z-type π/4 conjugations: one group for an odd number of Y's,
   two odd groups for an even number.  The bipartition maximises absorption by
   predecessors (an ``R`` rotation still commuting to the end of the emitted
   sequence with identical support) and then potential absorption by the next
   anticommuting successor; ties fall back to the prior-work split
   (singleton of the lowest Y index).  Absorbed pairs ``(Z_G)_{-π/4}
   (Z_G)_{+π/4}`` cancel exactly ("Pauli operator synthesis").
3. **Loose scheduling (§III-C, Alg. 2).**  Every iteration executes all
   executable frontier operators in one time step (node-disjoint routing
   trees grown by sequential BFS/Dijkstra from the already chosen tree, i.e.
   chosen tiles cost 0), then pops the lowest-index frontier operator and, while
   it is not executable, applies the patch operation with the best reward
   (#its patches whose required edge reaches the routing space; 0 if the
   operation disconnects any data patch or ``A``; ties → lower time cost).
4. **Edge-aware mapping (§III-D).**  Qubits are ranked by rotation demand
   (number of X↔Z basis switches along their chain in the synthesised PDAG)
   and mapped to patches ranked by how many edge types they expose.

Cost model (§II-B, Fig. 3, following Litinski 2019): π/8 rotation = P⊗Z
measurement with ``|m⟩`` (1 step); π/4 rotation = P⊗Y measurement with ``A``,
which needs both of ``A``'s edges in the routing tree (1 step); move to an
adjacent tile 1 step; patch rotation 3 steps; initialisation/shrink 0.

Matched-IR magic-state assumption (shared with HARVEST, Silva and Litinski
adapters): magic states are free and always ready at one port tile.  The port
sits outside the board next to ``A``'s X-side routing tile ``(0, 1)``, which
the layout search guarantees to be routing space; it is charged as one tile.
"""

from __future__ import annotations

import heapq
import random
import math
from collections import deque
from dataclasses import dataclass, field
from functools import lru_cache
from time import perf_counter
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple

from .base import BaselineAdapter, BaselineResult, BaselineRunConfig
from .cultivation import cultivation_mean, sample_cycles
from .litinski import parse_pauli_stream
from .metrics import space_time_volume

VARIANTS = ("o3ls_min", "o3ls_medium", "o3ls_large")
FIXED_VARIANT = "o3ls_fixed"  # requires config.options["board"] = [rows, cols]
AREA_FACTORS = {"o3ls_min": 1.0, "o3ls_medium": 1.5, "o3ls_large": 2.0}
ALPHA_E = 0.1
MAGIC_PORT_TILES = 1

# §II-B / Fig. 3 cost rules (time steps).
PPM_COST = 1
MOVE_COST = 1
ROTATE_COST = 3
INIT_COST = 0
SHRINK_COST = 0

ROUTING = -1
ANCILLA = -2
_SCORE_SCALE = 1000
_LIVE_CAP = 256  # bound on pending R rotations tracked for absorption
_SUCCESSOR_HORIZON = 64  # look-ahead window for successor absorption
_N, _S, _W, _E = 0, 1, 2, 3


class O3LSError(RuntimeError):
    """Raised when the reimplementation cannot make progress (fail loudly)."""


# --------------------------------------------------------------------------
# Cost rules
# --------------------------------------------------------------------------


def operation_cost(kind: str) -> int:
    """Time steps of one primitive of the §II-B cost model."""
    costs = {
        "init": INIT_COST,
        "shrink": SHRINK_COST,
        "expand": 1,
        "move": MOVE_COST,
        "rotate": ROTATE_COST,
        "pi8": PPM_COST,
        "pi4": PPM_COST,
    }
    if kind not in costs:
        raise ValueError(f"unknown O3LS operation {kind!r}")
    return costs[kind]


# --------------------------------------------------------------------------
# Module 2: Y-synthesis
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SynthOp:
    """One Y-free rotation: ``word`` over ``X``/``Z``/``_``; ``kind`` pi8 or pi4."""

    word: str
    kind: str
    source: int  # index of the input π/8 rotation it was derived from


@dataclass
class SynthesisStats:
    pi8: int = 0
    pi4_emitted: int = 0  # generated by Y-decomposition (2 per group)
    pi4_cancelled: int = 0
    odd_y_rotations: int = 0
    even_y_rotations: int = 0
    non_default_bipartitions: int = 0


def _z_anticommutes(group: FrozenSet[int], word: str) -> bool:
    """Whether ``Z_group`` anticommutes with Pauli ``word``."""
    return sum(1 for q in group if word[q] in "XY") % 2 == 1


def _valid_group(group: FrozenSet[int], yset: FrozenSet[int]) -> bool:
    """``group`` can be one Z-conjugation group of a rotation with Y set ``yset``."""
    if not group or len(group) % 2 == 0 or not group <= yset:
        return False
    if len(yset) % 2 == 1:
        return group == yset
    return group != yset  # even: proper odd subset (complement is then odd)


def y_synthesis(words: Sequence[str]) -> Tuple[List[SynthOp], SynthesisStats]:
    """Algorithm 1: Y-decompose with absorption-aware bipartitions, then cancel.

    Returns the Y-free rotation sequence (π/8 rotations and the surviving
    Z-type π/4 conjugations) in a valid topological order.
    """
    stats = SynthesisStats()
    ysets = [frozenset(i for i, p in enumerate(w) if p == "Y") for w in words]
    out: List[Optional[SynthOp]] = []
    live: Dict[FrozenSet[int], int] = {}  # pending R support -> output index

    def append(op: SynthOp) -> None:
        # Pending R rotations that anticommute with op can no longer be reached.
        for key in [k for k in live if _z_anticommutes(k, op.word)]:
            del live[key]
        out.append(op)

    def succ_potential(i: int, group: FrozenSet[int]) -> int:
        for k in range(i + 1, min(len(words), i + 1 + _SUCCESSOR_HORIZON)):
            if _z_anticommutes(group, words[k]):
                return 1 if _valid_group(group, ysets[k]) else 0
        return 0

    for i, word in enumerate(words):
        yset = ysets[i]
        if not yset:
            stats.pi8 += 1
            append(SynthOp(word, "pi8", i))
            continue
        ys = sorted(yset)
        if len(ys) % 2 == 1:
            stats.odd_y_rotations += 1
            groups: List[FrozenSet[int]] = [yset]
        else:
            stats.even_y_rotations += 1
            candidates: List[FrozenSet[int]] = [frozenset([ys[0]])]
            candidates += sorted(
                (k for k in live if _valid_group(k, yset)), key=lambda g: sorted(g)
            )
            for k in range(i + 1, min(len(words), i + 1 + _SUCCESSOR_HORIZON)):
                inter = yset & ysets[k]
                for g in (inter, yset - inter):
                    if _valid_group(g, yset):
                        candidates.append(g)
            best = None
            seen = set()
            for order, g1 in enumerate(candidates):
                g2 = yset - g1
                key = min(sorted(g1), sorted(g2))
                if tuple(key) in seen:
                    continue
                seen.add(tuple(key))
                score = (
                    (g1 in live) + (g2 in live),
                    succ_potential(i, g1) + succ_potential(i, g2),
                )
                if best is None or score > best[0]:
                    best = (score, order, g1, g2)
            assert best is not None
            if best[1] != 0:
                stats.non_default_bipartitions += 1
            g1, g2 = best[2], best[3]
            groups = sorted([g1, g2], key=lambda g: sorted(g))
        # Left conjugations (+π/4): absorbed by a matching pending R (−π/4).
        for g in groups:
            stats.pi4_emitted += 1
            if g in live:
                out[live.pop(g)] = None
                stats.pi4_cancelled += 2  # this L and the absorbed earlier R
            else:
                append(SynthOp(_z_word(g, len(word)), "pi4", i))
        stripped = word.replace("Y", "X")
        stats.pi8 += 1
        append(SynthOp(stripped, "pi8", i))
        for g in reversed(groups):
            stats.pi4_emitted += 1
            append(SynthOp(_z_word(g, len(word)), "pi4", i))
            live[g] = len(out) - 1
        if len(live) > _LIVE_CAP:
            for key in sorted(live, key=live.get)[: len(live) - _LIVE_CAP]:
                del live[key]
    return [op for op in out if op is not None], stats


def _z_word(group: FrozenSet[int], n: int) -> str:
    return "".join("Z" if q in group else "_" for q in range(n))


# --------------------------------------------------------------------------
# Board geometry
# --------------------------------------------------------------------------


def _sides(orientation: int, basis: str) -> Tuple[int, int]:
    """Directions of the ``basis`` edges.  o=0: X edges W/E, Z edges N/S."""
    if basis == "X":
        return (_W, _E) if orientation == 0 else (_N, _S)
    return (_N, _S) if orientation == 0 else (_W, _E)


class Grid:
    """Static N×M tile geometry with the ancilla in corner (0, 0)."""

    def __init__(self, rows: int, cols: int):
        if rows < 2 or cols < 2:
            raise ValueError("O3LS boards need at least 2x2 tiles")
        self.rows, self.cols = rows, cols
        self.size = rows * cols
        nbr4 = []
        for t in range(self.size):
            r, c = divmod(t, cols)
            nbr4.append(
                (
                    t - cols if r > 0 else -1,
                    t + cols if r < rows - 1 else -1,
                    t - 1 if c > 0 else -1,
                    t + 1 if c < cols - 1 else -1,
                )
            )
        self.nbr4 = nbr4
        self.nbrs = [tuple(u for u in n4 if u >= 0) for n4 in nbr4]
        self.ancilla = 0
        self.a_x = 1  # A's X edge (east) faces (0, 1); the |m> port attaches here
        self.a_z = cols  # A's Z edge (south) faces (1, 0)

    def component(self, occ: Sequence[int]) -> bytearray:
        """Routing tiles connected to A's X-side tile (empty if it is blocked)."""
        seen = bytearray(self.size)
        start = self.a_x
        if occ[start] != ROUTING:
            return seen
        seen[start] = 1
        stack = [start]
        nbrs = self.nbrs
        while stack:
            t = stack.pop()
            for u in nbrs[t]:
                if not seen[u] and occ[u] == ROUTING:
                    seen[u] = 1
                    stack.append(u)
        return seen

    def exposure(self, tile: int, orientation: int, main: bytearray) -> Tuple[bool, bool, int]:
        """``(X exposed, Z exposed, #edges on routing space)`` of one patch."""
        n4 = self.nbr4[tile]
        ex = ez = False
        ne = 0
        for basis in ("X", "Z"):
            for d in _sides(orientation, basis):
                u = n4[d]
                if u >= 0 and main[u]:
                    ne += 1
                    if basis == "X":
                        ex = True
                    else:
                        ez = True
        return ex, ez, ne

    def coords(self, tile: int) -> Tuple[int, int]:
        return divmod(tile, self.cols)


def score_board(
    grid: Grid, occ: Sequence[int], patches: Sequence[Tuple[int, int]], alpha_milli: int
) -> Optional[int]:
    """Scaled ``S(B)``; ``None`` when ``C(B) = 0``."""
    main = grid.component(occ)
    if not main[grid.a_z]:
        return None
    nx = nz = ne = 0
    for tile, orient in patches:
        ex, ez, k = grid.exposure(tile, orient, main)
        if not k:
            return None
        nx += ex
        nz += ez
        ne += k
    return _SCORE_SCALE * (nx + nz) - alpha_milli * ne


def board_is_valid(grid: Grid, occ: Sequence[int], patches: Sequence[Tuple[int, int]]) -> bool:
    """``C(B) = 1``: A has both edges on one routing component reaching all patches."""
    return score_board(grid, occ, patches, 0) is not None


# --------------------------------------------------------------------------
# Module 1: layout search
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Layout:
    rows: int
    cols: int
    patches: Tuple[Tuple[int, int], ...]  # (tile, orientation) in placement order
    score: int


def _terms(grid: Grid, main: bytearray, patches, skip: int = -1) -> Optional[Tuple[int, int]]:
    """``(N_x + N_z, N_e)`` over ``patches`` except ``skip``; ``None`` if C(B)=0."""
    if not main[grid.a_z]:
        return None
    nxz = ne = 0
    for pid, (tile, orient) in enumerate(patches):
        if pid == skip:
            continue
        ex, ez, k = grid.exposure(tile, orient, main)
        if not k:
            return None
        nxz += ex + ez
        ne += k
    return nxz, ne


def _candidate_scores(grid, main, patches, skip, tile, alpha_milli):
    """Scaled scores of the variable patch at ``tile`` for orientations 0 and 1."""
    base = _terms(grid, main, patches, skip)
    if base is None:
        return []
    out = []
    for o in (0, 1):
        ex, ez, k = grid.exposure(tile, o, main)
        if k:
            out.append((_SCORE_SCALE * (base[0] + ex + ez) - alpha_milli * (base[1] + k), o))
    return out


def layout_search(num_patches: int, rows: int, cols: int, alpha_e: float = ALPHA_E) -> Optional[Layout]:
    """Greedy §III-A layout design; ``None`` if some patch cannot keep C(B)=1."""
    grid = Grid(rows, cols)
    if num_patches + 3 > grid.size:
        return None
    alpha_milli = int(round(alpha_e * _SCORE_SCALE))
    occ = [ROUTING] * grid.size
    occ[grid.ancilla] = ANCILLA
    patches: List[Tuple[int, int]] = []
    score = 0
    for pid in range(num_patches):
        main = grid.component(occ)
        best = None
        for t in range(grid.size):
            if occ[t] != ROUTING or t in (grid.a_x, grid.a_z):
                continue
            if not any(main[u] for u in grid.nbrs[t]):
                continue  # necessary condition: the new main is a subset of the old
            occ[t] = pid
            for s, o in _candidate_scores(grid, grid.component(occ), patches, -1, t, alpha_milli):
                if best is None or s > best[0]:
                    best = (s, t, o)
            occ[t] = ROUTING
        if best is None:
            return None
        score, t, o = best
        occ[t] = pid
        patches.append((t, o))
        score = _improve_by_moves(grid, occ, patches, score, alpha_milli)
    return Layout(rows, cols, tuple(patches), score)


def _improve_by_moves(grid, occ, patches, score, alpha_milli) -> int:
    """Adopt the best strictly improving one-step relocation until none exists."""
    for _ in range(4 * len(patches) + 4):
        best = None
        current_main = grid.component(occ)
        for pid, (t, o) in enumerate(patches):
            targets = [t] + [u for u in grid.nbrs[t] if occ[u] == ROUTING]
            for u in targets:
                if u == t:
                    main = current_main
                else:
                    occ[t], occ[u] = ROUTING, pid
                    main = grid.component(occ)
                    occ[t], occ[u] = pid, ROUTING
                for s, o2 in _candidate_scores(grid, main, patches, pid, u, alpha_milli):
                    if u == t and o2 == o:
                        continue
                    if s > score and (best is None or s > best[0]):
                        best = (s, pid, u, o2)
        if best is None:
            return score
        score, pid, u, o2 = best
        occ[patches[pid][0]] = ROUTING
        occ[u] = pid
        patches[pid] = (u, o2)
    return score


def _near_square_boards(min_area: int, limit: int = 4096):
    """Near-square boards (rows ≤ cols ≤ rows + 1) by increasing area."""
    boards = []
    for r in range(2, int(math.isqrt(limit)) + 2):
        for c in (r, r + 1):
            if r * c >= min_area:
                boards.append((r * c, r, c))
    return [(r, c) for _, r, c in sorted(boards)]


@lru_cache(maxsize=64)
def minimum_layout(num_qubits: int, alpha_e: float = ALPHA_E) -> Layout:
    """Smallest near-square board on which the greedy search reaches C(B)=1."""
    for r, c in _near_square_boards(num_qubits + 3):
        layout = layout_search(num_qubits, r, c, alpha_e)
        if layout is not None:
            return layout
    raise O3LSError(f"no near-square board admits a valid layout for {num_qubits} qubits")


@lru_cache(maxsize=64)
def select_board(num_qubits: int, area_factor: float, alpha_e: float = ALPHA_E) -> Layout:
    """Smallest valid near-square board with area >= ``area_factor`` × minimum."""
    minimum = minimum_layout(num_qubits, alpha_e)
    if area_factor <= 1.0:
        return minimum
    target = math.ceil(area_factor * minimum.rows * minimum.cols)
    for r, c in _near_square_boards(target):
        layout = layout_search(num_qubits, r, c, alpha_e)
        if layout is not None:
            return layout
    raise O3LSError(f"no board with area >= {target} admits a valid layout")


# --------------------------------------------------------------------------
# Module 4: edge-aware initial mapping
# --------------------------------------------------------------------------


def rotation_demand(ops: Sequence[SynthOp], num_qubits: int) -> List[int]:
    """X↔Z basis switches along each qubit's PDAG chain."""
    last = [""] * num_qubits
    demand = [0] * num_qubits
    for op in ops:
        for q, p in enumerate(op.word):
            if p == "_":
                continue
            if last[q] and last[q] != p:
                demand[q] += 1
            last[q] = p
    return demand


def edge_aware_mapping(layout: Layout, ops: Sequence[SynthOp], num_qubits: int) -> List[Tuple[int, int]]:
    """Return ``(tile, orientation)`` per qubit."""
    grid = Grid(layout.rows, layout.cols)
    occ = [ROUTING] * grid.size
    occ[grid.ancilla] = ANCILLA
    for pid, (t, _) in enumerate(layout.patches):
        occ[t] = pid
    main = grid.component(occ)
    exposure = []
    for pid, (t, o) in enumerate(layout.patches):
        ex, ez, _ = grid.exposure(t, o, main)
        exposure.append(int(ex) + int(ez))
    patch_order = sorted(range(len(layout.patches)), key=lambda p: (-exposure[p], p))
    demand = rotation_demand(ops, num_qubits)
    qubit_order = sorted(range(num_qubits), key=lambda q: (-demand[q], q))
    mapping: List[Tuple[int, int]] = [(-1, 0)] * num_qubits
    for q, p in zip(qubit_order, patch_order):
        mapping[q] = layout.patches[p]
    return mapping


# --------------------------------------------------------------------------
# Module 3: loose scheduling
# --------------------------------------------------------------------------


@dataclass
class ScheduleStats:
    steps: int = 0
    measurement_steps: int = 0
    moves: int = 0
    rotations: int = 0
    wirelength: int = 0
    max_parallel: int = 0
    iterations: int = 0
    magic_wait: int = 0
    trace: List[Tuple[str, Any]] = field(default_factory=list)


class _BoardState:
    def __init__(self, grid: Grid, mapping: Sequence[Tuple[int, int]]):
        self.grid = grid
        self.pos = [t for t, _ in mapping]
        self.orient = [o for _, o in mapping]
        self.occ = [ROUTING] * grid.size
        self.occ[grid.ancilla] = ANCILLA
        for q, t in enumerate(self.pos):
            self.occ[t] = q
        self.main = grid.component(self.occ)

    def side_tiles(self, q: int, basis: str, main: bytearray, pos=None, orient=None) -> List[int]:
        tile = self.pos[q] if pos is None else pos
        o = self.orient[q] if orient is None else orient
        n4 = self.grid.nbr4[tile]
        return [n4[d] for d in _sides(o, basis) if n4[d] >= 0 and main[n4[d]]]

    def valid(self, main: bytearray) -> bool:
        if not main[self.grid.a_z]:
            return False
        nbr4 = self.grid.nbr4
        for t in self.pos:
            if not any(u >= 0 and main[u] for u in nbr4[t]):
                return False
        return True


def _requirements(word: str) -> List[Tuple[int, str]]:
    return [(q, p) for q, p in enumerate(word) if p != "_"]


def _route(state: _BoardState, op: SynthOp, used: bytearray) -> Optional[List[int]]:
    """Node-disjoint routing tree: sequential BFS from the tree (tree tiles cost 0)."""
    grid = state.grid
    if used[grid.a_x]:
        return None
    groups: List[set] = []
    for q, basis in _requirements(op.word):
        tiles = {t for t in state.side_tiles(q, basis, state.main) if not used[t]}
        if not tiles:
            return None
        groups.append(tiles)
    if op.kind == "pi4":
        if used[grid.a_z] or not state.main[grid.a_z]:
            return None
        groups.append({grid.a_z})
    tree = [grid.a_x]
    in_tree = {grid.a_x}
    remaining = [g for g in groups if not (g & in_tree)]
    occ, nbrs = state.occ, grid.nbrs
    while remaining:
        targets = set().union(*remaining)
        parent = {t: -1 for t in tree}
        queue = deque(tree)
        hit = -1
        while queue:
            t = queue.popleft()
            if t in targets:
                hit = t
                break
            for u in nbrs[t]:
                if u not in parent and occ[u] == ROUTING and not used[u]:
                    parent[u] = t
                    queue.append(u)
        if hit < 0:
            return None
        t = hit
        while t not in in_tree:
            in_tree.add(t)
            tree.append(t)
            t = parent[t]
        remaining = [g for g in remaining if not (g & in_tree)]
    return tree


def _satisfied(state: _BoardState, reqs, main, pos_override=None) -> int:
    count = 0
    for q, basis in reqs:
        if pos_override is not None and q == pos_override[0]:
            tiles = state.side_tiles(q, basis, main, pos_override[1], pos_override[2])
        else:
            tiles = state.side_tiles(q, basis, main)
        count += bool(tiles)
    return count


def _best_patch_operation(state: _BoardState, op: SynthOp):
    """Algorithm 2, Step 11: reward-maximising move/rotation (ties: lower cost)."""
    grid = state.grid
    reqs = _requirements(op.word)
    current = _satisfied(state, reqs, state.main)
    support = {q for q, _ in reqs}
    best = None  # (reward, -cost, -q, -kind_rank) maximised
    for q in range(len(state.pos)):
        t = state.pos[q]
        o = state.orient[q]
        # Rotation: needs a free neighbouring tile; occupancy is unchanged.
        if q in support and any(state.occ[u] == ROUTING for u in grid.nbrs[t]):
            r = _satisfied(state, reqs, state.main, (q, t, 1 - o))
            cand = (r, -ROTATE_COST, -q, -4)
            if r > current and (best is None or cand > best[0]):
                best = (cand, ("rotate", q, t, 1 - o, None))
        # Moves to an adjacent free tile (orientation kept).
        for d, u in enumerate(grid.nbr4[t]):
            if u < 0 or state.occ[u] != ROUTING:
                continue
            state.occ[t], state.occ[u] = ROUTING, q
            state.pos[q] = u
            main = grid.component(state.occ)
            r = _satisfied(state, reqs, main) if state.valid(main) else 0
            state.pos[q] = t
            state.occ[t], state.occ[u] = q, ROUTING
            cand = (r, -MOVE_COST, -q, -d)
            if r > current and (best is None or cand > best[0]):
                best = (cand, ("move", q, u, o, main))
    if best is None:
        raise O3LSError(f"no patch operation makes progress for operator {op.word!r}")
    return best[1]


def loose_schedule(
    grid: Grid,
    mapping: Sequence[Tuple[int, int]],
    ops: Sequence[SynthOp],
    deadline: Optional[float] = None,
    record_trace: bool = False,
    cultivation: Optional[Tuple[float, Optional[int]]] = None,
) -> ScheduleStats:
    """Algorithm 2 on the per-qubit-support PDAG of ``ops``.

    ``cultivation = (mean_cycles, seed)`` applies the common cultivation model to
    the single |m> port: after a pi/8 measurement consumes its state, the next
    state is ready Exp(mean) cycles later; pi/8 operators wait for it.
    """
    rng = random.Random(0 if cultivation is None or cultivation[1] is None else int(cultivation[1]))
    magic_ready = 0
    state = _BoardState(grid, mapping)
    if not state.valid(state.main):
        raise O3LSError("initial board violates C(B)=1")
    n = len(state.pos)
    L = len(ops)
    succs: List[List[int]] = [[] for _ in range(L)]
    indeg = [0] * L
    last = [-1] * n
    for i, op in enumerate(ops):
        preds = set()
        for q, p in enumerate(op.word):
            if p != "_":
                if last[q] >= 0:
                    preds.add(last[q])
                last[q] = i
        for p in preds:
            succs[p].append(i)
        indeg[i] = len(preds)
    ready = [i for i in range(L) if indeg[i] == 0]
    heapq.heapify(ready)
    stats = ScheduleStats()
    done = 0
    cap = (3 if cultivation else 2) * L + 10
    while done < L:
        stats.iterations += 1
        if stats.iterations > cap:
            raise O3LSError("loose scheduling exceeded its iteration cap")
        if deadline is not None and perf_counter() > deadline:
            raise TimeoutError("O3LS scheduling exceeded timeout")
        used = bytearray(grid.size)
        executed = []
        magic_ok = cultivation is None or stats.steps >= magic_ready
        for i in sorted(ready):
            if ops[i].kind == "pi8" and not magic_ok:
                continue
            tree = _route(state, ops[i], used)
            if tree is None:
                continue
            for t in tree:
                used[t] = 1
            executed.append(i)
            stats.wirelength += len(tree)
        if executed:
            if cultivation is not None and any(ops[i].kind == "pi8" for i in executed):
                magic_ready = stats.steps + sample_cycles(rng, cultivation[0])
            stats.steps += PPM_COST
            stats.measurement_steps += PPM_COST
            stats.max_parallel = max(stats.max_parallel, len(executed))
            if record_trace:
                stats.trace.append(("ppm", tuple(executed)))
            gone = set(executed)
            ready = [i for i in ready if i not in gone]
            for i in executed:
                for s in succs[i]:
                    indeg[s] -= 1
                    if indeg[s] == 0:
                        ready.append(s)
            heapq.heapify(ready)
            done += len(executed)
        if not ready:
            continue
        if not executed and not magic_ok and any(ops[i].kind == "pi8" for i in ready):
            # Nothing ran because the port is still cultivating: wait for it.
            stats.magic_wait += magic_ready - stats.steps
            stats.steps = magic_ready
            continue
        j = ready[0]
        reqs = _requirements(ops[j].word)
        guard = 0
        while _satisfied(state, reqs, state.main) < len(reqs):
            guard += 1
            if guard > len(reqs) + 1:
                raise O3LSError(f"operator {j} did not become executable")
            kind, q, u, o, main = _best_patch_operation(state, ops[j])
            if kind == "rotate":
                state.orient[q] = o
                stats.rotations += 1
                stats.steps += ROTATE_COST
            else:
                state.occ[state.pos[q]] = ROUTING
                state.occ[u] = q
                state.pos[q] = u
                state.main = main
                stats.moves += 1
                stats.steps += MOVE_COST
            if record_trace:
                stats.trace.append((kind, (q, u, o, j)))
    return stats


# --------------------------------------------------------------------------
# Adapter
# --------------------------------------------------------------------------


NOTES = [
    "Reimplementation from the paper text (no public O3LS code); deterministic, no seed.",
    "Input is the matched ±pi/8 stream (Cliffords absorbed); GridSynth/transpilation of the paper's pipeline is not rerun.",
    "Magic states are free (matched-IR); one |m> port tile outside the board, attached to A's X-side tile (0,1), is charged. No factory tiles/latency.",
    "Ancilla A fixed at corner (0,0) with X edge east, Z edge south; pi/8 trees contain (0,1), pi/4 trees contain (0,1) and (1,0), so they share A's neighbourhood.",
    "Patches are one tile; orientation o=0 exposes X on W/E and Z on N/S. Board-boundary sides are not routing space.",
    "Layout search: candidates are (free tile, orientation); ties -> lowest tile index then orientation 0; post-processing adopts the best strictly improving one-step move or free re-orientation, repeated until none.",
    "Board selection (not specified in the paper): o3ls_min = smallest near-square board (rows<=cols<=rows+1) on which the greedy search reaches C(B)=1; medium/large = smallest near-square valid board with area >= 1.5x / 2x that.",
    "Y-synthesis is applied to every Y-containing rotation (Alg. 1 has no layout condition); Y bases are never measured directly.",
    "Y-synthesis bipartition: candidates = prior-work singleton, live predecessor R supports, successor-derived splits; score = (#absorbed by predecessors, #potential successor absorptions); ties -> prior-work split. Absorption requires identical support and commutation with everything emitted in between (live set capped at 256, successor look-ahead 64).",
    "Absorbed (Z_G)_{-pi/4}(Z_G)_{+pi/4} pairs cancel exactly; pi/8 sign changes are irrelevant to the cost model and not tracked.",
    "Scheduling: frontier is processed in stream-index order; executed PPMs of one iteration share one time step with node-disjoint trees; patch operations then run sequentially (no overlap with PPMs).",
    "Patch-operation set per data patch: move to an adjacent free tile (1 step, orientation kept) and in-place rotation (3 steps, needs one free neighbouring tile). A patch is 'enabled' when a side of the required basis touches the routing component of A. Ties -> lower cost, then lower qubit index.",
    "pi/8 rotations are 1-step P(x)Z measurements with |m> (auto-correction/Clifford frame absorbed); pi/4 rotations are 1-step P(x)Y measurements with A.",
    "Edge-aware mapping: demand = #X<->Z switches per qubit in the synthesized stream; patches ranked by #edge types exposed, ties by placement order.",
]


class O3LSAdapter(BaselineAdapter):
    """O3LS layout search + Y-synthesis + loose scheduling on the matched stream."""

    name = "o3ls"

    def run(self, config: BaselineRunConfig, **kwargs: Any) -> BaselineResult:
        input_repr = "Pauli-product rotations (±pi/8), O3LS reimplementation"
        # Both regimes see the same input: the paper's own front end is the
        # Litinski transformation of the source (Pauli-based computation), which
        # is exactly HARVEST's matched Pauli-product stream.  matched_architecture
        # (a HARVEST layout) does not apply to this model.
        if config.comparison_mode not in ("matched_ir", "native_end_to_end"):
            return BaselineResult.failed(
                config,
                input_representation=input_repr,
                error="O3LS reimplementation is only qualified for matched_ir and native_end_to_end",
                seed=None,
            )
        if config.variant not in VARIANTS and config.variant != FIXED_VARIANT:
            raise ValueError(f"unknown O3LS variant {config.variant!r}")
        dag = kwargs.get("dag")
        if dag is None:
            raise ValueError("O3LSAdapter.run requires dag=")
        from .puremagic import UnsupportedPureMagicIR, pauli_dag_to_trans

        started = perf_counter()
        try:
            words = parse_pauli_stream(pauli_dag_to_trans(dag, config.num_qubits))
        except (UnsupportedPureMagicIR, ValueError) as exc:
            return BaselineResult.failed(
                config,
                input_representation=input_repr,
                error=f"O3LS matched-IR requires a ±pi/8 Pauli-product stream: {exc}",
                notes=["Unsupported operations were not discarded or rewritten."],
                seed=None,
            )
        n = int(config.num_qubits) if config.num_qubits else len(words[0])
        if any(len(w) != n for w in words):
            return BaselineResult.failed(
                config,
                input_representation=input_repr,
                error="Pauli word width does not match the register",
                seed=None,
            )
        alpha_e = float(config.options.get("alpha_e", ALPHA_E))
        deadline = started + float(config.timeout_s)
        try:
            ops, synth = y_synthesis(words)
            if config.variant == FIXED_VARIANT:
                board = config.options.get("board")
                if not board or len(board) != 2:
                    raise O3LSError("o3ls_fixed requires options['board'] = [rows, cols]")
                layout = layout_search(n, int(board[0]), int(board[1]), alpha_e)
                if layout is None:
                    raise O3LSError(f"layout search found no C(B)=1 layout on {board[0]}x{board[1]}")
            else:
                layout = select_board(n, AREA_FACTORS[config.variant], alpha_e)
            mapping = edge_aware_mapping(layout, ops, n)
            mean = cultivation_mean(config.options)
            sched = loose_schedule(
                Grid(layout.rows, layout.cols), mapping, ops, deadline,
                cultivation=None if mean is None else (mean, config.seed),
            )
        except TimeoutError as exc:
            return BaselineResult.failed(
                config, input_representation=input_repr, error=str(exc), timed_out=True,
                runtime_s=perf_counter() - started, seed=None, notes=list(NOTES),
            )
        except O3LSError as exc:
            return BaselineResult.failed(
                config, input_representation=input_repr, error=str(exc),
                runtime_s=perf_counter() - started, seed=None, notes=list(NOTES),
            )
        runtime = perf_counter() - started
        tiles = layout.rows * layout.cols + MAGIC_PORT_TILES
        return BaselineResult(
            baseline=config.baseline,
            variant=config.variant,
            circuit=config.circuit,
            num_qubits=config.num_qubits,
            input_representation=input_repr,
            comparison_mode=config.comparison_mode,
            schedule_length=sched.steps,
            logical_patches=tiles,
            space_time_volume=space_time_volume(tiles, sched.steps),
            routing_wirelength=sched.wirelength,
            compiler_runtime_s=runtime,
            completed=True,
            trial=config.trial,
            seed=config.seed if mean is not None else None,
            upstream_version="Zhu et al., ISCA 2026, arXiv:2604.15099 (reimplementation)",
            command="in-process reimplementation (harvest.baselines.o3ls)",
            notes=list(NOTES) + (
                []
                if mean is None
                else [f"Common cultivation model: the single |m> port waits Exp(mean={mean}) cycles "
                      "after each pi/8 measurement (overrides the free-magic note)."]
            ),
            initial_logical_patches=tiles,
            metadata={
                "board_rows": layout.rows,
                "board_cols": layout.cols,
                "board_tiles": layout.rows * layout.cols,
                "magic_port_tiles": MAGIC_PORT_TILES,
                "alpha_e": alpha_e,
                "layout_score": layout.score / _SCORE_SCALE,
                "num_input_rotations": len(words),
                "num_pi8": synth.pi8,
                "num_pi4_generated": synth.pi4_emitted,
                "num_pi4_cancelled": synth.pi4_cancelled,
                "num_pi4_scheduled": synth.pi4_emitted - synth.pi4_cancelled,
                "odd_y_rotations": synth.odd_y_rotations,
                "even_y_rotations": synth.even_y_rotations,
                "non_default_bipartitions": synth.non_default_bipartitions,
                "num_scheduled_ppms": len(ops),
                "measurement_steps": sched.measurement_steps,
                "patch_moves": sched.moves,
                "patch_rotations": sched.rotations,
                "move_steps": sched.moves * MOVE_COST,
                "rotation_steps": sched.rotations * ROTATE_COST,
                "max_parallel_ppms": sched.max_parallel,
                "magic_wait_steps": sched.magic_wait,
                "routing_wirelength_definition": "sum over PPMs of routing tiles in the tree",
                "space_time_definition": "(board tiles + magic port) * time steps",
            },
        )
