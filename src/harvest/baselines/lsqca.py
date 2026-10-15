"""LSQCA load/store architecture as a code-beat-level native end-to-end baseline.

Takada Kobori, Yasunari Suzuki, Yosuke Ueno, Teruo Tanimoto, Synge Todo and
Yuuki Tokunaga, "LSQCA: Resource-Efficient Load/Store Architecture for
Limited-Scale Fault-Tolerant Quantum Computing", HPCA 2025, arXiv:2412.20486v2.
The paper's code-beat-accurate simulator is not public, so this module is an
independent reimplementation of the published model.  Every rule below cites
the section it comes from; every choice the paper leaves open is marked
"(assumption)" and repeated in the result ``notes``.

Floorplan (§IV-C, §V-A, §VI-A, Figs. 10/12a)
--------------------------------------------
* Computational register (CR): two columns, exactly two register cells.
  Point SAM: the most compact CR, 2×3 = 6 cells (Fig. 10a); the two-bank point
  layout sandwiches the same 2×3 CR between the banks (Fig. 12a).
  Line SAM: the CR has the height ``h`` of a bank, 2×h cells (Fig. 10b, 12a).
  Four line banks (no figure; assumption): two stacked banks on each side of
  the CR, so the CR is 2×(2h) cells.
* Point-SAM bank holding ``k`` qubits: ``k + 1`` cells (k data + 1 scan cell),
  ``w = ceil(sqrt(k+1))`` columns × ``ceil((k+1)/w)`` rows; when ``k+1`` is not
  square the bottom line is shortened (footnote 1), keeping the cells next to
  the CR.  The scan cell starts at the CR-adjacent cell of the centre line.
* Line-SAM bank: ``w × h`` cells, one full row (the centre row) is the scan
  line, ``w·(h-1)`` data slots.  The bank shape is restricted to L×L or
  L×(L+1) (either orientation) and chosen to maximise memory density (§VI-A),
  i.e. to minimise SAM + CR cells.  For n = 400 this gives 20×21 + 2·21 = 462
  cells, reproducing the paper's 400/462 ≃ 87% (§VI-B).
* Qubits are distributed round-robin to the banks (§VI-A); inside a bank they
  fill data cells in row-major order (assumption).
* ``logical_patches`` = all SAM cells + CR cells + one magic-state port cell
  (§IV-A: "we assume a single cell as a port" for the MSF).  The paper's
  memory density (data / (SAM + CR), MSF excluded) is reported in metadata.

Latencies in code beats (Fig. 4, Table I, §IV-C)
------------------------------------------------
Lattice surgery (MZZ/MXX) 1, H 3, S 2, Pauli/preparation/single-qubit
measurement 0, move 1.  Point-SAM access of a cell displaced (W, H) from the
scan cell's home: ``W + H + 6·min(W,H) + 5·|W−H|`` (worst case 7√n), and
``W + H + 4·min + 3·|W−H|`` for the second cell picked while a first one is
out (one diagonal per 4 beats, two straight moves per 6 beats).  A load adds
the final move into the CR (+1, the "small constant").  Line-SAM load:
``|row − scan row|`` = (row shifts) + 1 long move.

Instruction translation (§VI-A "Benchmark program compilation")
---------------------------------------------------------------
* H, S (S† = S·Z, Pauli ignored): always in-memory (HD.M / PH.M).  Point SAM:
  scan-cell seek ``W + H`` + gate latency; line SAM: scan line moved next to
  the row (``|row − scan| − 1`` beats) + gate latency.
* T / T†: PM (magic state into a CR register), in-memory MZZ.M with the
  target, MX of the magic state, SK, then the outcome-dependent PH.M which is
  always executed ("we always choose the taken path").  The in-memory
  lattice surgery removes the final load move and the store (§V-C).
* CX: the optimised ``CX M1 M2`` instruction.  The operand with the cheaper
  load is loaded into a register, the other one is accessed in memory
  (assumption on the paper's runtime rule, ties keep the second operand
  in memory); the CNOT itself is the lattice-surgery sequence |+⟩ ancilla,
  MZZ, MXX, MZ in the CR = 2 beats; the loaded operand is then stored.
* CZ = H(t)·CX·H(t) and SWAP = 3 CX are decomposed explicitly; x/y/z/id are
  ignored like the paper's simulator ignores Pauli gates.  Anything else
  (including mid-circuit measurement, reset and classically conditioned
  gates) makes the run fail.

Memory state (§V-B, locality-aware store, always used in §VI-A)
---------------------------------------------------------------
* Point SAM: a pick slides the target along a diagonal-first path to the
  scan cell's home, shifting every cell on the path one step back (sliding
  puzzle; scan-cell seek moves are charged but not tracked).  Stores put the
  qubit into an empty cell touching the CR (1 beat); if none exists a hole is
  first slid there at one beat per cell (assumption).  Frequently used cells
  therefore migrate towards the CR, as described in §V-B.
* Line SAM: the scan line's current row is tracked; loading shifts the rows
  between it and the target.  Stores put the qubit into a row adjacent to the
  scan line (1 beat), preferring the row of the partner operand (spatial
  locality, Fig. 12b); otherwise a hole is shifted there at one beat per row.

Scheduling (assumption; the paper only names CPI and in-order translation)
--------------------------------------------------------------------------
Instructions are issued in program order; each gate starts as soon as its
qubits, the SAM banks it touches and the CR registers it needs (CX: both,
T: one) are free, but never before the previous gate started.  Gates on
different banks therefore overlap (the point of multi-bank SAM, §V-A).  Each
gate reserves its resources for its whole duration.  Different banks load in
parallel; two operands in the same bank are accessed sequentially.

Magic states: by default free and always ready at the single MSF port (the
matched assumption shared with the other baselines).  ``options
["msf_period_beats"] = P > 0`` models one factory finishing a state every P
beats (the paper uses Litinski's 15-beat factory) with a buffer of
``options["msf_buffer"]`` (default 2 = 2·factories, §VI-A).
"""

from __future__ import annotations

import math
from time import perf_counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .base import BaselineAdapter, BaselineResult, BaselineRunConfig
from .metrics import space_time_volume

VARIANTS = (
    "lsqca_point_b1",
    "lsqca_point_b2",
    "lsqca_line_b1",
    "lsqca_line_b2",
    "lsqca_line_b4",
)
MAGIC_PORT_CELLS = 1
POINT_CR_CELLS = 6  # 2×3, Fig. 10a / Fig. 12a

# Table I / Fig. 4 latencies (code beats).
LAT_H = 3
LAT_S = 2
LAT_LATTICE_SURGERY = 1
LAT_MOVE = 1
LAT_CNOT_IN_CR = 2  # |+> ancilla, MZZ (1), MXX (1), MZ (0)

SUPPORTED_GATES = frozenset(
    {"h", "s", "sdg", "t", "tdg", "x", "y", "z", "cx", "cz", "swap", "id"}
)
_PAULI = frozenset({"x", "y", "z", "id"})

NOTES = [
    "Independent reimplementation of the LSQCA model (arXiv:2412.20486v2); the paper's simulator is not public.",
    "Latencies from Fig. 4/Table I: lattice surgery 1, H 3, S 2, move 1, Pauli/prep/1q-measurement 0 beats.",
    "Point-SAM access cost W+H+6min(W,H)+5|W-H| (second cell: W+H+4min+3|W-H|), measured to the scan cell's home; loads add the final 1-beat move into the CR.",
    "Line-SAM load cost = row distance to the (tracked) scan line; in-memory ops skip the final long move.",
    "Single-qubit gates and the T-gate ZZ measurement are always in-memory (paper §VI-A); the conditional S of each T is always applied.",
    "CX: cheaper-to-load operand is loaded, the other accessed in memory (interpretation of the paper's runtime CX rule); CNOT in CR = MZZ+MXX via ancilla = 2 beats.",
    "CZ is decomposed as H-CX-H and SWAP as three CX; Pauli gates (x, y, z, id) are ignored as in the paper.",
    "Locality-aware store always used; when no free cell touches the CR/scan line, a hole is slid there at 1 beat per cell (assumption).",
    "Scan-cell seek moves are charged (W+H) but their one-cell displacements are not tracked; intra-bank path conflicts are abstracted by the closed forms.",
    "In-order issue with resource overlap (banks, two CR registers); each gate reserves its resources for its whole duration (assumption).",
    "Qubits distributed round-robin over banks, row-major inside a bank (paper leaves intra-bank placement open).",
    "Four-bank line SAM: two stacked banks per CR side, CR = 2 x (2h) cells (no figure in the paper; assumption).",
    "logical_patches = SAM cells (data + scan) + CR cells + 1 MSF port cell; factories excluded as in the paper's density.",
]


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def parse_variant(variant: str) -> Tuple[str, int]:
    """Return ``(sam_kind, banks)`` for a variant name."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown LSQCA variant {variant!r}")
    _, kind, banks = variant.split("_")
    return kind, int(banks[1:])


def bank_sizes(num_qubits: int, banks: int) -> List[int]:
    """Qubits per bank under round-robin distribution (§VI-A)."""
    return [len(range(b, num_qubits, banks)) for b in range(banks)]


def point_bank_shape(data_cells: int) -> Tuple[int, int, int]:
    """``(width, height, cells)`` of a point-SAM bank (§IV-C2, footnote 1).

    ``cells = data_cells + 1`` (one scan cell).  The bounding box is
    ``ceil(sqrt(cells))`` wide; a non-square ``cells`` shortens the bottom line.
    """
    cells = int(data_cells) + 1
    width = math.isqrt(cells - 1) + 1  # ceil(sqrt(cells))
    height = -(-cells // width)
    return width, height, cells


def line_cr_cells(height: int, banks: int) -> int:
    """CR cells for a line-SAM floorplan: two columns as tall as the bank(s)."""
    return 2 * height * (2 if banks == 4 else 1)


def line_bank_shape(data_cells: int, banks: int = 1) -> Tuple[int, int]:
    """``(width, height)`` of a line-SAM bank (L×L or L×(L+1), §VI-A).

    One full row is the scan line, so a bank stores ``width·(height-1)``
    qubits.  The shape minimises total SAM + CR cells, i.e. maximises the
    paper's memory density; ties prefer the shorter bank (lower latency).
    """
    need = max(1, int(data_cells))
    best = None
    lmax = math.isqrt(need) + 3
    for side in range(1, lmax + 1):
        for width, height in ((side, side), (side, side + 1), (side + 1, side)):
            if height < 2 or width * (height - 1) < need:
                continue
            total = banks * width * height + line_cr_cells(height, banks)
            key = (total, height, width)
            if best is None or key < best:
                best = key
    assert best is not None
    return best[2], best[1]


def point_access_beats(dx: int, dy: int, second: bool = False) -> int:
    """Seek + pick beats to bring a cell displaced (dx, dy) to the scan home."""
    w, h = abs(dx), abs(dy)
    diag, straight = (4, 3) if second else (6, 5)
    return w + h + diag * min(w, h) + straight * abs(w - h)


# ---------------------------------------------------------------------------
# Banks
# ---------------------------------------------------------------------------


class PointBank:
    """Point-SAM bank with sliding-puzzle state (§IV-C2, §V-B)."""

    kind = "point"

    def __init__(self, qubits: Sequence[int]):
        self.width, self.height, self.cells = point_bank_shape(len(qubits))
        last_row = self.cells - self.width * (self.height - 1)
        self.valid = {
            (x, y)
            for y in range(self.height)
            for x in range(self.width if y < self.height - 1 else last_row)
        }
        self.home = (0, (self.height - 1) // 2)  # CR-adjacent cell, centre line
        hx, hy = self.home
        self.store_cells = sorted(
            (c for c in ((hx, hy), (hx, hy - 1), (hx, hy + 1)) if c in self.valid),
            key=lambda c: (abs(c[1] - hy), c[1]),
        )
        self.grid: Dict[Tuple[int, int], Optional[int]] = {c: None for c in self.valid}
        self.pos: Dict[int, Tuple[int, int]] = {}
        free = sorted((c for c in self.valid if c != self.home), key=lambda c: (c[1], c[0]))
        for qubit, cell in zip(qubits, free):
            self.grid[cell] = qubit
            self.pos[qubit] = cell

    def data_cells(self) -> int:
        return self.cells - 1

    def scan_cells(self) -> int:
        return 1

    def describe(self) -> Dict[str, Any]:
        return {
            "kind": "point",
            "width": self.width,
            "height": self.height,
            "cells": self.cells,
            "data_cells": self.cells - 1,
            "scan_cells": 1,
            "qubits": len(self.pos),
        }

    def _disp(self, qubit: int) -> Tuple[int, int]:
        x, y = self.pos[qubit]
        return x - self.home[0], y - self.home[1]

    def load_beats(self, qubit: int, second: bool = False) -> int:
        return point_access_beats(*self._disp(qubit), second=second) + LAT_MOVE

    @staticmethod
    def _path(start: Tuple[int, int], end: Tuple[int, int], diagonal: bool):
        """Cells from ``start`` to ``end``: diagonal-first, or column-first."""
        path = [start]
        x, y = start
        while (x, y) != end:
            sx = (end[0] > x) - (end[0] < x)
            sy = (end[1] > y) - (end[1] < y)
            if diagonal:
                x, y = x + sx, y + sy
            elif sy:
                y += sy
            else:
                x += sx
            path.append((x, y))
        return path

    def _slide_hole(self, path: List[Tuple[int, int]]) -> None:
        """``path[-1]`` is empty; slide occupants so that ``path[0]`` is empty."""
        for i in range(len(path) - 1, 0, -1):
            if self.grid[path[i]] is None and self.grid[path[i - 1]] is not None:
                q = self.grid[path[i - 1]]
                self.grid[path[i]] = q
                self.grid[path[i - 1]] = None
                self.pos[q] = path[i]

    def pick(self, qubit: int, second: bool = False) -> int:
        """Slide ``qubit`` to the scan home and remove it; return access beats."""
        beats = point_access_beats(*self._disp(qubit), second=second)
        cell = self.pos.pop(qubit)
        self.grid[cell] = None
        # The target travels diagonal-first towards the home cell; the cells on
        # its path shift one step back into the vacated cells.
        self._slide_hole(self._path(cell, self.home, diagonal=True)[::-1])
        return beats

    def load(self, qubit: int, second: bool = False) -> int:
        return self.pick(qubit, second) + LAT_MOVE

    def access_in_memory(self, qubit: int, second: bool = False) -> int:
        """In-memory lattice-surgery access: pick without the final move (§V-C)."""
        beats = self.pick(qubit, second)
        self.grid[self.home] = qubit
        self.pos[qubit] = self.home
        return beats

    def single_qubit(self, qubit: int, latency: int) -> int:
        """In-memory H/S: scan-cell seek (W+H) then the gate (§V-C)."""
        dx, dy = self._disp(qubit)
        return abs(dx) + abs(dy) + latency

    def store(self, qubit: int, partner: Optional[int] = None) -> int:
        """Locality-aware store into an empty cell touching the CR (§V-B)."""
        del partner
        for cell in self.store_cells:
            if self.grid[cell] is None:
                self._place(qubit, cell)
                return LAT_MOVE
        holes = [c for c, q in self.grid.items() if q is None]
        best = None
        for hole in holes:
            for target in self.store_cells:
                d = abs(hole[0] - target[0]) + abs(hole[1] - target[1])
                key = (d, hole[1], hole[0], target[1])
                if best is None or key < best[0]:
                    best = (key, hole, target)
        assert best is not None, "point SAM bank has no free cell"
        _, hole, target = best
        self._slide_hole(self._path(target, hole, diagonal=False))
        self._place(qubit, target)
        return LAT_MOVE + best[0][0]

    def _place(self, qubit: int, cell: Tuple[int, int]) -> None:
        self.grid[cell] = qubit
        self.pos[qubit] = cell


class LineBank:
    """Line-SAM bank with a tracked scan line (§IV-C3, §V-B)."""

    kind = "line"
    SCAN = -1

    def __init__(self, qubits: Sequence[int], width: int, height: int):
        self.width, self.height = width, height
        scan_row = (height - 1) // 2
        data_rows = [r for r in range(height) if r != scan_row]
        self.order: List[int] = list(range(height))
        self.order[scan_row] = self.SCAN
        self.members: Dict[int, set] = {r: set() for r in data_rows}
        self.row_of: Dict[int, int] = {}
        for i, qubit in enumerate(qubits):
            row = data_rows[i // width]
            self.members[row].add(qubit)
            self.row_of[qubit] = row

    def data_cells(self) -> int:
        return self.width * (self.height - 1)

    def scan_cells(self) -> int:
        return self.width

    def describe(self) -> Dict[str, Any]:
        return {
            "kind": "line",
            "width": self.width,
            "height": self.height,
            "cells": self.width * self.height,
            "data_cells": self.data_cells(),
            "scan_cells": self.width,
            "qubits": len(self.row_of),
        }

    def _distance(self, qubit: int) -> Tuple[int, int, int]:
        r = self.order.index(self.row_of[qubit])
        s = self.order.index(self.SCAN)
        return abs(r - s), r, s

    def load_beats(self, qubit: int, second: bool = False) -> int:
        del second
        return self._distance(qubit)[0]

    def _seek(self, qubit: int) -> int:
        """Shift rows so the scan line neighbours the qubit's row; return beats."""
        d, r, s = self._distance(qubit)
        new_s = r - 1 if r > s else r + 1
        self.order.pop(s)
        self.order.insert(new_s, self.SCAN)
        return d - 1

    def load(self, qubit: int, second: bool = False) -> int:
        del second
        beats = self._seek(qubit) + LAT_MOVE
        self.members[self.row_of.pop(qubit)].discard(qubit)
        return beats

    def access_in_memory(self, qubit: int, second: bool = False) -> int:
        del second
        return self._seek(qubit)

    def single_qubit(self, qubit: int, latency: int) -> int:
        return self._seek(qubit) + latency

    def store(self, qubit: int, partner: Optional[int] = None) -> int:
        """Locality-aware store into a row next to the scan line (§V-B)."""
        s = self.order.index(self.SCAN)
        adjacent = [i for i in (s - 1, s + 1) if 0 <= i < self.height]
        partner_row = self.row_of.get(partner) if partner is not None else None
        adjacent.sort(key=lambda i: (self.order[i] != partner_row, i))
        for i in adjacent:
            row = self.order[i]
            if len(self.members[row]) < self.width:
                self._place(qubit, row)
                return LAT_MOVE
        best = None
        for i, row in enumerate(self.order):
            if row == self.SCAN or len(self.members[row]) >= self.width:
                continue
            j = s - 1 if i < s else s + 1  # same side: never shift across the scan line
            key = (abs(i - j), i, j)
            if best is None or key < best:
                best = key
        assert best is not None, "line SAM bank has no free slot"
        dist, i, j = best
        step = 1 if j > i else -1
        k = i  # row index holding the hole; pull one qubit per row toward it
        while k != j:
            src_row, dst_row = self.order[k + step], self.order[k]
            mover = min(self.members[src_row])
            self.members[src_row].discard(mover)
            self.members[dst_row].add(mover)
            self.row_of[mover] = dst_row
            k += step
        self._place(qubit, self.order[j])
        return LAT_MOVE + dist

    def _place(self, qubit: int, row: int) -> None:
        self.members[row].add(qubit)
        self.row_of[qubit] = row


class Floorplan:
    """Banks + CR for one variant."""

    def __init__(self, variant: str, num_qubits: int):
        self.kind, self.num_banks = parse_variant(variant)
        n = int(num_qubits)
        if n < 1:
            raise ValueError("num_qubits must be positive")
        assignment = [list(range(b, n, self.num_banks)) for b in range(self.num_banks)]
        self.bank_of = {q: q % self.num_banks for q in range(n)}
        if self.kind == "point":
            self.banks = [PointBank(qs) for qs in assignment]
            self.cr_cells = POINT_CR_CELLS
        else:
            width, height = line_bank_shape(max(bank_sizes(n, self.num_banks)), self.num_banks)
            self.banks = [LineBank(qs, width, height) for qs in assignment]
            self.cr_cells = line_cr_cells(height, self.num_banks)
        self.num_qubits = n
        self.sam_cells = sum(
            b.cells if isinstance(b, PointBank) else b.width * b.height for b in self.banks
        )

    @property
    def total_cells(self) -> int:
        return self.sam_cells + self.cr_cells + MAGIC_PORT_CELLS

    @property
    def memory_density(self) -> float:
        return self.num_qubits / (self.sam_cells + self.cr_cells)


def build_floorplan(variant: str, num_qubits: int) -> Floorplan:
    return Floorplan(variant, num_qubits)


# ---------------------------------------------------------------------------
# Circuit input
# ---------------------------------------------------------------------------


class UnsupportedLSQCAInput(ValueError):
    pass


def circuit_to_gates(circuit) -> List[Tuple[str, Tuple[int, ...]]]:
    """Flatten a Clifford+T circuit into LSQCA's gate set (H, S, T, CX, Pauli).

    Final measurements and barriers are removed; cz/swap are decomposed
    explicitly; any other operation raises :class:`UnsupportedLSQCAInput`.
    """
    circuit = circuit.remove_final_measurements(inplace=False)
    index = {qubit: i for i, qubit in enumerate(circuit.qubits)}
    gates: List[Tuple[str, Tuple[int, ...]]] = []
    for instruction in circuit.data:
        op = instruction.operation
        name = op.name
        if name == "barrier":
            continue
        if getattr(op, "condition", None) is not None:
            raise UnsupportedLSQCAInput(f"classically conditioned {name!r} is not supported")
        if name not in SUPPORTED_GATES:
            raise UnsupportedLSQCAInput(
                f"LSQCA input must be Clifford+T over {sorted(SUPPORTED_GATES)}; "
                f"unsupported operation {name!r}"
            )
        qs = tuple(index[q] for q in instruction.qubits)
        if name == "cz":
            gates += [("h", (qs[1],)), ("cx", qs), ("h", (qs[1],))]
        elif name == "swap":
            a, b = qs
            gates += [("cx", (a, b)), ("cx", (b, a)), ("cx", (a, b))]
        else:
            gates.append((name, qs))
    return gates


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------


class MagicSupply:
    """One factory finishing a state every ``period`` beats into a buffer."""

    def __init__(self, period: int, buffer: int):
        self.period = int(period)
        self.cap = max(1, int(buffer))
        self.buffer = 0
        self.next_done: Optional[int] = self.period if self.period > 0 else None
        self.wait_beats = 0

    def acquire(self, t: int) -> int:
        """Return the time a magic state is in the CR when requested at ``t``."""
        if self.period <= 0:
            return t
        while self.next_done is not None and self.next_done <= t and self.buffer < self.cap:
            self.buffer += 1
            self.next_done += self.period
        if self.buffer == self.cap:
            self.next_done = None  # factory stalls on a full buffer
        if self.buffer == 0:
            assert self.next_done is not None
            ready = self.next_done
            self.next_done += self.period
            self.wait_beats += ready - t
            return ready
        self.buffer -= 1
        if self.next_done is None:
            self.next_done = t + self.period
        return t


def simulate(
    gates: Sequence[Tuple[str, Tuple[int, ...]]],
    variant: str,
    num_qubits: int,
    *,
    msf_period_beats: int = 0,
    msf_buffer: int = 2,
) -> Dict[str, Any]:
    """Code-beat simulation of ``gates`` on one LSQCA floorplan."""
    plan = Floorplan(variant, num_qubits)
    banks = plan.banks
    qubit_ready = [0] * plan.num_qubits
    bank_free = [0] * len(banks)
    reg_free = [0, 0]
    magic = MagicSupply(msf_period_beats, msf_buffer)
    counts = {
        "instructions": 0,
        "loads": 0,
        "stores": 0,
        "in_memory_two_qubit_accesses": 0,
        "in_memory_single_qubit": 0,
        "magic_states": 0,
        "ignored_pauli": 0,
        "store_relocation_beats": 0,
    }
    gate_counts: Dict[str, int] = {}
    last_start = 0
    makespan = 0

    for name, qs in gates:
        gate_counts[name] = gate_counts.get(name, 0) + 1
        if name in _PAULI:
            counts["ignored_pauli"] += 1
            continue
        if name in ("h", "s", "sdg"):
            (q,) = qs
            b = plan.bank_of[q]
            start = max(last_start, qubit_ready[q], bank_free[b])
            latency = LAT_H if name == "h" else LAT_S
            end = start + banks[b].single_qubit(q, latency)
            bank_free[b] = qubit_ready[q] = end
            counts["instructions"] += 1  # HD.M / PH.M
            counts["in_memory_single_qubit"] += 1
        elif name in ("t", "tdg"):
            (q,) = qs
            b = plan.bank_of[q]
            r = 0 if reg_free[0] <= reg_free[1] else 1
            start = max(last_start, qubit_ready[q], bank_free[b], reg_free[r])
            t_magic = magic.acquire(start)  # PM: free/ready -> 0 beats
            access = banks[b].access_in_memory(q)  # scan path to the CR
            t = max(t_magic, start + access) + LAT_LATTICE_SURGERY  # MZZ.M
            reg_free[r] = t  # MX on the magic state (0 beats) frees the register
            t += banks[b].single_qubit(q, LAT_S)  # SK + PH.M, always taken
            bank_free[b] = qubit_ready[q] = t
            end = t
            counts["instructions"] += 5  # PM, MZZ.M, MX, SK, PH.M
            counts["in_memory_two_qubit_accesses"] += 1
            counts["in_memory_single_qubit"] += 1
            counts["magic_states"] += 1
        elif name == "cx":
            a, c = qs
            if a == c:
                raise UnsupportedLSQCAInput("cx with identical operands")
            ba, bc = plan.bank_of[a], plan.bank_of[c]
            # Runtime choice of the optimised CX: load the cheaper operand.
            if banks[ba].load_beats(a) < banks[bc].load_beats(c):
                loaded, mem = a, c
            elif banks[ba].load_beats(a) > banks[bc].load_beats(c):
                loaded, mem = c, a
            else:
                loaded, mem = a, c
            bl, bm = plan.bank_of[loaded], plan.bank_of[mem]
            start = max(
                last_start,
                qubit_ready[a],
                qubit_ready[c],
                bank_free[bl],
                bank_free[bm],
                reg_free[0],
                reg_free[1],
            )
            if bl == bm:
                t_access = banks[bl].load(loaded) + banks[bm].access_in_memory(mem, second=True)
                t_store_ready = start + t_access + LAT_CNOT_IN_CR
                store = banks[bl].store(loaded, partner=mem)
                end = t_store_ready + store
                bank_free[bl] = end
            else:
                load = banks[bl].load(loaded)
                access = banks[bm].access_in_memory(mem)
                t_store_ready = start + max(load, access) + LAT_CNOT_IN_CR
                store = banks[bl].store(loaded)
                end = t_store_ready + store
                bank_free[bl] = end
                bank_free[bm] = t_store_ready
            counts["store_relocation_beats"] += store - LAT_MOVE
            reg_free[0] = reg_free[1] = t_store_ready
            qubit_ready[loaded] = end
            qubit_ready[mem] = t_store_ready
            counts["instructions"] += 1  # CX M1 M2
            counts["loads"] += 1
            counts["stores"] += 1
            counts["in_memory_two_qubit_accesses"] += 1
        else:  # pragma: no cover - filtered by circuit_to_gates
            raise UnsupportedLSQCAInput(f"unsupported gate {name!r}")
        last_start = start
        makespan = max(makespan, end)

    instructions = counts["instructions"]
    return {
        "schedule_length": int(makespan),
        "logical_patches": plan.total_cells,
        "floorplan": plan,
        "counts": counts,
        "gate_counts": gate_counts,
        "cpi": (makespan / instructions) if instructions else None,
        "magic_wait_beats": magic.wait_beats,
    }


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class LSQCAAdapter(BaselineAdapter):
    """LSQCA point/line-SAM floorplans simulated on the native Clifford+T circuit."""

    name = "lsqca"

    def run(self, config: BaselineRunConfig, **kwargs: Any) -> BaselineResult:
        del kwargs
        input_repr = "OpenQASM 2 Clifford+T (H/S/T/CX), LSQCA load/store code-beat model"
        if config.comparison_mode != "native_end_to_end":
            return BaselineResult.failed(
                config,
                input_representation=input_repr,
                error="LSQCA is a gate-level floorplan model; only native_end_to_end is supported",
                seed=None,
            )
        if config.variant not in VARIANTS:
            return BaselineResult.failed(
                config,
                input_representation=input_repr,
                error=f"unknown LSQCA variant {config.variant!r}; expected one of {list(VARIANTS)}",
                seed=None,
            )
        period = int(config.options.get("msf_period_beats", 0) or 0)
        buffer = int(config.options.get("msf_buffer", 2) or 2)
        started = perf_counter()
        try:
            from qiskit import QuantumCircuit

            circuit = QuantumCircuit.from_qasm_file(str(config.input_path))
            gates = circuit_to_gates(circuit)
            sim = simulate(
                gates,
                config.variant,
                circuit.num_qubits,
                msf_period_beats=period,
                msf_buffer=buffer,
            )
        except Exception as exc:  # parse errors, unsupported gates, bad sizes
            return BaselineResult.failed(
                config,
                input_representation=input_repr,
                error=f"{type(exc).__name__}: {exc}",
                notes=["Unsupported operations were not discarded or rewritten."],
                runtime_s=perf_counter() - started,
                seed=None,
            )
        runtime = perf_counter() - started
        plan: Floorplan = sim["floorplan"]
        beats, cells = sim["schedule_length"], sim["logical_patches"]
        notes = list(NOTES)
        notes.append(
            "Magic states free and always ready at one MSF port (matched assumption)."
            if period <= 0
            else f"One magic-state factory, one state per {period} beats, buffer {buffer}."
        )
        return BaselineResult(
            baseline=config.baseline,
            variant=config.variant,
            circuit=config.circuit,
            num_qubits=circuit.num_qubits,
            input_representation=input_repr,
            comparison_mode=config.comparison_mode,
            schedule_length=beats,
            logical_patches=cells,
            space_time_volume=space_time_volume(cells, beats),
            routing_wirelength=None,
            compiler_runtime_s=runtime,
            completed=True,
            trial=config.trial,
            seed=None,
            upstream_version="Kobori et al., HPCA 2025, arXiv:2412.20486v2 (reimplementation)",
            command="code-beat simulation (harvest.baselines.lsqca)",
            notes=notes,
            initial_logical_patches=cells,
            metadata={
                "sam_kind": plan.kind,
                "banks": plan.num_banks,
                "bank_shapes": [b.describe() for b in plan.banks],
                "sam_cells": plan.sam_cells,
                "cr_cells": plan.cr_cells,
                "magic_port_cells": MAGIC_PORT_CELLS,
                "memory_density": plan.memory_density,
                "cpi": sim["cpi"],
                "instruction_count": sim["counts"]["instructions"],
                "load_count": sim["counts"]["loads"],
                "store_count": sim["counts"]["stores"],
                "counts": sim["counts"],
                "gate_counts": sim["gate_counts"],
                "msf_period_beats": period,
                "msf_buffer": buffer,
                "magic_wait_beats": sim["magic_wait_beats"],
                "time_unit": "code beats (d code cycles)",
                "space_time_definition": "(SAM + CR + MSF port cells) * code beats",
            },
        )
