from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple
import math

Coord = Tuple[int, int]

class TileKind(Enum):
    DATA = auto()
    ANCILLA = auto()

class AncillaState(Enum):
    CULTIVATING = auto()
    READY_MAGIC = auto()
    ROUTING_BUSY = auto()

@dataclass(frozen=True)
class DataPatch:
    qid: int
    pos: Coord  # occupies exactly one tile for v1

@dataclass
class Layout:
    W: int
    H: int
    data_at: Dict[Coord, DataPatch]             # occupied tiles
    ancilla_state: Dict[Coord, AncillaState]   # only for ancilla tiles
    cultivation_progress: Dict[Coord, int]      # simple deterministic model
    cultivation_latency: int = 5                # tiles become READY after 5 cycles

    def in_bounds(self, c: Coord) -> bool:
        x, y = c
        return 0 <= x < self.W and 0 <= y < self.H

    def is_data(self, c: Coord) -> bool:
        return c in self.data_at

    def is_ancilla(self, c: Coord) -> bool:
        return self.in_bounds(c) and (c not in self.data_at)

    def neighbors4(self, c: Coord) -> List[Coord]:
        x, y = c
        ns = [(x+1,y), (x-1,y), (x,y+1), (x,y-1)]
        return [n for n in ns if self.in_bounds(n)]

    # --- terminals: define where routing connects to a given Pauli on a data tile ---
    def terminal_for(self, qpos: Coord, pauli: str) -> Coord:
        x, y = qpos
        if pauli == "X":
            t = (x, y-1)      # north
        elif pauli == "Z":
            t = (x+1, y)      # east
        else:
            raise ValueError("v1 expects only X/Z (preprocess away Y).")
        if not self.is_ancilla(t):
            raise RuntimeError(f"Terminal {t} not available; adjust spacing/layout.")
        return t

    def ancilla_free_for_routing(self, c: Coord) -> bool:
        if not self.is_ancilla(c):
            return False
        return self.ancilla_state.get(c, AncillaState.CULTIVATING) != AncillaState.ROUTING_BUSY

    def mark_routing_busy(self, used: Set[Coord]) -> None:
        for c in used:
            if not self.is_ancilla(c):
                continue
            self.ancilla_state[c] = AncillaState.ROUTING_BUSY
            # interrupt cultivation progress on used tiles
            self.cultivation_progress[c] = 0

    def clear_routing_busy(self) -> None:
        for c, st in list(self.ancilla_state.items()):
            if st == AncillaState.ROUTING_BUSY:
                self.ancilla_state[c] = AncillaState.CULTIVATING

    def advance_cultivation(self) -> None:
        for x in range(self.W):
            for y in range(self.H):
                c = (x, y)
                if not self.is_ancilla(c):
                    continue
                st = self.ancilla_state.get(c, AncillaState.CULTIVATING)
                if st == AncillaState.ROUTING_BUSY:
                    continue
                if st == AncillaState.READY_MAGIC:
                    continue
                # cultivating
                p = self.cultivation_progress.get(c, 0) + 1
                self.cultivation_progress[c] = p
                if p >= self.cultivation_latency:
                    self.ancilla_state[c] = AncillaState.READY_MAGIC

    def find_any_ready_magic(self) -> Optional[Coord]:
        for c, st in self.ancilla_state.items():
            if st == AncillaState.READY_MAGIC:
                return c
        return None

    def consume_magic(self, c: Coord) -> None:
        if self.ancilla_state.get(c) != AncillaState.READY_MAGIC:
            raise RuntimeError("Not a ready magic tile.")
        self.ancilla_state[c] = AncillaState.CULTIVATING
        self.cultivation_progress[c] = 0

def make_layout(n_qubits: int, W: int = 25, H: int = 15) -> Layout:
    data_at: Dict[Coord, DataPatch] = {}
    ancilla_state: Dict[Coord, AncillaState] = {}
    cultivation_progress: Dict[Coord, int] = {}

    # place in a spaced grid (2-step spacing)
    import math
    cols = math.ceil(math.sqrt(n_qubits))
    rows = math.ceil(n_qubits / cols)
    x0, y0 = 2, 2  # margin
    q = 0
    for j in range(rows):
        for i in range(cols):
            if q >= n_qubits:
                break
            pos = (x0 + 2*i, y0 + 2*j)
            data_at[pos] = DataPatch(qid=q, pos=pos)
            q += 1

    # initialize all ancilla as cultivating
    for x in range(W):
        for y in range(H):
            c = (x, y)
            if c not in data_at:
                ancilla_state[c] = AncillaState.CULTIVATING
                cultivation_progress[c] = 0

    return Layout(W=W, H=H, data_at=data_at,
                  ancilla_state=ancilla_state,
                  cultivation_progress=cultivation_progress)




def construct_lattice_layout(lattice_type: str, dimensions: Tuple[int, int]) -> Layout:
    """
    Construct a lattice layout based on the specified type and dimensions.
    
    Args:
        lattice_type: Type of lattice ('square', 'rectangular', 'triangular')
        dimensions: (width, height) of the lattice
        
    Returns:
        Layout: The constructed lattice layout
    """
    W, H = dimensions
    
    if lattice_type.lower() in ['square', 'rectangular']:
        # For square/rectangular lattices, place data qubits in a regular grid
        # with spacing to allow for ancilla qubits between them
        data_at: Dict[Coord, DataPatch] = {}
        ancilla_state: Dict[Coord, AncillaState] = {}
        cultivation_progress: Dict[Coord, int] = {}
        
        # Place data qubits with 3-step spacing to ensure ancilla availability
        qid = 0
        for y in range(1, H, 3):  # Every 3rd row starting from 1
            for x in range(1, W, 3):  # Every 3rd column starting from 1
                if x < W and y < H:
                    pos = (x, y)
                    data_at[pos] = DataPatch(qid=qid, pos=pos)
                    qid += 1
        
        # Initialize all non-data positions as ancilla
        for x in range(W):
            for y in range(H):
                coord = (x, y)
                if coord not in data_at:
                    ancilla_state[coord] = AncillaState.CULTIVATING
                    cultivation_progress[coord] = 0
        
        return Layout(W=W, H=H, data_at=data_at,
                      ancilla_state=ancilla_state,
                      cultivation_progress=cultivation_progress)
    
    elif lattice_type.lower() == 'triangular':
        # For triangular lattices, use a hexagonal-like pattern
        # This is a simplified implementation
        return construct_lattice_layout('square', dimensions)  # Fallback to square for now
    
    else:
        raise ValueError(f"Unsupported lattice type: {lattice_type}")

def create_default_lattice_layout(number_of_qubits: int) -> Layout:
    """
    Create a default lattice layout that can accommodate the specified number of qubits.
    
    Args:
        number_of_qubits: Number of data qubits to place in the lattice
        
    Returns:
        Layout: A default layout with appropriate dimensions
    """
    # Aim for roughly square layout
    side_length = math.ceil(math.sqrt(number_of_qubits))
    
    # Account for 3x spacing and add some margin
    W = side_length * 3  # Minimum width of 15
    H = math.ceil(number_of_qubits / side_length) * 3
    
    return make_layout(n_qubits=number_of_qubits, W=W, H=H)

