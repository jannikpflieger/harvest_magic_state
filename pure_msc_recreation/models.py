from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Dict, Set, Optional
import random
import math

Coord = Tuple[int, int]

class CellType(str, Enum):
    DATA = "data"
    BUS = "bus"
    ANCILLA = "ancilla"   # Pure Magic: cultivator by default
    MAGIC = "magic"       # Bus baseline: dedicated cultivator

@dataclass
class Cell:
    typ: CellType
    occupied: bool = False

@dataclass
class Cultivator:
    coord: Coord
    remaining: int
    ready: bool = False

    def tick(self) -> None:
        if self.ready:
            return
        self.remaining -= 1
        if self.remaining <= 0:
            self.ready = True

@dataclass
class PauliTerm:
    qubit: int
    pauli: str  # 'X','Y','Z'

@dataclass
class PauliProduct:
    pid: int
    terms: List[PauliTerm]
    deps: Set[int] = field(default_factory=set)  # product IDs that must finish first

    def qubits(self) -> Set[int]:
        return {t.qubit for t in self.terms}

@dataclass
class Layout:
    w: int
    h: int
    grid: List[List[Cell]]
    data_pos: Dict[int, Coord]                 # qubit -> coord
    cultivators: Dict[Coord, Cultivator]       # coord -> cultivator

def sample_cultivation_cycles(lam: float, code_distance: int, rng: random.Random) -> int:
    # exponential with rate lam, then divide by code distance (paper-style abstraction)
    x = rng.expovariate(lam)
    #cycles = max(1, int(math.ceil(x / max(1, code_distance))))
    cycles = max(1, int(math.ceil(x * code_distance)))
    return cycles
