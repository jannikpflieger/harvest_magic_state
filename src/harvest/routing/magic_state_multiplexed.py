"""Multiplexed, buffered magic-state cultivation (RQ4 plug-ability model).

Each magic terminal is fed by ``units`` independent cultivation units that
run continuously; a unit finishes a state after an Exp(mean) number of
cycles (rounded, at least one) and starts the next attempt at once.  Finished
states wait in a per-terminal buffer of ``capacity`` states (a unit whose
state finds the buffer full discards it).  A terminal is ready while its
buffer is non-empty.  Every terminal draws from its own RNG seeded by
``(seed, terminal)``, so restricting the source to a subset of terminals
leaves the supply trace of each kept terminal unchanged.

Written only against the ``MagicStateSource`` interface; the scheduler, the
validator and the pruner are untouched.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional


class MultiplexedCultivator:
    def __init__(self, terminals: List[str], mean_cycles: float, units: int = 2,
                 capacity: int = 1, seed: int = 0) -> None:
        if mean_cycles <= 0 or units < 1 or capacity < 1:
            raise ValueError("mean_cycles > 0, units >= 1 and capacity >= 1 required")
        self._mean, self._units, self._capacity, self._seed = mean_cycles, units, capacity, seed
        self._terminals = list(terminals)
        self.reset()

    def _draw(self, terminal: str) -> int:
        return max(1, round(self._rng[terminal].expovariate(1.0 / self._mean)))

    def reset(self) -> None:
        self._rng = {t: random.Random(f"{self._seed}:{t}") for t in self._terminals}
        self._buffer: Dict[str, int] = {t: self._capacity for t in self._terminals}
        self._remaining = {t: [self._draw(t) for _ in range(self._units)] for t in self._terminals}
        self._consumed = self._waits = self._time = 0

    @property
    def unlimited(self) -> bool:
        return False

    @property
    def any_ready(self) -> bool:
        return any(self._buffer.values())

    @property
    def num_ready(self) -> int:
        return sum(1 for b in self._buffer.values() if b)

    def tick(self) -> None:
        self._time += 1
        for t, timers in self._remaining.items():
            for i in range(len(timers)):
                timers[i] -= 1
                if timers[i] <= 0:
                    self._buffer[t] = min(self._capacity, self._buffer[t] + 1)
                    timers[i] = self._draw(t)

    def get_ready_terminals(self) -> Optional[List[str]]:
        return [t for t in self._terminals if self._buffer[t]]

    def consume(self, terminal: str) -> bool:
        if not self._buffer.get(terminal):
            return False
        self._buffer[terminal] -= 1
        self._consumed += 1
        return True

    def record_wait_cycle(self) -> None:
        self._waits += 1

    def get_stats(self) -> dict:
        return {"source_type": "multiplexed", "num_terminals": len(self._terminals),
                "units": self._units, "capacity": self._capacity, "num_ready": self.num_ready,
                "total_consumed": self._consumed, "total_wait_cycles": self._waits,
                "current_time": self._time, "seed": self._seed}
