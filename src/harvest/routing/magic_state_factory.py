"""
Per-terminal magic-state preparation model for fault-tolerant quantum computation.

Each magic terminal on the lattice has its own independent distillation
pipeline.  After a magic state is consumed from a terminal, that terminal
enters a cooldown of ``preparation_cycles`` steps before it is ready again.

All terminals start in the *ready* state (cooldown = 0) so the first wave
of magic-consuming gates can proceed immediately.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

logger = logging.getLogger("HarvestMagicState.MagicStateFactory")


class MagicStateFactory:
    """Per-terminal magic state production.

    Parameters
    ----------
    terminals : list[str]
        Names of every magic terminal on the lattice.
    preparation_cycles : int
        Cooldown cycles after consumption before a terminal is ready again.
    """

    def __init__(
        self,
        terminals: List[str],
        preparation_cycles: int = 15,
    ) -> None:
        if preparation_cycles < 1:
            raise ValueError("preparation_cycles must be >= 1")
        self.preparation_cycles = preparation_cycles
        self._cooldowns: Dict[str, int] = {t: 0 for t in terminals}
        self._unlimited = False

        # bookkeeping
        self._current_time = 0
        self._total_consumed = 0
        self._total_wait_cycles = 0

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def create_unlimited(cls) -> MagicStateFactory:
        """Return a factory that never blocks (old behaviour)."""
        obj = cls.__new__(cls)
        obj.preparation_cycles = 0
        obj._cooldowns = {}
        obj._unlimited = True
        obj._current_time = 0
        obj._total_consumed = 0
        obj._total_wait_cycles = 0
        return obj

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def unlimited(self) -> bool:
        return self._unlimited

    def tick(self) -> None:
        """Advance every terminal's cooldown by one cycle."""
        self._current_time += 1
        if self._unlimited:
            return
        for t in self._cooldowns:
            if self._cooldowns[t] > 0:
                self._cooldowns[t] -= 1

    def get_ready_terminals(self) -> Optional[List[str]]:
        """Return terminals whose cooldown has reached zero.

        Returns ``None`` when the factory is unlimited (meaning *all*
        terminals are always available).
        """
        if self._unlimited:
            return None
        return [t for t, cd in self._cooldowns.items() if cd == 0]

    @property
    def any_ready(self) -> bool:
        if self._unlimited:
            return True
        return any(cd == 0 for cd in self._cooldowns.values())

    @property
    def num_ready(self) -> int:
        if self._unlimited:
            return 2**31
        return sum(1 for cd in self._cooldowns.values() if cd == 0)

    def consume(self, terminal: str) -> bool:
        """Consume a magic state from *terminal*, starting its cooldown.

        Returns ``True`` on success, ``False`` if the terminal is still
        cooling down.
        """
        if self._unlimited:
            self._total_consumed += 1
            return True
        if self._cooldowns.get(terminal, -1) != 0:
            return False
        self._cooldowns[terminal] = self.preparation_cycles
        self._total_consumed += 1
        return True

    def record_wait_cycle(self) -> None:
        """Record that the scheduler idled for one cycle."""
        self._total_wait_cycles += 1

    def reset(self) -> None:
        """Reset all cooldowns and counters."""
        for t in self._cooldowns:
            self._cooldowns[t] = 0
        self._current_time = 0
        self._total_consumed = 0
        self._total_wait_cycles = 0

    def get_stats(self) -> dict:
        num_ready = self.num_ready if not self._unlimited else "unlimited"
        return {
            "source_type": "factory",
            "preparation_cycles": self.preparation_cycles,
            "unlimited": self._unlimited,
            "num_terminals": len(self._cooldowns),
            "num_ready": num_ready,
            "total_consumed": self._total_consumed,
            "total_wait_cycles": self._total_wait_cycles,
            "current_time": self._current_time,
        }
