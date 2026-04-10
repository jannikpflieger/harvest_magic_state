"""
Stochastic magic-state cultivation model.

Unlike the deterministic :class:`MagicStateFactory` where every terminal
has a fixed cooldown, a *cultivator* models magic-state preparation as a
probabilistic process.  Each time a magic state is consumed, a new
cultivation attempt starts and the number of cycles until the next state
is ready is *sampled* from an injected probability distribution.

This captures the key architectural distinction from the cultivation
literature: readiness is stochastic and varies per attempt, whereas
factory-based production follows a deterministic cadence.

Implements the :class:`~harvest.routing.magic_state_source.MagicStateSource`
protocol so that the scheduler can use it interchangeably with a factory.
"""

from __future__ import annotations

import enum
import logging
import math
import random
from typing import Callable, Dict, List, Optional

logger = logging.getLogger("HarvestMagicState.MagicStateCultivator")

# Type alias for the sampler function the user injects.
# Given a ``random.Random`` instance it must return the number of cycles
# (>= 1) until the next magic state is ready.
ReadinessSampler = Callable[[random.Random], int]


# ======================================================================
# Distribution helper factories
# ======================================================================

def fixed_sampler(cycles: int) -> ReadinessSampler:
    """Deterministic sampler — always returns the same value.

    Useful for testing or for bridging to factory-like behaviour.
    """
    if cycles < 1:
        raise ValueError("cycles must be >= 1")

    def _sample(_rng: random.Random) -> int:
        return cycles

    return _sample


def geometric_sampler(p: float) -> ReadinessSampler:
    """Geometric-distribution sampler (number of trials until first success).

    ``p`` is the per-cycle success probability.  The returned value is
    always >= 1.  Mean = 1/p.
    """
    if not 0.0 < p <= 1.0:
        raise ValueError("p must be in (0, 1]")

    def _sample(rng: random.Random) -> int:
        # Inverse-CDF method:  ceil(log(1 - U) / log(1 - p))
        # For p == 1.0 every attempt succeeds immediately.
        if p == 1.0:
            return 1
        u = rng.random()
        # Clamp to avoid log(0)
        u = max(u, 1e-15)
        return max(1, math.ceil(math.log(1.0 - u) / math.log(1.0 - p)))

    return _sample


def lognormal_sampler(mean_cycles: float, std_cycles: float) -> ReadinessSampler:
    """Log-normal sampler (rounded to an integer, clamped to >= 1).

    Parameters are in *cycle* space (not log space).  Internally the
    log-normal μ/σ are derived so that the distribution has the
    requested mean and standard deviation.
    """
    if mean_cycles <= 0 or std_cycles < 0:
        raise ValueError("mean_cycles must be > 0 and std_cycles >= 0")

    # Derive underlying normal parameters
    variance = std_cycles ** 2
    mu = math.log(mean_cycles ** 2 / math.sqrt(variance + mean_cycles ** 2))
    sigma = math.sqrt(math.log(1.0 + variance / mean_cycles ** 2))

    def _sample(rng: random.Random) -> int:
        return max(1, round(rng.lognormvariate(mu, sigma)))

    return _sample


# ======================================================================
# Per-terminal cultivation status
# ======================================================================

class CultivationStatus(enum.Enum):
    """State of a single cultivating terminal."""

    IDLE = "idle"
    """No cultivation attempt is in progress."""

    CULTIVATING = "cultivating"
    """An attempt is underway; ``remaining_ticks`` counts down to zero."""

    READY = "ready"
    """A magic state has been successfully cultivated and is available."""


# ======================================================================
# Main class
# ======================================================================

class MagicStateCultivator:
    """Per-terminal stochastic magic-state cultivation.

    Parameters
    ----------
    terminals : list[str]
        Names of every cultivating terminal on the lattice.
    readiness_sampler : ReadinessSampler
        Callable ``(rng: random.Random) -> int`` that returns the number
        of cycles (>= 1) until the current cultivation attempt succeeds.
        Called once each time a new attempt starts.
    seed : int
        Seed for the internal RNG, ensuring deterministic reproducibility.
    max_cycles : int | None
        Optional hard cap on any sampled readiness time.  If ``None``,
        no cap is applied.  Useful to prevent extreme tail values from
        causing excessive stalling.
    """

    def __init__(
        self,
        terminals: List[str],
        readiness_sampler: ReadinessSampler,
        seed: int = 42,
        max_cycles: Optional[int] = None,
    ) -> None:
        self._sampler = readiness_sampler
        self._rng = random.Random(seed)
        self._seed = seed
        self._max_cycles = max_cycles
        self._unlimited = False

        # Per-terminal state
        self._statuses: Dict[str, CultivationStatus] = {
            t: CultivationStatus.READY for t in terminals
        }
        self._remaining: Dict[str, int] = {t: 0 for t in terminals}

        # Bookkeeping
        self._current_time: int = 0
        self._total_consumed: int = 0
        self._total_wait_cycles: int = 0

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def create_unlimited(cls) -> "MagicStateCultivator":
        """Return a cultivator that never blocks (old behaviour)."""
        obj = cls.__new__(cls)
        obj._sampler = fixed_sampler(1)
        obj._rng = random.Random(0)
        obj._seed = 0
        obj._max_cycles = None
        obj._unlimited = True
        obj._statuses = {}
        obj._remaining = {}
        obj._current_time = 0
        obj._total_consumed = 0
        obj._total_wait_cycles = 0
        return obj

    # ------------------------------------------------------------------
    # MagicStateSource protocol — properties
    # ------------------------------------------------------------------

    @property
    def unlimited(self) -> bool:
        return self._unlimited

    @property
    def any_ready(self) -> bool:
        if self._unlimited:
            return True
        return any(
            s == CultivationStatus.READY for s in self._statuses.values()
        )

    @property
    def num_ready(self) -> int:
        if self._unlimited:
            return 2**31
        return sum(
            1 for s in self._statuses.values() if s == CultivationStatus.READY
        )

    # ------------------------------------------------------------------
    # MagicStateSource protocol — time advancement
    # ------------------------------------------------------------------

    def tick(self) -> None:
        """Advance every cultivating terminal by one cycle.

        Terminals in ``CULTIVATING`` have their remaining count
        decremented.  When it reaches zero the terminal transitions
        to ``READY``.
        """
        self._current_time += 1
        if self._unlimited:
            return
        for t in self._statuses:
            if self._statuses[t] is CultivationStatus.CULTIVATING:
                self._remaining[t] -= 1
                if self._remaining[t] <= 0:
                    self._statuses[t] = CultivationStatus.READY
                    self._remaining[t] = 0

    # ------------------------------------------------------------------
    # MagicStateSource protocol — availability
    # ------------------------------------------------------------------

    def get_ready_terminals(self) -> Optional[List[str]]:
        """Return terminals with a cultivated magic state available.

        Returns ``None`` when ``unlimited``.
        """
        if self._unlimited:
            return None
        return [
            t for t, s in self._statuses.items()
            if s is CultivationStatus.READY
        ]

    # ------------------------------------------------------------------
    # MagicStateSource protocol — consumption
    # ------------------------------------------------------------------

    def consume(self, terminal: str) -> bool:
        """Consume the magic state from *terminal* and start the next attempt.

        Returns ``True`` on success.  Returns ``False`` if the terminal
        is not in the ``READY`` state.

        After consumption the terminal immediately transitions to
        ``CULTIVATING`` with a freshly sampled readiness duration.
        """
        if self._unlimited:
            self._total_consumed += 1
            return True

        if self._statuses.get(terminal) is not CultivationStatus.READY:
            return False

        self._total_consumed += 1
        self._start_cultivation(terminal)
        return True

    # ------------------------------------------------------------------
    # MagicStateSource protocol — bookkeeping
    # ------------------------------------------------------------------

    def record_wait_cycle(self) -> None:
        self._total_wait_cycles += 1

    def get_stats(self) -> dict:
        if self._unlimited:
            num_ready = "unlimited"
        else:
            num_ready = self.num_ready

        status_counts = {}
        for s in CultivationStatus:
            status_counts[s.value] = sum(
                1 for st in self._statuses.values() if st is s
            )

        return {
            "source_type": "cultivation",
            "unlimited": self._unlimited,
            "num_terminals": len(self._statuses),
            "num_ready": num_ready,
            "total_consumed": self._total_consumed,
            "total_wait_cycles": self._total_wait_cycles,
            "current_time": self._current_time,
            "seed": self._seed,
            "max_cycles": self._max_cycles,
            "status_counts": status_counts,
        }

    def reset(self) -> None:
        """Reset all state — all terminals become READY, counters zeroed."""
        self._rng = random.Random(self._seed)
        for t in self._statuses:
            self._statuses[t] = CultivationStatus.READY
            self._remaining[t] = 0
        self._current_time = 0
        self._total_consumed = 0
        self._total_wait_cycles = 0

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _start_cultivation(self, terminal: str) -> None:
        """Begin a new cultivation attempt on *terminal*."""
        duration = self._sample_readiness()
        self._statuses[terminal] = CultivationStatus.CULTIVATING
        self._remaining[terminal] = duration

    def _sample_readiness(self) -> int:
        """Draw the next readiness duration from the injected sampler."""
        value = self._sampler(self._rng)
        value = max(1, value)
        if self._max_cycles is not None:
            value = min(value, self._max_cycles)
        return value
