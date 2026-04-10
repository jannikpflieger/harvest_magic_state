"""
Protocol (structural type) shared by all magic-state preparation models.

Any class that exposes the methods listed in ``MagicStateSource`` can be
used by the scheduler without modification — no explicit inheritance is
required.

Implementations
---------------
* ``MagicStateFactory``  — deterministic per-terminal cooldown
* ``MagicStateCultivator`` — stochastic per-terminal cultivation
"""

from __future__ import annotations

from typing import List, Optional, runtime_checkable

try:
    from typing import Protocol          # Python ≥ 3.8
except ImportError:                       # pragma: no cover
    from typing_extensions import Protocol  # type: ignore[assignment]


@runtime_checkable
class MagicStateSource(Protocol):
    """Structural interface consumed by the DAG scheduler.

    Every magic-state source — whether a deterministic factory or a
    stochastic cultivator — must expose these members so that the
    scheduler can query availability, advance time, consume states,
    and collect statistics without knowing the concrete type.
    """

    # -- properties ------------------------------------------------

    @property
    def unlimited(self) -> bool:
        """``True`` when the source never blocks (old behaviour)."""
        ...

    @property
    def any_ready(self) -> bool:
        """At least one terminal has a magic state available."""
        ...

    @property
    def num_ready(self) -> int:
        """Count of terminals that currently have a magic state available."""
        ...

    # -- time advancement ------------------------------------------

    def tick(self) -> None:
        """Advance the internal clock by one scheduling cycle."""
        ...

    # -- availability query ----------------------------------------

    def get_ready_terminals(self) -> Optional[List[str]]:
        """Return terminals that have a state ready for consumption.

        Returns ``None`` when the source is *unlimited* (meaning all
        terminals are always available).
        """
        ...

    # -- consumption -----------------------------------------------

    def consume(self, terminal: str) -> bool:
        """Consume one magic state from *terminal*.

        Returns ``True`` on success.  Returns ``False`` if the terminal
        does not currently have a state available.
        """
        ...

    # -- bookkeeping -----------------------------------------------

    def record_wait_cycle(self) -> None:
        """Record that the scheduler idled for one cycle waiting."""
        ...

    def get_stats(self) -> dict:
        """Return a snapshot of source-specific statistics."""
        ...

    def reset(self) -> None:
        """Reset all internal state (cooldowns, counters, etc.)."""
        ...
