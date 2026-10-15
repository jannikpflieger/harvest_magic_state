"""Common magic-state cultivation model shared by the baseline adapters.

The default (``magic_model = "free"``) keeps the matched-IR assumption: magic
states are always ready at every port.  ``magic_model = "cultivation"`` gives
every compiler the same supply process instead: each magic port (HARVEST magic
patch, Silva storage site, Litinski/O3LS port) cultivates one state at a time,
and after a state is consumed the next one becomes ready after an
exponentially distributed number of logical cycles with mean
``cultivation_mean_cycles`` (rounded, at least one cycle).  PureMagic receives
the same distribution through its native ``--magic-state-lambda = 1/mean``.
The default mean of 3 logical cycles is the expected cultivation occupancy
FLASQ's cultivation data gives at d = 13, p = 1e-3 (v_cult = 3.0 blocks).
"""

from __future__ import annotations

import random
from typing import Any, List, Mapping, Optional

FREE = "free"
CULTIVATION = "cultivation"
DEFAULT_MEAN_CYCLES = 3.0


def cultivation_mean(options: Mapping[str, Any]) -> Optional[float]:
    """Mean cultivation cycles if the run uses the cultivation model, else None."""
    model = options.get("magic_model", FREE)
    if model == FREE:
        return None
    if model != CULTIVATION:
        raise ValueError(f"unknown magic_model {model!r}")
    mean = float(options.get("cultivation_mean_cycles", DEFAULT_MEAN_CYCLES))
    if mean <= 0:
        raise ValueError("cultivation_mean_cycles must be positive")
    return mean


def sample_cycles(rng: random.Random, mean: float) -> int:
    """One cultivation time: Exp(mean) rounded, clamped to >= 1 cycle."""
    return max(1, round(rng.expovariate(1.0 / mean)))


def make_cultivator(terminals: List[Any], mean: float, seed: Optional[int]):
    """HARVEST's per-terminal cultivator with the common distribution."""
    from harvest.routing.magic_state_cultivator import MagicStateCultivator, exponential_sampler

    return MagicStateCultivator(
        list(terminals), exponential_sampler(mean), seed=0 if seed is None else int(seed)
    )
