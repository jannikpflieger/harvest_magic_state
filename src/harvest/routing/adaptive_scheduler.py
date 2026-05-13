"""
Adaptive scheduler: selects among sequential / packing / pathfinder based on
interpretable circuit-structure heuristics, then delegates to the selected
scheduler.

Usage::

    from harvest.routing.adaptive_scheduler import schedule_adaptive, AdaptiveConfig

    results, mode, reason = schedule_adaptive(processor, dag)
    # or with custom thresholds:
    cfg = AdaptiveConfig(low_parallelism_threshold=3.0, high_weight_threshold=5.0)
    results, mode, reason = schedule_adaptive(processor, dag, config=cfg)
"""

import logging
from dataclasses import dataclass
from typing import List, Tuple

from qiskit.dagcircuit import DAGCircuit

from harvest.routing.circuit_features import CircuitFeatures, compute_circuit_features

logger = logging.getLogger("HarvestMagicState.AdaptiveScheduler")


@dataclass
class AdaptiveConfig:
    """Threshold parameters for the adaptive scheduler heuristic.

    Attributes:
        low_parallelism_threshold:
            If ``avg_layer_width`` is **below** this value, the circuit has
            limited parallelism and the sequential scheduler is preferred
            (packing overhead is not worth it).
        high_parallelism_threshold:
            If ``avg_layer_width`` is **at or above** this value, many
            operations are scheduled concurrently.  Their Steiner routes
            contend for the same grid resources, so the pathfinder's
            negotiated congestion resolution outperforms greedy packing.
    """

    low_parallelism_threshold: float = 2.0
    high_parallelism_threshold: float = 5.0


# Module-level default config — users may override without instantiating.
DEFAULT_CONFIG = AdaptiveConfig()


def select_scheduler(
    features: CircuitFeatures,
    config: AdaptiveConfig = DEFAULT_CONFIG,
) -> Tuple[str, str]:
    """Map circuit features to a scheduler mode via interpretable heuristics.

    Decision logic (evaluated in order):

    1. ``avg_layer_width < low_parallelism_threshold``
       → **steiner_tree** (sequential).
       Rationale: the DAG has very few independent operations per layer; the
       overhead of parallel packing yields no benefit.

    2. ``avg_layer_width >= high_parallelism_threshold``
       → **steiner_pathfinder** (negotiated congestion).
       Rationale: many operations are scheduled simultaneously, so their
       Steiner routes compete for the same grid cells.  Pathfinder's
       iterative rip-up-and-reroute resolves congestion far better than
       greedy packing, giving fewer total timesteps.

    3. Otherwise → **steiner_packing** (greedy parallel).
       Rationale: moderate parallelism — greedy packing is efficient and
       routing contention is low.

    Args:
        features: Precomputed :class:`~harvest.routing.circuit_features.CircuitFeatures`.
        config:   Threshold configuration (defaults to :data:`DEFAULT_CONFIG`).

    Returns:
        A ``(scheduler_mode, reason)`` tuple where *scheduler_mode* is one of
        ``"steiner_tree"``, ``"steiner_packing"``, ``"steiner_pathfinder"``
        and *reason* is a human-readable explanation string.
    """
    if features.avg_layer_width < config.low_parallelism_threshold:
        mode = "steiner_tree"
        reason = (
            f"avg_layer_width={features.avg_layer_width:.2f} < "
            f"{config.low_parallelism_threshold} "
            f"→ low parallelism, sequential routing is sufficient"
        )
    elif features.avg_layer_width >= config.high_parallelism_threshold:
        mode = "steiner_pathfinder"
        reason = (
            f"avg_layer_width={features.avg_layer_width:.2f} ≥ "
            f"{config.high_parallelism_threshold} "
            f"→ high parallelism, pathfinder resolves routing congestion"
        )
    else:
        mode = "steiner_packing"
        reason = (
            f"avg_layer_width={features.avg_layer_width:.2f} in "
            f"[{config.low_parallelism_threshold}, {config.high_parallelism_threshold}) "
            f"→ moderate parallelism, greedy packing is efficient"
        )

    return mode, reason


def schedule_adaptive(
    processor,
    dag: DAGCircuit,
    config: AdaptiveConfig = None,
) -> Tuple[List[dict], str, str]:
    """Compute circuit features, select a scheduler, and run it.

    This function exposes the same scheduling contract as the individual
    scheduler functions — it returns the standard list of per-node result
    dicts — but also returns the selection metadata for reporting.

    Args:
        processor: :class:`~harvest.routing.processor.DAGProcessor` instance.
                   Must be freshly initialised (no previously used state).
        dag:       The DAGCircuit to schedule (after PCB conversion).
        config:    Optional :class:`AdaptiveConfig`; falls back to
                   :data:`DEFAULT_CONFIG` when *None*.

    Returns:
        ``(results, selected_mode, reason)`` where:

        * *results* — list of per-node result dicts (same format as all
          scheduler functions).
        * *selected_mode* — the scheduler mode string that was chosen.
        * *reason* — human-readable explanation of the selection.
    """
    if config is None:
        config = DEFAULT_CONFIG

    from harvest.routing.scheduler import process_dag_adaptive

    features = compute_circuit_features(dag)
    logger.info(f"[AdaptiveScheduler] Circuit features: {features}")
    logger.info(
        f"[AdaptiveScheduler] Per-layer thresholds: "
        f"low={config.low_parallelism_threshold}, high={config.high_parallelism_threshold}"
    )

    results = process_dag_adaptive(
        processor,
        dag,
        low_threshold=config.low_parallelism_threshold,
        high_threshold=config.high_parallelism_threshold,
        visualize_each_step=False,
    )

    meta = getattr(processor, "_scheduling_metadata", {})
    counts = meta.get("adaptive_mode_counts", {})
    reason = (
        f"per-layer switching (pack if n<{config.high_parallelism_threshold}, "
        f"pathfinder if n≥{config.high_parallelism_threshold}): "
        f"pack={counts.get('steiner_packing', 0)}, "
        f"path={counts.get('steiner_pathfinder', 0)} steps"
    )
    mode = "steiner_adaptive"
    logger.info(f"[AdaptiveScheduler] Done — {reason}")
    return results, mode, reason
