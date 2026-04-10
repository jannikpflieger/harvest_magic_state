"""Tests for MagicStateCultivator and distribution sampler helpers."""

import random

import pytest

from harvest.routing.magic_state_cultivator import (
    CultivationStatus,
    MagicStateCultivator,
    ReadinessSampler,
    fixed_sampler,
    geometric_sampler,
    lognormal_sampler,
)
from harvest.routing.magic_state_source import MagicStateSource


TERMINALS = ["c0", "c1", "c2", "c3"]


# ======================================================================
# Sampler helpers
# ======================================================================

class TestFixedSampler:
    def test_returns_constant(self):
        s = fixed_sampler(7)
        rng = random.Random(0)
        assert all(s(rng) == 7 for _ in range(20))

    def test_rejects_zero(self):
        with pytest.raises(ValueError):
            fixed_sampler(0)

    def test_rejects_negative(self):
        with pytest.raises(ValueError):
            fixed_sampler(-1)


class TestGeometricSampler:
    def test_p_one_always_one(self):
        s = geometric_sampler(1.0)
        rng = random.Random(42)
        assert all(s(rng) == 1 for _ in range(50))

    def test_all_positive(self):
        s = geometric_sampler(0.3)
        rng = random.Random(42)
        samples = [s(rng) for _ in range(200)]
        assert all(v >= 1 for v in samples)

    def test_mean_roughly_correct(self):
        """Mean of geometric(p) should be ~1/p."""
        p = 0.25
        s = geometric_sampler(p)
        rng = random.Random(12345)
        samples = [s(rng) for _ in range(5_000)]
        empirical_mean = sum(samples) / len(samples)
        expected_mean = 1.0 / p
        assert abs(empirical_mean - expected_mean) < 1.0, (
            f"Empirical mean {empirical_mean:.2f} too far from expected {expected_mean}"
        )

    def test_rejects_p_zero(self):
        with pytest.raises(ValueError):
            geometric_sampler(0.0)

    def test_rejects_p_negative(self):
        with pytest.raises(ValueError):
            geometric_sampler(-0.1)

    def test_rejects_p_above_one(self):
        with pytest.raises(ValueError):
            geometric_sampler(1.1)


class TestLognormalSampler:
    def test_all_positive(self):
        s = lognormal_sampler(10.0, 3.0)
        rng = random.Random(42)
        samples = [s(rng) for _ in range(200)]
        assert all(v >= 1 for v in samples)

    def test_mean_roughly_correct(self):
        mean = 20.0
        s = lognormal_sampler(mean, 5.0)
        rng = random.Random(99)
        samples = [s(rng) for _ in range(10_000)]
        empirical_mean = sum(samples) / len(samples)
        assert abs(empirical_mean - mean) < 2.0

    def test_rejects_bad_params(self):
        with pytest.raises(ValueError):
            lognormal_sampler(0.0, 1.0)
        with pytest.raises(ValueError):
            lognormal_sampler(10.0, -1.0)


# ======================================================================
# Protocol conformance
# ======================================================================

class TestProtocol:
    def test_cultivator_satisfies_protocol(self):
        c = MagicStateCultivator(TERMINALS, fixed_sampler(5))
        assert isinstance(c, MagicStateSource)

    def test_unlimited_satisfies_protocol(self):
        c = MagicStateCultivator.create_unlimited()
        assert isinstance(c, MagicStateSource)


# ======================================================================
# Construction
# ======================================================================

class TestConstruction:
    def test_basic(self):
        c = MagicStateCultivator(TERMINALS, fixed_sampler(5))
        assert c.num_ready == 4
        assert c.any_ready is True
        assert c.unlimited is False

    def test_empty_terminals(self):
        c = MagicStateCultivator([], fixed_sampler(5))
        assert c.num_ready == 0
        assert c.any_ready is False

    def test_all_terminals_start_ready(self):
        c = MagicStateCultivator(TERMINALS, fixed_sampler(10))
        ready = c.get_ready_terminals()
        assert set(ready) == set(TERMINALS)


# ======================================================================
# State machine (READY → CULTIVATING → READY)
# ======================================================================

class TestCultivationStateMachine:
    def test_consume_transitions_to_cultivating(self):
        c = MagicStateCultivator(TERMINALS, fixed_sampler(3))
        assert c.consume("c0") is True
        assert "c0" not in c.get_ready_terminals()
        assert c._statuses["c0"] is CultivationStatus.CULTIVATING

    def test_cannot_consume_cultivating_terminal(self):
        c = MagicStateCultivator(TERMINALS, fixed_sampler(3))
        c.consume("c0")
        assert c.consume("c0") is False

    def test_ready_after_fixed_ticks(self):
        c = MagicStateCultivator(TERMINALS, fixed_sampler(4))
        c.consume("c0")
        for i in range(3):
            c.tick()
            assert "c0" not in c.get_ready_terminals(), f"Should not be ready after {i+1} ticks"
        c.tick()  # 4th tick
        assert "c0" in c.get_ready_terminals()

    def test_auto_restart_on_consume(self):
        """After readiness, consuming immediately starts next cultivation."""
        c = MagicStateCultivator(["x"], fixed_sampler(2))
        c.consume("x")
        c.tick()
        c.tick()
        assert c.consume("x") is True
        assert c._statuses["x"] is CultivationStatus.CULTIVATING

    def test_independent_terminals(self):
        c = MagicStateCultivator(TERMINALS, fixed_sampler(3))
        c.consume("c0")
        c.tick()
        c.consume("c1")  # c1 consumed 1 tick later
        c.tick()
        c.tick()  # c0 should be ready at tick 3
        assert "c0" in c.get_ready_terminals()
        assert "c1" not in c.get_ready_terminals()
        c.tick()  # c1 ready at tick 4
        assert "c1" in c.get_ready_terminals()

    def test_consume_all_then_wait(self):
        c = MagicStateCultivator(["a", "b"], fixed_sampler(2))
        c.consume("a")
        c.consume("b")
        assert c.any_ready is False
        assert c.num_ready == 0
        c.tick()
        c.tick()
        assert c.num_ready == 2


# ======================================================================
# Stochastic readiness
# ======================================================================

class TestStochasticReadiness:
    def test_geometric_varies_across_attempts(self):
        """Different consume/cultivation cycles should yield different durations."""
        c = MagicStateCultivator(["t0"], geometric_sampler(0.3), seed=42)
        durations = []
        for _ in range(10):
            c.consume("t0")
            ticks = 0
            while "t0" not in c.get_ready_terminals():
                c.tick()
                ticks += 1
            durations.append(ticks)
        # With p=0.3, not all durations should be identical
        assert len(set(durations)) > 1, f"All durations identical: {durations}"

    def test_not_immediately_available_after_consume(self):
        """With a low-p geometric sampler, it's unlikely to be ready after 1 tick every time."""
        c = MagicStateCultivator(["t0"], geometric_sampler(0.1), seed=7)
        immediately_ready = 0
        trials = 50
        for _ in range(trials):
            c.consume("t0")
            c.tick()
            if "t0" in c.get_ready_terminals():
                immediately_ready += 1
            # Drain remaining ticks to make terminal ready for next trial
            for _ in range(200):
                c.tick()
                if "t0" in c.get_ready_terminals():
                    break
        # With p=0.1 the probability of 1-tick readiness is 0.1
        # Getting immediately ready every time would be extremely unlikely
        assert immediately_ready < trials


# ======================================================================
# max_cycles cap
# ======================================================================

class TestMaxCyclesCap:
    def test_cap_limits_readiness_time(self):
        """With a very low success probability, max_cycles should cap the duration."""
        c = MagicStateCultivator(["t0"], geometric_sampler(0.01), seed=42, max_cycles=5)
        c.consume("t0")
        assert c._remaining["t0"] <= 5
        for _ in range(5):
            c.tick()
        assert "t0" in c.get_ready_terminals()


# ======================================================================
# Unlimited mode
# ======================================================================

class TestUnlimited:
    def test_create_unlimited(self):
        c = MagicStateCultivator.create_unlimited()
        assert c.unlimited is True

    def test_unlimited_always_ready(self):
        c = MagicStateCultivator.create_unlimited()
        assert c.any_ready is True

    def test_unlimited_get_ready_returns_none(self):
        c = MagicStateCultivator.create_unlimited()
        assert c.get_ready_terminals() is None

    def test_unlimited_consume_always_succeeds(self):
        c = MagicStateCultivator.create_unlimited()
        for _ in range(100):
            assert c.consume("any_terminal") is True

    def test_unlimited_tracks_consumed(self):
        c = MagicStateCultivator.create_unlimited()
        c.consume("t1")
        c.consume("t2")
        assert c._total_consumed == 2


# ======================================================================
# Bookkeeping
# ======================================================================

class TestBookkeeping:
    def test_total_consumed(self):
        c = MagicStateCultivator(TERMINALS, fixed_sampler(1))
        c.consume("c0")
        c.tick()
        c.consume("c0")
        assert c._total_consumed == 2

    def test_record_wait_cycle(self):
        c = MagicStateCultivator(TERMINALS, fixed_sampler(5))
        c.record_wait_cycle()
        c.record_wait_cycle()
        assert c._total_wait_cycles == 2

    def test_stats_snapshot(self):
        c = MagicStateCultivator(TERMINALS, fixed_sampler(3), seed=99, max_cycles=10)
        c.consume("c0")
        c.tick()
        c.record_wait_cycle()
        stats = c.get_stats()
        assert stats["source_type"] == "cultivation"
        assert stats["unlimited"] is False
        assert stats["num_terminals"] == 4
        assert stats["total_consumed"] == 1
        assert stats["total_wait_cycles"] == 1
        assert stats["seed"] == 99
        assert stats["max_cycles"] == 10
        assert stats["current_time"] == 1
        assert stats["status_counts"]["cultivating"] == 1
        assert stats["status_counts"]["ready"] == 3


# ======================================================================
# Reset
# ======================================================================

class TestReset:
    def test_reset_clears_state(self):
        c = MagicStateCultivator(TERMINALS, fixed_sampler(3), seed=42)
        c.consume("c0")
        c.consume("c1")
        c.tick()
        c.record_wait_cycle()

        c.reset()

        assert c.num_ready == 4
        assert set(c.get_ready_terminals()) == set(TERMINALS)
        assert c._total_consumed == 0
        assert c._total_wait_cycles == 0
        assert c._current_time == 0

    def test_reset_reseeds_rng(self):
        """After reset, the same sequence of sampled durations should repeat."""
        c = MagicStateCultivator(["t0"], geometric_sampler(0.3), seed=42)

        # First run
        durations_a = []
        for _ in range(5):
            c.consume("t0")
            ticks = 0
            while "t0" not in c.get_ready_terminals():
                c.tick()
                ticks += 1
            durations_a.append(ticks)

        c.reset()

        # Second run (same seed → same durations)
        durations_b = []
        for _ in range(5):
            c.consume("t0")
            ticks = 0
            while "t0" not in c.get_ready_terminals():
                c.tick()
                ticks += 1
            durations_b.append(ticks)

        assert durations_a == durations_b


# ======================================================================
# Deterministic reproducibility
# ======================================================================

class TestReproducibility:
    def test_same_seed_same_sequence(self):
        durations_a = self._run_cultivator(seed=123)
        durations_b = self._run_cultivator(seed=123)
        assert durations_a == durations_b

    def test_different_seed_different_sequence(self):
        durations_a = self._run_cultivator(seed=1)
        durations_b = self._run_cultivator(seed=2)
        # Extremely unlikely to be identical with different seeds over 10 samples
        assert durations_a != durations_b

    @staticmethod
    def _run_cultivator(seed):
        c = MagicStateCultivator(["t0"], geometric_sampler(0.3), seed=seed)
        durations = []
        for _ in range(10):
            c.consume("t0")
            ticks = 0
            while "t0" not in c.get_ready_terminals():
                c.tick()
                ticks += 1
            durations.append(ticks)
        return durations
