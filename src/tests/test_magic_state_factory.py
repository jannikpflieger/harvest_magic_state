"""Tests for per-terminal MagicStateFactory and node_needs_magic_state."""

import pytest

from harvest.routing.magic_state_factory import MagicStateFactory
from harvest.compilation.utils import node_needs_magic_state


# ======================================================================
# Helper: lightweight stub for DAG op-nodes
# ======================================================================

class _FakeOp:
    def __init__(self, name: str):
        self.name = name


class _FakeNode:
    def __init__(self, op_name: str):
        self.op = _FakeOp(op_name)


TERMINALS = ["m0", "m1", "m2", "m3"]


# ======================================================================
# node_needs_magic_state
# ======================================================================

class TestNodeNeedsMagicState:
    def test_pauli_evolution_needs_magic(self):
        assert node_needs_magic_state(_FakeNode("PauliEvolution")) is True

    def test_clifford_does_not_need_magic(self):
        for name in ("h", "cx", "cz", "s", "sdg", "x", "z", "swap"):
            assert node_needs_magic_state(_FakeNode(name)) is False

    def test_none_op(self):
        node = object()
        assert node_needs_magic_state(node) is False


# ======================================================================
# Construction / validation
# ======================================================================

class TestFactoryConstruction:
    def test_basic(self):
        f = MagicStateFactory(TERMINALS, preparation_cycles=15)
        assert f.preparation_cycles == 15
        assert f.num_ready == 4  # all start ready
        assert f.any_ready is True

    def test_invalid_preparation_cycles(self):
        with pytest.raises(ValueError):
            MagicStateFactory(TERMINALS, preparation_cycles=0)

    def test_empty_terminals(self):
        f = MagicStateFactory([], preparation_cycles=5)
        assert f.num_ready == 0
        assert f.any_ready is False


# ======================================================================
# Per-terminal cooldown cycle
# ======================================================================

class TestPerTerminalCooldown:
    def test_all_terminals_start_ready(self):
        f = MagicStateFactory(TERMINALS, preparation_cycles=5)
        ready = f.get_ready_terminals()
        assert set(ready) == set(TERMINALS)

    def test_consume_starts_cooldown(self):
        f = MagicStateFactory(TERMINALS, preparation_cycles=5)
        assert f.consume("m0") is True
        ready = f.get_ready_terminals()
        assert "m0" not in ready
        assert len(ready) == 3

    def test_cooldown_completes_after_prep_cycles(self):
        f = MagicStateFactory(TERMINALS, preparation_cycles=3)
        f.consume("m0")
        for _ in range(2):
            f.tick()
            assert "m0" not in f.get_ready_terminals()
        f.tick()  # 3rd tick
        assert "m0" in f.get_ready_terminals()

    def test_cannot_consume_cooling_terminal(self):
        f = MagicStateFactory(TERMINALS, preparation_cycles=5)
        f.consume("m0")
        assert f.consume("m0") is False

    def test_independent_cooldowns(self):
        f = MagicStateFactory(TERMINALS, preparation_cycles=3)
        f.consume("m0")
        f.tick()
        f.consume("m1")  # m1 consumed 1 tick later
        f.tick()
        f.tick()  # m0 should be ready at tick 3
        assert "m0" in f.get_ready_terminals()
        assert "m1" not in f.get_ready_terminals()
        f.tick()  # m1 ready at tick 4
        assert "m1" in f.get_ready_terminals()

    def test_consume_all_then_wait(self):
        f = MagicStateFactory(["a", "b"], preparation_cycles=2)
        f.consume("a")
        f.consume("b")
        assert f.any_ready is False
        assert f.num_ready == 0
        f.tick()
        f.tick()
        assert f.num_ready == 2

    def test_reconsume_after_ready(self):
        f = MagicStateFactory(["x"], preparation_cycles=2)
        f.consume("x")
        f.tick()
        f.tick()
        assert f.consume("x") is True
        assert "x" not in f.get_ready_terminals()


# ======================================================================
# Unlimited mode
# ======================================================================

class TestFactoryUnlimited:
    def test_create_unlimited(self):
        f = MagicStateFactory.create_unlimited()
        assert f.unlimited is True

    def test_unlimited_always_ready(self):
        f = MagicStateFactory.create_unlimited()
        assert f.any_ready is True

    def test_unlimited_get_ready_returns_none(self):
        f = MagicStateFactory.create_unlimited()
        assert f.get_ready_terminals() is None

    def test_unlimited_consume_always_succeeds(self):
        f = MagicStateFactory.create_unlimited()
        for _ in range(100):
            assert f.consume("any_terminal") is True

    def test_unlimited_tracks_consumed(self):
        f = MagicStateFactory.create_unlimited()
        f.consume("t1")
        f.consume("t2")
        assert f._total_consumed == 2


# ======================================================================
# Wait cycle tracking
# ======================================================================

class TestFactoryWaitTracking:
    def test_record_wait_cycle(self):
        f = MagicStateFactory(TERMINALS, preparation_cycles=5)
        assert f._total_wait_cycles == 0
        f.record_wait_cycle()
        f.record_wait_cycle()
        assert f._total_wait_cycles == 2


# ======================================================================
# Reset
# ======================================================================

class TestFactoryReset:
    def test_reset_clears_state(self):
        f = MagicStateFactory(TERMINALS, preparation_cycles=3)
        f.consume("m0")
        f.consume("m1")
        f.tick()
        f.record_wait_cycle()

        f.reset()
        assert f.num_ready == 4  # all ready again
        assert f._current_time == 0
        assert f._total_consumed == 0
        assert f._total_wait_cycles == 0


# ======================================================================
# Stats
# ======================================================================

class TestFactoryStats:
    def test_stats_snapshot(self):
        f = MagicStateFactory(TERMINALS, preparation_cycles=5)
        f.consume("m0")
        stats = f.get_stats()
        assert stats["preparation_cycles"] == 5
        assert stats["num_terminals"] == 4
        assert stats["num_ready"] == 3
        assert stats["total_consumed"] == 1
        assert stats["unlimited"] is False
