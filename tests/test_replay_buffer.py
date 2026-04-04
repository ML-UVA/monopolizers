"""Tests for uniform and prioritized replay buffers."""

import numpy as np
import pytest

from Monopoly.agents.ddqn_hybrid import (
    PrioritizedReplayBuffer,
    ReplayBuffer,
    Transition,
)


def _make_transition(obs_dim=4, action_dim=6, idx=0):
    """Create a dummy transition with a deterministic seed based on idx."""
    rng = np.random.RandomState(idx)
    return dict(
        state=rng.randn(obs_dim).astype(np.float32),
        action=rng.randint(action_dim),
        reward=float(rng.randn()),
        next_state=rng.randn(obs_dim).astype(np.float32),
        done=bool(rng.random() < 0.2),
        mask=np.ones(action_dim, dtype=np.int32),
        next_mask=np.ones(action_dim, dtype=np.int32),
    )


class TestUniformReplayBuffer:

    def test_capacity_respected(self):
        buf = ReplayBuffer(capacity=10)
        for i in range(20):
            buf.push(**_make_transition(idx=i))
        assert len(buf) == 10

    def test_sample_batch_size(self):
        buf = ReplayBuffer(capacity=100)
        for i in range(100):
            buf.push(**_make_transition(idx=i))
        batch = buf.sample(32)
        assert len(batch) == 32

    def test_sample_all_transitions(self):
        """All items are Transition instances."""
        buf = ReplayBuffer(capacity=50)
        for i in range(50):
            buf.push(**_make_transition(idx=i))
        batch = buf.sample(10)
        for t in batch:
            assert isinstance(t, Transition)

    def test_save_load_roundtrip(self):
        buf = ReplayBuffer(capacity=20)
        for i in range(15):
            buf.push(**_make_transition(idx=i))
        data = buf.get_data_for_save()

        buf2 = ReplayBuffer(capacity=20)
        buf2.load_data(data)
        assert len(buf2) == len(buf)

        # Verify data matches
        for orig, loaded in zip(buf.buffer, buf2.buffer):
            np.testing.assert_array_equal(orig.state, loaded.state)
            assert orig.action == loaded.action
            assert orig.reward == pytest.approx(loaded.reward)

    def test_buffer_deterministic_with_seed(self):
        """Same pushes should yield same sample with same random seed."""
        import random
        buf = ReplayBuffer(capacity=50)
        for i in range(50):
            buf.push(**_make_transition(idx=i))

        random.seed(123)
        sample1 = buf.sample(10)
        actions1 = [t.action for t in sample1]

        random.seed(123)
        sample2 = buf.sample(10)
        actions2 = [t.action for t in sample2]

        assert actions1 == actions2


class TestPrioritizedReplayBuffer:

    def test_capacity_respected(self):
        buf = PrioritizedReplayBuffer(capacity=10, alpha=0.6)
        for i in range(20):
            buf.push(**_make_transition(idx=i))
        assert len(buf) == 10

    def test_sample_returns_weights_and_indices(self):
        buf = PrioritizedReplayBuffer(capacity=50, alpha=0.6)
        for i in range(50):
            buf.push(**_make_transition(idx=i))

        batch, weights, indices = buf.sample(8, beta=0.4)
        assert len(batch) == 8
        assert weights.shape == (8,)
        assert indices.shape == (8,)
        # Weights should be positive and normalized (max = 1.0)
        assert np.all(weights > 0)
        assert np.max(weights) == pytest.approx(1.0, abs=1e-5)

    def test_per_high_priority_sampled_more(self):
        """Items with high priority should be sampled more frequently."""
        buf = PrioritizedReplayBuffer(capacity=100, alpha=1.0)  # full prioritization
        for i in range(100):
            buf.push(**_make_transition(idx=i))

        # Set 10 items with very high priority, rest with low
        high_indices = list(range(buf.tree.capacity - 1, buf.tree.capacity - 1 + 10))
        for idx in high_indices:
            buf.tree.update(idx, 100.0)
        for idx in range(buf.tree.capacity - 1 + 10, buf.tree.capacity - 1 + 100):
            buf.tree.update(idx, 1.0)

        # Sample 2000 times and count
        sample_counts = np.zeros(100)
        for _ in range(2000):
            _, _, indices = buf.sample(1, beta=1.0)
            data_idx = indices[0] - buf.tree.capacity + 1
            if 0 <= data_idx < 100:
                sample_counts[data_idx] += 1

        high_count = sample_counts[:10].mean()
        low_count = sample_counts[10:].mean()

        # High-priority items should be sampled much more often
        assert high_count > low_count * 3, \
            f"High-priority items ({high_count:.1f}) should be sampled >3x more than low ({low_count:.1f})"

    def test_per_is_weights_differ_with_beta(self):
        """IS weights at beta=0.4 should differ from beta=1.0."""
        buf = PrioritizedReplayBuffer(capacity=50, alpha=0.6)
        for i in range(50):
            buf.push(**_make_transition(idx=i))

        # Give different priorities
        for i in range(buf.tree.capacity - 1, buf.tree.capacity - 1 + 50):
            buf.tree.update(i, float(i + 1))

        np.random.seed(42)
        _, weights_04, _ = buf.sample(10, beta=0.4)
        np.random.seed(42)
        _, weights_10, _ = buf.sample(10, beta=1.0)

        # With beta=1.0, correction is stronger (weights closer to uniform)
        # With beta=0.4, correction is weaker (more variance in weights)
        assert not np.allclose(weights_04, weights_10)

    def test_update_priorities(self):
        buf = PrioritizedReplayBuffer(capacity=20, alpha=0.6)
        for i in range(20):
            buf.push(**_make_transition(idx=i))

        _, _, indices = buf.sample(5, beta=0.4)
        td_errors = np.array([0.1, 0.5, 2.0, 0.01, 1.0])
        buf.update_priorities(indices, td_errors)

        # max_priority should be updated
        assert buf.max_priority >= 2.0

    def test_save_load_roundtrip(self):
        buf = PrioritizedReplayBuffer(capacity=20, alpha=0.6)
        for i in range(15):
            buf.push(**_make_transition(idx=i))
        data = buf.get_data_for_save()

        buf2 = PrioritizedReplayBuffer(capacity=20, alpha=0.6)
        buf2.load_data(data)
        assert len(buf2) == len(buf)
