import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pytest
import torch

from Monopoly.agents.ddqn_hybrid import DDQNHybridTrainer


class _DummyUnwrapped:
    def __init__(self, env: "DummyMaskedEnv"):
        self._env = env

    def _get_legal_mask(self):
        return self._env._get_legal_mask()


class DummyMaskedEnv:
    """Tiny deterministic env with explicit action masks.

    Designed for unit tests of DDQN mechanics (masking, stepping, termination).
    """

    def __init__(
        self,
        obs_dim: int = 2,
        action_dim: int = 4,
        horizon: int = 3,
        mask_mode: str = "unwrapped",  # "unwrapped" | "direct"
    ):
        import gymnasium as gym

        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(action_dim)

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.mask_mode = mask_mode

        self._t = 0
        self._state = np.zeros((obs_dim,), dtype=np.float32)

        self._mask = np.ones((action_dim,), dtype=np.float32)
        self._last_action = None

        self.unwrapped = _DummyUnwrapped(self)

    def set_mask(self, mask: np.ndarray):
        mask = np.asarray(mask, dtype=np.float32)
        assert mask.shape == (self.action_dim,)
        self._mask = mask

    def _get_legal_mask(self):
        return self._mask.copy()

    def reset(self, seed=None) -> Tuple[np.ndarray, dict]:
        if seed is not None:
            np.random.seed(seed)
        self._t = 0
        self._state = np.zeros((self.obs_dim,), dtype=np.float32)
        self._last_action = None
        info = {
            "agent_net_worth": 100.0,
            "agent_status": "ACTIVE",
            "active_players": 4,
        }
        return self._state.copy(), info

    def step(self, action: int):
        self._last_action = int(action)
        # Reward is deterministic and depends on action to help learning signal.
        reward = float(action) * 0.1

        self._t += 1
        terminated = self._t >= self.horizon
        truncated = False

        self._state = np.zeros((self.obs_dim,), dtype=np.float32)
        self._state[min(self._t - 1, self.obs_dim - 1)] = 1.0

        info = {
            "agent_net_worth": 100.0 + 10.0 * reward,
            "agent_status": "ACTIVE",
            "active_players": 4,
        }

        return self._state.copy(), reward, terminated, truncated, info

    def close(self):
        return None


def _mk_trainer(tmp_path, env: DummyMaskedEnv, eval_env: DummyMaskedEnv, config=None):
    config = dict(config or {})
    config.setdefault("batch_size", 4)
    config.setdefault("buffer_size", 128)
    config.setdefault("learning_starts", 0)
    config.setdefault("gamma", 0.99)
    config.setdefault("lr", 1e-3)
    config.setdefault("target_update_interval", 50)

    trainer = DDQNHybridTrainer(
        env=env,
        eval_env=eval_env,
        output_dir=str(tmp_path),
        seed=123,
        config=config,
        verbose=0,
    )
    return trainer


def test_get_action_mask_prefers_unwrapped(tmp_path):
    env = DummyMaskedEnv(mask_mode="unwrapped")
    eval_env = DummyMaskedEnv(mask_mode="unwrapped")
    env.set_mask(np.array([1, 0, 1, 0], dtype=np.float32))

    trainer = _mk_trainer(tmp_path, env, eval_env)
    try:
        mask = trainer._get_action_mask(env)
        assert mask.shape == (env.action_space.n,)
        assert mask.dtype == np.float32
        assert mask.tolist() == [1.0, 0.0, 1.0, 0.0]
    finally:
        trainer.close()


def test_get_action_mask_direct_method(tmp_path):
    env = DummyMaskedEnv(mask_mode="direct")
    eval_env = DummyMaskedEnv(mask_mode="direct")

    # Remove unwrapped hook to force direct path.
    env.unwrapped = object()

    env.set_mask(np.array([0, 1, 1, 0], dtype=np.float32))

    trainer = _mk_trainer(tmp_path, env, eval_env)
    try:
        mask = trainer._get_action_mask(env)
        assert mask.tolist() == [0.0, 1.0, 1.0, 0.0]
    finally:
        trainer.close()


def test_epsilon_schedule_monotonic(tmp_path):
    env = DummyMaskedEnv()
    eval_env = DummyMaskedEnv()

    trainer = _mk_trainer(
        tmp_path,
        env,
        eval_env,
        config={
            "eps_start": 1.0,
            "eps_end": 0.1,
            "eps_decay": 100,
        },
    )
    try:
        trainer.total_steps = 0
        eps0 = trainer._get_epsilon()
        trainer.total_steps = 50
        eps50 = trainer._get_epsilon()
        trainer.total_steps = 100
        eps100 = trainer._get_epsilon()
        trainer.total_steps = 200
        eps200 = trainer._get_epsilon()

        assert 1.0 >= eps0 >= eps50 >= eps100 >= eps200
        assert eps200 >= 0.1 - 1e-12
        assert abs(eps0 - 1.0) < 1e-9
        assert eps200 == pytest.approx(0.1, abs=1e-12)
    finally:
        trainer.close()


def test_select_action_respects_mask_in_greedy_mode(tmp_path, monkeypatch):
    env = DummyMaskedEnv(action_dim=4)
    eval_env = DummyMaskedEnv(action_dim=4)

    env.set_mask(np.array([0, 1, 0, 1], dtype=np.float32))

    trainer = _mk_trainer(tmp_path, env, eval_env)
    try:
        # Make Q-values prefer illegal action 0 unless masking is applied.
        def fake_forward(state_tensor: torch.Tensor):
            batch = state_tensor.shape[0]
            out = torch.tensor([100.0, 2.0, 50.0, 3.0], device=trainer.device).view(1, -1)
            return out.repeat(batch, 1)

        monkeypatch.setattr(trainer.q_online, "forward", fake_forward)

        obs, _ = env.reset(seed=0)
        mask = trainer._get_action_mask(env)

        # Force greedy
        action = trainer.select_action(obs, mask, epsilon=0.0)
        assert action in [1, 3]
        assert action == 3  # among legal actions (1 and 3), q=3 for action 3.
    finally:
        trainer.close()


def test_train_step_returns_none_before_learning_starts(tmp_path):
    env = DummyMaskedEnv(action_dim=4)
    eval_env = DummyMaskedEnv(action_dim=4)

    trainer = _mk_trainer(tmp_path, env, eval_env, config={"learning_starts": 999, "batch_size": 4})
    try:
        loss = trainer.train_step()
        assert loss is None
    finally:
        trainer.close()


def test_train_step_updates_online_network_params(tmp_path):
    env = DummyMaskedEnv(obs_dim=2, action_dim=4)
    eval_env = DummyMaskedEnv(obs_dim=2, action_dim=4)

    trainer = _mk_trainer(
        tmp_path,
        env,
        eval_env,
        config={
            "learning_starts": 0,
            "batch_size": 8,
            "buffer_size": 256,
            "lr": 1e-2,
            "use_huber_loss": False,
        },
    )

    try:
        # Fill buffer with varied transitions.
        rng = np.random.default_rng(0)
        for _ in range(64):
            state = rng.normal(size=(2,)).astype(np.float32)
            next_state = rng.normal(size=(2,)).astype(np.float32)
            action = int(rng.integers(0, env.action_space.n))
            reward = float(rng.normal())
            done = bool(rng.integers(0, 2))
            mask = np.ones((env.action_space.n,), dtype=np.float32)
            next_mask = np.ones((env.action_space.n,), dtype=np.float32)
            trainer.replay_buffer.push(state, action, reward, next_state, done, mask, next_mask)

        before = [p.detach().clone() for p in trainer.q_online.parameters()]
        trainer.total_steps = 1000
        loss = trainer.train_step()
        after = [p.detach().clone() for p in trainer.q_online.parameters()]

        assert isinstance(loss, float)
        assert math.isfinite(loss)
        assert any((a - b).abs().max().item() > 0 for a, b in zip(after, before))
    finally:
        trainer.close()


def test_ddqn_target_math_uses_online_argmax_and_target_value(tmp_path):
    env = DummyMaskedEnv(obs_dim=2, action_dim=3)
    eval_env = DummyMaskedEnv(obs_dim=2, action_dim=3)

    trainer = _mk_trainer(
        tmp_path,
        env,
        eval_env,
        config={
            "learning_starts": 0,
            "batch_size": 2,
            "gamma": 0.99,
            "lr": 0.0,  # keep weights fixed
            "use_huber_loss": False,
        },
    )

    try:
        # Replace networks with tiny linear modules so we can control Q values exactly.
        class LinearQ(torch.nn.Module):
            def __init__(self, obs_dim: int, action_dim: int, weight: torch.Tensor):
                super().__init__()
                self.W = torch.nn.Parameter(weight.clone())

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x @ self.W

        # States are basis vectors so output is exactly a row of W.
        states = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        next_states = states.copy()

        # Online next Q rows -> choose action 1 for both, but masks differ.
        # row0: [1, 10, 100] but mask disallows action2 => argmax=1
        # row1: [50, 5, 4] but mask disallows action0 => argmax=1
        W_online = torch.tensor(
            [[1.0, 10.0, 100.0], [50.0, 5.0, 4.0]], dtype=torch.float32
        )

        # Target next Q rows used for value lookup at online argmax action.
        # row0: [7, 3, 9] => value at action1 = 3
        # row1: [1, 2, 30] => value at action1 = 2
        W_target = torch.tensor(
            [[7.0, 3.0, 9.0], [1.0, 2.0, 30.0]], dtype=torch.float32
        )

        trainer.q_online = LinearQ(2, 3, W_online).to(trainer.device)
        trainer.q_target = LinearQ(2, 3, W_target).to(trainer.device)
        trainer.optimizer = torch.optim.Adam(trainer.q_online.parameters(), lr=0.0)

        actions = np.array([0, 2], dtype=np.int64)
        rewards = np.array([0.5, -1.0], dtype=np.float32)
        dones = np.array([0.0, 1.0], dtype=np.float32)

        masks = np.ones((2, 3), dtype=np.float32)
        next_masks = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]], dtype=np.float32)

        for i in range(2):
            trainer.replay_buffer.push(states[i], int(actions[i]), float(rewards[i]), next_states[i], bool(dones[i]), masks[i], next_masks[i])

        trainer.total_steps = 1
        loss = trainer.train_step()
        assert isinstance(loss, float)
        assert math.isfinite(loss)

        # Manual expected targets:
        # sample0: target = 0.5 + 0.99*3 = 3.47
        # sample1: done => target = -1
        t0 = 0.5 + 0.99 * 3.0
        t1 = -1.0

        # Current Q(s,a): sample0 action0 => 1.0, sample1 action2 => 4.0
        q0 = 1.0
        q1 = 4.0

        expected_mse = float(((q0 - t0) ** 2 + (q1 - t1) ** 2) / 2.0)
        assert abs(loss - expected_mse) < 1e-5
    finally:
        trainer.close()


def test_checkpoint_roundtrip_restores_network_weights(tmp_path):
    env = DummyMaskedEnv(obs_dim=2, action_dim=4)
    eval_env = DummyMaskedEnv(obs_dim=2, action_dim=4)

    trainer = _mk_trainer(tmp_path, env, eval_env, config={"learning_starts": 0, "batch_size": 4})
    try:
        # Make a single update so weights change from init.
        rng = np.random.default_rng(0)
        for _ in range(32):
            state = rng.normal(size=(2,)).astype(np.float32)
            next_state = rng.normal(size=(2,)).astype(np.float32)
            action = int(rng.integers(0, env.action_space.n))
            reward = float(rng.normal())
            done = False
            mask = np.ones((env.action_space.n,), dtype=np.float32)
            next_mask = np.ones((env.action_space.n,), dtype=np.float32)
            trainer.replay_buffer.push(state, action, reward, next_state, done, mask, next_mask)

        trainer.total_steps = 1
        trainer.train_step()

        before = [p.detach().cpu().clone() for p in trainer.q_online.parameters()]
        trainer.save_checkpoint(name="pytest")

        # Perturb weights
        with torch.no_grad():
            for p in trainer.q_online.parameters():
                p.add_(torch.randn_like(p) * 0.01)

        trainer.load_checkpoint(name="pytest")
        after = [p.detach().cpu().clone() for p in trainer.q_online.parameters()]

        assert len(before) == len(after)
        assert all(torch.allclose(a, b, atol=0, rtol=0) for a, b in zip(before, after))
    finally:
        trainer.close()


def test_evaluate_writes_eval_rows_with_unique_episode_index(tmp_path, monkeypatch):
    env = DummyMaskedEnv(obs_dim=2, action_dim=4, horizon=2)
    eval_env = DummyMaskedEnv(obs_dim=2, action_dim=4, horizon=2)

    trainer = _mk_trainer(tmp_path, env, eval_env, config={"learning_starts": 0, "batch_size": 4})
    try:
        # Avoid depending on Q-network behavior.
        monkeypatch.setattr(trainer, "select_action", lambda obs, mask, epsilon=0.0: 0)

        metrics = trainer.evaluate(num_episodes=3)
        assert "mean_reward" in metrics

        csv_path = tmp_path / "results" / "ddqn_hybrid_metrics.csv"
        assert csv_path.exists()

        import csv

        rows = []
        with csv_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("eval_tag") == "eval":
                    rows.append(row)

        assert len(rows) == 3
        episodes = [int(r["episode"]) for r in rows]
        assert episodes == sorted(episodes)
        assert len(set(episodes)) == 3
    finally:
        trainer.close()
