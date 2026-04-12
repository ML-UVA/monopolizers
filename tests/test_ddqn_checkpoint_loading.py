"""Tests for DDQN checkpoint loading and resume semantics."""

from pathlib import Path

import numpy as np
import pytest
import torch

import train_dqn
from Monopoly.agents.network import QNetwork
from utils.trace_utils import load_ddqn_for_eval


def test_load_ddqn_for_eval_qnetwork_checkpoint(tmp_path):
    """DDQN eval loader should accept QNetwork.save() checkpoint files."""
    model = QNetwork(obs_dim=8, action_dim=6, hidden_dims=[16, 16], device='cpu')
    ckpt_path = tmp_path / 'qnetwork.pt'
    model.save(ckpt_path)

    loaded = load_ddqn_for_eval(str(ckpt_path), device='cpu')

    assert loaded.obs_dim == 8
    assert loaded.action_dim == 6

    action = loaded.select_action(
        np.zeros(8, dtype=np.float32),
        action_mask=np.ones(6, dtype=np.int32),
        epsilon=0.0,
    )
    assert 0 <= action < 6


def test_load_ddqn_for_eval_consolidated_checkpoint(tmp_path):
    """DDQN eval loader should accept consolidated trainer checkpoints."""
    base_model = QNetwork(obs_dim=10, action_dim=7, hidden_dims=[32, 16], device='cpu')
    consolidated_path = tmp_path / 'final.pt'

    torch.save(
        {
            'online_state_dict': base_model.state_dict(),
            'target_state_dict': base_model.state_dict(),
            'optimizer_state_dict': {},
            'obs_dim': 10,
            'action_dim': 7,
            'config': {'hidden_dims': [32, 16]},
            'total_steps': 123,
            'episodes_completed': 4,
        },
        consolidated_path,
    )

    loaded = load_ddqn_for_eval(str(consolidated_path), device='cpu')

    assert loaded.obs_dim == 10
    assert loaded.action_dim == 7


def test_load_ddqn_for_eval_raw_state_dict(tmp_path):
    """DDQN eval loader should accept raw model state_dict checkpoints."""
    base_model = QNetwork(obs_dim=9, action_dim=5, hidden_dims=[12, 12], device='cpu')
    raw_path = tmp_path / 'raw_state_dict.pt'
    torch.save(base_model.state_dict(), raw_path)

    loaded = load_ddqn_for_eval(str(raw_path), device='cpu')

    assert loaded.obs_dim == 9
    assert loaded.action_dim == 5


class _DummySpace:
    def __init__(self, shape=None, n=None):
        self.shape = shape
        self.n = n


class _DummyEnv:
    observation_space = _DummySpace(shape=(8,))
    action_space = _DummySpace(n=6)


class _FakeTrainer:
    last_train_total = None
    loaded_checkpoint = None

    def __init__(self, env, eval_env, output_dir, seed, device, config, verbose):
        self.total_steps = 0
        self.episodes_completed = 0

    def load_checkpoint(self, checkpoint):
        _FakeTrainer.loaded_checkpoint = checkpoint
        self.total_steps = 100

    def load_replay_buffer(self, filepath):
        raise AssertionError("Replay buffer should not be loaded in this test")

    def train(self, total_timesteps, eval_interval, eval_episodes, log_interval, save_interval, progress_bar):
        _FakeTrainer.last_train_total = total_timesteps
        return {'total_steps': total_timesteps}

    def close(self):
        return None


def test_train_ddqn_resume_uses_additional_steps(monkeypatch, tmp_path):
    """When resuming, total_timesteps should be interpreted as additional steps."""
    ckpt_path = tmp_path / 'resume.pt'
    ckpt_path.write_bytes(b'checkpoint')

    monkeypatch.setattr(train_dqn, 'make_monopoly_env', lambda **kwargs: _DummyEnv())

    import Monopoly.agents.ddqn_hybrid as ddqn_module
    monkeypatch.setattr(ddqn_module, 'DDQNHybridTrainer', _FakeTrainer)

    _FakeTrainer.last_train_total = None
    _FakeTrainer.loaded_checkpoint = None

    train_dqn.train_ddqn_hybrid(
        total_timesteps=250,
        output_dir=str(tmp_path / 'run'),
        seed=42,
        resume_checkpoint=str(ckpt_path),
        auto_resume_buffer=False,
        verbose=0,
    )

    assert _FakeTrainer.loaded_checkpoint == str(ckpt_path)
    assert _FakeTrainer.last_train_total == 350


def test_ddqn_diagnostics_disabled_by_default():
    """Q/TD diagnostics should be opt-in by default."""
    from Monopoly.agents.ddqn_hybrid import DDQNHybridTrainer

    assert DDQNHybridTrainer.DEFAULT_CONFIG['log_training_diagnostics'] is False
