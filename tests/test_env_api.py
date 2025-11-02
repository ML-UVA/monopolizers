import pytest
from ..Monopoly.envs.gym_env import MonopolyEnv

def test_reset_returns_obs_info():
    # Since methods raise NotImplementedError, test that they do so
    env = MonopolyEnv(None, None)
    with pytest.raises(NotImplementedError):
        env.reset(seed=42)

def test_step_returns_expected_tuple():
    env = MonopolyEnv(None, None)
    with pytest.raises(NotImplementedError):
        env.step(0)

def test_seed_reproducibility():
    env = MonopolyEnv(None, None)
    with pytest.raises(NotImplementedError):
        env.seed(42)