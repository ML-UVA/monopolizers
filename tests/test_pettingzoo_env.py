import pytest
from ..Monopoly.envs.pettingzoo_env import MonopolyAECEnv

def test_pettingzoo_env_creation():
    # Since it's a stub, test instantiation
    env = MonopolyAECEnv(None, [])
    assert env is not None
    # Further tests would require implementing the env
