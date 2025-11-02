import pytest
import numpy as np
from ..Monopoly.agents.random import RandomAgent

def test_random_agent_act():
    agent = RandomAgent()
    agent.rng = np.random.default_rng(42)  # For determinism in test
    legal_mask = np.array([True, False, True, False])
    action = agent.act(None, legal_mask)
    assert action in [0, 2]  # Only legal indices

def test_random_agent_no_legal_actions():
    agent = RandomAgent()
    legal_mask = np.array([False, False])
    action = agent.act(None, legal_mask)
    assert action == 0  # Default
