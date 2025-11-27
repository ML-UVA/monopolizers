import pytest
import numpy as np
from Monopoly.agents.random import RandomAgent

def test_random_agent_act():
    agent = RandomAgent(player_id=0)
    agent.rng = np.random.default_rng(42)  # For determinism in test
    legal_actions = [{'type': 'roll'}, {'type': 'pass'}]
    action = agent.select_action(None, legal_actions)
    assert action in legal_actions

def test_random_agent_no_legal_actions():
    agent = RandomAgent(player_id=0)
    legal_actions = []
    action = agent.select_action(None, legal_actions)
    assert action == {'type': 'pass'}  # Default fallback
