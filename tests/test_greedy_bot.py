import pytest
from ..Monopoly.agents.greedy import GreedyAgent

def test_greedy_buy_behavior():
    agent = GreedyAgent(safety_margin=50)
    observation = {
        'player_cash': 200,
        'legal_actions': [
            {'type': 'buy', 'cost': 100},
            {'type': 'pass'},
            {'type': 'collect_rent'}
        ]
    }
    action = agent.act(observation)
    assert action['type'] == 'buy'  # Should prefer buying since cash > cost + margin

def test_greedy_pass_when_insufficient_cash():
    agent = GreedyAgent(safety_margin=50)
    observation = {
        'player_cash': 140,
        'legal_actions': [
            {'type': 'buy', 'cost': 100},
            {'type': 'pass'},
            {'type': 'collect_rent'}
        ]
    }
    action = agent.act(observation)
    assert action['type'] == 'pass'  # Cash not enough with margin

def test_greedy_collect_rent():
    agent = GreedyAgent()
    observation = {
        'player_cash': 200,
        'legal_actions': [
            {'type': 'collect_rent'},
            {'type': 'pass'}
        ]
    }
    action = agent.act(observation)
    assert action['type'] == 'collect_rent'  # Prefers collecting rent

def test_greedy_no_legal_actions():
    agent = GreedyAgent()
    observation = {
        'player_cash': 200,
        'legal_actions': []
    }
    action = agent.act(observation)
    assert action == {}  # Returns empty dict if no actions