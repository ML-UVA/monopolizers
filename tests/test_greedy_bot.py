import pytest
from Monopoly.agents.greedy import GreedyAgent
from Monopoly.state import GameState, PlayerState, PropertyState, DeckState, PlayerStatus

def _make_test_state(cash=1500, player_id=0):
    """Create a minimal game state for testing."""
    players = [
        PlayerState(id=0, cash=cash, position=0, properties_owned=set(), 
                   houses_on_property={}, mortgaged_properties=set(),
                   jail_turns=0, get_out_of_jail_cards=0, status=PlayerStatus.ACTIVE),
        PlayerState(id=1, cash=1500, position=0, properties_owned=set(),
                   houses_on_property={}, mortgaged_properties=set(),
                   jail_turns=0, get_out_of_jail_cards=0, status=PlayerStatus.ACTIVE)
    ]
    properties = [PropertyState(owner=None) for _ in range(28)]
    return GameState(
        players=players,
        properties=properties,
        chance_deck=DeckState(pointer=0, seed=42),
        community_deck=DeckState(pointer=0, seed=42),
        current_player=player_id
    )

def test_greedy_buy_behavior():
    agent = GreedyAgent(player_id=0, safety_margin=50)
    state = _make_test_state(cash=200)
    legal_actions = [
        {'type': 'buy', 'property_idx': 0, 'cost': 100},
        {'type': 'pass'},
        {'type': 'end_turn'}
    ]
    action = agent.select_action(state, legal_actions)
    assert action['type'] == 'buy'  # Should prefer buying since cash > cost + margin

def test_greedy_pass_when_insufficient_cash():
    agent = GreedyAgent(player_id=0, safety_margin=50)
    state = _make_test_state(cash=140)
    legal_actions = [
        {'type': 'buy', 'property_idx': 0, 'cost': 100},
        {'type': 'pass'},
        {'type': 'end_turn'}
    ]
    action = agent.select_action(state, legal_actions)
    assert action['type'] == 'pass'  # Cash not enough with margin

def test_greedy_roll_preference():
    agent = GreedyAgent(player_id=0)
    state = _make_test_state(cash=200)
    legal_actions = [
        {'type': 'roll'},
        {'type': 'pass'},
        {'type': 'end_turn'}
    ]
    action = agent.select_action(state, legal_actions)
    assert action['type'] == 'roll'  # Prefers rolling

def test_greedy_no_legal_actions():
    agent = GreedyAgent(player_id=0)
    state = _make_test_state(cash=200)
    legal_actions = []
    action = agent.select_action(state, legal_actions)
    assert action == {'type': 'pass'}  # Returns pass as fallback