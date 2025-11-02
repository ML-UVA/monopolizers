import pytest
from ..Monopoly.trade import TradeProposal, TradeManager
from ..Monopoly.state import GameState, PlayerState, PropertyState

@pytest.fixture
def sample_state():
    players = [
        PlayerState(id=0, cash=500, position=0, properties_owned={1, 2}, houses_on_property={}, mortgaged_properties=set(), jail_turns=0, get_out_of_jail_cards=1),
        PlayerState(id=1, cash=300, position=0, properties_owned={3}, houses_on_property={}, mortgaged_properties=set(), jail_turns=0, get_out_of_jail_cards=0)
    ]
    properties = [PropertyState(owner=None) for _ in range(4)]
    properties[1].owner = 0
    properties[2].owner = 0
    properties[3].owner = 1
    return GameState(players=players, properties=properties, chance_deck=None, community_deck=None)

def test_trade_proposal_validation_valid(sample_state):
    proposal = TradeProposal(
        proposer=0, receiver=1,
        offer={'cash': 100, 'properties': [1], 'jail_cards': 0},
        ask={'cash': 50, 'properties': [3], 'jail_cards': 0}
    )
    assert proposal.validate(sample_state) == True

def test_trade_proposal_validation_invalid_insufficient_cash(sample_state):
    proposal = TradeProposal(
        proposer=0, receiver=1,
        offer={'cash': 600, 'properties': [], 'jail_cards': 0},  # More than proposer has
        ask={'cash': 0, 'properties': [], 'jail_cards': 0}
    )
    assert proposal.validate(sample_state) == False

def test_trade_proposal_validation_invalid_property_not_owned(sample_state):
    proposal = TradeProposal(
        proposer=0, receiver=1,
        offer={'cash': 0, 'properties': [3], 'jail_cards': 0},  # Property owned by receiver
        ask={'cash': 0, 'properties': [], 'jail_cards': 0}
    )
    assert proposal.validate(sample_state) == False

def test_trade_proposal_execution(sample_state):
    proposal = TradeProposal(
        proposer=0, receiver=1,
        offer={'cash': 100, 'properties': [1], 'jail_cards': 1},
        ask={'cash': 50, 'properties': [3], 'jail_cards': 0}
    )
    new_state = proposal.execute(sample_state)
    assert new_state.players[0].cash == 450  # 500 - 100 + 50
    assert new_state.players[1].cash == 450  # 300 + 100 - 50
    assert 1 not in new_state.players[0].properties_owned
    assert 3 in new_state.players[0].properties_owned
    assert 1 in new_state.players[1].properties_owned
    assert 3 not in new_state.players[1].properties_owned
    assert new_state.players[0].get_out_of_jail_cards == 0
    assert new_state.players[1].get_out_of_jail_cards == 1

def test_trade_manager_propose_valid(sample_state):
    manager = TradeManager()
    proposal = TradeProposal(
        proposer=0, receiver=1,
        offer={'cash': 100, 'properties': [1], 'jail_cards': 0},
        ask={'cash': 50, 'properties': [3], 'jail_cards': 0}
    )
    result = manager.propose_trade(sample_state, proposal)
    assert result['valid'] == True

def test_trade_manager_propose_invalid(sample_state):
    manager = TradeManager()
    proposal = TradeProposal(
        proposer=0, receiver=1,
        offer={'cash': 600, 'properties': [], 'jail_cards': 0},
        ask={'cash': 0, 'properties': [], 'jail_cards': 0}
    )
    result = manager.propose_trade(sample_state, proposal)
    assert result['valid'] == False

def test_trade_manager_accept_trade(sample_state):
    manager = TradeManager()
    proposal = TradeProposal(
        proposer=0, receiver=1,
        offer={'cash': 100, 'properties': [1], 'jail_cards': 0},
        ask={'cash': 50, 'properties': [3], 'jail_cards': 0}
    )
    new_state = manager.accept_trade(sample_state, proposal)
    assert new_state.players[0].cash == 450
    assert new_state.players[1].cash == 450

def test_trade_manager_decline_trade(sample_state):
    manager = TradeManager()
    proposal = TradeProposal(
        proposer=0, receiver=1,
        offer={'cash': 100, 'properties': [], 'jail_cards': 0},
        ask={'cash': 0, 'properties': [], 'jail_cards': 0}
    )
    new_state = manager.decline_trade(sample_state, proposal)
    assert new_state == sample_state  # No change