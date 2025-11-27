import pytest
from Monopoly.agents.mcts import MCTSAgent
from Monopoly.state import GameState, PlayerState, PropertyState, DeckState, PlayerStatus
from Monopoly.engine import GameEngine
from Monopoly.rules import RulesEngine
from Monopoly.board import Board
from Monopoly.property import load_property_specs
from Monopoly.cards import load_chance_cards, load_community_cards

@pytest.fixture
def sample_engine():
    """Create a game engine for MCTS testing."""
    board = Board.load_standard_board()
    property_specs = load_property_specs()
    chance_cards = load_chance_cards()
    community_cards = load_community_cards()
    rules = RulesEngine(board, property_specs, chance_cards, community_cards)
    return GameEngine(rules, seed=42)

@pytest.fixture
def sample_state():
    players = [
        PlayerState(id=0, cash=1500, position=0, properties_owned=set(), 
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
        current_player=0,
        has_rolled=False
    )

def test_mcts_agent_act(sample_state, sample_engine):
    agent = MCTSAgent(player_id=0, engine=sample_engine, rollouts=5, max_depth=5)
    legal_actions = [{'type': 'roll'}]
    action = agent.select_action(sample_state, legal_actions)
    assert isinstance(action, dict)
    assert action['type'] == 'roll'
