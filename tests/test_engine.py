import pytest
from ..Monopoly.engine import GameEngine
from ..Monopoly.rules import RulesEngine
from ..Monopoly.state import GameState, PlayerState, PropertyState
from ..Monopoly.board import Board
from ..Monopoly.property import load_property_specs
from ..Monopoly.cards import load_chance_cards, load_community_cards

@pytest.fixture
def game_engine():
    board = Board.load_standard_board()
    property_specs = load_property_specs()
    chance_cards = load_chance_cards()
    community_cards = load_community_cards()
    rules = RulesEngine(board, property_specs, chance_cards, community_cards)
    return GameEngine(rules, seed=42)

@pytest.fixture
def initial_state():
    players = [PlayerState(id=0, cash=1500, position=0, properties_owned=set(), houses_on_property={}, mortgaged_properties=set(), jail_turns=0, get_out_of_jail_cards=0)]
    properties = [PropertyState(owner=None) for _ in range(28)]
    return GameState(players=players, properties=properties, chance_deck=None, community_deck=None)

def test_run_turn_basic(game_engine, initial_state):
    new_state = game_engine.run_turn(initial_state)
    assert new_state.turn_number == 1
    assert new_state.current_player == 0  # Single player, stays 0
    # Position should have changed based on roll — or a card may have moved the player to GO (cash will change)
    assert (new_state.players[0].position != 0) or (new_state.players[0].cash != 1500)
