import pytest
from ..state import GameState, PlayerState, PropertyState
from ..board import Board
from ..property import load_property_specs
from ..rules import RulesEngine
from ..cards import load_chance_cards, load_community_cards

@pytest.fixture
def setup():
    board = Board.load_standard_board()
    property_specs = load_property_specs()
    chance_cards = load_chance_cards()
    community_cards = load_community_cards()
    rules = RulesEngine(board, property_specs, chance_cards, community_cards)
    state = GameState(
        players=[PlayerState(id=0, cash=1500, position=0, properties_owned=set(), houses_on_property={}, mortgaged_properties=set(), jail_turns=0, get_out_of_jail_cards=0)],
        properties=[PropertyState(owner=None) for _ in property_specs],
        chance_deck=None,  # Simplified
        community_deck=None,
    )
    return rules, state

def test_movement_passing_go(setup):
    rules, state = setup
    state = rules.move_player(state, 0, 5)  # From 0 to 5, no pass
    assert state.players[0].position == 5
    assert state.players[0].cash == 1500
    state = rules.move_player(state, 0, 35)  # From 5 to 40 (wrap to 0), pass GO
    assert state.players[0].position == 0
    assert state.players[0].cash == 1700

def test_basic_rent(setup):
    rules, state = setup
    state.properties[0].owner = 0  # Player owns Mediterranean
    rent = rules.calculate_rent(state, 0)
    assert rent == 2  # Base rent

def test_railroad_rent(setup):
    rules, state = setup
    # Assume property 2 is Reading Railroad
    state.properties[2].owner = 0
    rent = rules.calculate_rent(state, 2)
    assert rent == 25  # 1 owned

def test_utility_rent(setup):
    rules, state = setup
    # Assume property 7 is Electric Company
    state.properties[7].owner = 0
    rent = rules.calculate_rent(state, 7, dice_roll=6)
    assert rent == 24  # 4x6

def test_doubles_to_jail(setup):
    rules, state = setup
    roll = (3, 3)
    state = rules.handle_doubles_and_jail(state, 0, roll)
    assert state.doubles_count == 1
    state = rules.handle_doubles_and_jail(state, 0, roll)
    assert state.doubles_count == 2
    state = rules.handle_doubles_and_jail(state, 0, roll)
    assert state.players[0].position == 10  # Jail
    assert state.doubles_count == 0

def test_landing_on_property(setup):
    rules, state = setup
    state.properties[0].owner = 1  # Owned by another player
    state.players[0].position = 1  # Land on Mediterranean
    state = rules.handle_landing(state, 0, None)
    assert state.players[0].cash == 1500 - 2  # Paid rent
    assert state.players[1].cash == 1500 + 2

def test_bankruptcy(setup):
    rules, state = setup
    state.players[0].cash = -100
    state = rules.handle_bankruptcy(state, 0)
    assert state.players[0].status == PlayerStatus.BANKRUPT
    assert len(state.players[0].properties_owned) == 0

def test_card_draw(setup):
    rules, state = setup
    card = rules.draw_card(state, "chance", None)  # Mock rng
    assert isinstance(card, Card)

# Add more tests for actions, jail, etc.
