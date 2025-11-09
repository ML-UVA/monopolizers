import pytest
from ..Monopoly.state import GameState, PlayerState, PropertyState, DeckState
from ..Monopoly.board import Board
from ..Monopoly.property import load_property_specs
from ..Monopoly.rules import RulesEngine
from ..Monopoly.cards import load_chance_cards, load_community_cards

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
import pytest
from ..Monopoly.state import GameState, PlayerState, PropertyState, DeckState
from ..Monopoly.board import Board
from ..Monopoly.property import load_property_specs
from ..Monopoly.rules import RulesEngine
from ..Monopoly.cards import load_chance_cards, load_community_cards


@pytest.fixture
def setup():
    board = Board.load_standard_board()
    property_specs = load_property_specs()
    chance_cards = load_chance_cards()
    community_cards = load_community_cards()
    rules = RulesEngine(board, property_specs, chance_cards, community_cards)
    state = GameState(
        players=[
            PlayerState(id=0, cash=1500, position=0, properties_owned=set(), houses_on_property={}, mortgaged_properties=set(), jail_turns=0, get_out_of_jail_cards=0),
            PlayerState(id=1, cash=1500, position=0, properties_owned=set(), houses_on_property={}, mortgaged_properties=set(), jail_turns=0, get_out_of_jail_cards=0),
        ],
        properties=[PropertyState(owner=None) for _ in property_specs],
        chance_deck=DeckState(pointer=0, seed=42),
        community_deck=DeckState(pointer=0, seed=42),
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


def test_buy_property_success(rules_engine, sample_state):
    success = rules_engine.buy_property(sample_state, 0, 3)  # Buy Oriental Avenue
    assert success == True
    assert sample_state.players[0].cash == 1500 - 100  # Price of Oriental
    assert 3 in sample_state.players[0].properties_owned
    assert sample_state.properties[3].owner == 0


def test_buy_property_insufficient_cash(rules_engine, sample_state):
    sample_state.players[0].cash = 50
    success = rules_engine.buy_property(sample_state, 0, 3)
    assert success == False
    assert sample_state.players[0].cash == 50
    assert 3 not in sample_state.players[0].properties_owned


def test_bankruptcy(rules_engine, sample_state):
    sample_state.players[0].cash = -100
    new_state = rules_engine.handle_bankruptcy(sample_state, 0)
    assert new_state.players[0].status.value == 'BANKRUPT'
    assert len(new_state.players[0].properties_owned) == 0


def test_card_draw(rules_engine, sample_state):
    # Mock rng for determinism
    import random
    rng = random.Random(42)
    card = rules_engine.draw_card(sample_state, "chance", rng)
    assert card.id is not None  # Assuming cards have ids
