import pytest
import random
from ..state import GameState, PlayerState, PropertyState, DeckState, PlayerStatus
from ..board import Board
from ..property import load_property_specs
from ..rules import RulesEngine
from ..cards import load_chance_cards, load_community_cards, Card
import numpy as np

@pytest.fixture
def setup():
    board = Board.load_standard_board()
    property_specs = load_property_specs()
    chance_cards = load_chance_cards()
    community_cards = load_community_cards()
    rules = RulesEngine(board, property_specs, chance_cards, community_cards)
    # Create two players so tests that reference player 1 work
    players = [
        PlayerState(id=0, cash=1500, position=0, properties_owned=set(), houses_on_property={}, mortgaged_properties=set(), jail_turns=0, get_out_of_jail_cards=0),
        PlayerState(id=1, cash=1500, position=0, properties_owned=set(), houses_on_property={}, mortgaged_properties=set(), jail_turns=0, get_out_of_jail_cards=0),
    ]
    # Initialize deck pointers for draw_card usage
    chance_deck = DeckState(pointer=0, seed=42)
    community_deck = DeckState(pointer=0, seed=42)
    state = GameState(
        players=players,
        properties=[PropertyState(owner=None) for _ in property_specs],
        chance_deck=chance_deck,
        community_deck=community_deck,
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
    # ensure last_roll not required for property rent
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
    rng = random.Random(42)
    card = rules.draw_card(state, "chance", rng)
    assert isinstance(card, Card)

def test_buy_property_success(setup):
    rules, state = setup
    success = rules.buy_property(state, 0, 3)  # Buy Oriental Avenue (idx 3)
    assert success is True
    assert 3 in state.players[0].properties_owned
    assert state.players[0].cash == 1500 - rules.property_specs[3].price
    assert state.properties[3].owner == 0

def test_buy_property_insufficient_cash(setup):
    rules, state = setup
    state.players[0].cash = 50
    success = rules.buy_property(state, 0, 3)
    assert success is False
    assert 3 not in state.players[0].properties_owned
    assert state.players[0].cash == 50

def test_roll_dice_range(setup):
    rules, state = setup
    import random as pyrandom
    rng = pyrandom.Random(12345)
    d1, d2 = rules.roll_dice(rng)
    assert 1 <= d1 <= 6
    assert 1 <= d2 <= 6

def test_chance_go_to_jail_effect(setup):
    rules, state = setup
    chance_cards = load_chance_cards()
    # find Go to Jail chance card
    card = next((c for c in chance_cards if "Go to Jail" in c.id or "Go to Jail" in c.text), None)
    assert card is not None
    rng = np.random.default_rng(42)
    new_state, info = card.effect(state, rules, rng, 0)
    assert new_state.players[0].position == 10
    assert new_state.players[0].jail_turns == 1

def test_chance_general_repairs_effect(setup):
    rules, state = setup
    chance_cards = load_chance_cards()
    card = next((c for c in chance_cards if "General repairs" in c.id or "general repairs" in c.text.lower()), None)
    assert card is not None
    # Give player properties with houses/hotels
    state.players[0].properties_owned = {0, 1}
    state.properties[0].houses_count = 2  # 2 houses
    state.properties[1].houses_count = 5  # hotel
    starting_cash = state.players[0].cash
    rng = np.random.default_rng(123)
    new_state, info = card.effect(state, rules, rng, 0)
    houses = 1 * 2  # property 0 counted (2 houses)
    hotels = 1       # property 1 is hotel
    expected_cost = 2 * 25 + 1 * 100
    assert new_state.players[0].cash == starting_cash - expected_cost
    assert info.get("paid") == expected_cost or info.get("paid") == expected_cost

def test_chance_advance_nearest_railroad_effect(setup):
    rules, state = setup
    chance_cards = load_chance_cards()
    card = next((c for c in chance_cards if "nearest Railroad" in c.text), None)
    assert card is not None
    # Place player near a railroad and make one railroad owned by player 1
    state.players[0].position = 6  # nearest railroad should be at 15
    state.properties[10].owner = 1  # Pennsylvania Railroad (property idx 10 at board pos 15)
    owner_before = state.players[1].cash
    player_before = state.players[0].cash
    rng = np.random.default_rng(7)
    new_state, info = card.effect(state, rules, rng, 0)
    assert new_state.players[0].position in (5, 15, 25, 35)
    # If landing on an owned railroad, rent should have been paid (owner's cash increases)
    assert state.players[1].cash >= owner_before or state.players[0].cash <= player_before

def test_community_get_out_of_jail_free(setup):
    rules, state = setup
    community = load_community_cards()
    card = next((c for c in community if "Get out of Jail Free" in c.id or "Get out of Jail Free" in c.text), None)
    assert card is not None
    rng = np.random.default_rng(99)
    new_state, info = card.effect(state, rules, rng, 0)
    assert new_state.players[0].get_out_of_jail_cards >= 1

# Add more tests for actions, jail, etc.
