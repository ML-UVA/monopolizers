import pytest
from ..Monopoly.rules import RulesEngine
from ..Monopoly.state import GameState, PlayerState, PropertyState
from ..Monopoly.board import Board
from ..Monopoly.property import load_property_specs
from ..Monopoly.cards import load_chance_cards, load_community_cards

@pytest.fixture
def rules_engine():
    board = Board.load_standard_board()
    property_specs = load_property_specs()
    chance_cards = load_chance_cards()
    community_cards = load_community_cards()
    return RulesEngine(board, property_specs, chance_cards, community_cards)

@pytest.fixture
def sample_state():
    players = [
        PlayerState(id=0, cash=1500, position=0, properties_owned={0}, houses_on_property={}, mortgaged_properties=set(), jail_turns=0, get_out_of_jail_cards=0),
        PlayerState(id=1, cash=1500, position=0, properties_owned=set(), houses_on_property={}, mortgaged_properties=set(), jail_turns=0, get_out_of_jail_cards=0)
    ]
    # By default only Mediterranean (0) is owned; individual tests will set additional ownership as needed
    properties = [PropertyState(owner=0 if i == 0 else None) for i in range(28)]
    return GameState(players=players, properties=properties, chance_deck=None, community_deck=None)

def test_rent_calculation_with_houses_and_monopoly(rules_engine, sample_state):
    # Mediterranean (0) owned by 0, monopoly with Baltic (1), add houses
    # set up monopoly explicitly for this test
    sample_state.properties[1].owner = 0
    sample_state.players[0].properties_owned.add(1)
    sample_state.properties[0].houses_count = 2  # 2 houses
    rent = rules_engine.calculate_rent(sample_state, 0)
    assert rent == 30  # Rent table for 2 houses on Mediterranean

def test_rent_calculation_no_monopoly(rules_engine, sample_state):
    # Only Mediterranean owned, no monopoly
    sample_state.properties[0].houses_count = 0
    rent = rules_engine.calculate_rent(sample_state, 0)
    assert rent == 2  # Base rent

def test_rent_calculation_monopoly_no_houses(rules_engine, sample_state):
    # Monopoly but no houses
    # create monopoly explicitly
    sample_state.properties[1].owner = 0
    sample_state.players[0].properties_owned.add(1)
    rent = rules_engine.calculate_rent(sample_state, 0)
    assert rent == 4  # Doubled base rent for monopoly

def test_rent_calculation_railroad(rules_engine, sample_state):
    # Reading Railroad (2)
    sample_state.properties[2].owner = 0
    rent = rules_engine.calculate_rent(sample_state, 2)
    assert rent == 25  # 1 railroad

    # Add another railroad (10)
    sample_state.properties[10].owner = 0
    rent = rules_engine.calculate_rent(sample_state, 2)
    assert rent == 50  # 2 railroads

def test_rent_calculation_utility(rules_engine, sample_state):
    # Electric Company (7)
    sample_state.properties[7].owner = 0
    rent = rules_engine.calculate_rent(sample_state, 7, dice_roll=6)
    assert rent == 24  # 4x6

    # Add Water Works (20)
    sample_state.properties[20].owner = 0
    rent = rules_engine.calculate_rent(sample_state, 7, dice_roll=6)
    assert rent == 60  # 10x6

def test_rent_calculation_mortgaged(rules_engine, sample_state):
    sample_state.properties[0].mortgaged = True
    rent = rules_engine.calculate_rent(sample_state, 0)
    assert rent == 0  # No rent if mortgaged