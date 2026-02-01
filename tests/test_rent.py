"""
Minimal Rent Calculation Tests.

Rent affects net worth, which drives the H1 dense reward signal.
These tests validate rent calculation invariants.
"""

import pytest
from Monopoly.rules import RulesEngine
from Monopoly.state import GameState, PlayerState, PropertyState
from Monopoly.board import Board
from Monopoly.property import load_property_specs
from Monopoly.cards import load_chance_cards, load_community_cards


@pytest.fixture
def rules_engine():
    """Standard rules engine fixture."""
    board = Board.load_standard_board()
    property_specs = load_property_specs()
    chance_cards = load_chance_cards()
    community_cards = load_community_cards()
    return RulesEngine(board, property_specs, chance_cards, community_cards)


@pytest.fixture
def two_player_state():
    """Two-player game with Mediterranean owned by player 0."""
    players = [
        PlayerState(id=0, cash=1500, position=0, properties_owned={0},
                    houses_on_property={}, mortgaged_properties=set(),
                    jail_turns=0, get_out_of_jail_cards=0),
        PlayerState(id=1, cash=1500, position=0, properties_owned=set(),
                    houses_on_property={}, mortgaged_properties=set(),
                    jail_turns=0, get_out_of_jail_cards=0)
    ]
    properties = [PropertyState(owner=0 if i == 0 else None) for i in range(28)]
    return GameState(players=players, properties=properties,
                     chance_deck=None, community_deck=None)


# =============================================================================
# RENT INVARIANTS (Affect net worth → H1 reward)
# =============================================================================

class TestRentCalculation:
    """Core rent calculation invariants."""

    def test_base_rent(self, rules_engine, two_player_state):
        """Base rent without monopoly."""
        rent = rules_engine.calculate_rent(two_player_state, 0)
        assert rent == 2  # Mediterranean base rent

    def test_monopoly_doubles_rent(self, rules_engine, two_player_state):
        """Monopoly doubles base rent."""
        # Give player 0 Baltic to complete Purple monopoly
        two_player_state.properties[1].owner = 0
        two_player_state.players[0].properties_owned.add(1)
        
        rent = rules_engine.calculate_rent(two_player_state, 0)
        assert rent == 4  # Doubled base rent

    def test_houses_increase_rent(self, rules_engine, two_player_state):
        """Houses increase rent per rent table."""
        # Create monopoly
        two_player_state.properties[1].owner = 0
        two_player_state.players[0].properties_owned.add(1)
        # Add houses
        two_player_state.properties[0].houses_count = 2
        
        rent = rules_engine.calculate_rent(two_player_state, 0)
        assert rent == 30  # 2-house rent for Mediterranean

    def test_mortgaged_no_rent(self, rules_engine, two_player_state):
        """Mortgaged properties collect no rent."""
        two_player_state.properties[0].mortgaged = True
        rent = rules_engine.calculate_rent(two_player_state, 0)
        assert rent == 0

    def test_railroad_rent_scales(self, rules_engine, two_player_state):
        """Railroad rent scales with number owned."""
        # Reading Railroad (idx 2)
        two_player_state.properties[2].owner = 0
        two_player_state.players[0].properties_owned.add(2)
        
        rent1 = rules_engine.calculate_rent(two_player_state, 2)
        assert rent1 == 25  # 1 railroad
        
        # Add B&O Railroad (idx 10)
        two_player_state.properties[10].owner = 0
        two_player_state.players[0].properties_owned.add(10)
        
        rent2 = rules_engine.calculate_rent(two_player_state, 2)
        assert rent2 == 50  # 2 railroads

    def test_utility_rent_uses_dice(self, rules_engine, two_player_state):
        """Utility rent is multiplier × dice roll."""
        # Electric Company (idx 7)
        two_player_state.properties[7].owner = 0
        two_player_state.players[0].properties_owned.add(7)
        
        rent = rules_engine.calculate_rent(two_player_state, 7, dice_roll=6)
        assert rent == 24  # 4 × 6
