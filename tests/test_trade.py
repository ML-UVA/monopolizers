"""
Minimal Trade System Tests.

Tests for trading mechanics that affect net worth, which impacts H1 reward signal.
"""

import pytest
from Monopoly.trade import SimpleTrade, compute_trade_price, decode_trade_action, encode_trade_action
from Monopoly.state import GameState, PlayerState, PropertyState, PlayerStatus
from Monopoly.property import load_property_specs


@pytest.fixture
def property_specs():
    return load_property_specs()


@pytest.fixture
def two_player_state():
    """Minimal game state for trade testing."""
    players = [
        PlayerState(
            id=0, cash=1000, position=0,
            properties_owned={1},  # Owns Mediterranean (idx 1)
            houses_on_property={},
            mortgaged_properties=set(),
            jail_turns=0, get_out_of_jail_cards=0,
            status=PlayerStatus.ACTIVE
        ),
        PlayerState(
            id=1, cash=500, position=10,
            properties_owned=set(),
            houses_on_property={},
            mortgaged_properties=set(),
            jail_turns=0, get_out_of_jail_cards=0,
            status=PlayerStatus.ACTIVE
        ),
    ]
    properties = [PropertyState(owner=None, houses_count=0, mortgaged=False) for _ in range(28)]
    properties[1].owner = 0
    
    return GameState(
        players=players,
        properties=properties,
        chance_deck=None, community_deck=None,
        bank_houses_left=32, bank_hotels_left=12,
        current_player=0, last_roll=None, doubles_count=0,
        turn_number=0, seed=42, has_rolled=True,
        awaiting_buy_decision=False, pending_rent=0, rent_creditor=None
    )


# =============================================================================
# TRADE VALIDATION (Research invariant: trades must preserve game integrity)
# =============================================================================

class TestTradeValidation:
    """Test that invalid trades are rejected."""

    def test_valid_trade_passes(self, two_player_state, property_specs):
        """A valid trade should pass validation."""
        price = compute_trade_price(property_specs, 1)
        trade = SimpleTrade(seller_id=0, buyer_id=1, property_idx=1, price=price)
        assert trade.validate(two_player_state)

    def test_buyer_cannot_afford(self, two_player_state, property_specs):
        """Trade fails if buyer lacks funds."""
        # Price is 1.5 * mortgage = 1.5 * 30 = 45; buyer has 500
        # Force unaffordable
        two_player_state.players[1].cash = 10
        price = compute_trade_price(property_specs, 1)
        trade = SimpleTrade(seller_id=0, buyer_id=1, property_idx=1, price=price)
        assert not trade.validate(two_player_state)

    def test_seller_does_not_own(self, two_player_state, property_specs):
        """Trade fails if seller doesn't own property."""
        price = compute_trade_price(property_specs, 5)  # Not owned by player 0
        trade = SimpleTrade(seller_id=0, buyer_id=1, property_idx=5, price=price)
        assert not trade.validate(two_player_state)

    def test_mortgaged_property(self, two_player_state, property_specs):
        """Trade fails if property is mortgaged."""
        two_player_state.properties[1].mortgaged = True
        price = compute_trade_price(property_specs, 1)
        trade = SimpleTrade(seller_id=0, buyer_id=1, property_idx=1, price=price)
        assert not trade.validate(two_player_state)

    def test_improved_property(self, two_player_state, property_specs):
        """Trade fails if property has houses."""
        two_player_state.properties[1].houses_count = 1
        price = compute_trade_price(property_specs, 1)
        trade = SimpleTrade(seller_id=0, buyer_id=1, property_idx=1, price=price)
        assert not trade.validate(two_player_state)


# =============================================================================
# TRADE EXECUTION (Research invariant: net worth conservation)
# =============================================================================

class TestTradeExecution:
    """Test that trade execution correctly transfers assets."""

    def test_property_and_cash_transfer(self, two_player_state, property_specs):
        """Trade correctly transfers property and cash."""
        price = compute_trade_price(property_specs, 1)
        initial_seller_cash = two_player_state.players[0].cash
        initial_buyer_cash = two_player_state.players[1].cash
        
        trade = SimpleTrade(seller_id=0, buyer_id=1, property_idx=1, price=price)
        new_state = trade.execute(two_player_state)
        
        # Property transferred
        assert 1 not in new_state.players[0].properties_owned
        assert 1 in new_state.players[1].properties_owned
        assert new_state.properties[1].owner == 1
        
        # Cash transferred
        assert new_state.players[0].cash == initial_seller_cash + price
        assert new_state.players[1].cash == initial_buyer_cash - price


# =============================================================================
# ACTION ENCODING (Research invariant: bijective encoding)
# =============================================================================

class TestActionEncoding:
    """Test that action encoding is a bijection."""

    def test_encode_decode_roundtrip(self):
        """Encoding and decoding are inverse operations."""
        num_players = 4
        agent_id = 0
        
        for prop_idx in range(28):
            for buyer_id in range(1, num_players):  # Skip agent
                action = encode_trade_action(prop_idx, buyer_id, agent_id, num_players)
                decoded = decode_trade_action(action, agent_id, num_players)
                
                assert decoded is not None
                assert decoded[0] == prop_idx
                assert decoded[1] == buyer_id

    def test_action_range(self):
        """Trade actions occupy [90, 173]."""
        # First trade action
        action = encode_trade_action(0, 1, 0, 4)
        assert action == 90
        
        # Last trade action (property 27, buyer 3)
        action = encode_trade_action(27, 3, 0, 4)
        assert action == 173


# =============================================================================
# PRICING (Research invariant: deterministic pricing)
# =============================================================================

class TestTradePricing:
    """Test pricing formula consistency."""

    def test_price_is_1_5x_mortgage(self, property_specs):
        """Price = 1.5 × mortgage_value."""
        for i, spec in enumerate(property_specs):
            expected = int(spec.mortgage_value * 1.5)
            actual = compute_trade_price(property_specs, i)
            assert actual == expected, f"Property {i}: expected {expected}, got {actual}"
