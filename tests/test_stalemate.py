"""Tests for StalemateDetector."""

import pytest
from Monopoly.state import GameState, PlayerState, PropertyState, DeckState, PlayerStatus
from Monopoly.property import load_property_specs
from Monopoly.envs.stalemate import StalemateDetector


def _make_state(num_players=4, cash=1500):
    """Create a minimal GameState for testing."""
    players = [
        PlayerState(
            id=i, cash=cash, position=0,
            properties_owned=set(), houses_on_property={},
            mortgaged_properties=set(), jail_turns=0,
            get_out_of_jail_cards=0, status=PlayerStatus.ACTIVE,
        )
        for i in range(num_players)
    ]
    properties = [
        PropertyState(owner=None, houses_count=0, mortgaged=False)
        for _ in range(28)
    ]
    return GameState(
        players=players,
        properties=properties,
        chance_deck=DeckState(pointer=0, seed=42),
        community_deck=DeckState(pointer=0, seed=43),
        bank_houses_left=32,
        bank_hotels_left=12,
        current_player=0,
        last_roll=None,
        doubles_count=0,
        turn_number=0,
        seed=42,
        has_rolled=False,
        awaiting_buy_decision=False,
        pending_rent=0,
        rent_creditor=None,
    )


class TestStalemateDetector:

    def test_no_stalemate_with_property_changes(self):
        """Property ownership changes should reset the stale counter."""
        detector = StalemateDetector(threshold_rounds=3)
        specs = load_property_specs()
        state = _make_state()

        # First round: baseline
        assert not detector.on_round_complete(state, specs)

        # Second round: property changes
        state.properties[0].owner = 0
        state.players[0].properties_owned.add(0)
        assert not detector.on_round_complete(state, specs)

        # Even after many rounds, as long as things change, no stalemate
        for _ in range(10):
            assert not detector.on_round_complete(state, specs)
            state.properties[0].houses_count += 1
            if state.properties[0].houses_count > 5:
                state.properties[0].houses_count = 0

    def test_stalemate_after_threshold(self):
        """N identical rounds should trigger stalemate."""
        threshold = 3
        detector = StalemateDetector(threshold_rounds=threshold)
        specs = load_property_specs()
        state = _make_state()

        # First call: sets baseline, returns False
        assert not detector.on_round_complete(state, specs)

        # Next threshold calls with no changes
        for i in range(threshold - 1):
            assert not detector.on_round_complete(state, specs)

        # This one should trigger stalemate
        assert detector.on_round_complete(state, specs)

    def test_reset_clears_state(self):
        """After reset, stale counter should be 0."""
        detector = StalemateDetector(threshold_rounds=2)
        specs = load_property_specs()
        state = _make_state()

        # Build up stale rounds
        detector.on_round_complete(state, specs)
        detector.on_round_complete(state, specs)

        # Reset
        detector.reset()

        # Should need full threshold again
        assert not detector.on_round_complete(state, specs)
        assert not detector.on_round_complete(state, specs)
        assert detector.on_round_complete(state, specs)

    def test_bankruptcy_resets_counter(self):
        """A player going bankrupt should reset the stale counter."""
        detector = StalemateDetector(threshold_rounds=3)
        specs = load_property_specs()
        state = _make_state()

        # Build up 2 stale rounds
        detector.on_round_complete(state, specs)
        detector.on_round_complete(state, specs)

        # Player goes bankrupt — active_player_ids changes
        state.players[2].status = PlayerStatus.BANKRUPT
        assert not detector.on_round_complete(state, specs)

        # Counter should have reset; need full threshold again
        assert not detector.on_round_complete(state, specs)
        assert not detector.on_round_complete(state, specs)
        assert detector.on_round_complete(state, specs)
