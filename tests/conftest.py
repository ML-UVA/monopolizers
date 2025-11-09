import pytest
from monopolizers.Monopoly.board import Board
from monopolizers.Monopoly.property import load_property_specs
from monopolizers.Monopoly.cards import load_chance_cards, load_community_cards
from monopolizers.Monopoly.rules import RulesEngine
from monopolizers.Monopoly.state import GameState, PlayerState, PropertyState, DeckState


@pytest.fixture
def rules_engine():
    board = Board.load_standard_board()
    property_specs = load_property_specs()
    chance_cards = load_chance_cards()
    community_cards = load_community_cards()
    return RulesEngine(board, property_specs, chance_cards, community_cards)


@pytest.fixture
def sample_state():
    # Default sample state used across tests: two players, full property list and decks
    players = [
        PlayerState(id=0, cash=1500, position=0, properties_owned=set(), houses_on_property={}, mortgaged_properties=set(), jail_turns=0, get_out_of_jail_cards=0),
        PlayerState(id=1, cash=1500, position=0, properties_owned=set(), houses_on_property={}, mortgaged_properties=set(), jail_turns=0, get_out_of_jail_cards=0)
    ]
    property_specs = load_property_specs()
    properties = [PropertyState(owner=None) for _ in property_specs]
    chance_deck = DeckState(pointer=0, seed=42)
    community_deck = DeckState(pointer=0, seed=42)
    return GameState(players=players, properties=properties, chance_deck=chance_deck, community_deck=community_deck)
