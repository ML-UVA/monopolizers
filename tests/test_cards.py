import pytest
import numpy as np
from ..Monopoly.cards import CardDeck, load_chance_cards, load_community_cards
from ..Monopoly.state import GameState, PlayerState, PropertyState

def test_card_deck_shuffle_and_draw():
    cards = load_chance_cards()
    deck = CardDeck(cards)
    rng = np.random.default_rng(42)
    deck.shuffle(rng)
    card = deck.draw()
    assert card.id is not None

def test_card_deck_reset():
    cards = load_chance_cards()
    deck = CardDeck(cards)
    rng = np.random.default_rng(42)
    deck.reset(rng)
    assert deck.pointer == 0

def test_load_chance_cards():
    cards = load_chance_cards()
    assert len(cards) > 0
    assert cards[0].id == "Advance to GO"

def test_load_community_cards():
    cards = load_community_cards()
    assert len(cards) > 0
    assert cards[0].id == "Advance to GO"
