"""
Monopoly Reinforcement Learning Framework

A complete Monopoly game engine with Gymnasium environment support
for training reinforcement learning agents.
"""

from .state import GameState, PlayerState, PropertyState, DeckState, PlayerStatus
from .rules import RulesEngine, ActionType
from .engine import GameEngine, EngineConfig
from .board import Board, TileSpec, TileKind
from .property import PropertySpec, load_property_specs
from .cards import Card, load_chance_cards, load_community_cards
from .trade import TradeProposal, TradeManager
from .auction import run_auction, get_auction_starting_bid

__all__ = [
    # State
    'GameState',
    'PlayerState', 
    'PropertyState',
    'DeckState',
    'PlayerStatus',
    # Rules
    'RulesEngine',
    'ActionType',
    # Engine
    'GameEngine',
    'EngineConfig',
    # Board
    'Board',
    'TileSpec',
    'TileKind',
    # Property
    'PropertySpec',
    'load_property_specs',
    # Cards
    'Card',
    'load_chance_cards',
    'load_community_cards',
    # Trade
    'TradeProposal',
    'TradeManager',
    # Auction
    'run_auction',
    'get_auction_starting_bid',
]
