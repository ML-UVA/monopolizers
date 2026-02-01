"""
Monopoly Reinforcement Learning Framework

A complete Monopoly game engine with Gymnasium environment support
for training reinforcement learning agents.

RESEARCH FOCUS (H1): Comparing dense relative-net-worth reward vs sparse terminal
reward for long-horizon stochastic games.
"""

from .state import GameState, PlayerState, PropertyState, DeckState, PlayerStatus
from .rules import RulesEngine, ActionType
from .engine import GameEngine, EngineConfig
from .board import Board, TileSpec, TileKind
from .property import PropertySpec, load_property_specs
from .cards import Card, load_chance_cards, load_community_cards
from .trade import SimpleTrade, compute_trade_price, decode_trade_action, encode_trade_action
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
    'SimpleTrade',
    'compute_trade_price',
    'decode_trade_action',
    'encode_trade_action',
    # Auction
    'run_auction',
    'get_auction_starting_bid',
]
