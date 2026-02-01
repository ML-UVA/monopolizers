"""Auction logic for Monopoly properties.

NOTE: This module is NOT integrated into the Gym action space.
The current research focus (H1) examines reward shaping only.
When a player declines to buy a property, the property simply remains unowned
(official rules would auction it; this is a simplification).

This module provides auction mechanics that could be integrated in future work.
"""

from typing import Optional, List
from .state import GameState, PlayerState
from .property import load_property_specs, PropertySpec

def run_auction(state: GameState, property_idx: int, starting_bid: int = 1, rng: Optional[object] = None) -> GameState:
    """
    Runs an auction for the given property among all active players.
    Players bid in turn order starting from current_player.
    Each player can bid higher than the current bid or pass.
    Auction ends when all players pass after a bid.
    Winner pays the final bid amount and gains ownership.
    """
    property_specs = load_property_specs()
    spec = property_specs[property_idx]
    active_players = [p for p in state.players if p.status.name == 'ACTIVE']
    if not active_players:
        return state  # No one to auction to
    
    current_bid = starting_bid
    highest_bidder = None
    bidders_left = active_players.copy()
    current_bidder_idx = 0
    
    while bidders_left:
        bidder = bidders_left[current_bidder_idx % len(bidders_left)]
        # In a real implementation, this would prompt for bid; here, simulate simple logic
        # For RL/simulation, assume players bid up to their cash or a heuristic
        max_bid = min(bidder.cash, spec.price * 2)  # Simple cap
        if max_bid > current_bid:
            new_bid = min(max_bid, current_bid + 10)  # Increment bid
            if new_bid > current_bid:
                current_bid = new_bid
                highest_bidder = bidder
                bidders_left = active_players.copy()  # Reset passes
        else:
            bidders_left.remove(bidder)
        current_bidder_idx += 1
    
    if highest_bidder:
        highest_bidder.cash -= current_bid
        state.properties[property_idx].owner = highest_bidder.id
        # Optionally log the auction
    
    return state

def get_auction_starting_bid(property_idx: int) -> int:
    """Returns the starting bid for an auction (e.g., $1 or property price / 10)."""
    property_specs = load_property_specs()
    return max(1, property_specs[property_idx].price // 10)