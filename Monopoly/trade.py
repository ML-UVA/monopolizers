"""Trade proposal and execution logic for Monopoly.

RESEARCH INTEGRATION: Trading is part of the Gym action space.
The agent can propose to sell any owned (unmortgaged, unimproved) property
to a specific opponent for a fixed price (1.5x mortgage value).

This simplified trading model:
- Deterministic acceptance (opponent always accepts if they can afford it)
- Fixed pricing formula (removes negotiation complexity)
- Affects net worth directly (cash vs property value)

Trading is relevant to H1 because it allows the agent to convert assets
to cash, directly impacting the relative net worth reward signal.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
from .state import GameState, PlayerStatus


@dataclass
class SimpleTrade:
    """
    A simplified trade: agent sells one property to one opponent for cash.
    
    Trade encoding for action space:
        trade_action = 90 + (prop_idx * (num_players - 1)) + target_opponent_offset
        
    Where:
        - prop_idx: property index being sold (0-27)
        - target_opponent_offset: index of opponent in list of non-agent players
        
    For 4 players (3 opponents), this adds 28 * 3 = 84 trade actions.
    Total action space: 90 (base) + 84 (trades) = 174 actions.
    """
    seller_id: int
    buyer_id: int
    property_idx: int
    price: int  # Fixed at 1.5x mortgage value
    
    def validate(self, state: GameState) -> bool:
        """Check if trade is valid given current state."""
        seller = state.players[self.seller_id]
        buyer = state.players[self.buyer_id]
        
        # Both players must be active
        if seller.status != PlayerStatus.ACTIVE or buyer.status != PlayerStatus.ACTIVE:
            return False
        
        # Seller must own the property
        if self.property_idx not in seller.properties_owned:
            return False
        
        # Property must not be mortgaged
        prop = state.properties[self.property_idx]
        if prop.mortgaged:
            return False
        
        # Property must not have houses
        if prop.houses_count > 0:
            return False
        
        # Buyer must be able to afford the price
        if buyer.cash < self.price:
            return False
        
        return True
    
    def execute(self, state: GameState) -> GameState:
        """Execute the trade, transferring ownership and cash."""
        seller = state.players[self.seller_id]
        buyer = state.players[self.buyer_id]
        
        # Transfer property
        seller.properties_owned.remove(self.property_idx)
        buyer.properties_owned.add(self.property_idx)
        state.properties[self.property_idx].owner = self.buyer_id
        
        # Transfer cash
        buyer.cash -= self.price
        seller.cash += self.price
        
        return state


def compute_trade_price(property_specs, prop_idx: int) -> int:
    """
    Compute the fixed trade price for a property.
    
    Formula: price = 1.5 * mortgage_value
    
    This provides a fair exchange rate that:
    - Gives seller more than mortgaging (1.5x vs 1.0x mortgage value)
    - Gives buyer a discount vs market price (1.5x mortgage < purchase price typically)
    """
    spec = property_specs[prop_idx]
    return int(spec.mortgage_value * 1.5)


def decode_trade_action(action: int, agent_player_id: int, num_players: int) -> Optional[Tuple[int, int]]:
    """
    Decode a trade action index to (property_idx, buyer_id).
    
    Trade actions start at index 90.
    Encoding: action = 90 + prop_idx * (num_players - 1) + opponent_offset
    
    Args:
        action: The discrete action index (>= 90)
        agent_player_id: The agent's player ID (seller)
        num_players: Total number of players
        
    Returns:
        Tuple of (property_idx, buyer_id) or None if invalid
    """
    if action < 90:
        return None
    
    trade_idx = action - 90
    num_opponents = num_players - 1
    
    prop_idx = trade_idx // num_opponents
    opponent_offset = trade_idx % num_opponents
    
    if prop_idx >= 28:
        return None
    
    # Convert opponent offset to actual player ID
    # Opponents are all players except agent_player_id
    buyer_id = opponent_offset
    if buyer_id >= agent_player_id:
        buyer_id += 1
    
    return prop_idx, buyer_id


def encode_trade_action(prop_idx: int, buyer_id: int, agent_player_id: int, num_players: int) -> int:
    """
    Encode a trade into a discrete action index.
    
    Args:
        prop_idx: Property index being sold
        buyer_id: Player ID of buyer
        agent_player_id: The agent's player ID (seller)
        num_players: Total number of players
        
    Returns:
        Discrete action index
    """
    num_opponents = num_players - 1
    
    # Convert buyer_id to opponent offset
    opponent_offset = buyer_id if buyer_id < agent_player_id else buyer_id - 1
    
    return 90 + prop_idx * num_opponents + opponent_offset
