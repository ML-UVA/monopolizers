"""
Deterministic Trade Policy for Monopoly DDQN-Hybrid Training.

This module provides a fixed, deterministic trading policy used in the
DDQN-hybrid trainer. The policy is designed to be simple, reproducible,
and well-documented so that experiments are reproducible.

Trading Heuristics (deterministic):
1. Accept any incoming trade offer if it strictly increases agent net worth.
2. Propose trades only when agent has cash > CASH_THRESHOLD (default $500).
3. Proposal logic: if an opponent owns a property in a color group where
   agent owns other properties (potential monopoly completion), offer
   cash + property for the missing property if expected value exceeds threshold.
4. If multiple trades possible, pick the one with maximum immediate net worth gain.

The policy is deterministic given the game state, ensuring reproducible results.
"""

from typing import Any, Dict, List, Optional, Tuple, Set
from dataclasses import dataclass


# Configuration constants
CASH_THRESHOLD = 500  # Minimum cash to consider proposing trades
MONOPOLY_VALUE_THRESHOLD = 200  # Minimum expected monopoly value gain
MAX_CASH_OFFER_RATIO = 0.5  # Max ratio of cash to offer in trade


@dataclass
class TradeProposal:
    """Represents a trade proposal between two players."""
    proposer_id: int
    receiver_id: int
    proposer_gives_cash: int
    proposer_gives_properties: Set[int]
    receiver_gives_cash: int
    receiver_gives_properties: Set[int]


class TradePolicy:
    """
    Deterministic trade policy for Monopoly.
    
    This policy makes reproducible trading decisions based on simple heuristics:
    - Accept trades that increase net worth
    - Propose trades to complete monopolies
    - Never propose trades that decrease net worth
    
    All decisions are deterministic given the game state.
    """
    
    def __init__(
        self,
        agent_id: int = 0,
        cash_threshold: int = CASH_THRESHOLD,
        monopoly_value_threshold: int = MONOPOLY_VALUE_THRESHOLD,
        max_cash_offer_ratio: float = MAX_CASH_OFFER_RATIO
    ):
        self.agent_id = agent_id
        self.cash_threshold = cash_threshold
        self.monopoly_value_threshold = monopoly_value_threshold
        self.max_cash_offer_ratio = max_cash_offer_ratio
        
        # Property group mappings (from property.py)
        self.property_groups = {
            'Purple': [0, 1],
            'Light Blue': [3, 4, 5],
            'Pink': [6, 8, 9],
            'Orange': [11, 12, 13],
            'Red': [14, 15, 16],
            'Yellow': [18, 19, 21],
            'Green': [22, 23, 24],
            'Dark Blue': [26, 27],
            'Railroad': [2, 10, 17, 25],
            'Utility': [7, 20],
        }
        
        # Reverse mapping: property_idx -> group
        self.property_to_group = {}
        for group, props in self.property_groups.items():
            for prop in props:
                self.property_to_group[prop] = group
        
        # Base property values (simplified from property.py)
        self.property_values = {
            0: 60, 1: 60,  # Purple
            3: 100, 4: 100, 5: 120,  # Light Blue
            6: 140, 8: 140, 9: 160,  # Pink
            11: 180, 12: 180, 13: 200,  # Orange
            14: 220, 15: 220, 16: 240,  # Red
            18: 260, 19: 260, 21: 280,  # Yellow
            22: 300, 23: 300, 24: 320,  # Green
            26: 350, 27: 400,  # Dark Blue
            2: 200, 10: 200, 17: 200, 25: 200,  # Railroads
            7: 150, 20: 150,  # Utilities
        }
    
    def compute_net_worth(
        self,
        player_cash: int,
        player_properties: Set[int],
        property_states: List[Any],
        property_specs: List[Any] = None
    ) -> float:
        """
        Compute net worth for a player.
        
        Net worth = cash + sum(property_values)
        Property value considers base price and improvements.
        """
        net_worth = float(player_cash)
        
        for prop_idx in player_properties:
            # Base property value
            base_value = self.property_values.get(prop_idx, 100)
            
            # Check for improvements (if state available)
            if property_states and prop_idx < len(property_states):
                prop_state = property_states[prop_idx]
                if hasattr(prop_state, 'mortgaged') and prop_state.mortgaged:
                    # Mortgaged properties worth less
                    base_value = base_value * 0.5
                elif hasattr(prop_state, 'houses_count'):
                    # Add house/hotel value
                    houses = prop_state.houses_count
                    house_cost = 50 if prop_idx in [0, 1, 3, 4, 5] else 100 if prop_idx < 16 else 150 if prop_idx < 22 else 200
                    base_value += houses * house_cost * 0.5  # Houses sell for half
            
            net_worth += base_value
        
        return net_worth
    
    def should_accept_trade(
        self,
        proposal: TradeProposal,
        game_state: Any
    ) -> bool:
        """
        Determine if agent should accept an incoming trade.
        
        Decision rule (deterministic):
        Accept if and only if trade strictly increases agent net worth.
        
        Args:
            proposal: The trade proposal to evaluate
            game_state: Current game state
            
        Returns:
            True if trade should be accepted
        """
        if proposal.receiver_id != self.agent_id:
            return False
        
        player = game_state.players[self.agent_id]
        
        # Calculate current net worth
        current_nw = self.compute_net_worth(
            player.cash,
            player.properties_owned,
            game_state.properties
        )
        
        # Calculate net worth after trade
        new_cash = player.cash - proposal.receiver_gives_cash + proposal.proposer_gives_cash
        new_properties = player.properties_owned.copy()
        new_properties -= proposal.receiver_gives_properties
        new_properties |= proposal.proposer_gives_properties
        
        new_nw = self.compute_net_worth(
            new_cash,
            new_properties,
            game_state.properties
        )
        
        # Accept if strictly better
        return new_nw > current_nw
    
    def find_trade_opportunities(
        self,
        game_state: Any
    ) -> List[TradeProposal]:
        """
        Find all potential trade opportunities.
        
        Logic:
        - Look for color groups where agent owns some but not all properties
        - Check if opponents own the missing properties
        - Generate proposals to acquire missing properties
        
        Args:
            game_state: Current game state
            
        Returns:
            List of potential trade proposals (sorted by value)
        """
        player = game_state.players[self.agent_id]
        opportunities = []
        
        # Don't propose if low on cash
        if player.cash < self.cash_threshold:
            return []
        
        # For each color group, check if we can complete a monopoly
        for group, group_props in self.property_groups.items():
            # Skip railroads and utilities for monopoly completion
            if group in ['Railroad', 'Utility']:
                continue
            
            owned_in_group = player.properties_owned & set(group_props)
            
            # Only interested if we own at least one but not all
            if len(owned_in_group) == 0 or len(owned_in_group) == len(group_props):
                continue
            
            # Find missing properties
            missing = set(group_props) - owned_in_group
            
            for missing_prop in missing:
                # Find owner
                prop_state = game_state.properties[missing_prop]
                owner = prop_state.owner
                
                if owner is None or owner == self.agent_id:
                    continue
                
                # Calculate trade value
                prop_value = self.property_values.get(missing_prop, 100)
                
                # Offer: cash only (simplest deterministic strategy)
                # Offer up to 1.5x property value to complete monopoly
                max_offer = min(
                    int(player.cash * self.max_cash_offer_ratio),
                    int(prop_value * 1.5)
                )
                
                if max_offer < prop_value:
                    continue  # Can't afford
                
                # Calculate expected value gain from monopoly
                monopoly_bonus = sum(self.property_values.get(p, 100) for p in group_props) * 0.5
                
                if monopoly_bonus < self.monopoly_value_threshold:
                    continue
                
                # Create proposal
                proposal = TradeProposal(
                    proposer_id=self.agent_id,
                    receiver_id=owner,
                    proposer_gives_cash=max_offer,
                    proposer_gives_properties=set(),
                    receiver_gives_cash=0,
                    receiver_gives_properties={missing_prop}
                )
                
                # Calculate net worth change
                current_nw = self.compute_net_worth(
                    player.cash,
                    player.properties_owned,
                    game_state.properties
                )
                
                new_cash = player.cash - max_offer
                new_properties = player.properties_owned | {missing_prop}
                
                # Account for monopoly bonus in valuation
                new_nw = self.compute_net_worth(
                    new_cash,
                    new_properties,
                    game_state.properties
                )
                
                # Add monopoly completion bonus
                if len(new_properties & set(group_props)) == len(group_props):
                    new_nw += monopoly_bonus
                
                nw_gain = new_nw - current_nw
                
                if nw_gain > 0:
                    opportunities.append((nw_gain, proposal))
        
        # Sort by net worth gain (descending) for deterministic selection
        opportunities.sort(key=lambda x: -x[0])
        
        return [p for _, p in opportunities]
    
    def choose_trade(
        self,
        game_state: Any,
        incoming_proposal: Optional[TradeProposal] = None
    ) -> Optional[TradeProposal]:
        """
        Main entry point: choose trade action.
        
        Decision process (deterministic):
        1. If incoming proposal, decide accept/reject
        2. If proposing, find best opportunity
        3. Pick trade with maximum net worth gain
        
        Args:
            game_state: Current game state
            incoming_proposal: Optional incoming trade to respond to
            
        Returns:
            Trade proposal to execute, or None for no trade
        """
        # Handle incoming proposal
        if incoming_proposal is not None:
            if self.should_accept_trade(incoming_proposal, game_state):
                return incoming_proposal  # Accept
            return None  # Reject
        
        # Find our best trade opportunity
        opportunities = self.find_trade_opportunities(game_state)
        
        if opportunities:
            # Return best opportunity (already sorted by value)
            return opportunities[0]
        
        return None


def choose_trade(
    env_state: Any,
    agent_id: int = 0,
    incoming_proposal: Optional[TradeProposal] = None
) -> Optional[TradeProposal]:
    """
    Convenience function for trade decision.
    
    This is the main entry point called by the environment/trainer
    when a trade action is selected.
    
    Args:
        env_state: Current game state from environment
        agent_id: ID of the agent making the decision
        incoming_proposal: Optional incoming trade to respond to
        
    Returns:
        Trade proposal to execute, or None for no trade
    """
    policy = TradePolicy(agent_id=agent_id)
    return policy.choose_trade(env_state, incoming_proposal)


def compute_net_worth_after_trade(
    game_state: Any,
    player_id: int,
    proposal: TradeProposal
) -> float:
    """
    Compute a player's net worth after a trade is executed.
    
    Args:
        game_state: Current game state
        player_id: Player to compute net worth for
        proposal: Trade proposal to evaluate
        
    Returns:
        Estimated net worth after trade
    """
    policy = TradePolicy(agent_id=player_id)
    player = game_state.players[player_id]
    
    # Calculate changes based on role in trade
    if player_id == proposal.proposer_id:
        new_cash = player.cash - proposal.proposer_gives_cash + proposal.receiver_gives_cash
        new_properties = player.properties_owned.copy()
        new_properties -= proposal.proposer_gives_properties
        new_properties |= proposal.receiver_gives_properties
    elif player_id == proposal.receiver_id:
        new_cash = player.cash - proposal.receiver_gives_cash + proposal.proposer_gives_cash
        new_properties = player.properties_owned.copy()
        new_properties -= proposal.receiver_gives_properties
        new_properties |= proposal.proposer_gives_properties
    else:
        # Not involved in trade
        return policy.compute_net_worth(
            player.cash,
            player.properties_owned,
            game_state.properties
        )
    
    return policy.compute_net_worth(new_cash, new_properties, game_state.properties)
