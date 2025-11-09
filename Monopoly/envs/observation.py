"""
Observation utilities for encoding and decoding game state observations.

Provides functions to convert GameState into compact or verbose observation formats
for different types of agents (neural networks, rule-based, etc.).
"""

import numpy as np
from typing import Dict, Any, Optional
from ..state import GameState, PlayerStatus
from ..property import PropertySpec
from typing import List


def encode_observation_compact(
    state: GameState,
    player_id: int,
    property_specs: List[PropertySpec],
    num_players: int = 4,
    board_size: int = 40
) -> np.ndarray:
    """
    Encode game state into a compact flat numpy array.
    
    This is useful for simple neural network architectures that expect
    a fixed-size vector input.
    
    Args:
        state: Current game state
        player_id: ID of the player for whom we're generating the observation
        property_specs: List of property specifications
        num_players: Total number of players in the game
        board_size: Size of the game board (default 40)
    
    Returns:
        1D numpy array containing encoded state
    """
    features = []
    
    # Player-specific features (for the observing player)
    player = state.players[player_id]
    features.append(player.cash / 1000.0)  # Normalized cash
    features.append(player.position / board_size)  # Normalized position
    features.append(len(player.properties_owned) / len(property_specs))  # Property ratio
    features.append(player.jail_turns / 3.0)  # Normalized jail turns
    features.append(player.get_out_of_jail_cards)
    features.append(1.0 if player.status == PlayerStatus.ACTIVE else 0.0)
    
    # Other players' features (relative observations)
    for i in range(num_players):
        if i == player_id:
            continue
        other = state.players[i] if i < len(state.players) else None
        if other:
            features.append(other.cash / 1000.0)
            features.append(other.position / board_size)
            features.append(len(other.properties_owned) / len(property_specs))
            features.append(1.0 if other.status == PlayerStatus.ACTIVE else 0.0)
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])
    
    # Property features (for each property)
    for i, prop in enumerate(state.properties):
        # Owner encoding: -1=unowned, 0=self, 1=other
        if prop.owner is None:
            features.append(-1.0)
        elif prop.owner == player_id:
            features.append(0.0)
        else:
            features.append(1.0)
        
        # Houses (normalized)
        features.append(prop.houses_count / 5.0)
        
        # Mortgaged
        features.append(1.0 if prop.mortgaged else 0.0)
    
    # Global features
    features.append(state.bank_houses_left / 32.0)
    features.append(state.bank_hotels_left / 12.0)
    features.append(state.turn_number / 1000.0)  # Normalized turn number
    features.append(state.doubles_count / 3.0)
    
    # Last roll
    if state.last_roll:
        features.append(state.last_roll[0] / 6.0)
        features.append(state.last_roll[1] / 6.0)
    else:
        features.extend([0.0, 0.0])
    
    return np.array(features, dtype=np.float32)


def encode_observation_dict(
    state: GameState,
    player_id: int,
    property_specs: List[PropertySpec],
    legal_mask: Optional[np.ndarray] = None,
    num_players: int = 4
) -> Dict[str, np.ndarray]:
    """
    Encode game state into a dictionary of numpy arrays.
    
    This format is compatible with Gymnasium Dict observation spaces
    and provides more structured information.
    
    Args:
        state: Current game state
        player_id: ID of the player for whom we're generating the observation
        property_specs: List of property specifications
        legal_mask: Optional pre-computed legal action mask
        num_players: Total number of players in the game
    
    Returns:
        Dictionary mapping feature names to numpy arrays
    """
    obs = {}
    
    # Player ID
    obs['player_id'] = np.array([player_id], dtype=np.int32)
    
    # Player cash (for all players)
    obs['cash'] = np.array([
        p.cash if i < len(state.players) else 0
        for i, p in enumerate(state.players)
    ], dtype=np.float32)
    
    # Player positions
    obs['positions'] = np.array([
        p.position if i < len(state.players) else 0
        for i, p in enumerate(state.players)
    ], dtype=np.int32)
    
    # Property ownership (-1 for unowned, player_id for owned)
    obs['property_owner'] = np.array([
        p.owner if p.owner is not None else -1
        for p in state.properties
    ], dtype=np.int32)
    
    # Houses on properties
    obs['houses'] = np.array([
        p.houses_count for p in state.properties
    ], dtype=np.int32)
    
    # Mortgaged properties
    obs['mortgaged'] = np.array([
        int(p.mortgaged) for p in state.properties
    ], dtype=np.int32)
    
    # Jail status
    obs['jail_turns'] = np.array([
        p.jail_turns if i < len(state.players) else 0
        for i, p in enumerate(state.players)
    ], dtype=np.int32)
    
    # Get out of jail cards
    obs['get_out_cards'] = np.array([
        p.get_out_of_jail_cards if i < len(state.players) else 0
        for i, p in enumerate(state.players)
    ], dtype=np.int32)
    
    # Legal action mask (if provided)
    if legal_mask is not None:
        obs['legal_mask'] = legal_mask
    
    # Turn number
    obs['turn_number'] = np.array([state.turn_number], dtype=np.int32)
    
    # Last roll
    obs['last_roll'] = np.array(
        state.last_roll if state.last_roll else [1, 1],
        dtype=np.int32
    )
    
    # Bank resources
    obs['bank_houses'] = np.array([state.bank_houses_left], dtype=np.int32)
    obs['bank_hotels'] = np.array([state.bank_hotels_left], dtype=np.int32)
    
    return obs


def encode_observation_verbose(
    state: GameState,
    player_id: int,
    property_specs: List[PropertySpec],
    board
) -> Dict[str, Any]:
    """
    Encode game state into a verbose dictionary with interpretable features.
    
    This format is useful for debugging, rule-based agents, and human-readable
    game state inspection.
    
    Args:
        state: Current game state
        player_id: ID of the player for whom we're generating the observation
        property_specs: List of property specifications
        board: Board object for tile information
    
    Returns:
        Dictionary with interpretable game state information
    """
    player = state.players[player_id]
    
    obs = {
        'player_id': player_id,
        'player_cash': player.cash,
        'player_position': player.position,
        'player_position_name': board.get_tile(player.position).name,
        'player_properties': list(player.properties_owned),
        'player_property_names': [
            property_specs[idx].name for idx in player.properties_owned
        ],
        'player_jail_turns': player.jail_turns,
        'player_jail_cards': player.get_out_of_jail_cards,
        'player_status': player.status.value,
        
        # Opponents
        'opponents': [
            {
                'id': i,
                'cash': p.cash,
                'position': p.position,
                'position_name': board.get_tile(p.position).name,
                'num_properties': len(p.properties_owned),
                'status': p.status.value,
            }
            for i, p in enumerate(state.players) if i != player_id
        ],
        
        # Properties
        'properties': [
            {
                'idx': i,
                'name': spec.name,
                'group': spec.group,
                'price': spec.price,
                'owner': prop.owner,
                'owner_name': f"Player {prop.owner}" if prop.owner is not None else "Unowned",
                'houses': prop.houses_count,
                'mortgaged': prop.mortgaged,
            }
            for i, (spec, prop) in enumerate(zip(property_specs, state.properties))
        ],
        
        # Game state
        'turn_number': state.turn_number,
        'current_player': state.current_player,
        'last_roll': state.last_roll,
        'doubles_count': state.doubles_count,
        'bank_houses_left': state.bank_houses_left,
        'bank_hotels_left': state.bank_hotels_left,
        
        # Derived features
        'active_players': sum(1 for p in state.players if p.status == PlayerStatus.ACTIVE),
        'total_properties_owned': sum(len(p.properties_owned) for p in state.players),
        'unowned_properties': sum(1 for p in state.properties if p.owner is None),
    }
    
    return obs


def compute_property_value(
    state: GameState,
    player_id: int,
    property_specs: List[PropertySpec]
) -> float:
    """
    Compute total property value for a player.
    
    Includes:
    - Purchase price of owned properties
    - Investment in houses/hotels
    - Mortgage value of mortgaged properties
    
    Args:
        state: Current game state
        player_id: Player ID
        property_specs: List of property specifications
    
    Returns:
        Total property value
    """
    player = state.players[player_id]
    total_value = 0.0
    
    for prop_idx in player.properties_owned:
        spec = property_specs[prop_idx]
        prop = state.properties[prop_idx]
        
        if prop.mortgaged:
            # Mortgaged property contributes mortgage value
            total_value += spec.mortgage_value
        else:
            # Unmortgaged property contributes full price
            total_value += spec.price
            
            # Add value of houses/hotels
            if prop.houses_count > 0:
                total_value += prop.houses_count * spec.house_cost
    
    return total_value


def compute_net_worth(
    state: GameState,
    player_id: int,
    property_specs: List[PropertySpec]
) -> float:
    """
    Compute total net worth for a player (cash + property value).
    
    Args:
        state: Current game state
        player_id: Player ID
        property_specs: List of property specifications
    
    Returns:
        Total net worth
    """
    player = state.players[player_id]
    cash = player.cash
    property_value = compute_property_value(state, player_id, property_specs)
    return cash + property_value
