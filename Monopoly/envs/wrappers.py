import gymnasium as gym
import numpy as np
from typing import Optional, Dict, Any, Tuple

class MonopolyFlattenWrapper(gym.ObservationWrapper):
    """
    Flattens the dictionary observation into a single vector for dense layers.
    Performs basic normalization on features.
    
    This wrapper is essential for using standard RL algorithms that expect
    flat vector observations (like DQN with MLP policy).
    """
    def __init__(self, env):
        super().__init__(env)
        
        self.n_players = 4  # Default
        if hasattr(env.unwrapped, 'num_players'):
            self.n_players = env.unwrapped.num_players
            
        self.n_properties = 28
        self.n_actions = env.action_space.n
        
        # Calculate total dimension
        # player_id: 1
        # cash: n_players
        # positions: n_players
        # property_owner: n_properties
        # houses: n_properties
        # mortgaged: n_properties
        # jail_turns: n_players
        # get_out_cards: n_players
        # legal_mask: n_actions
        # turn_number: 1
        # last_roll: 2
        # bank_houses: 1
        # bank_hotels: 1
        
        flat_dim = (
            1 + 
            self.n_players + 
            self.n_players + 
            self.n_properties + 
            self.n_properties + 
            self.n_properties + 
            self.n_players + 
            self.n_players + 
            self.n_actions + 
            1 + 
            2 + 
            1 + 
            1
        )
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(flat_dim,), dtype=np.float32
        )

    def observation(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """Flatten and normalize the dictionary observation."""
        flat_list = []
        
        # Normalize and flatten each component
        # player_id: normalize to [0, 1]
        flat_list.append(obs['player_id'].flatten() / max(self.n_players - 1, 1))
        
        # cash: normalize by soft cap (can exceed 1.0 for rich players)
        flat_list.append(np.clip(obs['cash'].flatten() / 5000.0, -2.0, 10.0))
        
        # positions: normalize to [0, 1]
        flat_list.append(obs['positions'].flatten() / 39.0)
        
        # property_owner: encode as -1 (unowned), or player_id normalized
        # We'll keep it simple: -1 for unowned, else player_id / (n_players - 1)
        prop_owner = obs['property_owner'].flatten().astype(np.float32)
        prop_owner[prop_owner >= 0] = prop_owner[prop_owner >= 0] / max(self.n_players - 1, 1)
        flat_list.append(prop_owner)
        
        # houses: normalize to [0, 1]
        flat_list.append(obs['houses'].flatten() / 5.0)
        
        # mortgaged: already binary
        flat_list.append(obs['mortgaged'].flatten().astype(np.float32))
        
        # jail_turns: normalize
        flat_list.append(obs['jail_turns'].flatten() / 4.0)
        
        # get_out_cards: normalize
        flat_list.append(np.clip(obs['get_out_cards'].flatten() / 2.0, 0.0, 2.0))
        
        # legal_mask: already binary
        flat_list.append(obs['legal_mask'].flatten().astype(np.float32))
        
        # turn_number: normalize
        flat_list.append(np.clip(obs['turn_number'].flatten() / 1000.0, 0.0, 2.0))
        
        # last_roll: normalize to [0, 1]
        flat_list.append(obs['last_roll'].flatten() / 6.0)
        
        # bank_houses: normalize
        flat_list.append(obs['bank_houses'].flatten() / 32.0)
        
        # bank_hotels: normalize
        flat_list.append(obs['bank_hotels'].flatten() / 12.0)
        
        return np.concatenate(flat_list).astype(np.float32)


class ActionMaskWrapper(gym.Wrapper):
    """
    Wrapper that provides action masking functionality for algorithms
    that support it (like MaskablePPO from sb3-contrib).
    """
    def __init__(self, env):
        super().__init__(env)
        
    def action_masks(self) -> np.ndarray:
        """Return the current legal action mask."""
        if hasattr(self.env.unwrapped, '_get_legal_mask'):
            return self.env.unwrapped._get_legal_mask().astype(bool)
        # Fallback: all actions legal
        return np.ones(self.action_space.n, dtype=bool)
    
    def step(self, action: int) -> Tuple[Any, float, bool, bool, Dict]:
        return self.env.step(action)
    
    def reset(self, **kwargs) -> Tuple[Any, Dict]:
        return self.env.reset(**kwargs)


class RewardShapingWrapper(gym.Wrapper):
    """
    Wrapper that applies additional reward shaping to help with learning.
    """
    def __init__(self, env, 
                 cash_weight: float = 0.001,
                 property_weight: float = 0.5,
                 survival_bonus: float = 0.01):
        super().__init__(env)
        self.cash_weight = cash_weight
        self.property_weight = property_weight
        self.survival_bonus = survival_bonus
        self._prev_cash = 0
        self._prev_properties = 0
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        agent_id = self.env.unwrapped.agent_player_id
        self._prev_cash = self.env.unwrapped.state.players[agent_id].cash
        self._prev_properties = len(self.env.unwrapped.state.players[agent_id].properties_owned)
        return obs, info
    
    def step(self, action: int):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        agent_id = self.env.unwrapped.agent_player_id
        state = self.env.unwrapped.state
        
        # Cash change reward
        current_cash = state.players[agent_id].cash
        cash_delta = current_cash - self._prev_cash
        reward += cash_delta * self.cash_weight
        
        # Property acquisition reward
        current_properties = len(state.players[agent_id].properties_owned)
        prop_delta = current_properties - self._prev_properties
        reward += prop_delta * self.property_weight
        
        # Small survival bonus
        if not terminated:
            reward += self.survival_bonus
        
        self._prev_cash = current_cash
        self._prev_properties = current_properties
        
        return obs, reward, terminated, truncated, info
