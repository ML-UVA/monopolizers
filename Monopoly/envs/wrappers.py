import gymnasium as gym
import numpy as np

class MonopolyFlattenWrapper(gym.ObservationWrapper):
    """
    Flattens the dictionary observation into a single vector for dense layers.
    Performs basic normalization on features.
    """
    def __init__(self, env):
        super().__init__(env)
        
        self.n_players = 4 # Default, should ideally get from env
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
            low=-1.0, high=1.0, shape=(flat_dim,), dtype=np.float32
        )

    def observation(self, obs):
        flat_list = []
        
        # Normalize and flatten
        flat_list.append(obs['player_id'].flatten() / (self.n_players - 1) if self.n_players > 1 else obs['player_id'].flatten())
        flat_list.append(obs['cash'].flatten() / 5000.0) # Normalize cash (soft cap)
        flat_list.append(obs['positions'].flatten() / 39.0)
        flat_list.append(obs['property_owner'].flatten() / (self.n_players - 1) if self.n_players > 1 else obs['property_owner'].flatten())
        flat_list.append(obs['houses'].flatten() / 5.0)
        flat_list.append(obs['mortgaged'].flatten())
        flat_list.append(obs['jail_turns'].flatten() / 10.0)
        flat_list.append(obs['get_out_cards'].flatten() / 2.0)
        flat_list.append(obs['legal_mask'].flatten())
        flat_list.append(obs['turn_number'].flatten() / 1000.0)
        flat_list.append(obs['last_roll'].flatten() / 6.0)
        flat_list.append(obs['bank_houses'].flatten() / 32.0)
        flat_list.append(obs['bank_hotels'].flatten() / 12.0)
        
        return np.concatenate(flat_list).astype(np.float32)
