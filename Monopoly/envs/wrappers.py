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
        # auction_active: 1
        # auction_current_bid: 1
        # trade_phase: 1
        # trade_offer_properties: n_properties
        # trade_ask_properties: n_properties
        # trade_offer_cash: 1
        # trade_ask_cash: 1
        # trade_target: 1
        
        flat_dim = (
            1 +                     # player_id
            self.n_players +        # cash
            self.n_players +        # positions
            self.n_properties +     # property_owner
            self.n_properties +     # houses
            self.n_properties +     # mortgaged
            self.n_players +        # jail_turns
            self.n_players +        # get_out_cards
            self.n_actions +        # legal_mask
            1 +                     # turn_number
            2 +                     # last_roll
            1 +                     # bank_houses
            1 +                     # bank_hotels
            1 +                     # auction_active
            1 +                     # auction_current_bid
            1 +                     # trade_phase
            self.n_properties +     # trade_offer_properties
            self.n_properties +     # trade_ask_properties
            1 +                     # trade_offer_cash
            1 +                     # trade_ask_cash
            1 +                     # trade_target
            1 +                     # turn_phase
            self.n_properties +     # pending_trade_offer_properties
            self.n_properties +     # pending_trade_ask_properties
            1 +                     # pending_trade_offer_cash
            1 +                     # pending_trade_ask_cash
            1                       # pending_trade_proposer
        )
        
        self.observation_space = gym.spaces.Box(
            low=0.0, high=np.inf, shape=(flat_dim,), dtype=np.float32
        )

    def observation(self, obs):
        flat_list = []
        
        # Normalize and flatten
        flat_list.append(obs['player_id'].flatten() / (self.n_players - 1) if self.n_players > 1 else obs['player_id'].flatten())
        flat_list.append(obs['cash'].flatten() / 5000.0) # Normalize cash (soft cap)
        flat_list.append(obs['positions'].flatten() / 39.0)

        owner = obs['property_owner'].flatten().astype(np.float32)
        flat_list.append((owner + 1) / self.n_players) 

        flat_list.append(obs['houses'].flatten() / 5.0)
        flat_list.append(obs['mortgaged'].flatten().astype(np.float32))
        flat_list.append(obs['jail_turns'].flatten() / 10.0)
        flat_list.append(obs['get_out_cards'].flatten() / 2.0)
        flat_list.append(obs['legal_mask'].flatten().astype(np.float32))
        flat_list.append(obs['turn_number'].flatten() / 1000.0)
        flat_list.append(obs['last_roll'].flatten() / 6.0)
        flat_list.append(obs['bank_houses'].flatten() / 32.0)
        flat_list.append(obs['bank_hotels'].flatten() / 12.0)
        flat_list.append(obs['auction_active'].flatten().astype(np.float32))
        flat_list.append(obs['auction_current_bid'].flatten() / 5000.0)
        flat_list.append(obs['trade_phase'].flatten().astype(np.float32))
        flat_list.append(obs['trade_offer_properties'].flatten().astype(np.float32))
        flat_list.append(obs['trade_ask_properties'].flatten().astype(np.float32))
        flat_list.append(obs['trade_offer_cash'].flatten() / 5000.0)
        flat_list.append(obs['trade_ask_cash'].flatten() / 5000.0)
        
        target = obs['trade_target'].flatten().astype(np.float32)
        flat_list.append((target + 1) / self.n_players) 

        flat_list.append(obs['turn_phase'].flatten().astype(np.float32))

        flat_list.append(obs['pending_trade_offer_properties'].flatten().astype(np.float32))
        flat_list.append(obs['pending_trade_ask_properties'].flatten().astype(np.float32))
        flat_list.append(obs['pending_trade_offer_cash'].flatten() / 5000.0)
        flat_list.append(obs['pending_trade_ask_cash'].flatten() / 5000.0)

        proposer = obs['pending_trade_proposer'].flatten().astype(np.float32)
        flat_list.append((proposer + 1) / self.n_players)
        
        return np.concatenate(flat_list).astype(np.float32)
