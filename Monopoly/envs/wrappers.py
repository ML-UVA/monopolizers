import gymnasium as gym
import numpy as np
from typing import Optional, Dict, Any, Tuple, Callable


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
        # net_worth: n_players
        
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
            1 +
            self.n_players  # net_worth
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
        
        # net_worth: normalize by soft cap (can exceed 1.0 for rich players)
        if 'net_worth' in obs:
            flat_list.append(np.clip(obs['net_worth'].flatten() / 10000.0, 0.0, 10.0))
        else:
            # Fallback if net_worth not present
            flat_list.append(np.zeros(self.n_players, dtype=np.float32))
        
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


# NOTE: RewardShapingWrapper has been REMOVED.
#
# All reward logic for the H1 ablation study is handled directly in MonopolyEnv
# via the `reward_mode` parameter. Using a separate wrapper would create
# confounders and experimental ambiguity.
#
# If you need the old RewardShapingWrapper for other purposes, see git history.


class PettingZooToGymWrapper(gym.Env):
    """Single-agent Gymnasium wrapper around MonopolyAECEnv.

    One player is the learning agent; all others use provided opponent policies.
    This provides a gym-compatible interface backed by the PettingZoo AEC env.

    Usage:
        from Monopoly.envs.pettingzoo_env import MonopolyAECEnv
        aec = MonopolyAECEnv(seed=42)
        env = PettingZooToGymWrapper(aec, agent_id="player_0")
        obs, info = env.reset()
        obs, reward, term, trunc, info = env.step(action)
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 1}

    def __init__(
        self,
        aec_env,
        agent_id: str = "player_0",
        opponent_policies: Optional[Dict[str, Callable]] = None,
    ):
        super().__init__()
        self.aec_env = aec_env
        self.agent_id = agent_id
        self.opponent_policies = opponent_policies or {}

        # Mirror spaces from the AEC env
        self.observation_space = aec_env.observation_space(agent_id)
        self.action_space = aec_env.action_space(agent_id)

    def reset(self, seed=None, options=None) -> Tuple[Dict, Dict]:
        """Reset and simulate opponent turns until learning agent's turn."""
        self.aec_env.reset(seed=seed, options=options)

        # Simulate opponent turns if agent isn't first
        self._simulate_opponents()

        obs = self.aec_env.observe(self.agent_id)
        return obs, {}

    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """Take action for the learning agent, then simulate opponents."""
        self.aec_env.step(action)

        # Collect reward for this agent
        reward = self.aec_env.rewards.get(self.agent_id, 0.0)

        # Check if game ended
        terminated = self.aec_env.terminations.get(self.agent_id, False)
        truncated = self.aec_env.truncations.get(self.agent_id, False)

        if terminated or truncated:
            obs = self.aec_env.observe(self.agent_id)
            info = self.aec_env.infos.get(self.agent_id, {})
            return obs, reward, terminated, truncated, info

        # Simulate opponent turns until it's our agent's turn again
        self._simulate_opponents()

        # Accumulate rewards earned during opponent turns
        reward += self.aec_env.rewards.get(self.agent_id, 0.0)

        terminated = self.aec_env.terminations.get(self.agent_id, False)
        truncated = self.aec_env.truncations.get(self.agent_id, False)

        obs = self.aec_env.observe(self.agent_id)
        info = self.aec_env.infos.get(self.agent_id, {})
        return obs, reward, terminated, truncated, info

    def _simulate_opponents(self) -> None:
        """Step through opponent agents until it's the learning agent's turn."""
        max_opponent_steps = 200  # Safety limit
        steps = 0

        while (
            self.aec_env.agent_selection != self.agent_id
            and self.aec_env.agents
            and steps < max_opponent_steps
        ):
            current = self.aec_env.agent_selection
            terminated = self.aec_env.terminations.get(current, False)
            truncated = self.aec_env.truncations.get(current, False)

            if terminated or truncated:
                self.aec_env.step(None)  # dead step
                steps += 1
                continue

            # Get observation and select action
            obs = self.aec_env.observe(current)
            mask = obs["action_mask"]
            legal_actions = np.where(mask == 1)[0]

            if current in self.opponent_policies:
                action = self.opponent_policies[current](obs, legal_actions)
            elif len(legal_actions) > 0:
                action = int(legal_actions[0])  # Default: first legal action
            else:
                action = 89  # end_turn fallback

            self.aec_env.step(action)
            steps += 1

            # Check if game ended during opponent play
            if (
                self.aec_env.terminations.get(self.agent_id, False)
                or self.aec_env.truncations.get(self.agent_id, False)
            ):
                break

    def render(self):
        self.aec_env.render()

    def close(self):
        self.aec_env.close()
