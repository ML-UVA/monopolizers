from typing import Callable, Optional, Tuple, Dict, Any, List
import numpy as np
import gym
# Replace ambiguous relative import with parent-package relative + fallback to absolute.
try:
    from ..engine import GameEngine  # engine.py in parent Monopoly package
except Exception:
    try:
        from monopolizers.Monopoly.engine import GameEngine  # absolute path fallback (IDE/runtime)
    except Exception:
        # Last-resort: keep original relative form (may still warn if run as script)
        from .engine import GameEngine

"""Gym wrapper for the Monopoly game engine.

This file contains the skeleton Gym environment interface. Methods are intentionally
left unimplemented and will raise NotImplementedError to make intended API explicit
while you implement the engine and observation/action schemas.
"""

class MonopolyEnv(gym.Env):
    # Parameters:
    # game_engine_factory: Callable[[seed], GameEngine] OR game_engine instance
    # opponent_policies: List[Callable[[obs], action]]
    # obs_type: 'compact' or 'verbose'
    # max_turns: int
    # Attributes:
    # action_space: gym.spaces (e.g., Discrete or Dict)
    # observation_space: gym.spaces
    # engine: GameEngine
    # current_player_id: int (the learning agent id)

    def __init__(self, game_engine_factory, opponent_policies, obs_type='compact', max_turns=1000):
        # Minimal bookkeeping; full implementation should set action_space and observation_space.
        self.game_engine_factory = game_engine_factory
        self.opponent_policies = opponent_policies
        self.obs_type = obs_type
        self.max_turns = max_turns
        # Placeholder spaces; replace with concrete spaces once observation/action schemas are decided.
        self.action_space = None
        self.observation_space = None
        self.engine: GameEngine = None
        self.current_player_id = 0

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        raise NotImplementedError("MonopolyEnv.reset must be implemented with engine initialization and observation return.")

    def step(self, action) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        raise NotImplementedError("MonopolyEnv.step must apply action via the GameEngine and return (obs, reward, terminated, truncated, info).")

    def render(self, mode='human') -> None:
        raise NotImplementedError("MonopolyEnv.render should visualize the game state (text or pygame).")

    def close(self) -> None:
        # No-op by default; override if resources (e.g., pygame) need explicit cleanup.
        pass

    def seed(self, seed=None) -> List[int]:
        raise NotImplementedError("MonopolyEnv.seed should set RNG seed for reproducibility.")

    def compute_observation(self, player_id: int) -> Dict:
        raise NotImplementedError("Compute and return observation for the given player_id.")

    def compute_legal_actions_mask(self, player_id: int) -> np.ndarray:
        raise NotImplementedError("Return a boolean mask of legal actions for the current player.")