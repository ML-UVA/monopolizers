from typing import Callable, Optional, Tuple, Dict, Any, List
import numpy as np
import gym
from .engine import GameEngine

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
        pass

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        pass

    def step(self, action) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        pass

    def render(self, mode='human') -> None:
        pass

    def close(self) -> None:
        pass

    def seed(self, seed=None) -> List[int]:
        pass

    def compute_observation(self, player_id: int) -> Dict:
        pass

    def compute_legal_actions_mask(self, player_id: int) -> np.ndarray:
        pass