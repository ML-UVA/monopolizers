from typing import Optional, Tuple, Dict, Any
import numpy as np
from .rules import RulesEngine
from .state import GameState

class EngineConfig:
    # max_turns etc.
    pass

class GameEngine:
    rules: RulesEngine
    state: GameState
    rng: np.random.Generator
    config: EngineConfig

    def __init__(self, rules: RulesEngine, initial_state: GameState, rng: Optional[np.random.Generator] = None, config: Optional[EngineConfig] = None):
        pass

    def reset(self, seed: Optional[int] = None) -> GameState:
        pass

    def step(self, action: Dict) -> Tuple[GameState, float, bool, Dict]:
        pass

    def step_for_current_player(self, action: Dict) -> Tuple[GameState, float, bool, Dict]:
        pass

    def advance_to_next_active_player(self) -> None:
        pass

    def get_observation(self, player_id: int, obs_type: str = 'compact') -> Dict:
        pass

    def legal_actions_for_current_player(self) -> np.ndarray or List[Dict]:
        pass

    def save_replay(self) -> str:
        pass

    def load_replay(self, replay: str) -> None:
        pass