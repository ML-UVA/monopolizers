from typing import Optional, Callable, Any
from .state import GameState

class MCTSAgent:
    def __init__(self, rollouts: int, evaluator: Optional[Callable] = None):
        pass

    def act(self, state: GameState) -> Any:  # action
        pass

    def set_time_budget(self, seconds):
        pass