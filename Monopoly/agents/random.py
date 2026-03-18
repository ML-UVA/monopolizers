import numpy as np
from typing import List, Dict, Any
from ..state import GameState
from .agent import Agent
from ..rules import ActionType

class RandomAgent(Agent):
    """Agent that selects actions randomly."""
    
    def __init__(self, player_id: int = -1):
        super().__init__(player_id)
        self.rng = np.random.default_rng()

    def select_action(self, state: GameState, legal_actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not legal_actions:
            return {'type': ActionType.PASS.value}
        
        # Randomly choose an action
        # We can just pick a random index
        choice_idx = self.rng.integers(0, len(legal_actions))
        return legal_actions[choice_idx]
