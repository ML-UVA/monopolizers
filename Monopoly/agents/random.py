"""Random agent for baseline comparison.

Selects legal actions uniformly at random. Seed can be passed for reproducibility.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from ..state import GameState
from .agent import Agent


class RandomAgent(Agent):
    """Agent that selects actions uniformly at random.
    
    Args:
        player_id: The player index this agent controls
        seed: Random seed for reproducibility (critical for research experiments)
    """
    
    def __init__(self, player_id: int = -1, seed: Optional[int] = None):
        super().__init__(player_id)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def select_action(self, state: GameState, legal_actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Select a random legal action.
        
        Args:
            state: Current game state
            legal_actions: List of legal action dicts
            
        Returns:
            A randomly selected action from legal_actions
        """
        if not legal_actions:
            return {'type': 'pass'}
        
        choice_idx = self.rng.integers(0, len(legal_actions))
        return legal_actions[choice_idx]
    
    def reset(self, seed: Optional[int] = None):
        """Reset the agent's RNG for a new episode.
        
        Args:
            seed: New seed, or use original seed if None
        """
        if seed is not None:
            self.seed = seed
        self.rng = np.random.default_rng(self.seed)
