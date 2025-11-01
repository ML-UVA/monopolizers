import numpy as np
from typing import Any

class RandomAgent:
    def __init__(self):
        self.rng = np.random.default_rng()

    def act(self, observation, legal_mask) -> Any:  # action index
        legal_indices = np.where(legal_mask)[0]
        if len(legal_indices) > 0:
            return self.rng.choice(legal_indices)
        return 0  # Default if no legal actions

    def reset(self):
        pass