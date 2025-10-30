from typing import Optional, List
import pettingzoo
from .engine import GameEngine

class MonopolyAECEnv(pettingzoo.AECEnv):  # or ParallelEnv
    def __init__(self, game_engine_factory, agents: List[str], ...):
        pass

    def reset(self, seed: Optional[int] = None) -> None:
        pass

    def observe(self, agent: str) -> Any:  # observation
        pass

    def step(self, action) -> None:
        pass

    def agent_iter(self):  # generator of active agents
        pass

    def render(self):
        pass

    def close(self):
        pass

    # legal_actions, etc.