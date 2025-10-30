from dataclasses import dataclass
from typing import Callable, Tuple, Dict, List
import numpy as np

# Forward references
class GameState:
    pass

class GameEngine:
    pass

@dataclass
class Card:
    id: str
    text: str
    effect: Callable[[GameState, GameEngine, np.random.Generator, int], Tuple[GameState, Dict]]  # effect function

class CardDeck:
    cards: List[Card]
    pointer: int

    def shuffle(self, rng: np.random.Generator) -> None:
        pass

    def draw(self) -> Card:
        pass

    def reset(self, rng: np.random.Generator) -> None:
        pass

    def apply_card(self, card: Card, state: GameState, engine: GameEngine, rng: np.random.Generator) -> Tuple[GameState, Dict]:
        pass