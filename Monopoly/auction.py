from typing import List, Dict, Tuple
import numpy as np
from .state import GameState

class Auction:
    property_idx: int
    participants: List[int]
    current_bids: Dict[int, int]  # player -> bid

    def start_auction(self, state: GameState, starting_bid: int, rng: np.random.Generator) -> None:
        pass

    def place_bid(self, player_id: int, amount: int) -> bool:
        pass

    def resolve(self, state: GameState) -> Tuple[GameState, int]:  # returns updated state and winner id
        pass

    def abort(self) -> None:
        pass