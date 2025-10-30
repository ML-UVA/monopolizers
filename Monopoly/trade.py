from dataclasses import dataclass
from typing import Dict, Tuple
from .state import GameState

@dataclass
class TradeProposal:
    proposer: int
    receiver: int
    offer: Dict  # e.g., {'cash': 200, 'props':[1,3]}
    ask: Dict

    def validate(self, state: GameState) -> bool:
        pass

    def execute(self, state: GameState) -> GameState:
        pass

class TradeManager:
    def propose_trade(self, state: GameState, proposal: TradeProposal) -> Dict:  # returns validation/result
        pass

    def accept_trade(self, state: GameState, proposal: TradeProposal) -> GameState:
        pass

    def decline_trade(self, state: GameState, proposal: TradeProposal) -> GameState:
        pass