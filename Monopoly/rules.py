from typing import List, Dict, Tuple, Any
import numpy as np
from enum import Enum
from .state import GameState
from .trade import TradeProposal

class ActionType(Enum):
    ROLL = 'ROLL'
    BUY = 'BUY'
    PASS = 'PASS'
    AUCTION_BID = 'AUCTION_BID'
    BUILD = 'BUILD'
    MORTGAGE = 'MORTGAGE'
    UNMORTGAGE = 'UNMORTGAGE'
    PROPOSE_TRADE = 'PROPOSE_TRADE'
    ACCEPT_TRADE = 'ACCEPT_TRADE'
    DECLINE_TRADE = 'DECLINE_TRADE'
    USE_GET_OUT = 'USE_GET_OUT'
    PAY_FINE = 'PAY_FINE'
    END_TURN = 'END_TURN'
    DECLARE_BANKRUPTCY = 'DECLARE_BANKRUPTCY'

class RulesConfig:
    # settings: variant flags, taxes, house counts, etc.
    pass

class RulesEngine:
    config: RulesConfig

    def legal_actions(self, state: GameState, player_id: int) -> List[Dict] or np.ndarray:
        pass

    def apply_action(self, state: GameState, action: Dict, rng: np.random.Generator) -> Tuple[GameState, float, Dict]:
        pass

    def resolve_move(self, state: GameState, player_id: int, steps: int, rng: np.random.Generator) -> Tuple[GameState, Dict]:
        pass

    def handle_buy(self, state: GameState, player_id: int, property_idx: int) -> Tuple[GameState, Dict]:
        pass

    def handle_rent(self, state: GameState, payer_id: int, owner_id: int, property_idx: int) -> Tuple[GameState, Dict]:
        pass

    def handle_jail(self, state: GameState, player_id: int, action: Dict, rng: np.random.Generator) -> Tuple[GameState, Dict]:
        pass

    def handle_auction(self, state: GameState, property_idx: int, rng: np.random.Generator) -> Tuple[GameState, Dict]:
        pass

    def handle_trade(self, state: GameState, proposal: TradeProposal) -> Tuple[GameState, Dict]:
        pass

    def calculate_net_worth(self, state: GameState, player_id: int) -> int:
        pass