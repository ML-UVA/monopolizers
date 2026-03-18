from typing import List, Dict, Any
from ..state import GameState
from .agent import Agent
from ..rules import ActionType

class GreedyAgent(Agent):
    """Agent that uses simple heuristics to maximize assets."""
    
    def __init__(self, player_id: int = -1, safety_margin: int = 50):
        super().__init__(player_id)
        self.safety_margin = safety_margin

    def select_action(self, state: GameState, legal_actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not legal_actions:
            return {'type': ActionType.PASS.value}

        player = state.players[self.player_id]
        
        # 1. Always buy if affordable and safe
        buy_action = next((a for a in legal_actions if a['type'] == ActionType.BUY.value), None)
        if buy_action:
            cost = buy_action.get('cost', 0)
            if player.cash >= cost + self.safety_margin:
                return buy_action

        # 2. Build houses if affordable and safe
        build_action = next((a for a in legal_actions if a['type'] == ActionType.BUILD.value), None)
        if build_action:
            cost = build_action.get('cost', 0)
            if player.cash >= cost + self.safety_margin:
                return build_action
        
        # 3. Unmortgage if very rich
        unmortgage_action = next((a for a in legal_actions if a['type'] == ActionType.UNMORTGAGE.value), None)
        if unmortgage_action:
            cost = unmortgage_action.get('cost', 0)
            if player.cash >= cost + 500:  # High buffer for unmortgaging
                return unmortgage_action

        # 4. Get out of jail
        if player.jail_turns > 0:
            # Use card if available
            use_card = next((a for a in legal_actions if a['type'] == ActionType.USE_JAIL_CARD.value), None)
            if use_card:
                return use_card
            # Pay fine if rich
            pay_fine = next((a for a in legal_actions if a['type'] == ActionType.PAY_FINE.value), None)
            if pay_fine and player.cash >= 100:
                return pay_fine

        # 5. Default: Roll or Pass or End Turn
        # Prefer Roll > End Turn > Pass
        for type_pref in [ActionType.ROLL.value, ActionType.END_TURN.value, ActionType.PASS.value]:
            action = next((a for a in legal_actions if a['type'] == type_pref), None)
            if action:
                return action
        
        # Fallback
        return legal_actions[0]
