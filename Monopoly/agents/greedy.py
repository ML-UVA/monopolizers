from typing import Dict, Any

class GreedyAgent:
    def __init__(self, safety_margin: int = 50):
        self.safety_margin = safety_margin

    def act(self, observation: Dict) -> Dict:  # returns an Action dict
        legal_actions = observation.get('legal_actions', [])
        best_action = None
        best_value = -float('inf')
        for action in legal_actions:
            value = self._evaluate_action(action, observation)
            if value > best_value:
                best_value = value
                best_action = action
        return best_action or {}

    def reset(self):  # optional
        pass

    def _evaluate_action(self, action: Dict, observation: Dict) -> float:
        # Heuristic: prefer actions that increase cash or properties
        if action.get('type') == 'buy':
            cost = action.get('cost', 0)
            if observation['player_cash'] > cost + self.safety_margin:
                return 100  # High value for buying
        elif action.get('type') == 'collect_rent':
            return 50
        return 0