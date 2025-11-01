from typing import Optional, Callable, Any, Dict
from .state import GameState
import math
import time
import numpy as np

class MCTSNode:
    def __init__(self, state: GameState, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.value = 0.0

    def is_fully_expanded(self, legal_actions):
        return len(self.children) == len(legal_actions)

    def best_child(self, c=1.4):
        choices_weights = [
            (child.value / child.visits) + c * math.sqrt((2 * math.log(self.visits) / child.visits))
            for child in self.children
        ]
        return self.children[np.argmax(choices_weights)]

class MCTSAgent:
    def __init__(self, rollouts: int, evaluator: Optional[Callable] = None):
        self.rollouts = rollouts
        self.evaluator = evaluator or self._default_evaluator

    def act(self, state: GameState) -> Any:  # action
        root = MCTSNode(state)
        for _ in range(self.rollouts):
            node = self._select(root)
            if not node.state:  # Terminal
                reward = self.evaluator(node.state)
            else:
                node = self._expand(node)
                reward = self._simulate(node)
            self._backpropagate(node, reward)
        return root.best_child(c=0).action

    def set_time_budget(self, seconds):
        self.rollouts = max(10, int(seconds * 100))  # Estimate rollouts

    def _select(self, node):
        while node.is_fully_expanded(self._get_legal_actions(node.state)) and node.children:
            node = node.best_child()
        return node

    def _expand(self, node):
        legal_actions = self._get_legal_actions(node.state)
        for action in legal_actions:
            if action not in [child.action for child in node.children]:
                new_state = self._apply_action(node.state, action)
                child = MCTSNode(new_state, node, action)
                node.children.append(child)
                return child
        return node

    def _simulate(self, node):
        state = node.state
        depth = 0
        while not self._is_terminal(state) and depth < 50:
            action = np.random.choice(self._get_legal_actions(state))
            state = self._apply_action(state, action)
            depth += 1
        return self.evaluator(state)

    def _backpropagate(self, node, reward):
        while node:
            node.visits += 1
            node.value += reward
            node = node.parent

    def _default_evaluator(self, state):
        # Simple: net worth of current player
        return state.players[state.current_player].cash

    def _get_legal_actions(self, state):
        # Placeholder: return list of action indices
        return list(range(10))  # Assume 10 actions

    def _apply_action(self, state, action):
        # Placeholder: return new state
        return state

    def _is_terminal(self, state):
        # Placeholder
        return False