from abc import ABC, abstractmethod
from typing import List, Dict, Any
from ..state import GameState

class Agent(ABC):
    """Abstract base class for Monopoly agents."""
    
    def __init__(self, player_id: int = -1):
        self.player_id = player_id

    @abstractmethod
    def select_action(self, state: GameState, legal_actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Select an action from the list of legal actions.
        
        Args:
            state: The current game state.
            legal_actions: A list of legal action dictionaries.
            
        Returns:
            The selected action dictionary.
        """
        pass
