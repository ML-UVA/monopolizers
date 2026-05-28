"""Base agent class for Monopoly AI players."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from ..state import GameState


class Agent(ABC):
    """Abstract base class for Monopoly agents.
    
    All agent implementations should inherit from this class and implement
    the select_action method.
    """
    
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
    
    def reset(self, seed: Optional[int] = None):
        """Reset agent state for a new episode.
        
        Override this in subclasses that maintain internal state (e.g., RNG).
        
        Args:
            seed: Optional seed for reproducibility
        """
        pass  # Default is no-op
