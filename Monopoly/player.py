from .state import PlayerState
from .board import Board

class PlayerUtils:
    @staticmethod
    def can_afford(player: PlayerState, amount: int) -> bool:
        pass

    @staticmethod
    def pay(player: PlayerState, amount: int) -> None:  # mutates PlayerState
        pass

    @staticmethod
    def receive(player: PlayerState, amount: int) -> None:
        pass

    @staticmethod
    def move(player: PlayerState, steps: int, board: Board) -> int:  # returns new position
        pass