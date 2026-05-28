from .state import PlayerState
from .board import Board

class PlayerUtils:
    @staticmethod
    def can_afford(player: PlayerState, amount: int) -> bool:
        # PlayerState uses `cash` as the balance field
        return player.cash >= amount

    @staticmethod
    def pay(player: PlayerState, amount: int) -> None:  # mutates PlayerState
        player.cash -= amount

    @staticmethod
    def receive(player: PlayerState, amount: int) -> None:
        player.cash += amount

    @staticmethod
    def move(player: PlayerState, steps: int, board: Board) -> int:  # returns new position
        new_position = board.next_tile(player.position, steps)
        player.position = new_position
        return new_position