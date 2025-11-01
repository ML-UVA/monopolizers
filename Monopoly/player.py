from .state import PlayerState
from .board import Board

class PlayerUtils:
    @staticmethod
    def can_afford(player: PlayerState, amount: int) -> bool:
        return player.money >= amount

    @staticmethod
    def pay(player: PlayerState, amount: int) -> None:  # mutates PlayerState
        player.money -= amount

    @staticmethod
    def receive(player: PlayerState, amount: int) -> None:
        player.money += amount

    @staticmethod
    def move(player: PlayerState, steps: int, board: Board) -> int:  # returns new position
        new_position = board.next_tile(player.position, steps)
        player.position = new_position
        return new_position