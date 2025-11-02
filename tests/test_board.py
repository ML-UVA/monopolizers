import pytest
from ..Monopoly.board import Board, TileKind

def test_board_load_standard():
    board = Board.load_standard_board()
    assert board.board_size == 40
    assert board.get_tile(0).kind == TileKind.GO
    assert board.get_tile(10).kind == TileKind.JAIL

def test_next_tile():
    board = Board.load_standard_board()
    assert board.next_tile(0, 5) == 5
    assert board.next_tile(35, 10) == 5  # Wrap around

def test_property_index_for_board_idx():
    board = Board.load_standard_board()
    assert board.property_index_for_board_idx(1) == 0  # Mediterranean
    assert board.property_index_for_board_idx(0) is None  # GO
