import pytest
from ..Monopoly.player import PlayerUtils
from ..Monopoly.state import PlayerState
from ..Monopoly.board import Board

def test_can_afford():
    player = PlayerState(id=0, cash=100, position=0, properties_owned=set(), houses_on_property={}, mortgaged_properties=set(), jail_turns=0, get_out_of_jail_cards=0)
    assert PlayerUtils.can_afford(player, 50) == True
    assert PlayerUtils.can_afford(player, 150) == False

def test_pay():
    player = PlayerState(id=0, cash=100, position=0, properties_owned=set(), houses_on_property={}, mortgaged_properties=set(), jail_turns=0, get_out_of_jail_cards=0)
    PlayerUtils.pay(player, 30)
    assert player.cash == 70

def test_receive():
    player = PlayerState(id=0, cash=100, position=0, properties_owned=set(), houses_on_property={}, mortgaged_properties=set(), jail_turns=0, get_out_of_jail_cards=0)
    PlayerUtils.receive(player, 50)
    assert player.cash == 150

def test_move():
    board = Board.load_standard_board()
    player = PlayerState(id=0, cash=100, position=0, properties_owned=set(), houses_on_property={}, mortgaged_properties=set(), jail_turns=0, get_out_of_jail_cards=0)
    new_pos = PlayerUtils.move(player, 7, board)
    assert new_pos == 7
    assert player.position == 7
