import pytest
from ..Monopoly.auction import run_auction, get_auction_starting_bid
from ..Monopoly.state import GameState, PlayerState, PropertyState

@pytest.fixture
def auction_state():
    players = [
        PlayerState(id=0, cash=500, position=0, properties_owned=set(), houses_on_property={}, mortgaged_properties=set(), jail_turns=0, get_out_of_jail_cards=0),
        PlayerState(id=1, cash=300, position=0, properties_owned=set(), houses_on_property={}, mortgaged_properties=set(), jail_turns=0, get_out_of_jail_cards=0)
    ]
    properties = [PropertyState(owner=None) for _ in range(28)]
    return GameState(players=players, properties=properties, chance_deck=None, community_deck=None)

def test_auction_basic_flow(auction_state):
    new_state = run_auction(auction_state, 0, starting_bid=10)  # Auction Mediterranean
    # Assuming simple logic, player 0 wins with higher cash
    assert new_state.properties[0].owner == 0
    assert new_state.players[0].cash < 500  # Paid something

def test_auction_no_bidders(auction_state):
    auction_state.players[0].cash = 0
    auction_state.players[1].cash = 0
    new_state = run_auction(auction_state, 0)
    assert new_state.properties[0].owner is None  # No one can bid

def test_get_auction_starting_bid():
    bid = get_auction_starting_bid(0)  # Mediterranean price 60
    assert bid == 6  # 60 // 10