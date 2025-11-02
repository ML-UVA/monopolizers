import pytest
import json
from ..Monopoly.state import GameState, PlayerState, PropertyState, PlayerStatus, DeckState

def test_player_state_creation():
    player = PlayerState(id=0, cash=1500, position=0, properties_owned={1, 2}, houses_on_property={1: 2}, mortgaged_properties={3}, jail_turns=0, get_out_of_jail_cards=0)
    assert player.id == 0
    assert player.cash == 1500
    assert player.properties_owned == {1, 2}
    assert player.status == PlayerStatus.ACTIVE

def test_property_state_creation():
    prop = PropertyState(owner=0, houses_count=1, mortgaged=True)
    assert prop.owner == 0
    assert prop.houses_count == 1
    assert prop.mortgaged == True

def test_game_state_creation():
    players = [PlayerState(id=0, cash=1500, position=0, properties_owned=set(), houses_on_property={}, mortgaged_properties=set(), jail_turns=0, get_out_of_jail_cards=0)]
    properties = [PropertyState(owner=None) for _ in range(28)]
    chance_deck = DeckState(pointer=0, seed=42)
    community_deck = DeckState(pointer=0, seed=42)
    state = GameState(players=players, properties=properties, chance_deck=chance_deck, community_deck=community_deck)
    assert len(state.players) == 1
    assert len(state.properties) == 28
    assert state.current_player == 0

def test_serialization_roundtrip():
    players = [PlayerState(id=0, cash=1500, position=0, properties_owned={1}, houses_on_property={}, mortgaged_properties=set(), jail_turns=0, get_out_of_jail_cards=0)]
    properties = [PropertyState(owner=0 if i == 1 else None) for i in range(28)]
    chance_deck = DeckState(pointer=0, seed=42)
    community_deck = DeckState(pointer=0, seed=42)
    original = GameState(players=players, properties=properties, chance_deck=chance_deck, community_deck=community_deck)
    # Simulate JSON serialization (dataclasses can be serialized with custom encoder)
    data = {
        'players': [{'id': p.id, 'cash': p.cash, 'position': p.position, 'properties_owned': list(p.properties_owned), 'houses_on_property': p.houses_on_property, 'mortgaged_properties': list(p.mortgaged_properties), 'jail_turns': p.jail_turns, 'get_out_of_jail_cards': p.get_out_of_jail_cards, 'status': p.status.value} for p in original.players],
        'properties': [{'owner': p.owner, 'houses_count': p.houses_count, 'mortgaged': p.mortgaged} for p in original.properties],
        'chance_deck': {'pointer': original.chance_deck.pointer, 'seed': original.chance_deck.seed},
        'community_deck': {'pointer': original.community_deck.pointer, 'seed': original.community_deck.seed},
        'current_player': original.current_player,
        'last_roll': original.last_roll,
        'doubles_count': original.doubles_count,
        'turn_number': original.turn_number,
        'seed': original.seed
    }
    json_str = json.dumps(data)
    loaded_data = json.loads(json_str)
    # Reconstruct (simplified)
    loaded_players = [PlayerState(id=p['id'], cash=p['cash'], position=p['position'], properties_owned=set(p['properties_owned']), houses_on_property=p['houses_on_property'], mortgaged_properties=set(p['mortgaged_properties']), jail_turns=p['jail_turns'], get_out_of_jail_cards=p['get_out_of_jail_cards'], status=PlayerStatus(p['status'])) for p in loaded_data['players']]
    loaded_properties = [PropertyState(owner=p['owner'], houses_count=p['houses_count'], mortgaged=p['mortgaged']) for p in loaded_data['properties']]
    loaded_chance = DeckState(pointer=loaded_data['chance_deck']['pointer'], seed=loaded_data['chance_deck']['seed'])
    loaded_community = DeckState(pointer=loaded_data['community_deck']['pointer'], seed=loaded_data['community_deck']['seed'])
    loaded = GameState(players=loaded_players, properties=loaded_properties, chance_deck=loaded_chance, community_deck=loaded_community, current_player=loaded_data['current_player'], last_roll=loaded_data['last_roll'], doubles_count=loaded_data['doubles_count'], turn_number=loaded_data['turn_number'], seed=loaded_data['seed'])
    assert loaded.players[0].cash == original.players[0].cash
    assert loaded.properties[1].owner == original.properties[1].owner

def test_copy_independence():
    original = PlayerState(id=0, cash=1500, position=0, properties_owned={1}, houses_on_property={}, mortgaged_properties=set(), jail_turns=0, get_out_of_jail_cards=0)
    copy = PlayerState(id=original.id, cash=original.cash, position=original.position, properties_owned=original.properties_owned.copy(), houses_on_property=original.houses_on_property.copy(), mortgaged_properties=original.mortgaged_properties.copy(), jail_turns=original.jail_turns, get_out_of_jail_cards=original.get_out_of_jail_cards, status=original.status)
    copy.cash = 1000
    copy.properties_owned.add(2)
    assert original.cash == 1500
    assert 2 not in original.properties_owned