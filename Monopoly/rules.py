from typing import List, Dict, Tuple, Any
import numpy as np
from enum import Enum
from .state import GameState
from .trade import TradeProposal
import random
from typing import Optional, Tuple, List
from .state import GameState, PlayerState, PlayerStatus, PropertyState, DeckState
from .board import Board, TileKind
from .property import load_property_specs, PropertySpec
from .cards import load_chance_cards, load_community_cards, Card

class ActionType(Enum):
    ROLL = 'ROLL'
    BUY = 'BUY'
    PASS = 'PASS'
    AUCTION_BID = 'AUCTION_BID'
    BUILD = 'BUILD'
    MORTGAGE = 'MORTGAGE'
    UNMORTGAGE = 'UNMORTGAGE'
    PROPOSE_TRADE = 'PROPOSE_TRADE'
    ACCEPT_TRADE = 'ACCEPT_TRADE'
    DECLINE_TRADE = 'DECLINE_TRADE'
    USE_GET_OUT = 'USE_GET_OUT'
    PAY_FINE = 'PAY_FINE'
    END_TURN = 'END_TURN'
    DECLARE_BANKRUPTCY = 'DECLARE_BANKRUPTCY'

class RulesConfig:
    # settings: variant flags, taxes, house counts, etc.
    pass

class RulesEngine:
    config: RulesConfig

    def __init__(self, board: Board, property_specs: List[PropertySpec], chance_cards: List[Card], community_cards: List[Card]):
        self.board = board
        self.property_specs = property_specs
        self.chance_cards = chance_cards
        self.community_cards = community_cards

    def legal_actions(self, state: GameState, player_id: int) -> List[Dict]:
        """
        Return list of legal action dicts for the given player.
        
        Returns:
            List of action dicts with 'type' and optional parameters
        """
        actions = []
        player = state.players[player_id]
        
        # Roll action (if turn hasn't started yet)
        if state.last_roll is None or state.current_player == player_id:
            actions.append({'type': 'roll'})
        
        # Check current position for property transactions
        pos = player.position
        tile = self.board.get_tile(pos)
        
        if tile.property_idx is not None:
            prop = state.properties[tile.property_idx]
            spec = self.property_specs[tile.property_idx]
            
            # Buy action
            if prop.owner is None and player.cash >= spec.price:
                actions.append({'type': 'buy', 'property_idx': tile.property_idx, 'cost': spec.price})
        
        # Pass action (always available)
        actions.append({'type': 'pass'})
        
        # Build house actions (on monopolies)
        for prop_idx in player.properties_owned:
            if self._has_monopoly(state, prop_idx, player_id):
                spec = self.property_specs[prop_idx]
                prop = state.properties[prop_idx]
                if (prop.houses_count < 5 and 
                    player.cash >= spec.house_cost and 
                    not prop.mortgaged and
                    state.bank_houses_left > 0):
                    actions.append({
                        'type': 'build',
                        'property_idx': prop_idx,
                        'cost': spec.house_cost
                    })
        
        # Mortgage actions
        for prop_idx in player.properties_owned:
            prop = state.properties[prop_idx]
            spec = self.property_specs[prop_idx]
            
            if not prop.mortgaged and prop.houses_count == 0:
                actions.append({
                    'type': 'mortgage',
                    'property_idx': prop_idx,
                    'value': spec.mortgage_value
                })
            elif prop.mortgaged:
                unmortgage_cost = int(spec.mortgage_value * 1.1)
                if player.cash >= unmortgage_cost:
                    actions.append({
                        'type': 'unmortgage',
                        'property_idx': prop_idx,
                        'cost': unmortgage_cost
                    })
        
        # Jail actions
        if player.jail_turns > 0:
            if player.cash >= 50:
                actions.append({'type': 'pay_fine', 'cost': 50})
            if player.get_out_of_jail_cards > 0:
                actions.append({'type': 'use_jail_card'})
        
        # End turn action
        actions.append({'type': 'end_turn'})
        
        return actions

    def apply_action(self, state: GameState, action: dict, rng: random.Random) -> Tuple[GameState, float, bool, str]:
        action_type = action.get('type')
        player_id = state.current_player
        reward = 0.0
        done = False
        log = f"Player {player_id} performed {action_type}"
        
        if action_type == 'roll':
            roll = self.roll_dice(rng)
            state.last_roll = roll
            state = self.handle_doubles_and_jail(state, player_id, roll)
            if state.players[player_id].jail_turns == 0:
                steps = sum(roll)
                state = self.move_player(state, player_id, steps)
                state = self.handle_landing(state, player_id, rng)
        elif action_type == 'buy':
            prop_idx = action.get('property_idx')
            if self.buy_property(state, player_id, prop_idx):
                reward = 10  # Small reward for buying
        elif action_type == 'pay_rent':
            # Assume triggered on landing
            pass
        elif action_type == 'end_turn':
            state.current_player = (state.current_player + 1) % len(state.players)
            state.turn_number += 1
        elif action_type == 'use_jail_card':
            if state.players[player_id].get_out_of_jail_cards > 0:
                state.players[player_id].get_out_of_jail_cards -= 1
                state.players[player_id].jail_turns = 0
        # Add more actions as needed
        
        # Check for game end
        active_players = [p for p in state.players if p.status == PlayerStatus.ACTIVE]
        if len(active_players) <= 1:
            done = True
            reward = 100 if active_players[0].id == player_id else -100
        
        return state, reward, done, log

    def handle_landing(self, state: GameState, player_id: int, rng: random.Random, engine=None) -> GameState:
        pos = state.players[player_id].position
        tile = self.board.get_tile(pos)
        if tile.kind == TileKind.PROPERTY or tile.kind == TileKind.RAILROAD or tile.kind == TileKind.UTILITY:
            prop_idx = tile.property_idx
            if prop_idx is not None:
                owner = state.properties[prop_idx].owner
                if owner is not None and owner != player_id:
                    rent = self.calculate_rent(state, prop_idx, sum(state.last_roll) if tile.kind == TileKind.UTILITY else None)
                    state.players[player_id].cash -= rent
                    state.players[owner].cash += rent
                    if state.players[player_id].cash < 0:
                        state = self.handle_bankruptcy(state, player_id)
        elif tile.kind == TileKind.TAX:
            tax = 200 if pos == 4 else 100  # Income or Luxury
            state.players[player_id].cash -= tax
        elif tile.kind == TileKind.CHANCE:
            card = self.draw_card(state, "chance", rng)
            # Card.effect expects (state, engine, rng, player_id)
            state, _ = card.effect(state, engine, rng, player_id)
        elif tile.kind == TileKind.COMMUNITY:
            card = self.draw_card(state, "community", rng)
            state, _ = card.effect(state, engine, rng, player_id)
        elif tile.kind == TileKind.GO_TO_JAIL:
            state.players[player_id].position = 10
            state.players[player_id].jail_turns = 1
        return state

    def move_player(self, state: GameState, player_id: int, steps: int) -> GameState:
        player = state.players[player_id]
        old_pos = player.position
        new_pos = self.board.next_tile(old_pos, steps)
        player.position = new_pos
        # Pass GO: +200 if passed GO
        if new_pos < old_pos:
            player.cash += 200
        return state

    def calculate_rent(self, state: GameState, property_idx: int, dice_roll: Optional[int] = None) -> int:
        prop_state = state.properties[property_idx]
        spec = self.property_specs[property_idx]
        owner = prop_state.owner
        if owner is None or prop_state.mortgaged:
            return 0
        monopoly = self._has_monopoly(state, property_idx, owner)
        houses = prop_state.houses_count
        if spec.group in ["Railroad", "Utility"]:
            # Count owned properties by index to avoid dataclass equality collisions
            owned_count = sum(1 for i, prop in enumerate(state.properties) if prop.owner == owner and self.property_specs[i].group == spec.group)
            if spec.group == "Railroad":
                return spec.rent_table[owned_count - 1] if 1 <= owned_count <= 4 else 0
            elif spec.group == "Utility":
                multiplier = 10 if owned_count == 2 else 4
                return multiplier * dice_roll if dice_roll else 0
        else:
            return spec.rent_for(houses, monopoly, dice_roll)

    def _has_monopoly(self, state: GameState, property_idx: int, owner: int) -> bool:
        group = self.property_specs[property_idx].group
        group_props = [i for i, p in enumerate(self.property_specs) if p.group == group]
        return all(state.properties[i].owner == owner for i in group_props)

    def buy_property(self, state: GameState, player_id: int, property_idx: int) -> bool:
        player = state.players[player_id]
        spec = self.property_specs[property_idx]
        if player.cash >= spec.price and state.properties[property_idx].owner is None:
            player.cash -= spec.price
            player.properties_owned.add(property_idx)
            state.properties[property_idx].owner = player_id
            return True
        return False

    def handle_bankruptcy(self, state: GameState, player_id: int) -> GameState:
        player = state.players[player_id]
        # Transfer properties to bank or creditor if applicable
        for prop_idx in list(player.properties_owned):
            state.properties[prop_idx] = PropertyState(owner=None, houses_count=0, mortgaged=False)
        player.properties_owned.clear()
        player.status = PlayerStatus.BANKRUPT
        return state

    def roll_dice(self, rng: random.Random) -> Tuple[int, int]:
        return rng.randint(1, 6), rng.randint(1, 6)

    def handle_doubles_and_jail(self, state: GameState, player_id: int, roll: Tuple[int, int]) -> GameState:
        d1, d2 = roll
        if d1 == d2:
            state.doubles_count += 1
            if state.doubles_count == 3:
                # Go to jail
                state.players[player_id].position = 10  # Jail position
                state.players[player_id].jail_turns = 1
                state.doubles_count = 0
        else:
            state.doubles_count = 0
        return state

    def draw_card(self, state: GameState, deck: str, rng: random.Random) -> Card:
        # Ensure deck state exists on the GameState (tests/engines sometimes leave it None)
        if deck == "chance":
            if getattr(state, 'chance_deck', None) is None:
                state.chance_deck = DeckState(pointer=0, seed=state.seed)
            card = self.chance_cards[state.chance_deck.pointer]
            state.chance_deck.pointer = (state.chance_deck.pointer + 1) % len(self.chance_cards)
        elif deck == "community":
            if getattr(state, 'community_deck', None) is None:
                state.community_deck = DeckState(pointer=0, seed=state.seed)
            card = self.community_cards[state.community_deck.pointer]
            state.community_deck.pointer = (state.community_deck.pointer + 1) % len(self.community_cards)
        return card

    def apply_card_effect(self, state: GameState, player_id: int, card: Card, engine=None, rng: random.Random = None) -> GameState:
        # Backwards-compat wrapper: call the card's effect function which uses (state, engine, rng, player_id)
        return card.effect(state, engine, rng, player_id)