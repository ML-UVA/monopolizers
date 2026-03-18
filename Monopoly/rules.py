from __future__ import annotations
from typing import List, Dict, Tuple, Any, Optional, TypedDict
from enum import Enum
import random
from dataclasses import dataclass

from .state import GameState, PlayerState, PlayerStatus, PropertyState, DeckState
from .board import Board, TileKind
from .property import load_property_specs, PropertySpec
from .cards import load_chance_cards, load_community_cards, Card
from .trade import TradeProposal

class ActionDict(TypedDict, total=False):
    type: str
    property_idx: int
    cost: int
    value: int
    amount: int
    target_player_id: int

class ActionType(Enum):
    ROLL = 'roll'
    BUY = 'buy'
    PASS = 'pass'
    BUILD = 'build'
    SELL = 'sell'
    MORTGAGE = 'mortgage'
    UNMORTGAGE = 'unmortgage'
    PAY_FINE = 'pay_fine'
    USE_JAIL_CARD = 'use_jail_card'
    END_TURN = 'end_turn'
    AUCTION_BID = 'auction_bid'
    AUCTION_PASS = 'auction_pass'
    INITIATE_TRADE = 'initiate_trade'
    TOGGLE_OFFER_PROPERTY = 'toggle_offer_property'
    TOGGLE_ASK_PROPERTY = 'toggle_ask_property'
    OFFER_CASH = 'offer_cash'
    ASK_CASH = 'ask_cash'
    SELECT_TRADE_TARGET = 'select_trade_target'
    CONFIRM_TRADE = 'confirm_trade'
    ACCEPT_TRADE = 'accept_trade'
    DECLINE_TRADE = 'decline_trade'

@dataclass
class RulesConfig:
    # Tax rules
    income_tax_amount: int = 200
    luxury_tax_amount: int = 100

    # Jail rules
    jail_fine: int = 50
    max_jail_turns: int = 3  # turns before forced to pay fine

    # Building rules
    max_houses_per_property: int = 4
    hotel_requires_houses: bool = True  # must have 4 houses to build hotel
    even_building_rule: bool = True     # must build evenly across monopoly
    enable_houses_and_hotels: bool = True

    # GO rules
    salary_on_go: int = 200

    # Auction rules
    enable_auctions: bool = True
    auction_starting_bid: int = 1

    # Trading rules
    enable_trading: bool = True

    # Free parking rules
    free_parking_jackpot: bool = False  # house rule: taxes go to free parking pot

    # Bank limits
    bank_houses: int = 32
    bank_hotels: int = 12

    # Starting cash
    starting_cash: int = 1500

class RulesEngine:
    def __init__(
            self, 
            board: Board, 
            property_specs: List[PropertySpec], 
            chance_cards: List[Card], 
            community_cards: List[Card],
            config: RulesConfig = None
    ):
        self.config = config or RulesConfig()

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
        if state.last_roll is None:
            actions.append({'type': ActionType.ROLL.value})
            # Jail actions
            if player.jail_turns > 0:
                if player.cash >= self.config.jail_fine:
                    actions.append({'type': ActionType.PAY_FINE.value, 'cost': self.config.jail_fine})
                if player.get_out_of_jail_cards > 0:
                    actions.append({'type': ActionType.USE_JAIL_CARD.value})
            return actions
        
        # Check current position for property transactions
        pos = player.position
        tile = self.board.get_tile(pos)
        
        if tile.property_idx is not None:
            prop = state.properties[tile.property_idx]
            spec = self.property_specs[tile.property_idx]
            
            # Buy action
            if prop.owner is None and player.cash >= spec.price:
                actions.append({'type': ActionType.BUY.value, 'property_idx': tile.property_idx, 'cost': spec.price})

        # Pass action (always available)
        # actions.append({'type': ActionType.PASS.value})

        
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
                        'type': ActionType.BUILD.value,
                        'property_idx': prop_idx,
                        'cost': spec.house_cost
                    })
        
        # Mortgage actions
        for prop_idx in player.properties_owned:
            prop = state.properties[prop_idx]
            spec = self.property_specs[prop_idx]
            
            if not prop.mortgaged and prop.houses_count == 0:
                actions.append({
                    'type': ActionType.MORTGAGE.value,
                    'property_idx': prop_idx,
                    'value': spec.mortgage_value
                })
            # elif prop.mortgaged:
            #     unmortgage_cost = int(spec.mortgage_value * 1.1)
            #     if player.cash >= unmortgage_cost:
            #         actions.append({
            #             'type': ActionType.UNMORTGAGE.value,
            #             'property_idx': prop_idx,
            #             'cost': unmortgage_cost
            #         })
        
        
        
        # End turn action
        actions.append({'type': ActionType.END_TURN.value})
        
        return actions

    def apply_action(self, state: GameState, action: ActionDict, rng: random.Random, engine=None) -> Tuple[GameState, float, bool, str]:
        try:
            action_type = ActionType(action.get('type'))
        except ValueError:
            return state, 0.0, False, f"Unknown action type: {action.get('type')}"

        player_id = state.current_player
        reward = 0.0
        done = False
        log = f"Player {player_id} performed {action_type.value}"
        
        if action_type == ActionType.ROLL:
            roll = self.roll_dice(rng)
            state.last_roll = roll
            state = self.handle_doubles_and_jail(state, player_id, roll)
            if state.players[player_id].jail_turns == 0:
                steps = sum(roll)
                state = self.move_player(state, player_id, steps)
                state = self.handle_landing(state, player_id, rng, engine=engine)

        elif action_type == ActionType.BUY:
            prop_idx = action.get('property_idx')
            if prop_idx is None:
                return state, 0.0, False, "Missing property_idx"
            
            if self.buy_property(state, player_id, prop_idx):
                reward = 0  # Small reward for buying

        elif action_type == ActionType.BUILD:
            prop_idx = action.get('property_idx')
            if prop_idx is None:
                return state, 0.0, False, "Missing property_idx"
            
            player = state.players[player_id]
            spec = self.property_specs[prop_idx]
            prop = state.properties[prop_idx]

            if (prop.owner == player_id and
                self._has_monopoly(state, prop_idx, player_id) and
                prop.houses_count < 5 and
                player.cash >= spec.house_cost and 
                not prop.mortgaged and 
                state.bank_houses_left > 0):
                prop.houses_count += 1
                if prop.houses_count == 5:
                    state.bank_hotels_left -= 1
                    state.bank_houses_left += 4 # return 4 houses to bank
                else:
                    state.bank_houses_left -= 1

        elif action_type == ActionType.SELL:
            prop_idx = action.get('property_idx')
            if prop_idx is None:
                return state, 0.0, False, "Missing property_idx"
            
            player = state.players[player_id]
            spec = self.property_specs[prop_idx]
            prop = state.properties[prop_idx]

            if prop.owner == player_id and prop.houses_count > 0:
                sell_price = spec.house_cost // 2
                if prop.houses_count == 5:
                    # Selling hotel - return hotel, give back 4 houses
                    state.bank_hotels_left += 1
                    state.bank_houses_left -= 4
                    prop.houses_count = 4
                else:
                    prop.houses_count -= 1
                    state.bank_houses_left += 1
                player.cash += sell_price
        
        elif action_type == ActionType.MORTGAGE:
            prop_idx = action.get('property_idx')
            if prop_idx is None:
                return state, 0.0, False, "Missing property_idx"
            
            player = state.players[player_id]
            spec = self.property_specs[prop_idx]
            prop = state.properties[prop_idx]

            if (prop.owner == player_id and
                not prop.mortgaged and 
                prop.houses_count == 0):
                prop.mortgaged = True
                player.cash += spec.mortgage_value
        
        elif action_type == ActionType.UNMORTGAGE:
            prop_idx = action.get('property_idx')
            if prop_idx is None:
                return state, 0.0, False, "Missing property_idx"
            
            player = state.players[player_id]
            spec = self.property_specs[prop_idx]
            prop = state.properties[prop_idx]
            unmortgage_cost = int(spec.mortgage_value * 1.1)

            if (prop.owner == player_id and
                prop.mortgaged and
                player.cash >= unmortgage_cost):
                prop.mortgaged = False
                player.cash -= unmortgage_cost

        elif action_type == ActionType.PAY_FINE:
            player = state.players[player_id]

            if player.jail_turns > 0 and player.cash >= self.config.jail_fine:
                player.cash -= self.config.jail_fine
                player.jail_turns = 0
        
        elif action_type == ActionType.USE_JAIL_CARD:
            player = state.players[player_id]
            if player.get_out_of_jail_cards > 0:
                player.get_out_of_jail_cards -= 1
                player.jail_turns = 0

        elif action_type == ActionType.END_TURN:
            state.current_player = (state.current_player + 1) % len(state.players)
            state.turn_number += 1
            state.last_roll = None

        # ----- Auction actions -----

        elif action_type == ActionType.AUCTION_BID:
            amount = action.get('amount', 0)
            if (state.auction_active and
                state.players[player_id].cash >= amount and
                amount > state.auction_current_bid):
                state.auction_current_bid = amount
                state.auction_highest_bidder = player_id
        
        elif action_type == ActionType.AUCTION_PASS:
            # Handled externally by auction loop, single pass here just records intent
            pass

        # ----- Trade actions -----

        elif action_type == ActionType.INITIATE_TRADE:
            state.trade_phase = True
            state.trade_offer_properties = set()
            state.trade_ask_properties = set()
            state.trade_offer_cash = 0
            state.trade_ask_cash = 0
            state.trade_target = None
        
        elif action_type == ActionType.TOGGLE_OFFER_PROPERTY:
            prop_idx = action.get('property_idx')
            if prop_idx is None:
                return state, 0.0, False, "Missing property_idx"
            
            if prop_idx in state.trade_offer_properties:
                state.trade_offer_properties.remove(prop_idx)
            else:
                state.trade_offer_properties.add(prop_idx)
        
        elif action_type == ActionType.TOGGLE_ASK_PROPERTY:
            prop_idx = action.get('property_idx')
            if prop_idx is None:
                return state, 0.0, False, "Missing property_idx"

            if prop_idx in state.trade_ask_properties:
                state.trade_ask_properties.remove(prop_idx)
            else:
                state.trade_ask_properties.add(prop_idx)
        
        elif action_type == ActionType.OFFER_CASH:
            state.trade_offer_cash = action.get('amount', 0)

        elif action_type == ActionType.ASK_CASH:
            state.trade_ask_cash = action.get('amount', 0)
        
        elif action_type == ActionType.SELECT_TRADE_TARGET:
            target = action.get('target_player_id')
            if target is None:
                return state, 0.0, False, "Missing target_player_id"
            active_opponents = [
                p.id for p in state.players
                if p.id != player_id and p.status == PlayerStatus.ACTIVE
            ]
            if target in active_opponents:
                state.trade_target = target
                
        elif action_type == ActionType.CONFIRM_TRADE:
            if state.trade_target is not None:
                proposal = TradeProposal(
                    proposer=player_id,
                    receiver=state.trade_target,
                    offer={
                        'cash': state.trade_offer_cash,
                        'properties': list(state.trade_offer_properties)
                    },
                    ask={
                        'cash': state.trade_ask_cash,
                        'properties': list(state.trade_ask_properties)
                    }
                )
                if proposal.validate(state):
                    state.pending_trade = proposal
                # Reset trade construction state
                state.trade_phase = False
                state.trade_offer_properties = set()
                state.trade_ask_properties = set()
                state.trade_offer_cash = 0
                state.trade_ask_cash = 0
                state.trade_target = None
        
        elif action_type == ActionType.ACCEPT_TRADE:
            if state.pending_trade is not None:
                if state.pending_trade.validate(state):
                    traded_props = (
                        state.pending_trade.offer.get('properties', []) + 
                        state.pending_trade.ask.get('properties', [])
                    )
                    state = state.pending_trade.execute(state)
                    # Update monopoly status for all traded properties
                    for prop_idx in traded_props:
                        self._update_monopoly_status(state, prop_idx)
                state.pending_trade = None
        
        elif action_type == ActionType.DECLINE_TRADE:
            state.pending_trade = None
        
        elif action_type == ActionType.PASS:
            pass

        # Check for game end
        active_players = [p for p in state.players if p.status == PlayerStatus.ACTIVE]
        if len(active_players) <= 1:
            done = True
            if active_players:
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
            # pos 4 = Income Tax ($200), pos 38 = Luxury Tax ($100)
            tax = self.config.income_tax_amount if pos == 4 else self.config.luxury_tax_amount
            state.players[player_id].cash -= tax
        elif tile.kind == TileKind.CHANCE:
            # TODO: Include later in further training
            pass
            # card = self.draw_card(state, "chance", rng)
            # # Card.effect expects (state, engine, rng, player_id)
            # state, _ = card.effect(state, engine, rng, player_id)
        elif tile.kind == TileKind.COMMUNITY:
            # TODO: Include later in further training
            pass
            # card = self.draw_card(state, "community", rng)
            # state, _ = card.effect(state, engine, rng, player_id)
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
            player.cash += self.config.salary_on_go
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

    def _update_monopoly_status(self, state: GameState, property_idx: int):
        group = self.property_specs[property_idx].group
        group_props = [i for i, p in enumerate(self.property_specs) if p.group == group]
        owners = {state.properties[i].owner for i in group_props}
        if len(owners) == 1:
            owner = owners.pop()
            state.monopoly_status[group] = owner
        else:
            state.monopoly_status[group] = None

    def _has_monopoly(self, state: GameState, property_idx: int, owner: int) -> bool:
        group = self.property_specs[property_idx].group
        if group in state.monopoly_status:
            return state.monopoly_status[group] == owner
        
        # Fallback / Initialization
        group_props = [i for i, p in enumerate(self.property_specs) if p.group == group]
        is_monopoly = all(state.properties[i].owner == owner for i in group_props)
        if is_monopoly:
            state.monopoly_status[group] = owner
        else:
            # If we checked and it's not a monopoly, we can't easily say who owns it without checking all
            # But we know 'owner' doesn't have it.
            # Let's just run the update logic to be sure
            self._update_monopoly_status(state, property_idx)
            
        return is_monopoly

    def buy_property(self, state: GameState, player_id: int, property_idx: int) -> bool:
        player = state.players[player_id]
        spec = self.property_specs[property_idx]
        if player.cash >= spec.price and state.properties[property_idx].owner is None:
            player.cash -= spec.price
            player.properties_owned.add(property_idx)
            state.properties[property_idx].owner = player_id
            self._update_monopoly_status(state, property_idx)
            return True
        return False

    def handle_bankruptcy(self, state: GameState, player_id: int) -> GameState:
        player = state.players[player_id]
        # Transfer properties to bank or creditor if applicable
        # For now, just clear ownership (bankrupt to bank)
        properties_to_clear = list(player.properties_owned)
        for prop_idx in properties_to_clear:
            state.properties[prop_idx] = PropertyState(owner=None, houses_count=0, mortgaged=False)
            self._update_monopoly_status(state, prop_idx)
        player.properties_owned.clear()
        player.status = PlayerStatus.BANKRUPT
        return state

    def roll_dice(self, rng: random.Random) -> Tuple[int, int]:
        return rng.randint(1, 6), rng.randint(1, 6)

    def handle_doubles_and_jail(self, state: GameState, player_id: int, roll: Tuple[int, int]) -> GameState:
        d1, d2 = roll
        player = state.players[player_id]

        if player.jail_turns > 0:
            # Player is in jail
            if d1 == d2:
                # Escape jail via doubles
                player.jail_turns = 0
                state.doubles_count = 0  # no bonus roll for escaping jail
            else:
                # Failed to roll doubles
                player.jail_turns += 1
                if player.jail_turns > self.config.max_jail_turns:
                    player.cash -= self.config.jail_fine
                    player.jail_turns = 0
        else:
            # Normal doubles/triples handling
            if d1 == d2:
                state.doubles_count += 1
                if state.doubles_count == 3:
                    # Go to jail
                    player.position = 10  # Jail position
                    player.jail_turns = 1
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
