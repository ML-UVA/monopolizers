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
        
        Turn flow:
        1. Player must roll first (unless in jail with options)
        2. After rolling and landing, player may buy/pass if on unowned property
        3. Player may build/mortgage/unmortgage anytime during their turn (after rolling)
        4. Player must end turn when done
        
        Returns:
            List of action dicts with 'type' and optional parameters
        """
        actions = []
        player = state.players[player_id]
        
        # If player is bankrupt, no actions
        if player.status == PlayerStatus.BANKRUPT:
            return []
        
        # Check if it's this player's turn
        if state.current_player != player_id:
            return []
        
        # Handle jail situation
        if player.jail_turns > 0:
            # In jail - special actions available
            if not state.has_rolled:
                # Can try to roll doubles
                actions.append({'type': 'roll'})
                # Can pay $50 fine to get out and then roll
                if player.cash >= 50:
                    actions.append({'type': 'pay_fine', 'cost': 50})
                # Can use Get Out of Jail Free card and then roll
                if player.get_out_of_jail_cards > 0:
                    actions.append({'type': 'use_jail_card'})
            else:
                # Already rolled (and failed to get doubles), must end turn
                actions.append({'type': 'end_turn'})
            return actions
        
        # Normal turn flow
        if not state.has_rolled:
            # Must roll first
            actions.append({'type': 'roll'})
            return actions
        
        # After rolling - check if awaiting buy decision
        if state.awaiting_buy_decision:
            pos = player.position
            tile = self.board.get_tile(pos)
            if tile.property_idx is not None:
                prop = state.properties[tile.property_idx]
                spec = self.property_specs[tile.property_idx]
                if prop.owner is None and player.cash >= spec.price:
                    actions.append({'type': 'buy', 'property_idx': tile.property_idx, 'cost': spec.price})
            actions.append({'type': 'pass'})  # Decline to buy
            return actions
        
        # Post-roll phase - can do building/mortgage/etc and must eventually end turn
        
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
        
        # End turn action (always available after rolling)
        actions.append({'type': 'end_turn'})
        
        return actions

    def apply_action(self, state: GameState, action: dict, rng: random.Random, engine=None) -> Tuple[GameState, float, bool, str]:
        """
        Apply an action to the game state.
        
        Args:
            state: Current game state
            action: Action dict with 'type' and optional parameters
            rng: Random number generator
            engine: Optional GameEngine reference for card effects
            
        Returns:
            Tuple of (new_state, reward, done, log_message)
        """
        action_type = action.get('type')
        player_id = state.current_player
        reward = 0.0
        done = False
        log = f"Player {player_id} performed {action_type}"
        player = state.players[player_id]
        
        if action_type == 'roll':
            roll = self.roll_dice(rng)
            state.last_roll = roll
            state.has_rolled = True
            
            # Handle jail roll
            if player.jail_turns > 0:
                d1, d2 = roll
                if d1 == d2:
                    # Rolled doubles - get out of jail and move
                    player.jail_turns = 0
                    steps = sum(roll)
                    state = self.move_player(state, player_id, steps)
                    state = self.handle_landing(state, player_id, rng, engine)
                    log = f"Player {player_id} rolled doubles {roll}, got out of jail and moved"
                else:
                    # Failed to roll doubles
                    player.jail_turns += 1
                    if player.jail_turns >= 4:
                        # Must pay fine after 3 failed attempts, then move with this roll
                        player.cash -= 50
                        player.jail_turns = 0
                        steps = sum(roll)
                        state = self.move_player(state, player_id, steps)
                        state = self.handle_landing(state, player_id, rng, engine)
                        log = f"Player {player_id} paid $50 (3 failed attempts), rolled {roll} and moved"
                    else:
                        log = f"Player {player_id} rolled {roll} (in jail, attempt {player.jail_turns})"
            else:
                # Normal roll
                state = self.handle_doubles_and_jail(state, player_id, roll)
                if player.jail_turns == 0:  # Didn't get sent to jail for speeding
                    steps = sum(roll)
                    state = self.move_player(state, player_id, steps)
                    state = self.handle_landing(state, player_id, rng, engine)
                log = f"Player {player_id} rolled {roll}, moved to {player.position}"
                
        elif action_type == 'buy':
            prop_idx = action.get('property_idx')
            if self.buy_property(state, player_id, prop_idx):
                reward = 10  # Small reward for buying
                state.awaiting_buy_decision = False
            log = f"Player {player_id} bought property {prop_idx}"
            
        elif action_type == 'pass':
            # Pass on buying
            state.awaiting_buy_decision = False
            log = f"Player {player_id} passed on buying"
            
        elif action_type == 'end_turn':
            # Reset turn state and advance to next player
            state.has_rolled = False
            state.awaiting_buy_decision = False
            state.last_roll = None
            state.doubles_count = 0
            
            # Find next active player
            next_player = (state.current_player + 1) % len(state.players)
            safety = 0
            while state.players[next_player].status == PlayerStatus.BANKRUPT and safety < len(state.players):
                next_player = (next_player + 1) % len(state.players)
                safety += 1
            
            state.current_player = next_player
            state.turn_number += 1
            log = f"Player {player_id} ended turn, now player {next_player}'s turn"
            
        elif action_type == 'use_jail_card':
            if player.get_out_of_jail_cards > 0:
                player.get_out_of_jail_cards -= 1
                player.jail_turns = 0
                # Don't set has_rolled - player still needs to roll this turn
                log = f"Player {player_id} used Get Out of Jail Free card"
                
        elif action_type == 'pay_fine':
            if player.cash >= 50 and player.jail_turns > 0:
                player.cash -= 50
                player.jail_turns = 0
                # Don't set has_rolled - player still needs to roll this turn
                log = f"Player {player_id} paid $50 to get out of jail"
                
        elif action_type == 'build':
            prop_idx = action.get('property_idx')
            if prop_idx is not None and prop_idx in player.properties_owned:
                spec = self.property_specs[prop_idx]
                prop = state.properties[prop_idx]
                if (player.cash >= spec.house_cost and 
                    prop.houses_count < 5 and 
                    not prop.mortgaged and
                    self._has_monopoly(state, prop_idx, player_id)):
                    player.cash -= spec.house_cost
                    prop.houses_count += 1
                    if prop.houses_count == 5:
                        # Converting to hotel
                        state.bank_houses_left += 4
                        state.bank_hotels_left -= 1
                    else:
                        state.bank_houses_left -= 1
                    reward = 5  # Small reward for building
                    log = f"Player {player_id} built house on property {prop_idx}"
                    
        elif action_type == 'mortgage':
            prop_idx = action.get('property_idx')
            if prop_idx is not None and prop_idx in player.properties_owned:
                spec = self.property_specs[prop_idx]
                prop = state.properties[prop_idx]
                if not prop.mortgaged and prop.houses_count == 0:
                    prop.mortgaged = True
                    player.cash += spec.mortgage_value
                    player.mortgaged_properties.add(prop_idx)
                    log = f"Player {player_id} mortgaged property {prop_idx} for ${spec.mortgage_value}"
                    
        elif action_type == 'unmortgage':
            prop_idx = action.get('property_idx')
            if prop_idx is not None and prop_idx in player.properties_owned:
                spec = self.property_specs[prop_idx]
                prop = state.properties[prop_idx]
                unmortgage_cost = int(spec.mortgage_value * 1.1)
                if prop.mortgaged and player.cash >= unmortgage_cost:
                    prop.mortgaged = False
                    player.cash -= unmortgage_cost
                    player.mortgaged_properties.discard(prop_idx)
                    log = f"Player {player_id} unmortgaged property {prop_idx} for ${unmortgage_cost}"
        
        # Check for game end
        active_players = [p for p in state.players if p.status == PlayerStatus.ACTIVE]
        if len(active_players) <= 1:
            done = True
            if active_players and active_players[0].id == player_id:
                reward += 100
            else:
                reward -= 100
        
        return state, reward, done, log

    def handle_landing(self, state: GameState, player_id: int, rng: random.Random, engine=None) -> GameState:
        """Handle what happens when a player lands on a tile."""
        pos = state.players[player_id].position
        tile = self.board.get_tile(pos)
        player = state.players[player_id]
        
        if tile.kind == TileKind.PROPERTY or tile.kind == TileKind.RAILROAD or tile.kind == TileKind.UTILITY:
            prop_idx = tile.property_idx
            if prop_idx is not None:
                prop = state.properties[prop_idx]
                owner = prop.owner
                
                if owner is None:
                    # Unowned property - player can choose to buy
                    state.awaiting_buy_decision = True
                elif owner != player_id and not prop.mortgaged:
                    # Pay rent to owner
                    dice_roll = sum(state.last_roll) if state.last_roll else 7
                    rent = self.calculate_rent(state, prop_idx, dice_roll if tile.kind == TileKind.UTILITY else None)
                    player.cash -= rent
                    state.players[owner].cash += rent
                    if player.cash < 0:
                        state = self.handle_bankruptcy(state, player_id)
                        
        elif tile.kind == TileKind.TAX:
            tax = 200 if pos == 4 else 100  # Income Tax ($200) or Luxury Tax ($100)
            player.cash -= tax
            if player.cash < 0:
                state = self.handle_bankruptcy(state, player_id)
                
        elif tile.kind == TileKind.CHANCE:
            card = self.draw_card(state, "chance", rng)
            state, _ = card.effect(state, engine, rng, player_id)
            if player.cash < 0:
                state = self.handle_bankruptcy(state, player_id)
                
        elif tile.kind == TileKind.COMMUNITY:
            card = self.draw_card(state, "community", rng)
            state, _ = card.effect(state, engine, rng, player_id)
            if player.cash < 0:
                state = self.handle_bankruptcy(state, player_id)
                
        elif tile.kind == TileKind.GO_TO_JAIL:
            player.position = 10
            player.jail_turns = 1
            state.doubles_count = 0  # Reset doubles when going to jail
            
        # GO, FREE_PARKING, JAIL (just visiting) have no special effects
        
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