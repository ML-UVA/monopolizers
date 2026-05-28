from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import numpy as np
import random
from .rules import RulesEngine
from .state import GameState

@dataclass
class EngineConfig:
    max_turns: int = 1000  # Maximum turns before game ends
    enable_free_parking_money: bool = False  # Whether free parking collects fines/taxes
    income_tax_option: str = "200_or_10_percent"  # "200" or "10_percent" or "200_or_10_percent"
    luxury_tax_amount: int = 75  # Luxury tax amount
    salary_on_go: int = 200  # Amount collected passing/landing on GO
    jail_fee: int = 50  # Fee to get out of jail after rolls
    max_jail_attempts: int = 3  # Attempts to roll doubles before paying fee
    auction_start_bid: int = 1  # Starting bid for auctions
    enable_trading: bool = True  # Allow property trading between players
    enable_auctions: bool = True  # Enable auctions for declined properties
    max_houses_per_property: int = 4  # Houses before hotel
    hotel_requires_4_houses: bool = True  # Must have 4 houses to build hotel
    even_building_rule: bool = True  # Must build evenly across monopolies
    enable_houses_and_hotels: bool = True  # Allow building houses and hotels
    starting_cash: int = 1500  # Initial cash for each player
    max_players: int = 4  # Maximum number of players

class GameEngine:
    def __init__(self, rules_engine: RulesEngine, seed: int = 42):
        self.rules_engine = rules_engine
        self.rng = random.Random(seed)
        # Expose board and other components for card effects
        self.board = rules_engine.board
        self.property_specs = rules_engine.property_specs
    
    def calculate_rent(self, state: GameState, property_idx: int, dice_roll=None) -> int:
        """Wrapper to expose calculate_rent for card effects."""
        return self.rules_engine.calculate_rent(state, property_idx, dice_roll)

    def run_turn(self, state: GameState) -> GameState:
        player_id = state.current_player
        player = state.players[player_id]
        
        if player.jail_turns > 0:
            # Attempt to get out of jail
            roll = self.rules_engine.roll_dice(self.rng)
            if roll[0] == roll[1]:
                player.jail_turns = 0
                steps = sum(roll)
                state = self.rules_engine.move_player(state, player_id, steps)
                state = self.rules_engine.handle_landing(state, player_id, self.rng, engine=self)
            else:
                player.jail_turns += 1
                if player.jail_turns >= 4:  # After 3 attempts, pay $50
                    player.cash -= 50
                    player.jail_turns = 0
        else:
            # Normal turn
            roll = self.rules_engine.roll_dice(self.rng)
            state.last_roll = roll
            state = self.rules_engine.handle_doubles_and_jail(state, player_id, roll)
            if player.jail_turns == 0:
                steps = sum(roll)
                state = self.rules_engine.move_player(state, player_id, steps)
                state = self.rules_engine.handle_landing(state, player_id, self.rng, engine=self)
        
        # End turn
        state.current_player = (state.current_player + 1) % len(state.players)
        state.turn_number += 1
        return state

    def clone(self) -> 'GameEngine':
        # Create a new engine with the same rules and seed
        # Note: We don't need to deepcopy rules_engine as it is stateless (mostly)
        new_engine = GameEngine(self.rules_engine, seed=0)
        new_engine.rng.setstate(self.rng.getstate())
        return new_engine