from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
from enum import Enum
import numpy as np
import random
from .rules import RulesEngine
from .state import GameState

@dataclass
class EngineConfig:
    max_turns: int = 1000  # Maximum turns before game ends
    max_players: int = 4  # Maximum number of players

class GameEngine:
    def __init__(
            self, 
            rules_engine: RulesEngine, 
            config: EngineConfig = None,
            seed: int = 42
    ):  
        self.rules_engine = rules_engine
        self.config = config or EngineConfig()
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

        roll = self.rules_engine.roll_dice(self.rng)
        state.last_roll = roll

        state = self.rules_engine.handle_doubles_and_jail(state, player_id, roll)

        if player.jail_turns > 0:
            # Either failed jail roll or sent to jail by triple doubles
            state.current_player = (state.current_player + 1) % len(state.players)
            state.turn_number += 1
            return state

        steps = sum(roll)
        state = self.rules_engine.move_player(state, player_id, steps)
        state = self.rules_engine.handle_landing(state, player_id, self.rng, engine=self)

        # Simple heuristic: buy if possible
        pos = state.players[player_id].position
        tile = self.rules_engine.board.get_tile(pos)
        if tile.property_idx is not None:
            prop = state.properties[tile.property_idx]
            spec = self.rules_engine.property_specs[tile.property_idx]
            if prop.owner is None and state.players[player_id].cash >= spec.price:
                self.rules_engine.buy_property(state, player_id, tile.property_idx)

        # Doubles bonus roll (only when not in jail)
        rolled_doubles = (roll[0] == roll[1] and
                        state.players[player_id].jail_turns == 0)
        if rolled_doubles:
            return self.run_turn(state)

        state.current_player = (state.current_player + 1) % len(state.players)
        state.turn_number += 1
        return state

    def clone(self) -> 'GameEngine':
        # Create a new engine with the same rules and seed
        # Note: We don't need to deepcopy rules_engine as it is stateless (mostly)
        new_engine = GameEngine(self.rules_engine, config=self.config, seed=0)
        new_engine.rng.setstate(self.rng.getstate())
        return new_engine
