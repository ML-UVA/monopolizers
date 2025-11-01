from typing import Optional, Tuple, Dict, Any
import numpy as np
import random
from .rules import RulesEngine
from .state import GameState

class EngineConfig:
    # max_turns etc.
    pass

class GameEngine:
    def __init__(self, rules_engine: RulesEngine, seed: int = 42):
        self.rules_engine = rules_engine
        self.rng = random.Random(seed)

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
                state = self.rules_engine.handle_landing(state, player_id, self.rng)
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
                state = self.rules_engine.handle_landing(state, player_id, self.rng)
        
        # End turn
        state.current_player = (state.current_player + 1) % len(state.players)
        state.turn_number += 1
        return state