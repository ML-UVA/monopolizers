import sys
import os
import time
import random
import numpy as np

# Add the parent directory to sys.path to resolve the 'Monopoly' package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Monopoly.engine import GameEngine
from Monopoly.agents.random import RandomAgent
from Monopoly.agents.greedy import GreedyAgent
from Monopoly.agents.mcts import MCTSAgent
from Monopoly.envs.renderer import MonopolyRenderer
from Monopoly.agents.agent import Agent

class HardCodedRLAgent(Agent):
    """
    A template for a hard-coded Reinforcement Learning agent.
    This agent does not use Stable-Baselines3.
    """
    def __init__(self, player_id=-1, name="HardCodedRL"):
        super().__init__(player_id)
        self.name = name

    def select_action(self, state, legal_actions):
        """
        Select an action based on the current observation and legal actions.
        
        Args:
            state (GameState): The current state of the game.
            legal_actions (list): A list of legal action dictionaries.
            
        Returns:
            dict: The selected action dictionary.
        """
        # ---------------------------------------------------------
        # TODO: Implement your custom RL logic here.
        # You can access the state object to make decisions.
        # For now, this implementation selects a random legal action.
        # ---------------------------------------------------------
        
        if not legal_actions:
            return {'type': 'pass'}
            
        # Placeholder logic: Random choice
        return random.choice(legal_actions)

def run_game(render=True, speed=1.0):
    """
    Runs a game with the HardCodedRLAgent against policy bots.
    """
    # Configuration
    # config = EngineConfig(max_turns=1000) # EngineConfig not currently used by GameEngine
    
    # Initialize RulesEngine first
    from Monopoly.board import Board
    from Monopoly.property import load_property_specs
    from Monopoly.cards import load_chance_cards, load_community_cards
    from Monopoly.rules import RulesEngine
    from Monopoly.engine import GameEngine
    
    board = Board.load_standard_board()
    property_specs = load_property_specs()
    chance_cards = load_chance_cards()
    community_cards = load_community_cards()
    rules_engine = RulesEngine(board, property_specs, chance_cards, community_cards)
    
    engine = GameEngine(rules_engine, seed=42)
    
    # Initialize Agents
    agents = [
        HardCodedRLAgent(player_id=0, name="RL_Bot"),
        GreedyAgent(player_id=1),
        MCTSAgent(player_id=2, engine=engine, rollouts=20),
        GreedyAgent(player_id=3)
    ]
    
    # Initialize Renderer
    renderer = None
    if render:
        renderer = MonopolyRenderer(board, property_specs)

    # Game Loop
    # Note: engine.reset() isn't a method in GameEngine, we need to manually create state
    # or use the gym env's reset logic. Let's manually create state here for transparency.
    from Monopoly.state import GameState, PlayerState, PlayerStatus, PropertyState, DeckState
    
    players = [PlayerState(id=i, cash=1500, position=0, properties_owned=set(), 
                          houses_on_property={}, mortgaged_properties=set(), 
                          jail_turns=0, get_out_of_jail_cards=0, status=PlayerStatus.ACTIVE) 
               for i in range(4)]
    properties = [PropertyState(owner=None, houses_count=0, mortgaged=False) for _ in range(28)]
    
    state = GameState(
        players=players,
        properties=properties,
        chance_deck=DeckState(pointer=0, seed=42),
        community_deck=DeckState(pointer=0, seed=43),
        bank_houses_left=32,
        bank_hotels_left=12,
        current_player=0,
        last_roll=None,
        doubles_count=0,
        turn_number=0,
        seed=42
    )
    
    done = False
    print("Starting game: RL Bot vs Policy Bots...")
    
    while not done:
        # Handle Pygame events
        if renderer:
            renderer.render(state)
            # Check for quit event inside render or here
            for event in import_pygame().event.get():
                if event.type == import_pygame().QUIT:
                    return

        current_player_id = state.current_player
        agent = agents[current_player_id]
        
        # Get legal actions
        legal_actions = rules_engine.legal_actions(state, current_player_id)
        
        if not legal_actions:
            # Should not happen if 'end_turn' is always available, but just in case
            state.current_player = (state.current_player + 1) % 4
            continue

        # Agent selects action
        action = agent.select_action(state, legal_actions)
        
        # Apply action
        state, reward, done, log = rules_engine.apply_action(state, action, engine.rng)
        
        if renderer:
            time.sleep(0.5 / speed)
            
        if done:
            print(f"Game Over! Winner: Player {state.winner_id if hasattr(state, 'winner_id') else 'Unknown'}")
            if renderer:
                time.sleep(5)
                renderer.close()
            break

def import_pygame():
    import pygame
    return pygame


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run HardCodedRLAgent vs Policy Bots")
    parser.add_argument("--no-render", action="store_true", help="Disable visualization")
    parser.add_argument("--speed", type=float, default=2.0, help="Game speed multiplier")
    args = parser.parse_args()
    
    run_game(render=not args.no_render, speed=args.speed)
