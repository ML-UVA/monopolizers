"""
Simple smoke test script to verify the Monopoly game engine runs without crashes.

This script creates a basic game with random agents and runs a few turns
to ensure all components are integrated correctly.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from Monopoly.engine import GameEngine
from Monopoly.rules import RulesEngine, RulesConfig
from Monopoly.state import GameState, PlayerState, PropertyState, DeckState, PlayerStatus
from Monopoly.board import Board
from Monopoly.property import load_property_specs
from Monopoly.cards import load_chance_cards, load_community_cards


def run_smoke_game(num_players=4, max_turns=50, seed=42):
    """Run a simple game with random agents for smoke testing."""
    
    print(f"Starting smoke test game with {num_players} players...")
    
    # Initialize game components
    board = Board.load_standard_board()
    property_specs = load_property_specs()
    chance_cards = load_chance_cards()
    community_cards = load_community_cards()
    rules = RulesEngine(board, property_specs, chance_cards, community_cards, config=RulesConfig)
    engine = GameEngine(rules, seed=seed)
    
    # Create initial state
    players = []
    for i in range(num_players):
        players.append(PlayerState(
            id=i,
            cash=rules.config.starting_cash,
            position=0,
            properties_owned=set(),
            houses_on_property={},
            mortgaged_properties=set(),
            jail_turns=0,
            get_out_of_jail_cards=0,
            status=PlayerStatus.ACTIVE
        ))
    
    properties = [PropertyState(owner=None, houses_count=0, mortgaged=False) 
                  for _ in range(len(property_specs))]
    
    state = GameState(
        players=players,
        properties=properties,
        chance_deck=DeckState(pointer=0, seed=seed),
        community_deck=DeckState(pointer=0, seed=seed + 1),
        bank_houses_left=rules.config.bank_houses,
        bank_hotels_left=rules.config.bank_hotels,
        current_player=0,
        last_roll=None,
        doubles_count=0,
        turn_number=0,
        seed=seed
    )

    property_specs = load_property_specs()
    groups = set(spec.group for spec in property_specs)
    state.monopoly_status = {group: None for group in groups}
    
    # Run game for max_turns
    turn = 0
    while turn < max_turns:
        # Check if game is over
        active_players = [p for p in state.players if p.status == PlayerStatus.ACTIVE]
        if len(active_players) <= 1:
            print(f"Game ended at turn {turn}. Winner: Player {active_players[0].id if active_players else 'None'}")
            break
        
        # Run one turn
        try:
            state = engine.run_turn(state)
            turn += 1
            
            if turn % 10 == 0:
                active_count = len(active_players)
                print(f"Turn {turn}: {active_count} players active")
                
        except Exception as e:
            print(f"Error at turn {turn}: {e}")
            raise
    
    # Print final statistics
    print(f"\nGame completed after {turn} turns")
    print("Final player standings:")
    for i, player in enumerate(state.players):
        status = "ACTIVE" if player.status == PlayerStatus.ACTIVE else "BANKRUPT"
        print(f"  Player {i}: ${player.cash}, {len(player.properties_owned)} properties, {status}")
    
    print("\n[OK] Smoke test passed!")
    return 0


def main(seed=42, episodes=3, engine_factory=None, opponent_policies=None):
    """Run multiple smoke tests with different configurations."""
    try:
        # Test 1: 2-player game
        print("="*60)
        print("Test 1: 2-player game")
        print("="*60)
        run_smoke_game(num_players=2, max_turns=30, seed=seed)
        
        # Test 2: 4-player game
        print("\n" + "="*60)
        print("Test 2: 4-player game")
        print("="*60)
        run_smoke_game(num_players=4, max_turns=50, seed=seed + 1)
        
        # Test 3: Different seed
        print("\n" + "="*60)
        print("Test 3: Alternative seed")
        print("="*60)
        run_smoke_game(num_players=3, max_turns=40, seed=seed + 2)
        
        print("\n" + "="*60)
        print("All smoke tests passed successfully!")
        print("="*60)
        return 0
        
    except Exception as e:
        print(f"\n[ERROR] Smoke test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
