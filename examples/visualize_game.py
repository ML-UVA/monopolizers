"""
Example script demonstrating the pygame visualization for Monopoly.

This script shows how to run a game with visual rendering enabled,
allowing you to watch the game progress in real-time with a graphical interface.
"""

import sys
from pathlib import Path
import time

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from Monopoly.envs.gym_env import MonopolyEnv
from Monopoly.agents.random import RandomAgent
from Monopoly.agents.greedy import GreedyAgent


def visualize_random_game(num_episodes=3, num_players=4, steps_per_second=2):
    """
    Run and visualize games with random agents.
    
    Args:
        num_episodes: Number of games to play
        num_players: Number of players (2-8)
        steps_per_second: Speed of visualization (steps per second)
    """
    print("\n" + "="*60)
    print("Monopoly Game Visualization with Random Agents")
    print("="*60)
    print(f"\nSettings:")
    print(f"  - Episodes: {num_episodes}")
    print(f"  - Players: {num_players}")
    print(f"  - Speed: {steps_per_second} steps/second")
    print(f"\nClose the pygame window to exit.")
    print("="*60 + "\n")
    
    # Create environment with pygame rendering
    env = MonopolyEnv(
        num_players=num_players,
        agent_player_id=0,
        opponent_policies=[RandomAgent() for _ in range(num_players - 1)],
        max_turns=500,
        render_mode='human',  # Enable pygame visualization
        seed=42
    )
    
    step_delay = 1.0 / steps_per_second
    
    for episode in range(num_episodes):
        print(f"\n>>> Starting Episode {episode + 1}/{num_episodes} <<<\n")
        
        obs, info = env.reset(seed=42 + episode)
        env.render()
        time.sleep(1)  # Pause to see initial state
        
        terminated = False
        truncated = False
        step_count = 0
        
        while not (terminated or truncated):
            # Select random action from legal actions
            legal_mask = obs['legal_mask']
            legal_actions = [i for i, legal in enumerate(legal_mask) if legal]
            
            if legal_actions:
                action = legal_actions[0]  # Take first legal action (usually roll)
            else:
                action = 0
            
            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1
            
            # Render the updated state
            env.render()
            
            # Control visualization speed
            time.sleep(step_delay)
            
            # Prevent infinite loops
            if step_count > 1000:
                print("Max steps reached, ending episode.")
                break
        
        print(f"\nEpisode {episode + 1} finished:")
        print(f"  - Steps: {step_count}")
        print(f"  - Final cash: ${info['agent_cash']}")
        print(f"  - Properties owned: {info['agent_properties']}")
        print(f"  - Active players: {info['active_players']}")
        
        # Pause between episodes
        if episode < num_episodes - 1:
            print("\nStarting next episode in 3 seconds...")
            time.sleep(3)
    
    env.close()
    print("\n" + "="*60)
    print("Visualization Complete!")
    print("="*60 + "\n")


def visualize_vs_greedy(num_episodes=2, num_players=3, steps_per_second=1):
    """
    Visualize games where the agent plays against greedy opponents.
    
    Args:
        num_episodes: Number of games to play
        num_players: Number of players (2-8)
        steps_per_second: Speed of visualization
    """
    print("\n" + "="*60)
    print("Monopoly Game Visualization: Random vs Greedy Opponents")
    print("="*60)
    print(f"\nSettings:")
    print(f"  - Episodes: {num_episodes}")
    print(f"  - Players: {num_players}")
    print(f"  - Opponents: {num_players-1} Greedy Bots")
    print(f"  - Speed: {steps_per_second} steps/second")
    print(f"\nPlayer 0 (Red) is random, others are greedy.")
    print("="*60 + "\n")
    
    # Create environment with mixed policies
    env = MonopolyEnv(
        num_players=num_players,
        agent_player_id=0,
        opponent_policies=[GreedyAgent(safety_margin=100) for _ in range(num_players - 1)],
        max_turns=500,
        render_mode='human',
        seed=42
    )
    
    step_delay = 1.0 / steps_per_second
    
    for episode in range(num_episodes):
        print(f"\n>>> Starting Episode {episode + 1}/{num_episodes} <<<\n")
        
        obs, info = env.reset(seed=100 + episode)
        env.render()
        time.sleep(2)
        
        terminated = False
        truncated = False
        step_count = 0
        
        while not (terminated or truncated):
            legal_mask = obs['legal_mask']
            legal_actions = [i for i, legal in enumerate(legal_mask) if legal]
            
            if legal_actions:
                action = legal_actions[0]
            else:
                action = 0
            
            obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1
            
            env.render()
            time.sleep(step_delay)
            
            if step_count > 1000:
                break
        
        # Determine winner
        winner = None
        for i, player in enumerate(env.state.players):
            if player.status.value == 'ACTIVE':
                winner = i
                break
        
        print(f"\nEpisode {episode + 1} finished:")
        print(f"  - Steps: {step_count}")
        print(f"  - Winner: Player {winner}")
        print(f"  - Agent cash: ${info['agent_cash']}")
        print(f"  - Agent properties: {info['agent_properties']}")
        
        if episode < num_episodes - 1:
            print("\nStarting next episode in 3 seconds...")
            time.sleep(3)
    
    env.close()
    print("\n" + "="*60)
    print("Visualization Complete!")
    print("="*60 + "\n")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize Monopoly games with pygame")
    parser.add_argument('--mode', choices=['random', 'greedy'], default='random',
                       help='Game mode: random agents or random vs greedy')
    parser.add_argument('--episodes', type=int, default=2,
                       help='Number of episodes to play')
    parser.add_argument('--players', type=int, default=4,
                       help='Number of players (2-8)')
    parser.add_argument('--speed', type=float, default=2.0,
                       help='Steps per second for visualization')
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'random':
            visualize_random_game(
                num_episodes=args.episodes,
                num_players=args.players,
                steps_per_second=args.speed
            )
        else:
            visualize_vs_greedy(
                num_episodes=args.episodes,
                num_players=args.players,
                steps_per_second=args.speed
            )
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\nVisualization interrupted by user.")
        return 0
    except Exception as e:
        print(f"\n[ERROR] Visualization failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
