"""
Training script for Monopoly agent vs GreedyBot opponents.

This script demonstrates how to:
1. Create a Monopoly Gymnasium environment with opponent policies
2. Train a simple baseline agent
3. Evaluate agent performance

For production training, use stable-baselines3 or similar RL libraries.
"""

import sys
from pathlib import Path
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from monopolizers.Monopoly.envs.gym_env import MonopolyEnv
from monopolizers.Monopoly.agents.random import RandomAgent
from monopolizers.Monopoly.agents.greedy import GreedyAgent


def make_env(num_players=2, agent_id=0, seed=42):
    """Create a Monopoly environment with greedy opponents."""
    # Create opponent policies (all greedy)
    opponent_policies = [GreedyAgent(safety_margin=100) for _ in range(num_players - 1)]
    
    env = MonopolyEnv(
        num_players=num_players,
        agent_player_id=agent_id,
        opponent_policies=opponent_policies,
        max_turns=300,
        seed=seed
    )
    return env


def evaluate_random_vs_greedy(num_episodes=10, num_players=2, seed=42):
    """Evaluate a random agent against greedy opponents."""
    print(f"\n{'='*60}")
    print(f"Evaluating Random Agent vs {num_players-1} Greedy Opponent(s)")
    print(f"{'='*60}\n")
    
    env = make_env(num_players=num_players, agent_id=0, seed=seed)
    
    episode_rewards = []
    episode_lengths = []
    wins = 0
    
    for episode in range(num_episodes):
        obs, info = env.reset(seed=seed + episode)
        episode_reward = 0
        episode_length = 0
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            # Random agent: sample from legal actions
            legal_mask = obs['legal_mask']
            legal_actions = np.where(legal_mask)[0]
            
            if len(legal_actions) > 0:
                action = np.random.choice(legal_actions)
            else:
                action = 0
            
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            
            if episode_length > 1000:  # Prevent infinite loops
                truncated = True
                break
        
        # Check if agent won
        if env.state.players[0].status.value == 'ACTIVE':
            active_count = sum(1 for p in env.state.players if p.status.value == 'ACTIVE')
            if active_count == 1:
                wins += 1
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        print(f"Episode {episode + 1}/{num_episodes}: "
              f"Reward={episode_reward:.2f}, Length={episode_length}, "
              f"Cash=${info['agent_cash']}, Props={info['agent_properties']}")
    
    # Print summary statistics
    win_rate = wins / num_episodes
    print(f"\n{'-'*60}")
    print(f"Results Summary:")
    print(f"  Win Rate: {win_rate*100:.1f}% ({wins}/{num_episodes})")
    print(f"  Avg Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"  Avg Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"{'-'*60}\n")
    
    env.close()
    return win_rate, episode_rewards


def train_simple_q_table(num_episodes=100, num_players=2):
    """
    Simple Q-learning baseline (for demonstration only).
    
    Note: This is a toy example. For real training, use stable-baselines3 or similar.
    """
    print(f"\n{'='*60}")
    print("Simple Q-Learning Training (Baseline)")
    print(f"{'='*60}\n")
    
    env = make_env(num_players=num_players, agent_id=0, seed=42)
    
    # Hyperparameters
    learning_rate = 0.1
    discount_factor = 0.95
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.01
    
    episode_rewards = []
    
    for episode in range(num_episodes):
        obs, info = env.reset(seed=42 + episode)
        episode_reward = 0
        episode_length = 0
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            # Epsilon-greedy action selection
            legal_mask = obs['legal_mask']
            legal_actions = np.where(legal_mask)[0]
            
            if len(legal_actions) == 0:
                action = 0
            elif np.random.random() < epsilon:
                # Explore: random legal action
                action = np.random.choice(legal_actions)
            else:
                # Exploit: prefer rolling/buying (simple heuristic)
                if 0 in legal_actions:  # Roll
                    action = 0
                elif 1 in legal_actions:  # Buy
                    action = 1
                else:
                    action = legal_actions[0]
            
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            
            if episode_length > 500:
                truncated = True
                break
        
        episode_rewards.append(episode_reward)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        
        if (episode + 1) % 10 == 0:
            recent_avg = np.mean(episode_rewards[-10:])
            print(f"Episode {episode + 1}/{num_episodes}: "
                  f"Avg Reward (last 10)={recent_avg:.2f}, epsilon={epsilon:.3f}")
    
    print(f"\nTraining complete!")
    print(f"  Final avg reward (last 20): {np.mean(episode_rewards[-20:]):.2f}")
    
    env.close()
    return episode_rewards


def main():
    """Main training and evaluation pipeline."""
    try:
        print("\n" + "="*60)
        print("Monopoly Training: Random Agent vs Greedy Opponents")
        print("="*60)
        
        # Keep this example fast (it's used as a smoke test in CI).
        win_rate_2p, _ = evaluate_random_vs_greedy(num_episodes=2, num_players=2, seed=42)
        
        # Evaluate with more opponents
        win_rate_4p, _ = evaluate_random_vs_greedy(num_episodes=2, num_players=4, seed=123)
        
        # Simple Q-learning baseline
        train_simple_q_table(num_episodes=5, num_players=2)
        
        print("\n" + "="*60)
        print("Training & Evaluation Complete!")
        print(f"  2-player win rate: {win_rate_2p*100:.1f}%")
        print(f"  4-player win rate: {win_rate_4p*100:.1f}%")
        print("="*60 + "\n")
        
        return 0
        
    except Exception as e:
        print(f"\n[ERROR] Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())