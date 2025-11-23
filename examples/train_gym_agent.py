"""
Example script demonstrating how to train an RL agent on Monopoly using Gymnasium.

This script shows:
1. Environment creation and configuration
2. Random agent baseline
3. Integration pattern for stable-baselines3 (commented out if not installed)
"""

import numpy as np
from Monopoly.envs.gym_env import MonopolyEnv
from Monopoly.agents.random import RandomAgent
from Monopoly.agents.greedy import GreedyAgent


def random_agent_rollout(env, num_episodes=5, render=False):
    """Run episodes with a random agent to test the environment."""
    print(f"\n{'='*60}")
    print("Running Random Agent Baseline")
    print(f"{'='*60}\n")
    
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        obs, info = env.reset(seed=42 + episode)
        episode_reward = 0
        episode_length = 0
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            # Sample random legal action
            legal_mask = obs['legal_mask']
            legal_actions = np.where(legal_mask)[0]
            
            if len(legal_actions) > 0:
                action = np.random.choice(legal_actions)
            else:
                action = 0  # Default to roll
            
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            
            if render and episode_length % 10 == 0:
                env.render()
            
            # Prevent infinite loops
            if episode_length > 1000:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        print(f"Episode {episode + 1}: Reward={episode_reward:.2f}, Length={episode_length}, "
              f"Final Cash=${info['agent_cash']}, Properties={info['agent_properties']}")
    
    print(f"\nAverage Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average Episode Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    
    return episode_rewards, episode_lengths


def train_with_stable_baselines3(env, total_timesteps=10000):
    """
    Example of training with stable-baselines3 PPO.
    
    Requires: pip install stable-baselines3
    """
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
        from stable_baselines3.common.callbacks import EvalCallback
        
        print(f"\n{'='*60}")
        print("Training with Stable-Baselines3 PPO")
        print(f"{'='*60}\n")
        
        # Wrap environment
        vec_env = DummyVecEnv([lambda: env])
        
        # Create evaluation callback
        eval_env = DummyVecEnv([lambda: MonopolyEnv(num_players=2, agent_player_id=0, seed=42)])
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path='./models/',
            log_path='./logs/',
            eval_freq=1000,
            deterministic=True,
            render=False
        )
        
        # Initialize PPO agent
        model = PPO(
            "MultiInputPolicy",  # For Dict observation space
            vec_env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            tensorboard_log="./tensorboard/"
        )
        
        # Train
        print("Starting training...")
        model.learn(total_timesteps=total_timesteps, callback=eval_callback)
        
        # Save final model
        model.save("monopoly_ppo_final")
        print("\nTraining complete! Model saved as 'monopoly_ppo_final'")
        
        # Evaluate trained model
        print("\nEvaluating trained model...")
        obs = vec_env.reset()
        episode_reward = 0
        for _ in range(100):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            episode_reward += reward[0]
            if done[0]:
                print(f"Trained agent episode reward: {episode_reward:.2f}")
                break
        
        return model
        
    except ImportError:
        print("\n⚠️  stable-baselines3 not installed. Skipping SB3 training.")
        print("Install with: pip install stable-baselines3")
        return None


def compare_agents(num_episodes=10):
    """Compare random agent vs environment with multiple opponents."""
    print(f"\n{'='*60}")
    print("Comparing Agent Configurations")
    print(f"{'='*60}\n")
    
    configs = [
        ("2 Players (Agent vs 1 Random)", 2, 0, [RandomAgent()]),
        ("3 Players (Agent vs 2 Random)", 3, 0, [RandomAgent(), RandomAgent()]),
        ("4 Players (Agent vs 3 Random)", 4, 0, [RandomAgent(), RandomAgent(), RandomAgent()]),
    ]
    
    for config_name, n_players, agent_id, opponents in configs:
        env = MonopolyEnv(
            num_players=n_players,
            agent_player_id=agent_id,
            opponent_policies=opponents,
            max_turns=500,
            seed=42
        )
        
        print(f"\nConfiguration: {config_name}")
        rewards, lengths = random_agent_rollout(env, num_episodes=num_episodes, render=False)
        env.close()


def test_environment_speed():
    """Test environment step speed."""
    import time
    
    print(f"\n{'='*60}")
    print("Environment Speed Test")
    print(f"{'='*60}\n")
    
    env = MonopolyEnv(num_players=2, agent_player_id=0, seed=42)
    
    n_steps = 1000
    obs, info = env.reset(seed=42)
    
    start_time = time.time()
    for _ in range(n_steps):
        legal_mask = obs['legal_mask']
        legal_actions = np.where(legal_mask)[0]
        action = np.random.choice(legal_actions) if len(legal_actions) > 0 else 0
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            obs, info = env.reset()
    
    elapsed = time.time() - start_time
    steps_per_sec = n_steps / elapsed
    
    print(f"Completed {n_steps} steps in {elapsed:.2f} seconds")
    print(f"Speed: {steps_per_sec:.1f} steps/second")
    print(f"Average time per step: {(elapsed/n_steps)*1000:.2f} ms")


def main():
    """Main entry point."""
    print("\n" + "="*60)
    print("Monopoly Gymnasium Environment - Example & Training")
    print("="*60)
    
    # Create environment
    env = MonopolyEnv(
        num_players=2,
        agent_player_id=0,
        opponent_policies=[RandomAgent()],
        max_turns=500,
        render_mode='human',
        seed=42
    )
    
    print("\n✓ Environment created successfully!")
    print(f"  - Observation space: {env.observation_space}")
    print(f"  - Action space: {env.action_space}")
    print(f"  - Number of players: {env.num_players}")
    
    # Test basic functionality
    print("\nTesting basic environment functionality...")
    obs, info = env.reset(seed=42)
    print(f"✓ Reset successful. Initial state: Turn {info['turn_number']}, "
          f"Cash ${info['agent_cash']}")
    
    # Random agent baseline
    random_agent_rollout(env, num_episodes=3, render=False)
    
    # Speed test
    test_environment_speed()
    
    # Compare configurations
    compare_agents(num_episodes=5)
    
    # Train with SB3 (if available)
    # Uncomment the following line to train with stable-baselines3
    # model = train_with_stable_baselines3(env, total_timesteps=10000)
    
    print("\n" + "="*60)
    print("Example complete!")
    print("="*60 + "\n")
    
    env.close()


if __name__ == "__main__":
    main()
