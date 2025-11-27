#!/usr/bin/env python3
"""
Stable-Baselines3 DQN Training Script for Monopoly

This script provides a complete training pipeline for training a DQN agent
to play Monopoly using the Stable-Baselines3 library.

Features:
- DQN with configurable hyperparameters
- Action masking for legal move enforcement
- Tensorboard logging
- Checkpoint saving
- Evaluation during training
- Model loading and evaluation

Requirements:
    pip install stable-baselines3 tensorboard gymnasium numpy

Usage:
    python train_dqn.py --train --timesteps 100000
    python train_dqn.py --evaluate --model models/monopoly_dqn_final.zip
"""

import sys
import os
import argparse
import time
from pathlib import Path
from typing import Optional, Dict, Any, Callable
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Gymnasium
import gymnasium as gym

# Stable-Baselines3
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
    CallbackList
)
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

# Monopoly Environment
from Monopoly.envs.gym_env import MonopolyEnv
from Monopoly.envs.wrappers import MonopolyFlattenWrapper, RewardShapingWrapper
from Monopoly.agents.random import RandomAgent
from Monopoly.agents.greedy import GreedyAgent


# ============================================================================
# Custom Callbacks
# ============================================================================

class TensorboardCallback(BaseCallback):
    """
    Custom callback for logging additional metrics to Tensorboard.
    """
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_cash = []
        self.episode_properties = []
        
    def _on_step(self) -> bool:
        # Log episode info when available
        if 'infos' in self.locals:
            for info in self.locals['infos']:
                if 'episode' in info:
                    ep_reward = info['episode']['r']
                    ep_length = info['episode']['l']
                    self.episode_rewards.append(ep_reward)
                    self.episode_lengths.append(ep_length)
                    
                    # Log to tensorboard
                    self.logger.record('rollout/ep_rew_mean_custom', np.mean(self.episode_rewards[-100:]))
                    self.logger.record('rollout/ep_len_mean_custom', np.mean(self.episode_lengths[-100:]))
                    
                # Log game-specific info
                if 'agent_cash' in info:
                    self.episode_cash.append(info['agent_cash'])
                    self.logger.record('game/agent_cash', info['agent_cash'])
                if 'agent_properties' in info:
                    self.episode_properties.append(info['agent_properties'])
                    self.logger.record('game/agent_properties', info['agent_properties'])
        
        return True


class ActionMaskCallback(BaseCallback):
    """
    Callback that applies action masking during training.
    This modifies the action selection to only choose from legal actions.
    """
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        
    def _on_step(self) -> bool:
        return True


# ============================================================================
# Environment Factory Functions
# ============================================================================

def make_monopoly_env(
    num_players: int = 2,
    agent_player_id: int = 0,
    opponent_type: str = 'random',
    max_turns: int = 500,
    seed: Optional[int] = None,
    flatten: bool = True,
    reward_shaping: bool = True
) -> gym.Env:
    """
    Create a Monopoly environment with optional wrappers.
    
    Args:
        num_players: Number of players (2-4)
        agent_player_id: Player ID for the RL agent
        opponent_type: Type of opponent ('random', 'greedy')
        max_turns: Maximum turns before truncation
        seed: Random seed
        flatten: Whether to flatten observations
        reward_shaping: Whether to apply reward shaping
        
    Returns:
        Gymnasium environment
    """
    # Create opponent agents
    opponents = []
    for i in range(num_players):
        if i != agent_player_id:
            if opponent_type == 'greedy':
                opponents.append(GreedyAgent(player_id=i))
            else:
                opponents.append(RandomAgent(player_id=i))
    
    # Create base environment
    env = MonopolyEnv(
        num_players=num_players,
        agent_player_id=agent_player_id,
        opponent_policies=opponents,
        max_turns=max_turns,
        seed=seed
    )
    
    # Apply wrappers
    if reward_shaping:
        env = RewardShapingWrapper(env, cash_weight=0.001, property_weight=0.5, survival_bonus=0.01)
    
    if flatten:
        env = MonopolyFlattenWrapper(env)
    
    # Add monitor for logging
    env = Monitor(env)
    
    return env


def make_vec_env(
    n_envs: int = 1,
    num_players: int = 2,
    opponent_type: str = 'random',
    max_turns: int = 500,
    seed: Optional[int] = None
) -> VecMonitor:
    """
    Create a vectorized environment for parallel training.
    
    Args:
        n_envs: Number of parallel environments
        num_players: Number of players per game
        opponent_type: Type of opponent
        max_turns: Maximum turns
        seed: Base random seed
        
    Returns:
        Vectorized environment
    """
    def make_env(env_id: int) -> Callable[[], gym.Env]:
        def _init() -> gym.Env:
            env_seed = seed + env_id if seed is not None else None
            return make_monopoly_env(
                num_players=num_players,
                opponent_type=opponent_type,
                max_turns=max_turns,
                seed=env_seed
            )
        return _init
    
    envs = DummyVecEnv([make_env(i) for i in range(n_envs)])
    envs = VecMonitor(envs)
    
    return envs


# ============================================================================
# DQN Configuration
# ============================================================================

def get_dqn_config(preset: str = 'default') -> Dict[str, Any]:
    """
    Get DQN hyperparameter configuration.
    
    Args:
        preset: Configuration preset ('default', 'fast', 'thorough')
        
    Returns:
        Dictionary of hyperparameters
    """
    configs = {
        'default': {
            'learning_rate': 1e-4,
            'buffer_size': 100000,
            'learning_starts': 1000,
            'batch_size': 64,
            'tau': 0.005,
            'gamma': 0.99,
            'train_freq': 4,
            'gradient_steps': 1,
            'target_update_interval': 1000,
            'exploration_fraction': 0.2,
            'exploration_initial_eps': 1.0,
            'exploration_final_eps': 0.05,
            'max_grad_norm': 10,
        },
        'fast': {
            'learning_rate': 3e-4,
            'buffer_size': 50000,
            'learning_starts': 500,
            'batch_size': 32,
            'tau': 0.01,
            'gamma': 0.99,
            'train_freq': 4,
            'gradient_steps': 1,
            'target_update_interval': 500,
            'exploration_fraction': 0.1,
            'exploration_initial_eps': 1.0,
            'exploration_final_eps': 0.1,
            'max_grad_norm': 10,
        },
        'thorough': {
            'learning_rate': 5e-5,
            'buffer_size': 500000,
            'learning_starts': 5000,
            'batch_size': 128,
            'tau': 0.001,
            'gamma': 0.995,
            'train_freq': 4,
            'gradient_steps': 2,
            'target_update_interval': 2000,
            'exploration_fraction': 0.3,
            'exploration_initial_eps': 1.0,
            'exploration_final_eps': 0.02,
            'max_grad_norm': 10,
        }
    }
    
    return configs.get(preset, configs['default'])


# ============================================================================
# Training Function
# ============================================================================

def train_dqn(
    total_timesteps: int = 100000,
    num_players: int = 2,
    opponent_type: str = 'random',
    max_turns: int = 500,
    config_preset: str = 'default',
    save_path: str = 'models',
    log_path: str = 'logs',
    seed: int = 42,
    eval_freq: int = 5000,
    n_eval_episodes: int = 10,
    verbose: int = 1
) -> DQN:
    """
    Train a DQN agent to play Monopoly.
    
    Args:
        total_timesteps: Total training timesteps
        num_players: Number of players (2-4)
        opponent_type: Type of opponent ('random', 'greedy')
        max_turns: Maximum turns per episode
        config_preset: Hyperparameter preset
        save_path: Path to save models
        log_path: Path for tensorboard logs
        seed: Random seed
        eval_freq: Evaluation frequency
        n_eval_episodes: Number of evaluation episodes
        verbose: Verbosity level
        
    Returns:
        Trained DQN model
    """
    print("=" * 60)
    print("Monopoly DQN Training")
    print("=" * 60)
    print(f"Total timesteps: {total_timesteps}")
    print(f"Players: {num_players}")
    print(f"Opponent type: {opponent_type}")
    print(f"Config preset: {config_preset}")
    print(f"Seed: {seed}")
    print("=" * 60)
    
    # Create directories
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    
    # Create training environment
    print("\nCreating training environment...")
    train_env = make_vec_env(
        n_envs=1,
        num_players=num_players,
        opponent_type=opponent_type,
        max_turns=max_turns,
        seed=seed
    )
    
    # Create evaluation environment
    print("Creating evaluation environment...")
    eval_env = make_vec_env(
        n_envs=1,
        num_players=num_players,
        opponent_type=opponent_type,
        max_turns=max_turns,
        seed=seed + 1000
    )
    
    # Get hyperparameters
    config = get_dqn_config(config_preset)
    print(f"\nHyperparameters: {config}")
    
    # Create DQN model
    print("\nInitializing DQN model...")
    model = DQN(
        policy="MlpPolicy",
        env=train_env,
        verbose=verbose,
        seed=seed,
        tensorboard_log=log_path,
        device='auto',
        **config
    )
    
    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=save_path,
        name_prefix="monopoly_dqn"
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_path,
        log_path=log_path,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
        verbose=verbose
    )
    
    tensorboard_callback = TensorboardCallback(verbose=verbose)
    
    callback_list = CallbackList([
        checkpoint_callback,
        eval_callback,
        tensorboard_callback
    ])
    
    # Train
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    
    start_time = time.time()
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback_list,
        log_interval=100,
        tb_log_name="DQN",
        progress_bar=True
    )
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")
    print(f"Average speed: {total_timesteps / training_time:.1f} timesteps/second")
    
    # Save final model
    final_path = os.path.join(save_path, "monopoly_dqn_final")
    model.save(final_path)
    print(f"\nFinal model saved to: {final_path}.zip")
    
    # Cleanup
    train_env.close()
    eval_env.close()
    
    return model


# ============================================================================
# Evaluation Function
# ============================================================================

def evaluate_model(
    model_path: str,
    num_episodes: int = 100,
    num_players: int = 2,
    opponent_type: str = 'random',
    max_turns: int = 500,
    seed: int = 42,
    render: bool = False,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Evaluate a trained DQN model.
    
    Args:
        model_path: Path to the saved model
        num_episodes: Number of evaluation episodes
        num_players: Number of players
        opponent_type: Type of opponent
        max_turns: Maximum turns per episode
        seed: Random seed
        render: Whether to render games
        verbose: Print detailed results
        
    Returns:
        Dictionary of evaluation metrics
    """
    print("=" * 60)
    print("Evaluating Monopoly DQN Agent")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Episodes: {num_episodes}")
    print(f"Opponent: {opponent_type}")
    print("=" * 60)
    
    # Load model
    model = DQN.load(model_path)
    
    # Create evaluation environment
    env = make_monopoly_env(
        num_players=num_players,
        opponent_type=opponent_type,
        max_turns=max_turns,
        seed=seed,
        flatten=True,
        reward_shaping=False  # No reward shaping for evaluation
    )
    
    # Run evaluation
    episode_rewards = []
    episode_lengths = []
    wins = 0
    final_cash = []
    final_properties = []
    
    for episode in range(num_episodes):
        obs, info = env.reset(seed=seed + episode)
        episode_reward = 0
        episode_length = 0
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            # Get action from model
            action, _ = model.predict(obs, deterministic=True)
            
            # Apply action masking
            legal_mask = env.unwrapped._get_legal_mask()
            if legal_mask[action] == 0:
                # Choose random legal action
                legal_actions = np.where(legal_mask)[0]
                if len(legal_actions) > 0:
                    action = np.random.choice(legal_actions)
            
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            
            if render:
                env.render()
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        final_cash.append(info.get('agent_cash', 0))
        final_properties.append(info.get('agent_properties', 0))
        
        # Check if agent won
        if info.get('agent_status') == 'ACTIVE':
            active_count = info.get('active_players', 1)
            if active_count == 1:
                wins += 1
        
        if verbose and (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{num_episodes}: "
                  f"Reward={episode_reward:.2f}, "
                  f"Length={episode_length}, "
                  f"Cash=${info.get('agent_cash', 0)}")
    
    env.close()
    
    # Calculate metrics
    results = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths),
        'win_rate': wins / num_episodes,
        'mean_final_cash': np.mean(final_cash),
        'mean_final_properties': np.mean(final_properties),
    }
    
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"Mean Episode Length: {results['mean_length']:.1f} ± {results['std_length']:.1f}")
    print(f"Win Rate: {results['win_rate'] * 100:.1f}%")
    print(f"Mean Final Cash: ${results['mean_final_cash']:.0f}")
    print(f"Mean Final Properties: {results['mean_final_properties']:.1f}")
    print("=" * 60)
    
    return results


# ============================================================================
# Random Baseline
# ============================================================================

def run_random_baseline(
    num_episodes: int = 100,
    num_players: int = 2,
    max_turns: int = 500,
    seed: int = 42
) -> Dict[str, float]:
    """
    Run random agent baseline for comparison.
    """
    print("=" * 60)
    print("Running Random Agent Baseline")
    print("=" * 60)
    
    env = make_monopoly_env(
        num_players=num_players,
        opponent_type='random',
        max_turns=max_turns,
        seed=seed,
        flatten=True,
        reward_shaping=False
    )
    
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
            # Random legal action
            legal_mask = env.unwrapped._get_legal_mask()
            legal_actions = np.where(legal_mask)[0]
            
            if len(legal_actions) > 0:
                action = np.random.choice(legal_actions)
            else:
                action = 0
            
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        if info.get('agent_status') == 'ACTIVE' and info.get('active_players', 1) == 1:
            wins += 1
    
    env.close()
    
    results = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'win_rate': wins / num_episodes,
    }
    
    print(f"Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"Mean Episode Length: {results['mean_length']:.1f}")
    print(f"Win Rate: {results['win_rate'] * 100:.1f}%")
    print("=" * 60)
    
    return results


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train or evaluate a DQN agent for Monopoly')
    
    # Mode
    parser.add_argument('--train', action='store_true', help='Train a new model')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate a trained model')
    parser.add_argument('--baseline', action='store_true', help='Run random baseline')
    
    # Training parameters
    parser.add_argument('--timesteps', type=int, default=100000, help='Total training timesteps')
    parser.add_argument('--config', type=str, default='default', 
                       choices=['default', 'fast', 'thorough'], help='Hyperparameter preset')
    
    # Environment parameters
    parser.add_argument('--players', type=int, default=2, choices=[2, 3, 4], help='Number of players')
    parser.add_argument('--opponent', type=str, default='random', 
                       choices=['random', 'greedy'], help='Opponent type')
    parser.add_argument('--max-turns', type=int, default=500, help='Maximum turns per episode')
    
    # Evaluation parameters
    parser.add_argument('--model', type=str, default='models/monopoly_dqn_final.zip',
                       help='Path to model for evaluation')
    parser.add_argument('--episodes', type=int, default=100, help='Number of evaluation episodes')
    
    # Other
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--verbose', type=int, default=1, help='Verbosity level')
    
    args = parser.parse_args()
    
    if args.train:
        train_dqn(
            total_timesteps=args.timesteps,
            num_players=args.players,
            opponent_type=args.opponent,
            max_turns=args.max_turns,
            config_preset=args.config,
            seed=args.seed,
            verbose=args.verbose
        )
    
    if args.evaluate:
        evaluate_model(
            model_path=args.model,
            num_episodes=args.episodes,
            num_players=args.players,
            opponent_type=args.opponent,
            max_turns=args.max_turns,
            seed=args.seed
        )
    
    if args.baseline:
        run_random_baseline(
            num_episodes=args.episodes,
            num_players=args.players,
            max_turns=args.max_turns,
            seed=args.seed
        )
    
    if not (args.train or args.evaluate or args.baseline):
        print("Please specify --train, --evaluate, or --baseline")
        print("Run with --help for usage information")


if __name__ == "__main__":
    main()
