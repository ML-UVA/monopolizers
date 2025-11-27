#!/usr/bin/env python3
"""
Stable-Baselines3 DQN Training Script for Monopoly

This script provides a complete training pipeline for training a DQN agent
to play Monopoly using the Stable-Baselines3 library.

Four-Player Mode:
- Player 0: RL Agent (DQN - generates training data)
- Player 1: Random Bot
- Player 2: MCTS Bot
- Player 3: Greedy Bot

Reward System (Net Worth Based):
- No reward for winning/losing
- Reward = agent_net_worth / sum(other_active_players_net_worth)
- Net worth = cash + sum(property_values)
- Property value = (base_price - mortgage_value) * bonus + houses * house_cost + hotels * hotel_cost
- Bonus = 1.5 (no monopoly) or 2.0 (has monopoly)
- Mortgaged properties have value 0

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
from Monopoly.envs.wrappers import MonopolyFlattenWrapper
from Monopoly.agents.random import RandomAgent
from Monopoly.agents.greedy import GreedyAgent
from Monopoly.agents.mcts import MCTSAgent
from Monopoly.engine import GameEngine
from Monopoly.rules import RulesEngine
from Monopoly.board import Board
from Monopoly.property import load_property_specs
from Monopoly.cards import load_chance_cards, load_community_cards


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
        self.episode_net_worths = []
        
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
                if 'agent_net_worth' in info:
                    self.episode_net_worths.append(info['agent_net_worth'])
                    self.logger.record('game/agent_net_worth', info['agent_net_worth'])
                if 'agent_cash' in info:
                    self.logger.record('game/agent_cash', info['agent_cash'])
                if 'agent_properties' in info:
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

def create_opponent_agents(seed: int = 42) -> list:
    """
    Create the three opponent agents for 4-player mode:
    - Player 1: Random Bot
    - Player 2: MCTS Bot (with lightweight engine for simulations)
    - Player 3: Greedy Bot
    
    Args:
        seed: Random seed for reproducibility
        
    Returns:
        List of opponent agents
    """
    # Create a lightweight engine for MCTS simulations
    board = Board.load_standard_board()
    property_specs = load_property_specs()
    chance_cards = load_chance_cards()
    community_cards = load_community_cards()
    rules_engine = RulesEngine(board, property_specs, chance_cards, community_cards)
    mcts_engine = GameEngine(rules_engine, seed=seed)
    
    opponents = [
        RandomAgent(player_id=1),
        MCTSAgent(player_id=2, engine=mcts_engine, rollouts=5, max_depth=5),  # Lightweight MCTS
        GreedyAgent(player_id=3),
    ]
    
    return opponents


def make_monopoly_env(
    num_players: int = 4,
    agent_player_id: int = 0,
    max_turns: int = 500,
    seed: Optional[int] = None,
    flatten: bool = True,
) -> gym.Env:
    """
    Create a Monopoly environment with 4 players:
    - Player 0: RL Agent
    - Player 1: Random Bot
    - Player 2: MCTS Bot
    - Player 3: Greedy Bot
    
    Uses net worth-based reward system (no win/lose rewards).
    
    Args:
        num_players: Number of players (always 4 for this setup)
        agent_player_id: Player ID for the RL agent (always 0)
        max_turns: Maximum turns before truncation
        seed: Random seed
        flatten: Whether to flatten observations
        
    Returns:
        Gymnasium environment
    """
    # Create opponent agents
    opponents = create_opponent_agents(seed=seed or 42)
    
    # Create base environment
    env = MonopolyEnv(
        num_players=4,
        agent_player_id=0,
        opponent_policies=opponents,
        max_turns=max_turns,
        seed=seed
    )
    
    # Apply flatten wrapper (no reward shaping - using net worth reward)
    if flatten:
        env = MonopolyFlattenWrapper(env)
    
    # Add monitor for logging
    env = Monitor(env)
    
    return env


def make_vec_env(
    n_envs: int = 1,
    max_turns: int = 500,
    seed: Optional[int] = None
) -> VecMonitor:
    """
    Create a vectorized environment for parallel training.
    
    Args:
        n_envs: Number of parallel environments
        max_turns: Maximum turns
        seed: Base random seed
        
    Returns:
        Vectorized environment
    """
    def make_env(env_id: int) -> Callable[[], gym.Env]:
        def _init() -> gym.Env:
            env_seed = seed + env_id if seed is not None else None
            return make_monopoly_env(
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
    Train a DQN agent to play Monopoly in 4-player mode.
    
    Players:
    - Player 0: RL Agent (DQN)
    - Player 1: Random Bot
    - Player 2: MCTS Bot
    - Player 3: Greedy Bot
    
    Reward: Net worth relative to other players (no win/lose bonus)
    
    Args:
        total_timesteps: Total training timesteps
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
    print("Monopoly DQN Training (4-Player Mode)")
    print("=" * 60)
    print(f"Total timesteps: {total_timesteps}")
    print(f"Players: 4 (RL Agent vs Random, MCTS, Greedy)")
    print(f"Reward: Net worth relative to opponents")
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
        max_turns=max_turns,
        seed=seed
    )
    
    # Create evaluation environment
    print("Creating evaluation environment...")
    eval_env = make_vec_env(
        n_envs=1,
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
    max_turns: int = 500,
    seed: int = 42,
    render: bool = False,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Evaluate a trained DQN model in 4-player mode.
    
    Args:
        model_path: Path to the saved model
        num_episodes: Number of evaluation episodes
        max_turns: Maximum turns per episode
        seed: Random seed
        render: Whether to render games
        verbose: Print detailed results
        
    Returns:
        Dictionary of evaluation metrics
    """
    print("=" * 60)
    print("Evaluating Monopoly DQN Agent (4-Player Mode)")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Episodes: {num_episodes}")
    print(f"Opponents: Random, MCTS, Greedy")
    print("=" * 60)
    
    # Load model
    model = DQN.load(model_path)
    
    # Create evaluation environment
    env = make_monopoly_env(
        max_turns=max_turns,
        seed=seed,
        flatten=True,
    )
    
    # Run evaluation
    episode_rewards = []
    episode_lengths = []
    wins = 0
    final_net_worths = []
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
        final_net_worths.append(info.get('agent_net_worth', 0))
        
        # Check if agent won
        if info.get('agent_status') == 'ACTIVE':
            active_count = info.get('active_players', 1)
            if active_count == 1:
                wins += 1
        
        if verbose and (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{num_episodes}: "
                  f"Reward={episode_reward:.2f}, "
                  f"Length={episode_length}, "
                  f"NW=${info.get('agent_net_worth', 0):.0f}")
    
    env.close()
    
    # Calculate metrics
    results = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths),
        'win_rate': wins / num_episodes,
        'mean_final_net_worth': np.mean(final_net_worths),
        'mean_final_cash': np.mean(final_cash),
        'mean_final_properties': np.mean(final_properties),
    }
    
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"Mean Episode Length: {results['mean_length']:.1f} ± {results['std_length']:.1f}")
    print(f"Win Rate: {results['win_rate'] * 100:.1f}%")
    print(f"Mean Final Net Worth: ${results['mean_final_net_worth']:.0f}")
    print(f"Mean Final Cash: ${results['mean_final_cash']:.0f}")
    print(f"Mean Final Properties: {results['mean_final_properties']:.1f}")
    print("=" * 60)
    
    return results


# ============================================================================
# Random Baseline
# ============================================================================

def run_random_baseline(
    num_episodes: int = 100,
    max_turns: int = 500,
    seed: int = 42
) -> Dict[str, float]:
    """
    Run random agent baseline for comparison in 4-player mode.
    """
    print("=" * 60)
    print("Running Random Agent Baseline (4-Player Mode)")
    print("=" * 60)
    
    env = make_monopoly_env(
        max_turns=max_turns,
        seed=seed,
        flatten=True,
    )
    
    episode_rewards = []
    episode_lengths = []
    wins = 0
    final_net_worths = []
    
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
        final_net_worths.append(info.get('agent_net_worth', 0))
        
        if info.get('agent_status') == 'ACTIVE' and info.get('active_players', 1) == 1:
            wins += 1
    
    env.close()
    
    results = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'win_rate': wins / num_episodes,
        'mean_final_net_worth': np.mean(final_net_worths),
    }
    
    print(f"Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"Mean Episode Length: {results['mean_length']:.1f}")
    print(f"Win Rate: {results['win_rate'] * 100:.1f}%")
    print(f"Mean Final Net Worth: ${results['mean_final_net_worth']:.0f}")
    print("=" * 60)
    
    return results


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train or evaluate a DQN agent for Monopoly (4-Player Mode)')
    
    # Mode
    parser.add_argument('--train', action='store_true', help='Train a new model')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate a trained model')
    parser.add_argument('--baseline', action='store_true', help='Run random baseline')
    
    # Training parameters
    parser.add_argument('--timesteps', type=int, default=100000, help='Total training timesteps')
    parser.add_argument('--config', type=str, default='default', 
                       choices=['default', 'fast', 'thorough'], help='Hyperparameter preset')
    
    # Environment parameters
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
            max_turns=args.max_turns,
            config_preset=args.config,
            seed=args.seed,
            verbose=args.verbose
        )
    
    if args.evaluate:
        evaluate_model(
            model_path=args.model,
            num_episodes=args.episodes,
            max_turns=args.max_turns,
            seed=args.seed
        )
    
    if args.baseline:
        run_random_baseline(
            num_episodes=args.episodes,
            max_turns=args.max_turns,
            seed=args.seed
        )
    
    if not (args.train or args.evaluate or args.baseline):
        print("Please specify --train, --evaluate, or --baseline")
        print("Run with --help for usage information")
        print("\n4-Player Mode:")
        print("  Player 0: RL Agent (DQN)")
        print("  Player 1: Random Bot")
        print("  Player 2: MCTS Bot")
        print("  Player 3: Greedy Bot")
        print("\nReward: Net worth relative to other active players")


if __name__ == "__main__":
    main()
