#!/usr/bin/env python3
"""
Monopoly RL Training Script - DQN and DDQN-Hybrid Modes

RESEARCH FOCUS (H1): Comparing reward shaping strategies for long-horizon games.
- 'dense_networth': r = nw_agent / sum(nw_others) every step
- 'sparse_terminal': +1 win, -1 lose, 0 otherwise

This script provides training pipelines for both:
- SB3 DQN (Stable-Baselines3 implementation)
- DDQN-Hybrid (Custom PyTorch Double-DQN with action masking)

Four-Player Mode:
- Player 0: RL Agent (DQN/DDQN - generates training data)
- Player 1: Random Bot
- Player 2: MCTS Bot
- Player 3: Greedy Bot

Example Usage:
    # Train with dense net-worth reward (default)
    python train_dqn.py --train --agent dqn --reward_mode dense_networth --seed 42
    
    # Train with sparse terminal reward (ablation)
    python train_dqn.py --train --agent dqn --reward_mode sparse_terminal --seed 42

    # Train with DDQN-Hybrid
    python train_dqn.py --train --agent ddqn_hybrid --reward_mode dense_networth --seed 42

    # Continue DDQN-Hybrid training from a .pt checkpoint
    python train_dqn.py --train --agent ddqn_hybrid --resume_checkpoint runs/ddqn_hybrid_dense_networth_seed42/models/ddqn_hybrid/final.pt --total_timesteps 2000000

    # Evaluate a trained model
    python train_dqn.py --evaluate --model runs/dqn_seed42/models/dqn/monopoly_dqn_final.zip

    # Evaluate DDQN-Hybrid .pt checkpoint
    python train_dqn.py --evaluate --agent ddqn_hybrid --model runs/ddqn_hybrid_dense_networth_seed42/models/ddqn_hybrid/final.pt

    # Run random baseline
    python train_dqn.py --baseline --episodes 100

Requirements:
    pip install stable-baselines3 tensorboard gymnasium numpy torch
"""

import sys
import os
import argparse
import time
import random
from pathlib import Path
from typing import Optional, Dict, Any, Callable
from datetime import datetime
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Gymnasium
import gymnasium as gym

# PyTorch (for seeding)
import torch

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
from Monopoly.envs.gym_env import MonopolyEnv, RewardMode
from Monopoly.envs.wrappers import MonopolyFlattenWrapper
from Monopoly.agents.random import RandomAgent
from Monopoly.agents.greedy import GreedyAgent
from Monopoly.agents.mcts import MCTSAgent
from Monopoly.engine import GameEngine
from Monopoly.rules import RulesEngine
from Monopoly.board import Board
from Monopoly.property import load_property_specs
from Monopoly.cards import load_chance_cards, load_community_cards

# Utilities
from utils.save_utils import (
    save_metrics_csv,
    append_metrics_csv,
    ensure_dir,
    get_run_id,
    CSV_COLUMNS,
)


# ============================================================================
# Seeding Utilities
# ============================================================================

def set_global_seeds(seed: int) -> None:
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ============================================================================
# Custom Callbacks for SB3 DQN
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


class MetricsCSVCallback(BaseCallback):
    """
    Callback that logs episode metrics to CSV file.
    """
    def __init__(
        self, 
        csv_path: str,
        run_id: str,
        seed: int,
        verbose: int = 0
    ):
        super().__init__(verbose)
        self.csv_path = Path(csv_path)
        self.run_id = run_id
        self.seed = seed
        self.episode_count = 0
        
    def _on_step(self) -> bool:
        if 'infos' in self.locals:
            for info in self.locals['infos']:
                if 'episode' in info:
                    self.episode_count += 1
                    
                    metric = {
                        'timestamp': datetime.now().isoformat(),
                        'episode': self.episode_count,
                        'episode_length': info['episode']['l'],
                        'final_net_worth': info.get('agent_net_worth', 0),
                        'agent_rank': 0,
                        'win_flag': 1 if (info.get('agent_status') == 'ACTIVE' and 
                                         info.get('active_players', 1) == 1) else 0,
                        'total_steps': self.num_timesteps,
                        'mean_episode_reward': info['episode']['r'],
                        'eval_tag': 'train',
                    }
                    
                    append_metrics_csv(self.csv_path, metric, self.run_id, self.seed)
        
        return True


# ============================================================================
# Environment Factory Functions
# ============================================================================

def create_opponent_agents(seed: int = 42) -> list:
    """
    Create the three opponent agents for 4-player mode:
    - Player 1: Random Bot (seeded)
    - Player 2: MCTS Bot (seeded, with lightweight engine for simulations)
    - Player 3: Greedy Bot (deterministic)
    
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
    
    # Use derived seeds for each opponent for reproducibility
    opponents = [
        RandomAgent(player_id=1, seed=seed + 100),
        MCTSAgent(player_id=2, engine=mcts_engine, rollouts=5, max_depth=5, seed=seed + 200),
        GreedyAgent(player_id=3),  # Deterministic, no seed needed
    ]
    
    return opponents


def make_monopoly_env(
    num_players: int = 4,
    agent_player_id: int = 0,
    max_turns: int = 500,
    seed: Optional[int] = None,
    flatten: bool = True,
    reward_mode: RewardMode = 'dense_networth',
    trace_dir: Optional[str] = None,
    render_mode: Optional[str] = None,
) -> gym.Env:
    """
    Create a Monopoly environment with 4 players:
    - Player 0: RL Agent
    - Player 1: Random Bot
    - Player 2: MCTS Bot
    - Player 3: Greedy Bot

    Args:
        num_players: Number of players (always 4 for this setup)
        agent_player_id: Player ID for the RL agent (always 0)
        max_turns: Maximum turns before truncation
        seed: Random seed
        flatten: Whether to flatten observations
        reward_mode: Reward strategy (H1 ablation)
            - 'dense_networth': r = nw_agent / sum(nw_others) every step
            - 'sparse_terminal': +1 win, -1 lose, 0 otherwise
        trace_dir: If set, wraps env with EpisodeTracer writing to this directory
        render_mode: Render mode ('human', 'ansi', or None to disable)

    Returns:
        Gymnasium environment
    """
    # Create opponent agents with derived seeds
    opponents = create_opponent_agents(seed=seed or 42)

    # Create base environment with configurable reward mode
    env = MonopolyEnv(
        num_players=4,
        agent_player_id=0,
        opponent_policies=opponents,
        max_turns=max_turns,
        seed=seed,
        reward_mode=reward_mode,
        render_mode=render_mode,
    )

    # Optional episode tracing (must go before flatten)
    if trace_dir:
        from Monopoly.envs.tracing import EpisodeTracer
        env = EpisodeTracer(env, output_dir=trace_dir)

    # Apply flatten wrapper
    if flatten:
        env = MonopolyFlattenWrapper(env)

    # Add monitor for logging
    env = Monitor(env)

    return env


def make_vec_env(
    n_envs: int = 1,
    max_turns: int = 500,
    seed: Optional[int] = None,
    reward_mode: RewardMode = 'dense_networth',
    render_mode: Optional[str] = None,
) -> VecMonitor:
    """
    Create a vectorized environment for parallel training.
    
    Args:
        n_envs: Number of parallel environments
        max_turns: Maximum turns
        seed: Base random seed
        reward_mode: Reward strategy (H1 ablation)
        render_mode: Render mode ('human', 'ansi', or None to disable)
        
    Returns:
        Vectorized environment
    """
    def make_env(env_id: int) -> Callable[[], gym.Env]:
        def _init() -> gym.Env:
            env_seed = seed + env_id if seed is not None else None
            return make_monopoly_env(
                max_turns=max_turns,
                seed=env_seed,
                reward_mode=reward_mode,
                render_mode=render_mode,
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
# SB3 DQN Training Function
# ============================================================================

def train_dqn(
    total_timesteps: int = 100000,
    max_turns: int = 500,
    config_preset: str = 'default',
    output_dir: str = 'runs/dqn',
    seed: int = 42,
    eval_interval: int = 5000,
    n_eval_episodes: int = 10,
    verbose: int = 1,
    reward_mode: RewardMode = 'dense_networth',
    render: bool = False,
) -> DQN:
    """
    Train a DQN agent using Stable-Baselines3.
    
    Players:
    - Player 0: RL Agent (DQN)
    - Player 1: Random Bot
    - Player 2: MCTS Bot
    - Player 3: Greedy Bot
    
    Args:
        total_timesteps: Total training timesteps
        max_turns: Maximum turns per episode
        config_preset: Hyperparameter preset
        output_dir: Output directory for models/logs/metrics
        seed: Random seed
        eval_interval: Evaluation frequency
        n_eval_episodes: Number of evaluation episodes
        verbose: Verbosity level
        reward_mode: Reward strategy (H1 ablation)
            - 'dense_networth': r = nw_agent / sum(nw_others) every step
            - 'sparse_terminal': +1 win, -1 lose, 0 otherwise
        render: Whether to enable visualization (slows training)
        
    Returns:
        Trained DQN model
    """
    # Set seeds
    set_global_seeds(seed)
    
    # Setup output directories (include reward_mode in path)
    output_dir = Path(output_dir)
    models_dir = output_dir / 'models' / 'dqn'
    results_dir = output_dir / 'results'
    tensorboard_dir = output_dir / 'tensorboard' / 'dqn'
    
    ensure_dir(models_dir)
    ensure_dir(results_dir)
    ensure_dir(tensorboard_dir)
    
    # Generate run ID
    run_id = get_run_id('dqn', seed)
    csv_path = results_dir / 'dqn_metrics.csv'
    
    # Determine render mode
    render_mode = 'human' if render else None
    
    print("=" * 60)
    print("Monopoly DQN Training (4-Player Mode)")
    print("=" * 60)
    print(f"Total timesteps: {total_timesteps}")
    print(f"Players: 4 (RL Agent vs Random, MCTS, Greedy)")
    print(f"Reward mode: {reward_mode}")
    print(f"Config preset: {config_preset}")
    print(f"Seed: {seed}")
    print(f"Render: {render}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
    
    # Create training environment with configured reward mode
    print(f"\nCreating training environment (reward_mode={reward_mode})...")
    train_env = make_vec_env(
        n_envs=1,
        max_turns=max_turns,
        seed=seed,
        reward_mode=reward_mode,
        render_mode=render_mode,
    )
    
    # Create evaluation environment with same reward mode (no render for eval)
    print("Creating evaluation environment...")
    eval_env = make_vec_env(
        n_envs=1,
        max_turns=max_turns,
        seed=seed + 1000,
        reward_mode=reward_mode,
        render_mode=None,  # Eval env never renders
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
        tensorboard_log=str(tensorboard_dir),
        device='auto',
        **config
    )
    
    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=str(models_dir),
        name_prefix="monopoly_dqn"
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(models_dir),
        log_path=str(output_dir / 'logs'),
        eval_freq=eval_interval,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
        verbose=verbose
    )
    
    tensorboard_callback = TensorboardCallback(verbose=verbose)
    
    metrics_callback = MetricsCSVCallback(
        csv_path=str(csv_path),
        run_id=run_id,
        seed=seed,
        verbose=verbose
    )
    
    callback_list = CallbackList([
        checkpoint_callback,
        eval_callback,
        tensorboard_callback,
        metrics_callback
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
    final_path = models_dir / "monopoly_dqn_final"
    model.save(str(final_path))
    print(f"\nFinal model saved to: {final_path}.zip")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Model saved to: {models_dir}")
    print(f"Metrics saved to: {csv_path}")
    print(f"TensorBoard logs: {tensorboard_dir}")
    print("=" * 60)
    
    # Cleanup
    train_env.close()
    eval_env.close()
    
    return model


# ============================================================================
# DDQN-Hybrid Training Function
# ============================================================================

def train_ddqn_hybrid(
    total_timesteps: int = 100000,
    max_turns: int = 500,
    output_dir: str = 'runs/ddqn_hybrid',
    seed: int = 42,
    eval_interval: int = 1000000,
    eval_episodes: int = 100,
    verbose: int = 1,
    config: Optional[Dict[str, Any]] = None,
    reward_mode: RewardMode = 'dense_networth',
    render: bool = False,
    resume_checkpoint: Optional[str] = None,
    resume_replay_buffer: Optional[str] = None,
    auto_resume_buffer: bool = True,
) -> None:
    """
    Train using custom DDQN-Hybrid trainer with PyTorch.
    
    Uses the same environment and action masking as the SB3 DQN path,
    but with a custom Double-DQN implementation.
    
    Args:
        total_timesteps: Total training timesteps
        max_turns: Maximum turns per episode
        output_dir: Output directory for models/logs/metrics
        seed: Random seed
        eval_interval: Steps between evaluations
        eval_episodes: Number of evaluation episodes
        verbose: Verbosity level
        config: Optional hyperparameter overrides
        reward_mode: Reward strategy (H1 ablation)
            - 'dense_networth': r = nw_agent / sum(nw_others) every step
            - 'sparse_terminal': +1 win, -1 lose, 0 otherwise
        render: Whether to enable visualization (slows training)
        resume_checkpoint: Optional path to DDQN .pt checkpoint to resume from
        resume_replay_buffer: Optional path to replay buffer .pkl to restore
        auto_resume_buffer: When resuming, auto-load output_dir buffer if present
    """
    # Set seeds
    set_global_seeds(seed)
    
    # Import DDQN trainer
    from Monopoly.agents.ddqn_hybrid import DDQNHybridTrainer
    
    # Determine render mode
    render_mode = 'human' if render else None
    
    print("=" * 60)
    print("Monopoly DDQN-Hybrid Training (4-Player Mode)")
    print("=" * 60)
    print(f"Total timesteps: {total_timesteps}")
    print(f"Reward mode: {reward_mode}")
    print(f"Seed: {seed}")
    print(f"Render: {render}")
    print("=" * 60)
    
    # Create environments with configured reward mode
    print(f"\nCreating training environment (reward_mode={reward_mode})...")
    train_env = make_monopoly_env(
        max_turns=max_turns,
        seed=seed,
        flatten=True,
        reward_mode=reward_mode,
        render_mode=render_mode,
    )
    
    print("Creating evaluation environment...")
    eval_env = make_monopoly_env(
        max_turns=max_turns,
        seed=seed + 1000,
        flatten=True,
        reward_mode=reward_mode,
        render_mode=None,  # Eval env never renders
    )
    
    # Create trainer
    trainer = DDQNHybridTrainer(
        env=train_env,
        eval_env=eval_env,
        output_dir=output_dir,
        seed=seed,
        device='auto',
        config=config,
        verbose=verbose
    )

    # Optional resume from checkpoint
    if resume_checkpoint:
        checkpoint_path = Path(resume_checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")

        print(f"\nResuming from checkpoint: {checkpoint_path}")
        trainer.load_checkpoint(str(checkpoint_path))
        print(
            f"Loaded trainer state | total_steps={trainer.total_steps} | "
            f"episodes={trainer.episodes_completed}"
        )

        if resume_replay_buffer:
            replay_path = Path(resume_replay_buffer)
            if not replay_path.exists():
                raise FileNotFoundError(f"Replay buffer not found: {replay_path}")
            trainer.load_replay_buffer(str(replay_path))
            print(f"Loaded replay buffer from: {replay_path}")
        elif auto_resume_buffer:
            default_replay_path = Path(output_dir) / 'buffers' / 'ddqn_hybrid_replay.pkl'
            if default_replay_path.exists():
                trainer.load_replay_buffer(str(default_replay_path))
                print(f"Auto-loaded replay buffer from: {default_replay_path}")
            else:
                print(f"No replay buffer found at {default_replay_path}; continuing with empty buffer.")

        # Resume semantics: run for additional timesteps from checkpoint step count.
        target_total_timesteps = trainer.total_steps + total_timesteps
        print(
            f"Resume mode: running {total_timesteps} additional steps "
            f"to reach total_steps={target_total_timesteps}."
        )
    else:
        target_total_timesteps = total_timesteps
    
    # Train
    results = trainer.train(
        total_timesteps=target_total_timesteps,
        eval_interval=eval_interval,
        eval_episodes=min(eval_episodes, 50),
        log_interval=1000,
        save_interval=50000,
        progress_bar=True
    )
    
    # Cleanup
    trainer.close()
    
    return results


# ============================================================================
# Evaluation Function
# ============================================================================

def evaluate_model(
    model_path: str,
    num_episodes: int = 100,
    max_turns: int = 500,
    seed: int = 42,
    render: bool = False,
    verbose: bool = True,
    trace_dir: Optional[str] = None,
    agent: str = 'dqn',
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
        agent: Model family ('dqn' or 'ddqn_hybrid')
        
    Returns:
        Dictionary of evaluation metrics
    """
    set_global_seeds(seed)
    
    print("=" * 60)
    print("Evaluating Monopoly RL Agent (4-Player Mode)")
    print("=" * 60)
    print(f"Model: {model_path}")
    resolved_agent = agent
    model_suffix = Path(model_path).suffix.lower()
    if resolved_agent == 'dqn' and model_suffix == '.pt':
        resolved_agent = 'ddqn_hybrid'
        print("Agent type auto-detected from .pt model: ddqn_hybrid")
    else:
        print(f"Agent type: {resolved_agent}")
    print(f"Episodes: {num_episodes}")
    print(f"Opponents: Random, MCTS, Greedy")
    print("=" * 60)
    
    # Load model
    if resolved_agent == 'dqn':
        model = DQN.load(model_path)
        model_type = 'sb3'
    elif resolved_agent == 'ddqn_hybrid':
        from utils.trace_utils import load_ddqn_for_eval
        model = load_ddqn_for_eval(model_path)
        model_type = 'ddqn'
    else:
        raise ValueError(f"Unsupported agent type for evaluation: {resolved_agent}")
    
    # Create evaluation environment
    env = make_monopoly_env(
        max_turns=max_turns,
        seed=seed,
        flatten=True,
        trace_dir=trace_dir,
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
            if model_type == 'sb3':
                action, _ = model.predict(obs, deterministic=True)
                action = int(action)
            else:
                legal_mask = env.unwrapped._get_legal_mask()
                action = model.select_action(obs, action_mask=legal_mask, epsilon=0.0)
            
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
    set_global_seeds(seed)
    
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
    parser = argparse.ArgumentParser(
        description='Train or evaluate DQN/DDQN agents for Monopoly (4-Player Mode)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train with SB3 DQN
    python train_dqn.py --agent dqn --total_timesteps 2000000 --output_dir runs/dqn_seed42
    
    # Train with DDQN-Hybrid
    python train_dqn.py --agent ddqn_hybrid --total_timesteps 2000000 --output_dir runs/ddqn_hybrid_seed42

    # Resume DDQN-Hybrid from a checkpoint for additional timesteps
    python train_dqn.py --train --agent ddqn_hybrid --resume_checkpoint runs/ddqn_hybrid_seed42/models/ddqn_hybrid/final.pt --total_timesteps 500000
    
    # Evaluate a model
    python train_dqn.py --evaluate --model runs/dqn_seed42/models/dqn/monopoly_dqn_final.zip

    # Evaluate DDQN-Hybrid .pt checkpoint
    python train_dqn.py --evaluate --agent ddqn_hybrid --model runs/ddqn_hybrid_seed42/models/ddqn_hybrid/final.pt
    
    # Run baseline
    python train_dqn.py --baseline --episodes 100
"""
    )
    
    # Mode selection
    parser.add_argument('--train', action='store_true', help='Train a new model')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate a trained model')
    parser.add_argument('--baseline', action='store_true', help='Run random baseline')
    
    # Agent selection
    parser.add_argument('--agent', type=str, default='dqn', 
                       choices=['dqn', 'ddqn_hybrid'],
                       help='Agent type: dqn (SB3) or ddqn_hybrid (custom PyTorch)')
    
    # Training parameters
    parser.add_argument('--total_timesteps', type=int, default=100000, 
                       help='Total training timesteps')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--eval_interval', type=int, default=1000000,
                       help='Steps between evaluations')
    parser.add_argument('--output_dir', type=str, default='runs/',
                       help='Output directory for models/logs/metrics')

    # DDQN resume options
    parser.add_argument('--resume_checkpoint', type=str, default=None,
                       help='Path to DDQN .pt checkpoint for continuing training')
    parser.add_argument('--resume_replay_buffer', type=str, default=None,
                       help='Path to DDQN replay buffer .pkl file for resume')
    parser.add_argument('--no_resume_buffer_autoload', action='store_true',
                       help='Disable auto-loading output_dir/buffers/ddqn_hybrid_replay.pkl when resuming DDQN')
    parser.add_argument('--ddqn_log_diagnostics', action='store_true',
                       help='Enable DDQN Q/TD diagnostic TensorBoard metrics (disabled by default)')
    
    # DQN-specific
    parser.add_argument('--config', type=str, default='default', 
                       choices=['default', 'fast', 'thorough'],
                       help='Hyperparameter preset (DQN only)')
    
    # Environment parameters
    parser.add_argument('--max_turns', type=int, default=500, 
                       help='Maximum turns per episode')
    
    # H1 Ablation: Reward mode
    parser.add_argument('--reward_mode', type=str, default='dense_networth',
                       choices=['dense_networth', 'sparse_terminal', 'modular'],
                       help='Reward strategy: dense_networth (shaped), sparse_terminal, or modular (component-based)')

    # YAML config file for DDQN-Hybrid
    parser.add_argument('--config_file', type=str, default=None,
                       help='Path to YAML config file for DDQN-Hybrid (CLI args override YAML values)')
    
    # Evaluation parameters
    parser.add_argument('--model', type=str, default='models/monopoly_dqn_final.zip',
                       help='Path to model for evaluation (.zip for dqn, .pt for ddqn_hybrid)')
    parser.add_argument('--episodes', type=int, default=100, 
                       help='Number of evaluation episodes')
    
    # Tracing
    parser.add_argument('--trace_eval', action='store_true',
                        help='Enable episode tracing during evaluation (writes to output_dir/analysis/traces)')

    # Rendering
    parser.add_argument('--no-render', action='store_true', dest='no_render',
                        help='Disable visualization during training/evaluation (default: rendering disabled)')
    parser.add_argument('--render', action='store_true',
                        help='Enable visualization during training/evaluation (slows down training)')

    # Other
    parser.add_argument('--verbose', type=int, default=1, help='Verbosity level')
    
    args = parser.parse_args()
    
    # Resolve render flag (--render enables, --no-render disables, default is disabled)
    enable_render = args.render and not args.no_render
    
    # Determine output directory based on agent type and reward mode
    if args.output_dir == 'runs/':
        args.output_dir = f'runs/{args.agent}_{args.reward_mode}_seed{args.seed}'
    
    if args.train:
        if args.agent == 'dqn':
            train_dqn(
                total_timesteps=args.total_timesteps,
                max_turns=args.max_turns,
                config_preset=args.config,
                output_dir=args.output_dir,
                seed=args.seed,
                eval_interval=args.eval_interval,
                n_eval_episodes=10,
                reward_mode=args.reward_mode,
                verbose=args.verbose,
                render=enable_render,
            )
        elif args.agent == 'ddqn_hybrid':
            # Build config overrides from YAML + CLI
            ddqn_config = None
            if args.config_file:
                from Monopoly.agents.ddqn_hybrid import load_yaml_config
                ddqn_config = load_yaml_config(args.config_file)

            if ddqn_config is None:
                ddqn_config = {}
            if args.ddqn_log_diagnostics:
                ddqn_config['log_training_diagnostics'] = True

            train_ddqn_hybrid(
                total_timesteps=args.total_timesteps,
                max_turns=args.max_turns,
                output_dir=args.output_dir,
                seed=args.seed,
                eval_interval=args.eval_interval,
                eval_episodes=100,
                reward_mode=args.reward_mode,
                verbose=args.verbose,
                config=ddqn_config,
                render=enable_render,
                resume_checkpoint=args.resume_checkpoint,
                resume_replay_buffer=args.resume_replay_buffer,
                auto_resume_buffer=not args.no_resume_buffer_autoload,
            )
    
    if args.evaluate:
        trace_dir = None
        if args.trace_eval:
            trace_dir = str(Path(args.output_dir) / 'analysis' / 'traces')
        evaluate_model(
            model_path=args.model,
            num_episodes=args.episodes,
            max_turns=args.max_turns,
            seed=args.seed,
            trace_dir=trace_dir,
            render=enable_render,
            agent=args.agent,
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
        print("  Player 0: RL Agent (DQN/DDQN)")
        print("  Player 1: Random Bot")
        print("  Player 2: MCTS Bot")
        print("  Player 3: Greedy Bot")
        print("\nH1 Ablation - Reward Modes:")
        print("  --reward_mode dense_networth  : Shaped reward (default)")
        print("  --reward_mode sparse_terminal : Sparse +1/-1 terminal reward")
        print("\nAgent types:")
        print("  --agent dqn         : Stable-Baselines3 DQN")
        print("  --agent ddqn_hybrid : Custom PyTorch Double-DQN")


if __name__ == "__main__":
    main()
