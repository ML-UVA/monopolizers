"""
Double DQN Hybrid Trainer for Monopoly.

This module implements a custom Double-DQN training loop using PyTorch,
designed to work with the same environment wrapper, action masking, and
reward function as the SB3 DQN path.

Key Features:
- Double DQN update rule (reduces overestimation bias)
- Action masking for legal moves only
- Epsilon-greedy exploration with annealing
- Experience replay buffer with action masks
- Periodic target network updates
- TensorBoard logging
- CSV metrics export
"""

import os
import time
import random
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from .network import QNetwork


@dataclass
class Transition:
    """Single transition in replay buffer."""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    mask: np.ndarray  # Legal action mask at state
    next_mask: np.ndarray  # Legal action mask at next_state


class ReplayBuffer:
    """
    Experience replay buffer for DDQN training.
    
    Stores transitions including action masks for proper action selection.
    """
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
    
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        mask: np.ndarray,
        next_mask: np.ndarray
    ) -> None:
        """Add a transition to the buffer."""
        self.buffer.append(Transition(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            mask=mask,
            next_mask=next_mask
        ))
    
    def sample(self, batch_size: int) -> List[Transition]:
        """Sample a batch of transitions."""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def get_data_for_save(self) -> List[Dict]:
        """Get buffer data in a picklable format."""
        return [
            {
                'state': t.state,
                'action': t.action,
                'reward': t.reward,
                'next_state': t.next_state,
                'done': t.done,
                'mask': t.mask,
                'next_mask': t.next_mask
            }
            for t in self.buffer
        ]
    
    def load_data(self, data: List[Dict]) -> None:
        """Load buffer data from saved format."""
        self.buffer.clear()
        for d in data:
            self.buffer.append(Transition(**d))


class DDQNHybridTrainer:
    """
    Custom Double-DQN Trainer for Monopoly.
    
    Implements Double DQN update rule:
    - Uses online network to select best action
    - Uses target network to evaluate Q-value of that action
    - This reduces overestimation bias compared to standard DQN
    
    # Double-DQN target calculation (per minibatch)
    # a_max = argmax_a Q_online(next_states)[a]  # mask invalid actions by setting to -inf
    # target_q = rewards + (1 - dones) * gamma * Q_target(next_states)[range(batch), a_max]
    # loss = MSE(Q_online(states)[range(batch), actions], target_q.detach())
    
    Args:
        env: Gymnasium environment (should be wrapped with MonopolyFlattenWrapper)
        eval_env: Separate environment for evaluation
        output_dir: Directory for saving outputs
        seed: Random seed for reproducibility
        device: PyTorch device ('cuda' or 'cpu')
        
    Hyperparameters (defaults provided):
        gamma: Discount factor (0.999)
        lr: Learning rate (1e-4)
        batch_size: Training batch size (128)
        buffer_size: Replay buffer capacity (1_000_000)
        start_training_after: Steps before training begins (10_000)
        target_update: Steps between target network updates (5_000)
        train_freq: Steps between training updates (4)
        eps_start: Initial exploration rate (1.0)
        eps_end: Final exploration rate (0.01)
        eps_decay: Steps for epsilon annealing (500_000)
        hidden_dims: Network hidden layer sizes ([256, 256])
        grad_clip: Maximum gradient norm (10.0)
    """
    
    # Default hyperparameters (well-tuned for Monopoly)
    # References: DQN (Mnih et al., 2015), DDQN (van Hasselt et al., 2016)
    DEFAULT_CONFIG = {
        'gamma': 0.99,           # Discount factor (0.99 is standard)
        'lr': 1e-4,              # Learning rate (Adam)
        'batch_size': 64,        # Mini-batch size
        'buffer_size': 100_000,  # Replay buffer capacity
        'learning_starts': 1000, # Steps before training begins
        'target_update_interval': 1000,  # Steps between target updates
        'train_freq': 4,         # Steps between gradient updates
        'eps_start': 1.0,        # Initial exploration rate
        'eps_end': 0.05,         # Final exploration rate  
        'eps_decay': 100_000,    # Linear annealing steps
        'hidden_dims': [256, 256],  # MLP hidden layer sizes
        'grad_clip': 10.0,       # Max gradient norm
        'use_huber_loss': True,  # Huber loss for stability (vs MSE)
    }
    
    def __init__(
        self,
        env,
        eval_env=None,
        output_dir: str = 'runs/ddqn_hybrid',
        seed: int = 42,
        device: str = 'auto',
        config: Optional[Dict[str, Any]] = None,
        verbose: int = 1
    ):
        # Merge config with defaults
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        
        self.env = env
        self.eval_env = eval_env or env
        self.output_dir = Path(output_dir)
        self.seed = seed
        self.verbose = verbose
        
        # Set up device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Set random seeds for reproducibility
        self._set_seeds(seed)
        
        # Get dimensions from environment
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        
        # Initialize networks
        self.q_online = QNetwork(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_dims=self.config['hidden_dims']
        ).to(self.device)
        
        self.q_target = QNetwork(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_dims=self.config['hidden_dims']
        ).to(self.device)
        
        # Copy weights to target network
        self.q_target.load_state_dict(self.q_online.state_dict())
        self.q_target.eval()  # Target network is not trained directly
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.q_online.parameters(),
            lr=self.config['lr']
        )
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(self.config['buffer_size'])
        
        # Training state
        self.total_steps = 0
        self.episodes_completed = 0
        self.training_started = False
        
        # Metrics tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_net_worths = []
        self.losses = []
        
        # Create output directories
        self._setup_output_dirs()
        
        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=str(self.output_dir / 'tensorboard' / 'ddqn_hybrid'))
        
        # Run ID for CSV
        self.run_id = f"ddqn_hybrid_seed{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def _set_seeds(self, seed: int) -> None:
        """Set random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    
    def _setup_output_dirs(self) -> None:
        """Create output directory structure."""
        (self.output_dir / 'models' / 'ddqn_hybrid').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'buffers').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'results').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'tensorboard' / 'ddqn_hybrid').mkdir(parents=True, exist_ok=True)
    
    def _get_epsilon(self) -> float:
        """Calculate current epsilon for exploration."""
        # Linear annealing from eps_start to eps_end over eps_decay steps
        progress = min(1.0, self.total_steps / self.config['eps_decay'])
        epsilon = self.config['eps_start'] + progress * (self.config['eps_end'] - self.config['eps_start'])
        return epsilon
    
    def _get_action_mask(self, env) -> np.ndarray:
        """Get legal action mask from environment."""
        if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, '_get_legal_mask'):
            return env.unwrapped._get_legal_mask()
        elif hasattr(env, '_get_legal_mask'):
            return env._get_legal_mask()
        else:
            # Fallback: all actions legal
            return np.ones(self.action_dim, dtype=np.int32)
    
    def select_action(self, state: np.ndarray, mask: np.ndarray, epsilon: float = 0.0) -> int:
        """
        Select action using epsilon-greedy policy with action masking.
        
        Args:
            state: Current observation
            mask: Legal action mask
            epsilon: Exploration probability
            
        Returns:
            Selected action index
        """
        # Get legal actions
        legal_actions = np.where(mask == 1)[0]
        
        if len(legal_actions) == 0:
            return 0  # Fallback (should not happen)
        
        # Epsilon-greedy
        if random.random() < epsilon:
            # Random exploration among legal actions
            return int(np.random.choice(legal_actions))
        else:
            # Greedy action selection
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                mask_tensor = torch.tensor(mask, dtype=torch.float32, device=self.device).unsqueeze(0)
                
                q_values = self.q_online.get_q_values(state_tensor, mask_tensor)
                # q_values is now a numpy array, use axis instead of dim
                return int(q_values.argmax(axis=1).item())
    
    def train_step(self) -> Optional[float]:
        """
        Perform one training step (update networks from replay buffer).
        
        Returns:
            Loss value if training occurred, None otherwise
        """
        # Check if we have enough samples to start training
        min_samples = self.config.get('learning_starts', self.config.get('start_training_after', 1000))
        if len(self.replay_buffer) < min_samples:
            return None
        
        if len(self.replay_buffer) < self.config['batch_size']:
            return None
        
        # Sample batch
        batch = self.replay_buffer.sample(self.config['batch_size'])
        
        # Convert to tensors
        states = torch.tensor(
            np.array([t.state for t in batch]), 
            dtype=torch.float32, device=self.device
        )
        actions = torch.tensor(
            [t.action for t in batch], 
            dtype=torch.long, device=self.device
        )
        rewards = torch.tensor(
            [t.reward for t in batch], 
            dtype=torch.float32, device=self.device
        )
        next_states = torch.tensor(
            np.array([t.next_state for t in batch]), 
            dtype=torch.float32, device=self.device
        )
        dones = torch.tensor(
            [t.done for t in batch], 
            dtype=torch.float32, device=self.device
        )
        next_masks = torch.tensor(
            np.array([t.next_mask for t in batch]), 
            dtype=torch.float32, device=self.device
        )
        
        # ---------------------------------------------------------------------
        # Double-DQN target calculation (per minibatch)
        # a_max = argmax_a Q_online(next_states)[a]  # mask invalid actions by setting to -inf
        # target_q = rewards + (1 - dones) * gamma * Q_target(next_states)[range(batch), a_max]
        # loss = MSE(Q_online(states)[range(batch), actions], target_q.detach())
        # ---------------------------------------------------------------------
        
        with torch.no_grad():
            # Use online network to select best action (with masking)
            next_q_online = self.q_online(next_states)
            # Mask illegal actions for action selection
            next_q_online_masked = next_q_online.clone()
            next_q_online_masked[next_masks == 0] = float('-inf')
            a_max = next_q_online_masked.argmax(dim=1)
            
            # Use target network to evaluate Q-value of selected action
            next_q_target = self.q_target(next_states)
            target_q_values = next_q_target[torch.arange(len(batch)), a_max]
            
            # Compute TD target
            targets = rewards + (1 - dones) * self.config['gamma'] * target_q_values
        
        # Get current Q-values
        current_q = self.q_online(states)
        current_q_actions = current_q[torch.arange(len(batch)), actions]
        
        # Compute loss (Huber for stability, MSE as fallback)
        if self.config.get('use_huber_loss', True):
            # Huber loss is less sensitive to outliers than MSE
            loss = nn.functional.smooth_l1_loss(current_q_actions, targets)
        else:
            loss = nn.functional.mse_loss(current_q_actions, targets)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if self.config['grad_clip'] is not None:
            nn.utils.clip_grad_norm_(self.q_online.parameters(), self.config['grad_clip'])
        
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self) -> None:
        """Copy online network weights to target network."""
        self.q_target.load_state_dict(self.q_online.state_dict())
    
    def train(
        self,
        total_timesteps: int,
        eval_interval: int = 100000,
        eval_episodes: int = 100,
        log_interval: int = 1000,
        save_interval: int = 50000,
        progress_bar: bool = True
    ) -> Dict[str, Any]:
        """
        Main training loop.
        
        Args:
            total_timesteps: Total environment steps to train for
            eval_interval: Steps between evaluations
            eval_episodes: Number of episodes per evaluation
            log_interval: Steps between logging
            save_interval: Steps between model saves
            progress_bar: Whether to show progress bar
            
        Returns:
            Dictionary of training results
        """
        print("=" * 60)
        print("DDQN-Hybrid Training (4-Player Mode)")
        print("=" * 60)
        print(f"Total timesteps: {total_timesteps}")
        print(f"Device: {self.device}")
        print(f"Observation dim: {self.obs_dim}")
        print(f"Action dim: {self.action_dim}")
        print(f"Config: {self.config}")
        print("=" * 60)

        # Cap evaluation episodes to avoid excessive evaluation time
        eval_episodes = min(eval_episodes, 50)
        
        start_time = time.time()
        
        # Initialize environment
        obs, info = self.env.reset(seed=self.seed)
        mask = self._get_action_mask(self.env)
        
        episode_reward = 0.0
        episode_length = 0
        
        # Progress tracking
        if progress_bar:
            try:
                from tqdm import tqdm
                pbar = tqdm(total=total_timesteps, desc="Training")
            except ImportError:
                pbar = None
        else:
            pbar = None
        
        while self.total_steps < total_timesteps:
            # Get epsilon for exploration
            epsilon = self._get_epsilon()
            
            # Select action
            action = self.select_action(obs, mask, epsilon)
            
            # Take step
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            next_mask = self._get_action_mask(self.env)
            
            # Store transition
            self.replay_buffer.push(
                state=obs,
                action=action,
                reward=reward,
                next_state=next_obs,
                done=done,
                mask=mask,
                next_mask=next_mask
            )
            
            episode_reward += reward
            episode_length += 1
            self.total_steps += 1
            
            # Training update
            if self.total_steps % self.config['train_freq'] == 0:
                loss = self.train_step()
                if loss is not None:
                    self.losses.append(loss)
                    self.writer.add_scalar('train/loss', loss, self.total_steps)
            
            # Target network update (handle both config key names)
            target_interval = self.config.get('target_update_interval', self.config.get('target_update', 1000))
            if self.total_steps % target_interval == 0:
                self.update_target_network()
                if self.verbose >= 2:
                    print(f"Step {self.total_steps}: Target network updated")
            
            # Episode end
            if done:
                self.episodes_completed += 1
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                
                # Get net worth from info
                net_worth = info.get('agent_net_worth', 0)
                self.episode_net_worths.append(net_worth)
                
                # Log to TensorBoard
                self.writer.add_scalar('train/episode_reward', episode_reward, self.total_steps)
                self.writer.add_scalar('train/episode_length', episode_length, self.total_steps)
                self.writer.add_scalar('train/net_worth', net_worth, self.total_steps)
                self.writer.add_scalar('train/epsilon', epsilon, self.total_steps)
                self.writer.add_scalar('train/buffer_size', len(self.replay_buffer), self.total_steps)
                
                # Save training metric to CSV
                self._save_episode_metric(
                    episode=self.episodes_completed,
                    episode_length=episode_length,
                    final_net_worth=net_worth,
                    mean_episode_reward=episode_reward,
                    win_flag=1 if info.get('agent_status') == 'ACTIVE' and info.get('active_players', 1) == 1 else 0,
                    eval_tag='train'
                )
                
                # Reset
                obs, info = self.env.reset()
                mask = self._get_action_mask(self.env)
                episode_reward = 0.0
                episode_length = 0
            else:
                obs = next_obs
                mask = next_mask
            
            # Logging
            if self.total_steps % log_interval == 0 and self.verbose >= 1:
                mean_reward = np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0
                mean_length = np.mean(self.episode_lengths[-100:]) if self.episode_lengths else 0
                mean_loss = np.mean(self.losses[-100:]) if self.losses else 0
                print(f"Step {self.total_steps}/{total_timesteps} | "
                      f"Episodes: {self.episodes_completed} | "
                      f"Mean Reward: {mean_reward:.2f} | "
                      f"Mean Length: {mean_length:.1f} | "
                      f"Loss: {mean_loss:.4f} | "
                      f"Epsilon: {epsilon:.3f}")
            
            # Evaluation
            if self.total_steps % eval_interval == 0:
                eval_results = self.evaluate(eval_episodes)
                
                # Log to TensorBoard
                self.writer.add_scalar('eval/mean_reward', eval_results['mean_reward'], self.total_steps)
                self.writer.add_scalar('eval/mean_net_worth', eval_results['mean_net_worth'], self.total_steps)
                self.writer.add_scalar('eval/win_rate', eval_results['win_rate'], self.total_steps)
            
            # Save checkpoint
            if self.total_steps % save_interval == 0:
                self.save_checkpoint(f"checkpoint_{self.total_steps}")
            
            # Update progress bar
            if pbar is not None:
                pbar.update(1)
        
        if pbar is not None:
            pbar.close()
        
        # Final evaluation
        print("\n" + "=" * 60)
        print("Final Evaluation")
        print("=" * 60)
        final_results = self.evaluate(eval_episodes)
        
        # Save final model and buffer
        self.save_checkpoint("final")
        self.save_replay_buffer()
        
        training_time = time.time() - start_time
        
        # Print summary
        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
        print(f"Total time: {training_time:.2f}s")
        print(f"Total steps: {self.total_steps}")
        print(f"Episodes completed: {self.episodes_completed}")
        print(f"Final mean eval net worth: ${final_results['mean_net_worth']:.0f}")
        print(f"Final win rate: {final_results['win_rate'] * 100:.1f}%")
        print(f"Model saved to: {self.output_dir / 'models' / 'ddqn_hybrid'}")
        print(f"Buffer saved to: {self.output_dir / 'buffers' / 'ddqn_hybrid_replay.pkl'}")
        print(f"Metrics saved to: {self.output_dir / 'results' / 'ddqn_hybrid_metrics.csv'}")
        print("=" * 60)
        
        return {
            'total_steps': self.total_steps,
            'episodes': self.episodes_completed,
            'training_time': training_time,
            'final_eval_results': final_results,
        }
    
    def evaluate(self, num_episodes: int = 100) -> Dict[str, float]:
        """
        Run evaluation episodes with deterministic policy.
        
        Args:
            num_episodes: Number of episodes to evaluate
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.verbose >= 1:
            print(f"\nRunning evaluation ({num_episodes} episodes)...")
        
        episode_rewards = []
        episode_lengths = []
        net_worths = []
        wins = 0
        
        for ep in range(num_episodes):
            obs, info = self.eval_env.reset(seed=self.seed + 10000 + ep)
            mask = self._get_action_mask(self.eval_env)
            
            episode_reward = 0.0
            episode_length = 0
            done = False
            
            while not done:
                # Deterministic action (epsilon=0)
                action = self.select_action(obs, mask, epsilon=0.0)
                
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                done = terminated or truncated
                mask = self._get_action_mask(self.eval_env)
                
                episode_reward += reward
                episode_length += 1
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            net_worths.append(info.get('agent_net_worth', 0))
            
            # Check win
            if info.get('agent_status') == 'ACTIVE' and info.get('active_players', 1) == 1:
                wins += 1
            
            # Save eval metric to CSV
            self._save_episode_metric(
                # Use a stable, unique episode index for evaluation rows.
                # We offset by completed training episodes to keep monotonic ordering.
                episode=self.episodes_completed + ep + 1,
                episode_length=episode_length,
                final_net_worth=info.get('agent_net_worth', 0),
                mean_episode_reward=episode_reward,
                win_flag=1 if info.get('agent_status') == 'ACTIVE' and info.get('active_players', 1) == 1 else 0,
                eval_tag='eval'
            )
        
        results = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'mean_net_worth': np.mean(net_worths),
            'win_rate': wins / num_episodes,
        }
        
        if self.verbose >= 1:
            print(f"Eval Results: Reward={results['mean_reward']:.2f} | "
                  f"Net Worth=${results['mean_net_worth']:.0f} | "
                  f"Win Rate={results['win_rate']*100:.1f}%")
        
        return results
    
    def _save_episode_metric(
        self,
        episode: int,
        episode_length: int,
        final_net_worth: float,
        mean_episode_reward: float,
        win_flag: int,
        eval_tag: str
    ) -> None:
        """Save single episode metric to CSV."""
        from utils.save_utils import append_metrics_csv
        
        metric = {
            'timestamp': datetime.now().isoformat(),
            'episode': episode,
            'episode_length': episode_length,
            'final_net_worth': final_net_worth,
            'agent_rank': 0,  # Could compute rank among players
            'win_flag': win_flag,
            'total_steps': self.total_steps,
            'mean_episode_reward': mean_episode_reward,
            'eval_tag': eval_tag,
        }
        
        csv_path = self.output_dir / 'results' / 'ddqn_hybrid_metrics.csv'
        append_metrics_csv(csv_path, metric, self.run_id, self.seed)
    
    def save_checkpoint(self, name: str = "checkpoint") -> None:
        """Save model checkpoint."""
        model_dir = self.output_dir / 'models' / 'ddqn_hybrid'
        
        # Save online network
        self.q_online.save(model_dir / f"{name}_online.pt")
        
        # Save target network
        self.q_target.save(model_dir / f"{name}_target.pt")
        
        # Save optimizer state
        torch.save({
            'optimizer_state_dict': self.optimizer.state_dict(),
            'total_steps': self.total_steps,
            'episodes_completed': self.episodes_completed,
            'config': self.config,
            'seed': self.seed,
        }, model_dir / f"{name}_trainer_state.pt")
    
    def save_replay_buffer(self) -> None:
        """Save replay buffer to file."""
        from utils.save_utils import save_replay_buffer
        
        buffer_path = self.output_dir / 'buffers' / 'ddqn_hybrid_replay.pkl'
        save_replay_buffer(self.replay_buffer.get_data_for_save(), buffer_path)
    
    def load_checkpoint(self, name: str = "checkpoint") -> None:
        """Load model checkpoint."""
        model_dir = self.output_dir / 'models' / 'ddqn_hybrid'
        
        # Load networks
        self.q_online = QNetwork.load(model_dir / f"{name}_online.pt", device=self.device)
        self.q_target = QNetwork.load(model_dir / f"{name}_target.pt", device=self.device)
        
        # Load trainer state
        state = torch.load(model_dir / f"{name}_trainer_state.pt", map_location=self.device)
        self.optimizer.load_state_dict(state['optimizer_state_dict'])
        self.total_steps = state['total_steps']
        self.episodes_completed = state['episodes_completed']
    
    def close(self) -> None:
        """Clean up resources."""
        self.writer.close()
        if hasattr(self.env, 'close'):
            self.env.close()
        if self.eval_env is not self.env and hasattr(self.eval_env, 'close'):
            self.eval_env.close()
