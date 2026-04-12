"""
Double DQN Hybrid Trainer for Monopoly.

This module implements a custom Double-DQN training loop using PyTorch,
designed to work with the same environment wrapper, action masking, and
reward function as the SB3 DQN path.

Key Features:
- Double DQN update rule (reduces overestimation bias)
- Action masking for legal moves only
- Epsilon-greedy exploration with annealing
- Experience replay buffer with action masks (uniform or PER)
- Hard or soft (Polyak) target network updates
- Optional reward normalization
- Enhanced TensorBoard diagnostics (TD error, grad norm, Q-values)
- CSV metrics export
- Consolidated checkpointing with best-model tracking
"""

import os
import time
import random
import warnings
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


class SumTree:
    """Binary sum-tree for O(log n) proportional sampling in PER."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data = [None] * capacity
        self.size = 0
        self.write_pos = 0

    def _propagate(self, idx: int, change: float) -> None:
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        return self._retrieve(right, s - self.tree[left])

    @property
    def total(self) -> float:
        return self.tree[0]

    def add(self, priority: float, data: Any) -> None:
        idx = self.write_pos + self.capacity - 1
        self.data[self.write_pos] = data
        self.update(idx, priority)
        self.write_pos = (self.write_pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, idx: int, priority: float) -> None:
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s: float) -> Tuple[int, float, Any]:
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    """Proportional Prioritized Experience Replay (Schaul et al., 2016).

    Uses a SumTree for O(log n) sampling proportional to TD-error priority.

    Args:
        capacity: Maximum number of transitions.
        alpha: Priority exponent (0 = uniform, 1 = full prioritization).
    """

    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.tree = SumTree(capacity)
        self.max_priority = 1.0
        self._epsilon = 1e-6

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
        """Add transition with max priority (ensures it gets sampled at least once)."""
        transition = Transition(
            state=state, action=action, reward=reward,
            next_state=next_state, done=done, mask=mask, next_mask=next_mask
        )
        priority = (self.max_priority + self._epsilon) ** self.alpha
        self.tree.add(priority, transition)

    def sample(
        self, batch_size: int, beta: float = 0.4
    ) -> Tuple[List[Transition], np.ndarray, np.ndarray]:
        """Sample batch with importance-sampling weights.

        Returns:
            (transitions, is_weights, tree_indices)
        """
        batch = []
        indices = np.empty(batch_size, dtype=np.int64)
        priorities = np.empty(batch_size, dtype=np.float64)
        segment = self.tree.total / batch_size

        for i in range(batch_size):
            lo = segment * i
            hi = segment * (i + 1)
            s = np.random.uniform(lo, hi)
            idx, priority, data = self.tree.get(s)
            indices[i] = idx
            priorities[i] = priority
            batch.append(data)

        # Importance-sampling weights
        probs = priorities / (self.tree.total + 1e-10)
        n = len(self)
        is_weights = (n * probs) ** (-beta)
        is_weights /= is_weights.max()

        return batch, is_weights.astype(np.float32), indices

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        """Update priorities from absolute TD errors."""
        for idx, td_err in zip(indices, td_errors):
            priority = (abs(td_err) + self._epsilon) ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, abs(td_err))

    def __len__(self) -> int:
        return self.tree.size

    def get_data_for_save(self) -> List[Dict]:
        """Get buffer data in a picklable format."""
        result = []
        for i in range(self.tree.size):
            t = self.tree.data[i]
            if t is not None:
                result.append({
                    'state': t.state, 'action': t.action, 'reward': t.reward,
                    'next_state': t.next_state, 'done': t.done,
                    'mask': t.mask, 'next_mask': t.next_mask
                })
        return result

    def load_data(self, data: List[Dict]) -> None:
        """Load buffer data from saved format."""
        for d in data:
            self.push(**d)


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load YAML config file and return as dict."""
    import yaml
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


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
        # Target update mode
        'target_update_mode': 'hard',  # 'hard' or 'soft'
        'tau': 0.005,            # Polyak averaging coefficient (soft mode)
        # PER (opt-in)
        'use_per': False,
        'per_alpha': 0.6,
        'per_beta_start': 0.4,
        'per_beta_frames': 100_000,
        # Reward normalization
        'normalize_rewards': False,
        'reward_clip': 10.0,
        # Logging policy
        'log_training_diagnostics': False,
    }

    KNOWN_KEYS = set(DEFAULT_CONFIG.keys()) | {
        'start_training_after', 'target_update',
        # Config-file-only keys (not hyperparams but allowed in YAML)
        'reward_mode', 'reward_weights', 'total_timesteps',
        'eval_interval', 'eval_episodes', 'log_interval',
        'save_interval', 'max_turns', 'seed',
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
        # Warn on unknown config keys
        merged_config = config or {}
        unknown = set(merged_config.keys()) - self.KNOWN_KEYS
        if unknown:
            warnings.warn(f"Unknown config keys (possible typos): {unknown}")

        # Merge config with defaults
        self.config = {**self.DEFAULT_CONFIG, **merged_config}

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

        # Replay buffer (uniform or PER)
        if self.config.get('use_per', False):
            self.replay_buffer = PrioritizedReplayBuffer(
                capacity=self.config['buffer_size'],
                alpha=self.config.get('per_alpha', 0.6),
            )
        else:
            self.replay_buffer = ReplayBuffer(self.config['buffer_size'])

        # Reward normalizer (optional)
        self.reward_normalizer = None
        if self.config.get('normalize_rewards', False):
            from .reward import RunningNormalizer
            self.reward_normalizer = RunningNormalizer(
                clip=self.config.get('reward_clip', 10.0)
            )

        # Training state
        self.total_steps = 0
        self.episodes_completed = 0
        self.training_started = False
        self.best_win_rate = -1.0

        # Metrics tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_net_worths = []
        self.losses = []

        # Diagnostic caches for histograms
        self._last_td_errors: Optional[torch.Tensor] = None
        self._last_q_values: Optional[torch.Tensor] = None

        # Create output directories
        self._setup_output_dirs()

        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=str(self.output_dir / 'tensorboard' / 'ddqn_hybrid'))

        # Run ID for CSV
        self.run_id = f"ddqn_hybrid_seed{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    @classmethod
    def from_config(
        cls,
        config_path: str,
        env,
        eval_env=None,
        output_dir: str = 'runs/ddqn_hybrid',
        seed: int = 42,
        device: str = 'auto',
        overrides: Optional[Dict[str, Any]] = None,
        verbose: int = 1,
    ) -> 'DDQNHybridTrainer':
        """Create trainer from a YAML config file with optional overrides."""
        yaml_config = load_yaml_config(config_path)
        if overrides:
            yaml_config.update(overrides)
        return cls(
            env=env, eval_env=eval_env, output_dir=output_dir,
            seed=seed, device=device, config=yaml_config, verbose=verbose,
        )

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

    def _get_per_beta(self) -> float:
        """Calculate current PER beta for importance-sampling correction."""
        beta_start = self.config.get('per_beta_start', 0.4)
        beta_frames = self.config.get('per_beta_frames', 100_000)
        progress = min(1.0, self.total_steps / beta_frames)
        return beta_start + progress * (1.0 - beta_start)

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

    def train_step(self) -> Optional[Dict[str, float]]:
        """
        Perform one training step (update networks from replay buffer).

        Returns:
            Dict of diagnostics if training occurred, None otherwise.
            Keys: loss, mean_td_error, max_td_error, mean_q_value, max_q_value, grad_norm
        """
        # Check if we have enough samples to start training
        min_samples = self.config.get('learning_starts', self.config.get('start_training_after', 1000))
        if len(self.replay_buffer) < min_samples:
            return None

        if len(self.replay_buffer) < self.config['batch_size']:
            return None

        # Sample batch — PER returns extra info
        use_per = self.config.get('use_per', False)
        if use_per:
            beta = self._get_per_beta()
            batch, is_weights, per_indices = self.replay_buffer.sample(
                self.config['batch_size'], beta=beta
            )
            is_weights_t = torch.tensor(is_weights, dtype=torch.float32, device=self.device)
        else:
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

        # Compute element-wise TD errors for diagnostics and PER
        td_errors = (current_q_actions - targets).detach()

        # Compute loss (Huber for stability, MSE as fallback)
        if self.config.get('use_huber_loss', True):
            if use_per:
                element_loss = nn.functional.smooth_l1_loss(
                    current_q_actions, targets, reduction='none'
                )
                loss = (is_weights_t * element_loss).mean()
            else:
                loss = nn.functional.smooth_l1_loss(current_q_actions, targets)
        else:
            if use_per:
                element_loss = nn.functional.mse_loss(
                    current_q_actions, targets, reduction='none'
                )
                loss = (is_weights_t * element_loss).mean()
            else:
                loss = nn.functional.mse_loss(current_q_actions, targets)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping (capture returned norm)
        grad_norm = 0.0
        if self.config['grad_clip'] is not None:
            grad_norm = nn.utils.clip_grad_norm_(
                self.q_online.parameters(), self.config['grad_clip']
            ).item()

        self.optimizer.step()

        # Update PER priorities
        if use_per:
            self.replay_buffer.update_priorities(
                per_indices, td_errors.abs().cpu().numpy()
            )

        # Cache for histogram logging
        self._last_td_errors = td_errors
        self._last_q_values = current_q.detach()

        return {
            'loss': loss.item(),
            'mean_td_error': td_errors.mean().item(),
            'max_td_error': td_errors.abs().max().item(),
            'mean_q_value': current_q_actions.mean().item(),
            'max_q_value': current_q_actions.max().item(),
            'grad_norm': grad_norm,
        }

    def update_target_network(self) -> None:
        """Update target network (hard copy or soft Polyak averaging)."""
        if self.config.get('target_update_mode', 'hard') == 'soft':
            tau = self.config.get('tau', 0.005)
            for target_param, online_param in zip(
                self.q_target.parameters(), self.q_online.parameters()
            ):
                target_param.data.copy_(
                    tau * online_param.data + (1 - tau) * target_param.data
                )
        else:
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

        use_soft_target = self.config.get('target_update_mode', 'hard') == 'soft'
        start_time = time.time()

        # Initialize environment
        obs, info = self.env.reset(seed=self.seed)
        mask = self._get_action_mask(self.env)

        episode_reward = 0.0
        episode_length = 0
        episode_houses_built = 0
        episode_trades_executed = 0
        episode_buy_opportunities = 0
        episode_buy_actions = 0

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

            # Optional reward normalization
            if self.reward_normalizer is not None:
                self.reward_normalizer.update(reward)
                reward = self.reward_normalizer.normalize(reward)

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

            # Track gameplay-oriented episode metrics.
            if 3 <= action <= 30:
                episode_houses_built += 1
            if action >= 90:
                episode_trades_executed += 1
            if action in (1, 2):
                episode_buy_opportunities += 1
            if action == 1:
                episode_buy_actions += 1

            # DQN-style Monopoly game metrics (logged each step).
            self.writer.add_scalar('game/agent_net_worth', info.get('agent_net_worth', 0), self.total_steps)
            self.writer.add_scalar('game/agent_cash', info.get('agent_cash', 0), self.total_steps)
            self.writer.add_scalar('game/agent_properties', info.get('agent_properties', 0), self.total_steps)
            self.writer.add_scalar('game/active_players', info.get('active_players', 0), self.total_steps)

            # Training update
            if self.total_steps % self.config['train_freq'] == 0:
                diagnostics = self.train_step()
                if diagnostics is not None:
                    self.losses.append(diagnostics['loss'])
                    # Keep primary optimization loss visible by default.
                    self.writer.add_scalar('train/loss', diagnostics['loss'], self.total_steps)

                    # Optional detailed diagnostics (Q-values, TD-error, grad norm).
                    if self.config.get('log_training_diagnostics', False):
                        for key in ('mean_td_error', 'max_td_error', 'mean_q_value', 'max_q_value', 'grad_norm'):
                            self.writer.add_scalar(f'train/{key}', diagnostics[key], self.total_steps)

            # Target network update
            if use_soft_target:
                # Soft update every step
                self.update_target_network()
            else:
                # Hard update at interval
                target_interval = self.config.get(
                    'target_update_interval',
                    self.config.get('target_update', 1000)
                )
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

                buy_rate = episode_buy_actions / max(episode_buy_opportunities, 1)
                self.writer.add_scalar('game/houses_built', episode_houses_built, self.total_steps)
                self.writer.add_scalar('game/trades_executed', episode_trades_executed, self.total_steps)
                self.writer.add_scalar('game/buy_actions', episode_buy_actions, self.total_steps)
                self.writer.add_scalar('game/buy_opportunities', episode_buy_opportunities, self.total_steps)
                self.writer.add_scalar('game/buy_rate', buy_rate, self.total_steps)

                # Log modular reward components if available
                reward_components = info.get('reward_components')
                if reward_components is not None:
                    for comp_name, comp_val in reward_components.items():
                        self.writer.add_scalar(
                            f'train/reward_components/{comp_name}',
                            comp_val, self.total_steps
                        )

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
                episode_houses_built = 0
                episode_trades_executed = 0
                episode_buy_opportunities = 0
                episode_buy_actions = 0
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

            # Learning rate logging
            if self.total_steps % 10000 == 0:
                lr = self.optimizer.param_groups[0]['lr']
                self.writer.add_scalar('train/learning_rate', lr, self.total_steps)

            # Evaluation
            if self.total_steps % eval_interval == 0:
                eval_results = self.evaluate(eval_episodes)

                # Log to TensorBoard
                self.writer.add_scalar('eval/mean_reward', eval_results['mean_reward'], self.total_steps)
                self.writer.add_scalar('eval/mean_net_worth', eval_results['mean_net_worth'], self.total_steps)
                self.writer.add_scalar('eval/win_rate', eval_results['win_rate'], self.total_steps)

                # Best-model tracking
                if eval_results['win_rate'] > self.best_win_rate:
                    self.best_win_rate = eval_results['win_rate']
                    self.save_checkpoint("best")

            # Save checkpoint + histograms
            if self.total_steps % save_interval == 0:
                self.save_checkpoint(f"checkpoint_{self.total_steps}")
                self._log_histograms()

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

    def _log_histograms(self) -> None:
        """Log weight and diagnostic histograms to TensorBoard."""
        # Network weight histograms
        for i, (name, param) in enumerate(self.q_online.named_parameters()):
            self.writer.add_histogram(f'weights/online/{name}', param.data, self.total_steps)
        for i, (name, param) in enumerate(self.q_target.named_parameters()):
            self.writer.add_histogram(f'weights/target/{name}', param.data, self.total_steps)

        # Q-value and TD error distributions from last batch
        if self._last_q_values is not None:
            self.writer.add_histogram('q_values/distribution', self._last_q_values, self.total_steps)
        if self._last_td_errors is not None:
            self.writer.add_histogram('td_errors/distribution', self._last_td_errors, self.total_steps)

    def evaluate(self, num_episodes: int = 100, trace_dir: str = None) -> Dict[str, float]:
        """
        Run evaluation episodes with deterministic policy.

        Args:
            num_episodes: Number of episodes to evaluate
            trace_dir: If set, wrap eval env with EpisodeTracer for this run

        Returns:
            Dictionary of evaluation metrics
        """
        if self.verbose >= 1:
            print(f"\nRunning evaluation ({num_episodes} episodes)...")

        eval_env = self.eval_env
        if trace_dir:
            from Monopoly.envs.tracing import EpisodeTracer
            # Unwrap to raw env, wrap with tracer, re-flatten
            raw = eval_env.unwrapped
            traced = EpisodeTracer(raw, output_dir=trace_dir)
            from Monopoly.envs.wrappers import MonopolyFlattenWrapper
            eval_env = MonopolyFlattenWrapper(traced)

        episode_rewards = []
        episode_lengths = []
        net_worths = []
        wins = 0

        for ep in range(num_episodes):
            obs, info = eval_env.reset(seed=self.seed + 10000 + ep)
            mask = self._get_action_mask(eval_env)

            episode_reward = 0.0
            episode_length = 0
            done = False

            while not done:
                # Deterministic action (epsilon=0)
                action = self.select_action(obs, mask, epsilon=0.0)

                obs, reward, terminated, truncated, info = eval_env.step(action)
                done = terminated or truncated
                mask = self._get_action_mask(eval_env)

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
        """Save consolidated model checkpoint."""
        model_dir = self.output_dir / 'models' / 'ddqn_hybrid'

        checkpoint = {
            'online_state_dict': self.q_online.state_dict(),
            'target_state_dict': self.q_target.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'total_steps': self.total_steps,
            'episodes_completed': self.episodes_completed,
            'config': self.config,
            'seed': self.seed,
            'best_win_rate': self.best_win_rate,
            'obs_dim': self.obs_dim,
            'action_dim': self.action_dim,
        }
        if self.reward_normalizer is not None:
            checkpoint['reward_normalizer'] = self.reward_normalizer.state_dict()

        torch.save(checkpoint, model_dir / f"{name}.pt")

        # Backward-compatible 3-file format
        self.q_online.save(model_dir / f"{name}_online.pt")
        self.q_target.save(model_dir / f"{name}_target.pt")
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

    def load_replay_buffer(self, filepath: Union[str, Path]) -> None:
        """Load replay buffer transitions from file."""
        from utils.save_utils import load_replay_buffer

        buffer_data = load_replay_buffer(filepath)
        self.replay_buffer.load_data(buffer_data)

    def _load_checkpoint_dict(self, checkpoint: Dict[str, Any]) -> None:
        """Load trainer/network state from an in-memory checkpoint dictionary."""
        if 'online_state_dict' in checkpoint:
            # Consolidated DDQN trainer checkpoint
            self.q_online.load_state_dict(checkpoint['online_state_dict'])
            self.q_target.load_state_dict(checkpoint.get('target_state_dict', checkpoint['online_state_dict']))

            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            self.total_steps = int(checkpoint.get('total_steps', self.total_steps))
            self.episodes_completed = int(checkpoint.get('episodes_completed', self.episodes_completed))
            self.best_win_rate = float(checkpoint.get('best_win_rate', self.best_win_rate))

            if self.reward_normalizer is not None and 'reward_normalizer' in checkpoint:
                self.reward_normalizer.load_state_dict(checkpoint['reward_normalizer'])
            return

        if 'model_state_dict' in checkpoint:
            # QNetwork checkpoint format
            self.q_online.load_state_dict(checkpoint['model_state_dict'])
            self.q_target.load_state_dict(checkpoint['model_state_dict'])

            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            self.total_steps = int(checkpoint.get('total_steps', self.total_steps))
            self.episodes_completed = int(checkpoint.get('episodes_completed', self.episodes_completed))
            return

        # Raw state dict fallback
        if checkpoint and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
            self.q_online.load_state_dict(checkpoint)
            self.q_target.load_state_dict(checkpoint)
            return

        raise ValueError(
            "Unsupported checkpoint format. Expected consolidated DDQN checkpoint, "
            "QNetwork checkpoint, or raw state_dict."
        )

    def load_checkpoint(self, checkpoint: str = "checkpoint") -> None:
        """Load model checkpoint from a name in run dir or an explicit checkpoint path."""
        model_dir = self.output_dir / 'models' / 'ddqn_hybrid'

        checkpoint_path = Path(checkpoint)
        if checkpoint_path.suffix == '.pt' and not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        if checkpoint_path.exists() and checkpoint_path.is_file():
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            self._load_checkpoint_dict(ckpt)
            return

        consolidated_path = model_dir / f"{checkpoint}.pt"
        if consolidated_path.exists():
            consolidated_ckpt = torch.load(consolidated_path, map_location=self.device)
            self._load_checkpoint_dict(consolidated_ckpt)
        else:
            # Legacy 3-file format
            self.q_online = QNetwork.load(model_dir / f"{checkpoint}_online.pt", device=self.device)
            self.q_target = QNetwork.load(model_dir / f"{checkpoint}_target.pt", device=self.device)
            state = torch.load(model_dir / f"{checkpoint}_trainer_state.pt", map_location=self.device)
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
