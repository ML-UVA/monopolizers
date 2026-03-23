from typing import Optional

import torch
import torch.nn as nn
import numpy as np

from Monopoly.agents.replay_buffer import ReplayBuffer

class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class DQNAgent:
    def __init__(
            self,
            obs_dim: int,
            n_actions: int,
            lr: float = 1e-4,
            gamma: float = 0.99,
            epsilon_start: float = 1.0,
            epsilon_end: float = 0.05,
            epsilon_decay: int = 100_000,
            buffer_capacity: int = 50_000,
            batch_size: int = 64,
            target_update_freq: int = 1000,
            device: str = 'cpu'
    ):
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0
        self.device = torch.device(device)

        self.q_network = QNetwork(obs_dim, n_actions).to(self.device)
        self.target_network = QNetwork(obs_dim, n_actions).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        for param in self.target_network.parameters():
            # Only update target network using load_state_dict, 
            # Don't use gradient descent
            param.requires_grad = False

        self.optimizer = torch.optim.Adam(
            self.q_network.parameters(), lr=lr
        )
        self.loss_fn = nn.SmoothL1Loss()

        self.buffer = ReplayBuffer(buffer_capacity)
    
    @property
    def epsilon(self) -> float:
        """Current epsilon value, uses epsilon decay"""
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
               np.exp(-self.steps_done / self.epsilon_decay) 

    def select_action(self, obs: np.ndarray, legal_mask: np.ndarray) -> int:
        """Epsilon-greedy action selection with legal action masking."""
        self.steps_done += 1

        if np.random.random() < self.epsilon:
            # Explore: random legal action
            legal_actions = np.where(legal_mask)[0]
            return int(np.random.choice(legal_actions))
        else:
            # Exploit: highest Q-value among legal actions
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                
                output: torch.Tensor = self.q_network(obs_tensor)
                q_values = output.squeeze(0)

                # Mask illegal actions
                mask_tensor = torch.FloatTensor(legal_mask).to(self.device)
                q_values[mask_tensor == 0] = float('-inf')

                return int(q_values.argmax().item())
        
    def store_transition(
            self,
            obs: np.ndarray,
            action: int,
            reward: float,
            next_obs: np.ndarray,
            done: float,
            next_legal_mask: np.ndarray
    ):
        """Store a transition in the replay buffer"""
        self.buffer.push(obs, action, reward, next_obs, done, next_legal_mask)

    def update(self) -> Optional[float]:
        """Sample a batch and perform one gradient update. Returns loss or None."""
        if not self.buffer.is_ready(self.batch_size):
            return None
        
        obs, actions, rewards, next_obs, dones, next_legal_masks = \
            self.buffer.sample(self.batch_size)
        
        obs = obs.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_obs = next_obs.to(self.device)
        dones = dones.to(self.device)
        next_legal_masks = next_legal_masks.to(self.device)

        q_values: torch.Tensor = self.q_network(obs)            # (batch, n_actions)
        q_values = q_values.gather(1, actions.unsqueeze(1))     # (batch, 1)
        q_values = q_values.squeeze(1)                          # (batch,)


        with torch.no_grad():
            next_q_values: torch.Tensor = self.target_network(next_obs)

            # Mask illegal actions in next state
            next_q_values[next_legal_masks == 0] = float('-inf')

            max_next_q = next_q_values.max(dim=1).values

            # If done, no future reward (state terminates)
            targets = rewards + self.gamma * max_next_q * (1 - dones)

        loss = self.loss_fn(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        if self.steps_done % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        return loss.item()

    def save(self, path: str):
        """Save network weights and training state."""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
        }, path)
    
    def load(self, path: str):
        """Load network weights and training state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.steps_done = checkpoint['steps_done']
