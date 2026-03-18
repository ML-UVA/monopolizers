from collections import deque

import numpy as np
import random
import torch

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(
            self, 
            obs: np.ndarray, 
            action: int, 
            reward: float, 
            next_obs: np.ndarray, 
            done: float,
            next_legal_mask: np.ndarray
        ):
        # done: binary flag, stores whether the transition leads to
        # a terminal state
        self.buffer.append(
            (obs, action, reward, next_obs, done, next_legal_mask)
        )
    
    def sample(self, batch_size: int):
        """Randomly sample a batch of transitions"""
        batch = random.sample(self.buffer, batch_size)
        obs, actions, rewards, next_obs, dones, next_legal_masks = zip(*batch)
        return (
            torch.FloatTensor(np.array(obs)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_obs)),
            torch.FloatTensor(dones),
            torch.FloatTensor(np.array(next_legal_masks)),
        )

    def is_ready(self, batch_size: int):
        """Check if buffer has enough samples to train"""
        return len(self.buffer) >= batch_size

    def __len__(self):
        return len(self.buffer)
