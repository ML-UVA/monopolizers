"""
PyTorch MLP networks for Q-value estimation.

This module provides the neural network architecture used by the DDQN-hybrid trainer.
The network takes flattened observations (same as wrappers.py) and outputs Q-values.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Union
from pathlib import Path


class QNetwork(nn.Module):
    """
    Multi-layer Perceptron for Q-value estimation.
    
    Takes a flattened observation vector and outputs Q-values for all actions.
    Uses the same observation normalization as MonopolyFlattenWrapper.
    
    Architecture:
        Input -> [hidden_dims] layers with ReLU -> Output (action_dim)
    
    Args:
        obs_dim: Dimension of the flattened observation space
        action_dim: Number of possible actions (90 for Monopoly)
        hidden_dims: List of hidden layer dimensions (default: [256, 256])
        device: Device to place the network on ('cpu', 'cuda', or 'auto')
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: List[int] = None,
        device: str = 'auto'
    ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 256]
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Build network layers
        layers = []
        prev_dim = obs_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        # Output layer (Q-values for each action)
        layers.append(nn.Linear(prev_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
        
        # Move to device
        self.to(self.device)
    
    def _init_weights(self):
        """Initialize network weights.
        
        Uses Kaiming (He) initialization for layers followed by ReLU,
        which is more appropriate than Xavier for ReLU activations.
        The output layer uses smaller initialization for stable Q-values.
        """
        for i, module in enumerate(self.network):
            if isinstance(module, nn.Linear):
                # Check if this is the output layer (last linear)
                is_output = (i == len(self.network) - 1)
                if is_output:
                    # Smaller init for output layer (Q-values should start near 0)
                    nn.init.orthogonal_(module.weight, gain=0.01)
                else:
                    # Kaiming init for hidden layers with ReLU
                    nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: compute Q-values for all actions given state.
        
        Args:
            state: Flattened observation tensor of shape (batch_size, obs_dim)
            
        Returns:
            Q-values tensor of shape (batch_size, action_dim)
        """
        return self.network(state)
    
    def get_q_values(
        self, 
        state: torch.Tensor, 
        action_mask: Optional[torch.Tensor] = None
    ) -> np.ndarray:
        """
        Get Q-values with optional action masking.
        
        Args:
            state: Flattened observation (numpy array or tensor)
            action_mask: Binary mask (1 = legal, 0 = illegal)
            
        Returns:
            Q-values as numpy array with illegal actions masked to -inf
        """
        # Convert numpy to tensor if needed
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
        
        # Ensure 2D
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        with torch.no_grad():
            q_values = self.forward(state)
        
        if action_mask is not None:
            # Convert mask if needed
            if isinstance(action_mask, np.ndarray):
                action_mask = torch.tensor(action_mask, dtype=torch.float32, device=self.device)
            if action_mask.dim() == 1:
                action_mask = action_mask.unsqueeze(0)
            
            # Mask illegal actions with large negative value
            masked_q = q_values.clone()
            masked_q[action_mask == 0] = float('-inf')
            return masked_q.cpu().numpy()
        
        return q_values.cpu().numpy()
    
    def select_action(
        self,
        state: np.ndarray,
        action_mask: Optional[np.ndarray] = None,
        epsilon: float = 0.0
    ) -> int:
        """
        Select action using epsilon-greedy policy with action masking.
        
        Args:
            state: Single observation (numpy array) of shape (obs_dim,) or (1, obs_dim)
            action_mask: Binary mask (1 = legal, 0 = illegal)
            epsilon: Exploration probability
            
        Returns:
            Selected action index
        """
        # Get legal actions from mask
        if action_mask is not None:
            if isinstance(action_mask, torch.Tensor):
                legal_actions = torch.where(action_mask.flatten() == 1)[0].cpu().numpy()
            else:
                legal_actions = np.where(action_mask.flatten() == 1)[0]
        else:
            legal_actions = np.arange(self.action_dim)
        
        if len(legal_actions) == 0:
            # Fallback: return action 0 (should not happen in practice)
            return 0
        
        # Epsilon-greedy with action masking
        if np.random.random() < epsilon:
            # Random exploration among legal actions only
            return int(np.random.choice(legal_actions))
        else:
            # Greedy: select best Q-value among legal actions
            # Convert numpy to tensor if needed
            if isinstance(state, np.ndarray):
                state_t = torch.tensor(state, dtype=torch.float32, device=self.device)
            else:
                state_t = state
            
            if state_t.dim() == 1:
                state_t = state_t.unsqueeze(0)
                
            with torch.no_grad():
                q_values = self.forward(state_t).squeeze(0)
                
                # Mask illegal actions
                if action_mask is not None:
                    if isinstance(action_mask, np.ndarray):
                        action_mask_t = torch.tensor(action_mask, dtype=torch.float32, device=q_values.device)
                    else:
                        action_mask_t = action_mask
                    q_values = q_values.clone()
                    q_values[action_mask_t.flatten() == 0] = float('-inf')
                
                return int(q_values.argmax().item())
    
    def save(self, filepath: Union[str, Path]) -> None:
        """Save model weights to file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        # Ensure metadata is stored as plain Python types so that
        # PyTorch 2.6+ default `weights_only=True` loading works.
        obs_dim = int(self.obs_dim)
        action_dim = int(self.action_dim)
        hidden_dims = [int(x) for x in self.hidden_dims]
        torch.save({
            'model_state_dict': self.state_dict(),
            'obs_dim': obs_dim,
            'action_dim': action_dim,
            'hidden_dims': hidden_dims,
        }, filepath)
        print(f"Q-Network saved to: {filepath}")
    
    @classmethod
    def load(cls, filepath: Union[str, Path], device: Optional[torch.device] = None) -> 'QNetwork':
        """Load model weights from file."""
        filepath = Path(filepath)
        checkpoint = torch.load(filepath, map_location=device)
        
        model = cls(
            obs_dim=checkpoint['obs_dim'],
            action_dim=checkpoint['action_dim'],
            hidden_dims=checkpoint['hidden_dims']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if device is not None:
            model = model.to(device)
        
        print(f"Q-Network loaded from: {filepath}")
        return model


class DuelingQNetwork(nn.Module):
    """
    Dueling DQN architecture (Wang et al., 2016).
    
    Separates value and advantage streams:
        Q(s, a) = V(s) + (A(s, a) - mean(A(s, .)))
    
    This improves learning by decoupling state value estimation from
    action advantage estimation, leading to better policy evaluation.
    
    Reference: https://arxiv.org/abs/1511.06581
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: List[int] = None,
        device: str = 'auto'
    ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 256]
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        
        # Shared feature extraction
        feature_layers = []
        prev_dim = obs_dim
        
        for hidden_dim in hidden_dims[:-1]:
            feature_layers.append(nn.Linear(prev_dim, hidden_dim))
            feature_layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        self.feature_net = nn.Sequential(*feature_layers)
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], action_dim)
        )
        
        self._init_weights()
        
        # Move to device
        self.to(self.device)
    
    def _init_weights(self):
        """Initialize network weights with Kaiming for ReLU layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with dueling architecture.
        
        Q(s, a) = V(s) + (A(s, a) - mean(A(s, .)))
        """
        features = self.feature_net(state)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Combine value and advantage
        # Subtract mean advantage to ensure identifiability
        q_values = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        
        return q_values
    
    def get_q_values(
        self, 
        state: torch.Tensor, 
        action_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Get Q-values with optional action masking."""
        q_values = self.forward(state)
        
        if action_mask is not None:
            masked_q = q_values.clone()
            masked_q[action_mask == 0] = float('-inf')
            return masked_q
        
        return q_values
    
    def save(self, filepath: Union[str, Path]) -> None:
        """Save model weights to file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.state_dict(),
            'obs_dim': self.obs_dim,
            'action_dim': self.action_dim,
            'hidden_dims': self.hidden_dims,
            'architecture': 'dueling',
        }, filepath)
    
    @classmethod
    def load(cls, filepath: Union[str, Path], device: Optional[torch.device] = None) -> 'DuelingQNetwork':
        """Load model weights from file."""
        filepath = Path(filepath)
        checkpoint = torch.load(filepath, map_location=device)
        
        model = cls(
            obs_dim=checkpoint['obs_dim'],
            action_dim=checkpoint['action_dim'],
            hidden_dims=checkpoint['hidden_dims']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if device is not None:
            model = model.to(device)
        
        return model
