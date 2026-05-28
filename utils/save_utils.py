"""
Utility functions for saving and loading training artifacts.

This module provides helpers for:
- CSV writing of per-episode metrics (append-safe)
- Pickle save/load for replay buffers
- Model save/load for PyTorch networks
"""

import os
import csv
import pickle
import torch
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime


# CSV column schema (shared by DQN and DDQN-hybrid paths)
CSV_COLUMNS = [
    'run_id',
    'seed', 
    'timestamp',
    'episode',
    'episode_length',
    'final_net_worth',
    'agent_rank',
    'win_flag',
    'total_steps',
    'mean_episode_reward',
    'eval_tag',  # 'train' or 'eval'
]


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if necessary."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_metrics_csv(
    filepath: Union[str, Path],
    metrics: List[Dict[str, Any]],
    run_id: str,
    seed: int,
    overwrite: bool = False
) -> None:
    """
    Save metrics to CSV file.
    
    Args:
        filepath: Path to CSV file
        metrics: List of metric dictionaries
        run_id: Unique identifier for this run
        seed: Random seed used
        overwrite: If True, overwrite existing file; else append
    """
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    
    file_exists = filepath.exists() and not overwrite
    mode = 'a' if file_exists else 'w'
    
    with open(filepath, mode, newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        
        if not file_exists:
            writer.writeheader()
        
        for metric in metrics:
            row = {
                'run_id': run_id,
                'seed': seed,
                'timestamp': metric.get('timestamp', datetime.now().isoformat()),
                'episode': metric.get('episode', 0),
                'episode_length': metric.get('episode_length', 0),
                'final_net_worth': metric.get('final_net_worth', 0.0),
                'agent_rank': metric.get('agent_rank', 0),
                'win_flag': metric.get('win_flag', 0),
                'total_steps': metric.get('total_steps', 0),
                'mean_episode_reward': metric.get('mean_episode_reward', 0.0),
                'eval_tag': metric.get('eval_tag', 'train'),
            }
            writer.writerow(row)


def append_metrics_csv(
    filepath: Union[str, Path],
    metric: Dict[str, Any],
    run_id: str,
    seed: int
) -> None:
    """
    Append a single metric row to CSV file.
    
    Args:
        filepath: Path to CSV file
        metric: Single metric dictionary
        run_id: Unique identifier for this run
        seed: Random seed used
    """
    save_metrics_csv(filepath, [metric], run_id, seed, overwrite=False)


def save_replay_buffer(
    buffer: Any,
    filepath: Union[str, Path]
) -> None:
    """
    Save replay buffer to pickle file.
    
    Args:
        buffer: Replay buffer object (must be picklable)
        filepath: Path to save file
    """
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    
    with open(filepath, 'wb') as f:
        pickle.dump(buffer, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"Replay buffer saved to: {filepath}")


def load_replay_buffer(filepath: Union[str, Path]) -> Any:
    """
    Load replay buffer from pickle file.
    
    Args:
        filepath: Path to pickle file
        
    Returns:
        Loaded replay buffer object
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Replay buffer not found: {filepath}")
    
    with open(filepath, 'rb') as f:
        buffer = pickle.load(f)
    
    print(f"Replay buffer loaded from: {filepath}")
    return buffer


def save_model(
    model: torch.nn.Module,
    filepath: Union[str, Path],
    optimizer: Optional[torch.optim.Optimizer] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save PyTorch model checkpoint.
    
    Args:
        model: PyTorch model to save
        filepath: Path to save file
        optimizer: Optional optimizer state to save
        metadata: Optional metadata dictionary
    """
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'timestamp': datetime.now().isoformat(),
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    if metadata is not None:
        checkpoint['metadata'] = metadata
    
    torch.save(checkpoint, filepath)
    print(f"Model saved to: {filepath}")


def load_model(
    model: torch.nn.Module,
    filepath: Union[str, Path],
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Load PyTorch model checkpoint.
    
    Args:
        model: PyTorch model to load weights into
        filepath: Path to checkpoint file
        optimizer: Optional optimizer to load state into
        device: Device to map tensors to
        
    Returns:
        Checkpoint dictionary with metadata
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {filepath}")
    
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Model loaded from: {filepath}")
    return checkpoint


def get_run_id(agent_type: str, seed: int) -> str:
    """Generate a unique run ID."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{agent_type}_seed{seed}_{timestamp}"
