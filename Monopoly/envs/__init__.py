"""
Monopoly Gymnasium Environment Package

This package provides a Gymnasium-compatible environment for the Monopoly board game,
suitable for training reinforcement learning agents.

NOTE: RewardShapingWrapper has been removed. All reward logic for the H1 ablation 
study is handled directly in MonopolyEnv via the `reward_mode` parameter.
"""

from .gym_env import MonopolyEnv
from .wrappers import MonopolyFlattenWrapper, ActionMaskWrapper, PettingZooToGymWrapper
from .observation import encode_observation_compact, encode_observation_dict
from .tracing import EpisodeTracer
from .pettingzoo_env import MonopolyAECEnv
from .stalemate import StalemateDetector

__all__ = [
    'MonopolyEnv',
    'MonopolyAECEnv',
    'MonopolyFlattenWrapper',
    'ActionMaskWrapper',
    'PettingZooToGymWrapper',
    'StalemateDetector',
    'EpisodeTracer',
    'encode_observation_compact',
    'encode_observation_dict',
]
