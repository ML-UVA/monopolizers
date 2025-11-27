"""
Monopoly Gymnasium Environment Package

This package provides a Gymnasium-compatible environment for the Monopoly board game,
suitable for training reinforcement learning agents.
"""

from .gym_env import MonopolyEnv
from .wrappers import MonopolyFlattenWrapper, ActionMaskWrapper, RewardShapingWrapper
from .observation import encode_observation_compact, encode_observation_dict

__all__ = [
    'MonopolyEnv',
    'MonopolyFlattenWrapper',
    'ActionMaskWrapper',
    'RewardShapingWrapper',
    'encode_observation_compact',
    'encode_observation_dict',
]
