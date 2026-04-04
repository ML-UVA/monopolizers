"""Modular reward function for Monopoly RL.

Provides interpretable, component-based reward computation with optional
online normalization. Designed as a third reward_mode ('modular') alongside
'dense_networth' and 'sparse_terminal' for the H1 ablation study.

Components:
    - delta_networth: Change in agent net worth since last step
    - buy_property: Bonus when agent buys a property
    - build: Bonus when agent builds a house/hotel
    - bankrupt: Penalty at terminal if agent is bankrupt
    - time_penalty: Small per-step cost to encourage faster play
"""

import math
from typing import Any, Dict, Optional, Tuple


DEFAULT_REWARD_WEIGHTS = {
    'delta_networth': 1.0,
    'buy_property': 0.1,
    'build': 0.05,
    'bankrupt': -1.0,
    'time_penalty': -0.001,
}


class RunningNormalizer:
    """Online reward normalizer using Welford's algorithm for running mean/variance."""

    def __init__(self, clip: float = 10.0):
        self.clip = clip
        self.mean = 0.0
        self.var = 1.0
        self.count = 0
        self._m2 = 0.0

    def update(self, x: float) -> None:
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self._m2 += delta * delta2
        self.var = self._m2 / max(self.count, 1)

    def normalize(self, x: float) -> float:
        std = math.sqrt(self.var) if self.var > 1e-8 else 1.0
        normalized = (x - self.mean) / std
        if self.clip is not None:
            normalized = max(-self.clip, min(self.clip, normalized))
        return normalized

    def state_dict(self) -> Dict[str, Any]:
        return {
            'mean': self.mean,
            'var': self.var,
            'count': self.count,
            '_m2': self._m2,
            'clip': self.clip,
        }

    def load_state_dict(self, d: Dict[str, Any]) -> None:
        self.mean = d['mean']
        self.var = d['var']
        self.count = d['count']
        self._m2 = d['_m2']
        self.clip = d['clip']


class ModularRewardCalculator:
    """Computes reward as a weighted sum of interpretable components.

    Each component is computed independently and returned alongside the
    total for diagnostic logging.

    Args:
        weights: Dict mapping component names to scalar weights.
        normalize: If True, apply online normalization to total reward.
        clip: Clip value for the normalizer (None = no clipping).
    """

    # Action indices for component detection
    ACTION_BUY = 1
    ACTION_BUILD_START = 3
    ACTION_BUILD_END = 30  # inclusive

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        normalize: bool = False,
        clip: Optional[float] = 10.0,
    ):
        self.weights = {**DEFAULT_REWARD_WEIGHTS, **(weights or {})}
        self.normalizer = RunningNormalizer(clip=clip) if normalize else None
        self._prev_net_worth: Optional[float] = None

    def reset(self) -> None:
        """Call at episode start to clear per-episode state."""
        self._prev_net_worth = None

    def compute(
        self,
        info: Dict[str, Any],
        action: int,
        terminated: bool,
        truncated: bool,
    ) -> Tuple[float, Dict[str, float]]:
        """Compute modular reward from environment info and action taken.

        Args:
            info: Info dict from MonopolyEnv.step() containing 'agent_net_worth',
                  'agent_status', etc.
            action: The discrete action that was executed.
            terminated: Whether the episode terminated.
            truncated: Whether the episode was truncated.

        Returns:
            (total_reward, components_dict) where components_dict maps each
            component name to its weighted value for this step.
        """
        agent_nw = info.get('agent_net_worth', 0.0)
        components = {}

        # delta_networth: change in net worth since last step
        if self._prev_net_worth is not None:
            delta = agent_nw - self._prev_net_worth
        else:
            delta = 0.0
        components['delta_networth'] = delta * self.weights['delta_networth']
        self._prev_net_worth = agent_nw

        # buy_property: +1 when a buy action was executed
        bought = 1.0 if action == self.ACTION_BUY else 0.0
        components['buy_property'] = bought * self.weights['buy_property']

        # build: +1 when a build action was executed
        built = 1.0 if self.ACTION_BUILD_START <= action <= self.ACTION_BUILD_END else 0.0
        components['build'] = built * self.weights['build']

        # bankrupt: penalty at terminal if agent went bankrupt
        if (terminated or truncated) and info.get('agent_status') == 'BANKRUPT':
            components['bankrupt'] = self.weights['bankrupt']
        else:
            components['bankrupt'] = 0.0

        # time_penalty: constant per-step cost
        components['time_penalty'] = self.weights['time_penalty']

        # Total reward
        total = sum(components.values())

        # Optional online normalization
        if self.normalizer is not None:
            self.normalizer.update(total)
            total = self.normalizer.normalize(total)

        return total, components

    def state_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            'weights': self.weights.copy(),
            '_prev_net_worth': self._prev_net_worth,
        }
        if self.normalizer is not None:
            d['normalizer'] = self.normalizer.state_dict()
        return d

    def load_state_dict(self, d: Dict[str, Any]) -> None:
        self.weights = d['weights']
        self._prev_net_worth = d['_prev_net_worth']
        if self.normalizer is not None and 'normalizer' in d:
            self.normalizer.load_state_dict(d['normalizer'])
