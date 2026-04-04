"""Rollout recording and replay for Monopoly episodes.

Provides dataclasses for step-by-step game state recording,
gzip-compressed pickle serialization, and a recorder class
for integration with MonopolyEnv.
"""

import gzip
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..state import GameState


@dataclass
class StepRecord:
    """A single recorded step in a Monopoly episode."""
    step_idx: int
    state: GameState
    action: Optional[int]       # None for initial frame (step 0)
    action_label: str           # human-readable, e.g. "Roll Dice"
    reward: float
    net_worths: List[float]     # per player
    terminated: bool
    truncated: bool


@dataclass
class Rollout:
    """A complete recorded Monopoly episode."""
    steps: List[StepRecord]
    metadata: Dict[str, Any] = field(default_factory=dict)


class RolloutRecorder:
    """Accumulates StepRecords during an episode."""

    def __init__(self, metadata: Optional[Dict[str, Any]] = None):
        self.metadata = metadata or {}
        self.metadata.setdefault('timestamp', datetime.now().isoformat())
        self._steps: List[StepRecord] = []

    def record_initial(self, state: GameState, net_worths: List[float]) -> None:
        """Record the initial state after reset (step 0, no action)."""
        self._steps.append(StepRecord(
            step_idx=0,
            state=state.clone(),
            action=None,
            action_label="Game Start",
            reward=0.0,
            net_worths=list(net_worths),
            terminated=False,
            truncated=False,
        ))

    def record_step(
        self,
        state: GameState,
        action: int,
        action_label: str,
        reward: float,
        net_worths: List[float],
        terminated: bool,
        truncated: bool,
    ) -> None:
        """Record a step after env.step()."""
        self._steps.append(StepRecord(
            step_idx=len(self._steps),
            state=state.clone(),
            action=action,
            action_label=action_label,
            reward=reward,
            net_worths=list(net_worths),
            terminated=terminated,
            truncated=truncated,
        ))

    def finalize(self) -> Rollout:
        """Return the completed Rollout."""
        return Rollout(steps=list(self._steps), metadata=dict(self.metadata))


def save_rollout(rollout: Rollout, path: Path) -> None:
    """Save a Rollout to a gzip-compressed pickle file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, 'wb') as f:
        pickle.dump(rollout, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_rollout(path: Path) -> Rollout:
    """Load a Rollout from a gzip-compressed pickle file."""
    with gzip.open(path, 'rb') as f:
        return pickle.load(f)
