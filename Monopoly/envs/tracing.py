"""Episode tracing wrapper for recording per-turn game state snapshots.

This module provides a Gymnasium wrapper that intercepts step() calls to record
detailed per-turn data (NPZ) and per-episode summaries (JSON). It must wrap
the raw MonopolyEnv *before* the MonopolyFlattenWrapper so it can access dict
observations with property_owner, houses, etc.

Wrapper ordering: MonopolyEnv -> EpisodeTracer -> MonopolyFlattenWrapper -> Monitor

Usage:
    env = MonopolyEnv(...)
    env = EpisodeTracer(env, output_dir="runs/analysis/traces")
    env = MonopolyFlattenWrapper(env)
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import gymnasium as gym
import numpy as np

from Monopoly.property import load_property_specs


def build_color_groups() -> Dict[str, List[int]]:
    """Build mapping from color group name to list of property indices.

    Only includes buildable color groups (excludes Railroad, Utility).
    """
    specs = load_property_specs()
    groups: Dict[str, List[int]] = {}
    for spec in specs:
        if spec.group not in ("Railroad", "Utility"):
            groups.setdefault(spec.group, []).append(spec.idx)
    return groups


COLOR_GROUPS = build_color_groups()


class EpisodeTracer(gym.Wrapper):
    """Records per-turn snapshots (NPZ) and per-episode summaries (JSON).

    Must wrap raw MonopolyEnv (before flattening) so it sees dict observations.

    Args:
        env: The MonopolyEnv instance (must have Dict observation space).
        output_dir: Directory where trace files are written.
        enabled: When False, wrapper is pure passthrough with zero overhead.
        model_path: Optional path to the model being evaluated (for metadata).
    """

    def __init__(
        self,
        env: gym.Env,
        output_dir: str,
        enabled: bool = True,
        model_path: str = "",
    ):
        super().__init__(env)

        # Validate wrapper ordering
        assert isinstance(
            env.observation_space, gym.spaces.Dict
        ), (
            "EpisodeTracer must wrap the raw MonopolyEnv (Dict obs space), "
            "not a flattened environment. Place it before MonopolyFlattenWrapper."
        )

        self.output_dir = Path(output_dir)
        self.enabled = enabled
        self.model_path = model_path
        self._episode_count = 0

        # Per-turn buffers (reset each episode)
        self._turn_buffers: Dict[str, list] = {}
        self._actions: List[int] = []
        self._rewards: List[float] = []
        self._prev_active: int = 0
        self._bankruptcies: List[Dict[str, int]] = []
        self._first_monopoly_turn: Optional[int] = None
        self._total_houses_built: int = 0
        self._total_trades: int = 0
        self._max_properties_held: int = 0
        self._buy_opportunities: int = 0
        self._buy_actions: int = 0

        if self.enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def reset(self, **kwargs):
        """Reset environment and initialize new trace buffers."""
        obs, info = self.env.reset(**kwargs)

        if self.enabled:
            # Flush previous episode if there was one
            # (handled in step on term/trunc, but safety net for forced resets)

            # Initialize buffers
            self._turn_buffers = {
                "net_worth": [],
                "cash": [],
                "property_owner": [],
                "houses": [],
                "positions": [],
            }
            self._actions = []
            self._rewards = []
            self._prev_active = info.get("active_players", 4)
            self._bankruptcies = []
            self._first_monopoly_turn = None
            self._total_houses_built = 0
            self._total_trades = 0
            self._max_properties_held = 0
            self._buy_opportunities = 0
            self._buy_actions = 0

            # Record initial state
            self._record_obs(obs, info)

        return obs, info

    def step(self, action: int):
        """Execute step and record trace data."""
        obs, reward, terminated, truncated, info = self.env.step(action)

        if self.enabled:
            self._record_turn(action, obs, reward, info)

            if terminated or truncated:
                self._flush_episode(obs, info, terminated)

        return obs, reward, terminated, truncated, info

    def _record_obs(self, obs: Dict[str, Any], info: Dict[str, Any]):
        """Record observation arrays into turn buffers."""
        self._turn_buffers["net_worth"].append(obs["net_worth"].copy())
        self._turn_buffers["cash"].append(obs["cash"].copy())
        self._turn_buffers["property_owner"].append(obs["property_owner"].copy())
        self._turn_buffers["houses"].append(obs["houses"].copy())
        self._turn_buffers["positions"].append(obs["positions"].copy())

    def _record_turn(self, action: int, obs: Dict[str, Any], reward: float, info: Dict[str, Any]):
        """Record one turn of data."""
        self._record_obs(obs, info)
        self._actions.append(action)
        self._rewards.append(float(reward))

        turn = info.get("turn_number", len(self._actions))

        # Detect bankruptcies
        current_active = info.get("active_players", 0)
        if current_active < self._prev_active:
            self._bankruptcies.append({"player": -1, "turn": turn})
        self._prev_active = current_active

        # Detect trades (actions 90-173)
        if action >= 90:
            self._total_trades += 1

        # Detect buy opportunities and actions
        if action == 1:
            self._buy_actions += 1
            self._buy_opportunities += 1
        elif action == 2:
            self._buy_opportunities += 1

        # Detect house building (actions 3-30)
        if 3 <= action <= 30:
            self._total_houses_built += 1

        # Track max properties held by agent
        agent_id = obs["player_id"][0]
        props_owned = int(np.sum(obs["property_owner"] == agent_id))
        self._max_properties_held = max(self._max_properties_held, props_owned)

        # Detect first monopoly
        if self._first_monopoly_turn is None:
            owners = obs["property_owner"]
            for group_name, indices in COLOR_GROUPS.items():
                if all(owners[idx] == agent_id for idx in indices):
                    self._first_monopoly_turn = turn
                    break

    def _flush_episode(self, final_obs: Dict[str, Any], final_info: Dict[str, Any], terminated: bool):
        """Save episode trace to NPZ + JSON files."""
        ep_id = self._episode_count
        self._episode_count += 1

        # Stack turn buffers into arrays
        trace_arrays = {}
        for key, buf in self._turn_buffers.items():
            if buf:
                trace_arrays[key] = np.array(buf)

        trace_arrays["action_taken"] = np.array(self._actions, dtype=np.int32)
        trace_arrays["reward"] = np.array(self._rewards, dtype=np.float32)

        # Save NPZ
        npz_path = self.output_dir / f"episode_{ep_id:04d}.npz"
        np.savez_compressed(npz_path, **trace_arrays)

        # Build episode summary
        final_nw = final_obs["net_worth"]
        agent_id = int(final_obs["player_id"][0])
        agent_nw = float(final_nw[agent_id])

        # Determine winner and rank
        nw_list = [float(v) for v in final_nw]
        sorted_nws = sorted(enumerate(nw_list), key=lambda x: -x[1])
        agent_rank = next(i + 1 for i, (pid, _) in enumerate(sorted_nws) if pid == agent_id)

        if terminated:
            # Winner is the last active player (highest NW if multiple active)
            winner = sorted_nws[0][0]
        else:
            winner = -1  # truncated

        # Get reward mode from env
        reward_mode = getattr(self.env.unwrapped, "reward_mode", "unknown")

        summary = {
            "episode_id": ep_id,
            "seed": getattr(self.env.unwrapped, "_seed", None),
            "model_path": self.model_path,
            "reward_mode": str(reward_mode),
            "num_players": len(nw_list),
            "episode_length": len(self._actions),
            "winner": winner,
            "final_net_worths": nw_list,
            "agent_win": agent_rank == 1,
            "agent_rank": agent_rank,
            "total_reward": sum(self._rewards),
            "bankruptcies": self._bankruptcies,
            "first_monopoly_turn": self._first_monopoly_turn,
            "total_houses_built": self._total_houses_built,
            "total_trades": self._total_trades,
            "max_properties_held": self._max_properties_held,
        }

        # Save JSON
        json_path = self.output_dir / f"episode_{ep_id:04d}.json"
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)
