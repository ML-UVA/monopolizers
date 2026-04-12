"""Trace loading and feature extraction utilities.

Provides functions to load episode traces (NPZ/JSON), extract strategy features,
and load DDQN models for standalone evaluation.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from Monopoly.envs.tracing import COLOR_GROUPS


# ---------------------------------------------------------------------------
# Trace I/O
# ---------------------------------------------------------------------------

def load_episode_summary(json_path: str) -> dict:
    """Load a single episode summary JSON."""
    with open(json_path) as f:
        return json.load(f)


def load_episode_trace(npz_path: str) -> Dict[str, np.ndarray]:
    """Load a single episode trace NPZ and return arrays as dict."""
    with np.load(npz_path) as data:
        return {key: data[key] for key in data.files}


def load_all_summaries(trace_dir: str) -> pd.DataFrame:
    """Load all episode summary JSONs from a trace directory into a DataFrame."""
    trace_dir = Path(trace_dir)
    summaries = []
    for json_path in sorted(trace_dir.glob("episode_*.json")):
        summaries.append(load_episode_summary(str(json_path)))
    if not summaries:
        return pd.DataFrame()
    return pd.DataFrame(summaries)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    "time_to_first_monopoly",
    "build_rate",
    "trade_frequency",
    "peak_net_worth_ratio",
    "aggression_score",
    "mortgage_rate",
    "property_diversity",
    "cash_conservation",
    "endgame_dominance",
]


def extract_features(summary: dict, trace: Dict[str, np.ndarray]) -> dict:
    """Compute a strategy feature vector from one episode's summary + trace.

    Args:
        summary: Loaded JSON summary dict.
        trace: Loaded NPZ arrays dict.

    Returns:
        Dict mapping feature name to float value.
    """
    ep_len = summary["episode_length"]
    if ep_len == 0:
        return {name: 0.0 for name in FEATURE_NAMES}

    # time_to_first_monopoly (normalized, 1.0 if never achieved)
    fmt = summary.get("first_monopoly_turn")
    time_to_first_monopoly = fmt / ep_len if fmt is not None else 1.0

    # build_rate
    build_rate = summary.get("total_houses_built", 0) / ep_len

    # trade_frequency
    trade_frequency = summary.get("total_trades", 0) / ep_len

    # peak_net_worth_ratio
    nw = trace.get("net_worth")
    if nw is not None and nw.shape[0] > 0:
        agent_id = 0  # agent is always player 0
        agent_max_nw = float(np.max(nw[:, agent_id]))
        # Best opponent peak NW
        opponent_ids = [i for i in range(nw.shape[1]) if i != agent_id]
        opp_max_nw = float(np.max(nw[:, opponent_ids])) if opponent_ids else 1.0
        peak_net_worth_ratio = agent_max_nw / max(opp_max_nw, 1.0)
    else:
        peak_net_worth_ratio = 1.0

    # aggression_score (buy actions / buy opportunities)
    actions = trace.get("action_taken", np.array([]))
    if len(actions) > 0:
        buy_mask = actions == 1
        pass_mask = actions == 2
        buy_opps = int(np.sum(buy_mask)) + int(np.sum(pass_mask))
        aggression_score = int(np.sum(buy_mask)) / max(buy_opps, 1)
    else:
        aggression_score = 0.0

    # mortgage_rate
    if len(actions) > 0:
        mortgage_actions = np.sum((actions >= 31) & (actions <= 58))
        mortgage_rate = float(mortgage_actions) / ep_len
    else:
        mortgage_rate = 0.0

    # property_diversity (distinct color groups with >= 1 property)
    if "property_owner" in trace and trace["property_owner"].shape[0] > 0:
        final_owners = trace["property_owner"][-1]
        agent_props = set(np.where(final_owners == 0)[0])
        groups_touched = sum(
            1 for indices in COLOR_GROUPS.values()
            if any(idx in agent_props for idx in indices)
        )
        property_diversity = float(groups_touched)
    else:
        property_diversity = 0.0

    # cash_conservation
    cash = trace.get("cash")
    if cash is not None and nw is not None and cash.shape[0] > 0:
        mean_cash = float(np.mean(cash[:, 0]))
        mean_nw = float(np.mean(nw[:, 0]))
        cash_conservation = mean_cash / max(mean_nw, 1.0)
    else:
        cash_conservation = 0.0

    # endgame_dominance
    final_nws = summary.get("final_net_worths", [])
    if final_nws:
        total_nw = sum(final_nws)
        endgame_dominance = final_nws[0] / max(total_nw, 1.0)
    else:
        endgame_dominance = 0.0

    return {
        "time_to_first_monopoly": time_to_first_monopoly,
        "build_rate": build_rate,
        "trade_frequency": trade_frequency,
        "peak_net_worth_ratio": peak_net_worth_ratio,
        "aggression_score": aggression_score,
        "mortgage_rate": mortgage_rate,
        "property_diversity": property_diversity,
        "cash_conservation": cash_conservation,
        "endgame_dominance": endgame_dominance,
    }


def extract_all_features(trace_dir: str) -> pd.DataFrame:
    """Extract feature vectors for all episodes in a trace directory.

    Returns:
        DataFrame with one row per episode, columns = FEATURE_NAMES + episode_id + agent_win.
    """
    trace_dir = Path(trace_dir)
    rows = []
    for json_path in sorted(trace_dir.glob("episode_*.json")):
        summary = load_episode_summary(str(json_path))
        npz_path = json_path.with_suffix(".npz")
        if npz_path.exists():
            trace = load_episode_trace(str(npz_path))
        else:
            trace = {}
        features = extract_features(summary, trace)
        features["episode_id"] = summary.get("episode_id", 0)
        features["agent_win"] = summary.get("agent_win", False)
        rows.append(features)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# DDQN model loading (standalone, without DDQNHybridTrainer)
# ---------------------------------------------------------------------------

def load_ddqn_for_eval(model_path: str, device: str = "auto"):
    """Load a DDQN QNetwork for evaluation without the full trainer.

    Args:
        model_path: Path to the .pt checkpoint file.
        device: PyTorch device.

    Returns:
        QNetwork instance with loaded weights in eval mode.
    """
    from Monopoly.agents.network import QNetwork

    def _resolve_device(device_name: str) -> torch.device:
        if device_name == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device_name)

    def _infer_dims(state_dict: Dict[str, torch.Tensor]) -> tuple:
        """Infer obs/action/hidden dims from QNetwork-style state_dict."""
        weight_keys = [
            key for key in state_dict.keys()
            if key.startswith("network.") and key.endswith(".weight")
        ]
        if not weight_keys:
            raise ValueError("Could not infer network dimensions from state_dict")

        def layer_index(key: str) -> int:
            # network.0.weight -> 0
            return int(key.split(".")[1])

        ordered_keys = sorted(weight_keys, key=layer_index)
        ordered_weights = [state_dict[key] for key in ordered_keys]

        obs_dim = int(ordered_weights[0].shape[1])
        action_dim = int(ordered_weights[-1].shape[0])
        hidden_dims = [int(weight.shape[0]) for weight in ordered_weights[:-1]]
        return obs_dim, action_dim, hidden_dims

    resolved_device = _resolve_device(device)
    checkpoint = torch.load(model_path, map_location=resolved_device, weights_only=False)

    # Consolidated DDQN trainer checkpoint.
    if isinstance(checkpoint, dict) and "online_state_dict" in checkpoint:
        state_dict = checkpoint["online_state_dict"]
        obs_dim = checkpoint.get("obs_dim")
        action_dim = checkpoint.get("action_dim")

        hidden_dims = checkpoint.get("hidden_dims")
        if hidden_dims is None:
            cfg = checkpoint.get("config")
            if isinstance(cfg, dict):
                hidden_dims = cfg.get("hidden_dims")

        if obs_dim is None or action_dim is None or hidden_dims is None:
            inferred_obs_dim, inferred_action_dim, inferred_hidden_dims = _infer_dims(state_dict)
            obs_dim = inferred_obs_dim if obs_dim is None else int(obs_dim)
            action_dim = inferred_action_dim if action_dim is None else int(action_dim)
            hidden_dims = inferred_hidden_dims if hidden_dims is None else hidden_dims

        model = QNetwork(
            obs_dim=int(obs_dim),
            action_dim=int(action_dim),
            hidden_dims=[int(x) for x in hidden_dims],
            device=str(resolved_device),
        )
        model.load_state_dict(state_dict)
        model.eval()
        return model

    # QNetwork checkpoint format ({model_state_dict, obs_dim, action_dim, hidden_dims}).
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        if all(k in checkpoint for k in ("obs_dim", "action_dim", "hidden_dims")):
            model = QNetwork(
                obs_dim=int(checkpoint["obs_dim"]),
                action_dim=int(checkpoint["action_dim"]),
                hidden_dims=[int(x) for x in checkpoint["hidden_dims"]],
                device=str(resolved_device),
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            return model

        inferred_obs_dim, inferred_action_dim, inferred_hidden_dims = _infer_dims(checkpoint["model_state_dict"])
        model = QNetwork(
            obs_dim=inferred_obs_dim,
            action_dim=inferred_action_dim,
            hidden_dims=inferred_hidden_dims,
            device=str(resolved_device),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model

    # Raw state dict fallback.
    if isinstance(checkpoint, dict) and checkpoint and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
        inferred_obs_dim, inferred_action_dim, inferred_hidden_dims = _infer_dims(checkpoint)
        model = QNetwork(
            obs_dim=inferred_obs_dim,
            action_dim=inferred_action_dim,
            hidden_dims=inferred_hidden_dims,
            device=str(resolved_device),
        )
        model.load_state_dict(checkpoint)
        model.eval()
        return model

    raise ValueError(
        "Unsupported DDQN checkpoint format. Expected consolidated trainer .pt, "
        "QNetwork .pt, or raw PyTorch state_dict."
    )
