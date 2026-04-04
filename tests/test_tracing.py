"""Tests for episode tracing wrapper and feature extraction pipeline.

Validates:
1. EpisodeTracer produces NPZ + JSON files with correct schema
2. Tracer does not alter environment behavior (passthrough)
3. Feature extraction computes expected values from synthetic data
4. Clustering pipeline runs without errors
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from Monopoly.envs.gym_env import MonopolyEnv
from Monopoly.envs.tracing import EpisodeTracer, COLOR_GROUPS


# =============================================================================
# TRACER FILE PRODUCTION
# =============================================================================

class TestTracerFileProduction:
    """Verify that the tracer writes expected output files."""

    def _run_episodes(self, env, n_episodes=2):
        """Run n episodes with random legal actions."""
        for _ in range(n_episodes):
            obs, info = env.reset(seed=42)
            done = False
            while not done:
                legal = np.where(obs["legal_mask"] == 1)[0]
                action = int(np.random.choice(legal)) if len(legal) > 0 else 0
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

    def test_tracer_produces_files(self, tmp_path):
        """Run 2 short episodes, assert NPZ+JSON files exist."""
        env = MonopolyEnv(num_players=2, max_turns=10, seed=42)
        env = EpisodeTracer(env, output_dir=str(tmp_path), enabled=True)

        self._run_episodes(env, n_episodes=2)

        npz_files = sorted(tmp_path.glob("episode_*.npz"))
        json_files = sorted(tmp_path.glob("episode_*.json"))
        assert len(npz_files) == 2, f"Expected 2 NPZ files, got {len(npz_files)}"
        assert len(json_files) == 2, f"Expected 2 JSON files, got {len(json_files)}"

    def test_trace_schema_completeness(self, tmp_path):
        """Load NPZ file, verify all expected keys with correct shapes."""
        env = MonopolyEnv(num_players=2, max_turns=15, seed=42)
        env = EpisodeTracer(env, output_dir=str(tmp_path))

        self._run_episodes(env, n_episodes=1)

        npz_path = list(tmp_path.glob("episode_*.npz"))[0]
        trace = dict(np.load(npz_path))

        expected_keys = {
            "net_worth", "cash", "property_owner", "houses",
            "positions", "action_taken", "reward",
        }
        assert expected_keys.issubset(set(trace.keys())), (
            f"Missing keys: {expected_keys - set(trace.keys())}"
        )

        # All observation arrays should have T+1 rows (initial + T steps)
        # action_taken and reward should have T rows
        t_actions = trace["action_taken"].shape[0]
        assert t_actions > 0, "Episode produced no actions"

        # Obs arrays have one more row than actions (initial obs included)
        for key in ["net_worth", "cash", "positions"]:
            assert trace[key].shape[0] == t_actions + 1, (
                f"{key} has {trace[key].shape[0]} rows, expected {t_actions + 1}"
            )

    def test_trace_json_fields(self, tmp_path):
        """Load JSON sidecar, verify required fields exist with correct types."""
        env = MonopolyEnv(num_players=2, max_turns=15, seed=42)
        env = EpisodeTracer(env, output_dir=str(tmp_path))

        self._run_episodes(env, n_episodes=1)

        json_path = list(tmp_path.glob("episode_*.json"))[0]
        with open(json_path) as f:
            summary = json.load(f)

        required_fields = {
            "episode_id": int,
            "episode_length": int,
            "winner": int,
            "final_net_worths": list,
            "agent_win": bool,
            "agent_rank": int,
            "total_reward": (int, float),
            "bankruptcies": list,
            "total_houses_built": int,
            "total_trades": int,
            "max_properties_held": int,
        }

        for field, expected_type in required_fields.items():
            assert field in summary, f"Missing field: {field}"
            assert isinstance(summary[field], expected_type), (
                f"Field {field}: expected {expected_type}, got {type(summary[field])}"
            )


# =============================================================================
# TRACER DISABLED / PASSTHROUGH
# =============================================================================

class TestTracerBehavior:
    """Verify tracer does not alter environment behavior."""

    def test_tracer_disabled(self, tmp_path):
        """enabled=False → no files written."""
        env = MonopolyEnv(num_players=2, max_turns=10, seed=42)
        env = EpisodeTracer(env, output_dir=str(tmp_path), enabled=False)

        obs, _ = env.reset(seed=42)
        done = False
        while not done:
            legal = np.where(obs["legal_mask"] == 1)[0]
            action = int(np.random.choice(legal)) if len(legal) > 0 else 0
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

        assert len(list(tmp_path.glob("*"))) == 0, "Files written when tracer disabled"

    def test_tracer_passthrough(self, tmp_path):
        """Same seed with/without tracer produces identical trajectories."""
        # Run without tracer
        env1 = MonopolyEnv(num_players=2, max_turns=20, seed=42)
        obs1, _ = env1.reset(seed=42)
        rng = np.random.RandomState(99)
        trajectory1 = []
        done = False
        while not done:
            legal = np.where(obs1["legal_mask"] == 1)[0]
            action = int(rng.choice(legal)) if len(legal) > 0 else 0
            obs1, reward1, term1, trunc1, _ = env1.step(action)
            trajectory1.append((action, reward1, term1, trunc1))
            done = term1 or trunc1

        # Run with tracer
        env2 = MonopolyEnv(num_players=2, max_turns=20, seed=42)
        env2 = EpisodeTracer(env2, output_dir=str(tmp_path))
        obs2, _ = env2.reset(seed=42)
        rng2 = np.random.RandomState(99)
        trajectory2 = []
        done = False
        while not done:
            legal = np.where(obs2["legal_mask"] == 1)[0]
            action = int(rng2.choice(legal)) if len(legal) > 0 else 0
            obs2, reward2, term2, trunc2, _ = env2.step(action)
            trajectory2.append((action, reward2, term2, trunc2))
            done = term2 or trunc2

        assert len(trajectory1) == len(trajectory2), "Trajectory lengths differ"
        for i, (t1, t2) in enumerate(zip(trajectory1, trajectory2)):
            assert t1[0] == t2[0], f"Step {i}: actions differ"
            assert abs(t1[1] - t2[1]) < 1e-6, f"Step {i}: rewards differ"
            assert t1[2] == t2[2], f"Step {i}: terminated differs"
            assert t1[3] == t2[3], f"Step {i}: truncated differs"


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

class TestFeatureExtraction:
    """Validate feature computation from trace data."""

    def test_feature_extraction(self):
        """Synthetic data → verify computed features match expected values."""
        from utils.trace_utils import extract_features

        summary = {
            "episode_id": 0,
            "episode_length": 100,
            "first_monopoly_turn": 50,
            "total_houses_built": 10,
            "total_trades": 5,
            "agent_win": True,
            "agent_rank": 1,
            "final_net_worths": [3000.0, 1000.0],
            "max_properties_held": 6,
        }

        # Build synthetic trace arrays
        trace = {
            "net_worth": np.column_stack([
                np.linspace(1500, 3000, 101),  # agent NW rising
                np.linspace(1500, 1000, 101),  # opponent NW falling
            ]).astype(np.float32),
            "cash": np.column_stack([
                np.full(101, 500, dtype=np.float32),
                np.full(101, 800, dtype=np.float32),
            ]),
            "property_owner": np.zeros((101, 28), dtype=np.int32) - 1,  # all unowned
            "action_taken": np.zeros(100, dtype=np.int32),  # all rolls
            "reward": np.ones(100, dtype=np.float32),
        }

        # Set some actions to buy (1) and pass (2)
        trace["action_taken"][10] = 1  # buy
        trace["action_taken"][20] = 1  # buy
        trace["action_taken"][30] = 2  # pass
        trace["action_taken"][40] = 1  # buy

        features = extract_features(summary, trace)

        assert abs(features["time_to_first_monopoly"] - 0.5) < 1e-6
        assert abs(features["build_rate"] - 0.1) < 1e-6
        assert abs(features["trade_frequency"] - 0.05) < 1e-6
        assert features["peak_net_worth_ratio"] == 3000.0 / 1500.0
        assert abs(features["aggression_score"] - 3 / 4) < 1e-6  # 3 buys / 4 buy-or-pass
        assert features["mortgage_rate"] == 0.0
        assert abs(features["endgame_dominance"] - 3000 / 4000) < 1e-6

    def test_feature_extraction_no_monopoly(self):
        """When no monopoly achieved, time_to_first_monopoly should be 1.0."""
        from utils.trace_utils import extract_features

        summary = {
            "episode_id": 1,
            "episode_length": 50,
            "first_monopoly_turn": None,
            "total_houses_built": 0,
            "total_trades": 0,
            "agent_win": False,
            "agent_rank": 2,
            "final_net_worths": [1000.0, 2000.0],
            "max_properties_held": 2,
        }
        trace = {
            "net_worth": np.ones((51, 2), dtype=np.float32) * 1500,
            "cash": np.ones((51, 2), dtype=np.float32) * 1500,
            "property_owner": np.full((51, 28), -1, dtype=np.int32),
            "action_taken": np.zeros(50, dtype=np.int32),
            "reward": np.zeros(50, dtype=np.float32),
        }

        features = extract_features(summary, trace)
        assert features["time_to_first_monopoly"] == 1.0


# =============================================================================
# CLUSTERING SMOKE TEST
# =============================================================================

class TestClusteringSmoke:
    """Verify clustering pipeline runs without errors."""

    def test_clustering_smoke(self):
        """20 synthetic feature vectors → KMeans produces assignments."""
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        rng = np.random.RandomState(42)
        n_episodes = 20
        n_features = 9
        features = rng.rand(n_episodes, n_features)

        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features_scaled)

        assert labels.shape == (n_episodes,)
        assert len(set(labels)) >= 2  # at least 2 clusters used
        assert all(0 <= l < 3 for l in labels)
