"""Tests for rollout recording, headless rendering, overlays, and export."""

import os
import tempfile
import random
from pathlib import Path

import pytest
import numpy as np

from Monopoly.envs.gym_env import MonopolyEnv
from Monopoly.envs.rollout import save_rollout, load_rollout
from Monopoly.envs.renderer import MonopolyRenderer
from Monopoly.board import Board
from Monopoly.property import load_property_specs


def _run_episode(env, steps=10):
    """Run a short episode returning (obs, done) after at most `steps` actions."""
    obs, info = env.reset()
    for _ in range(steps):
        legal = obs['legal_mask']
        legal_indices = [i for i, v in enumerate(legal) if v == 1]
        action = random.choice(legal_indices) if legal_indices else 89
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    return obs, terminated or truncated


class TestRolloutRecordAndLoad:
    def test_roundtrip(self, tmp_path):
        """Record 10 steps, save, load, assert step count and state fields."""
        env = MonopolyEnv(num_players=4, seed=42, record_rollout=True)
        _run_episode(env, steps=10)

        path = tmp_path / "test.rollout.gz"
        env.save_rollout(str(path))
        env.close()

        rollout = load_rollout(path)
        assert len(rollout.steps) >= 2  # at least initial + 1 step
        assert rollout.metadata['seed'] == 42
        assert rollout.metadata['num_players'] == 4

    def test_step_record_fields(self):
        """Each StepRecord has valid state, net_worths, action_label."""
        env = MonopolyEnv(num_players=4, seed=99, record_rollout=True)
        _run_episode(env, steps=5)
        rollout = env.current_rollout
        env.close()

        for step in rollout.steps:
            assert step.state is not None
            assert len(step.net_worths) == 4
            assert isinstance(step.action_label, str) and len(step.action_label) > 0
            # Step 0 has action=None
            if step.step_idx == 0:
                assert step.action is None
            else:
                assert isinstance(step.action, int)

    def test_no_recording_raises(self):
        """save_rollout raises when recording is not enabled."""
        env = MonopolyEnv(num_players=4, seed=42, record_rollout=False)
        _run_episode(env, steps=2)
        with pytest.raises(RuntimeError, match="not enabled"):
            env.save_rollout("test.rollout.gz")
        env.close()


class TestActionToLabel:
    def test_representative_actions(self):
        """Representative actions return non-empty strings."""
        env = MonopolyEnv(num_players=4, seed=42)
        env.reset()

        test_actions = [0, 1, 2, 3, 31, 59, 87, 88, 89, 90]
        for action in test_actions:
            label = env._action_to_label(action)
            assert isinstance(label, str)
            assert len(label) > 0
        env.close()


class TestHeadlessRenderer:
    def test_capture_frame(self):
        """Headless renderer returns correct shape array."""
        board = Board.load_standard_board()
        specs = load_property_specs()
        renderer = MonopolyRenderer(board, specs, headless=True)

        # Create a minimal game state for rendering
        env = MonopolyEnv(num_players=4, seed=42)
        obs, info = env.reset()

        renderer.render(env.state, show_stats=True)
        frame = renderer.capture_frame()

        assert isinstance(frame, np.ndarray)
        assert frame.dtype == np.uint8
        assert frame.shape == (960, 1400, 3)

        renderer.close()
        env.close()


class TestOverlays:
    def test_overlay_no_crash(self):
        """Render with full overlay_data dict, assert no exception."""
        board = Board.load_standard_board()
        specs = load_property_specs()
        renderer = MonopolyRenderer(board, specs, headless=True)

        env = MonopolyEnv(num_players=4, seed=42)
        env.reset()

        overlay = {
            'net_worths': [1500.0, 1500.0, 1500.0, 1500.0],
            'action_label': 'Roll Dice',
            'net_worth_history': [
                [1500.0, 1500.0, 1500.0, 1500.0],
                [1600.0, 1400.0, 1500.0, 1500.0],
                [1700.0, 1300.0, 1550.0, 1450.0],
            ],
            'turn_number': 10,
            'total_turns': 500,
        }

        # Should not raise
        renderer.render(env.state, show_stats=True, overlay_data=overlay)
        frame = renderer.capture_frame()
        assert frame.shape == (960, 1400, 3)

        renderer.close()
        env.close()


class TestExportSmoke:
    def test_export_mp4(self, tmp_path):
        """Record a short rollout, export to mp4, assert file exists and size > 0."""
        imageio = pytest.importorskip("imageio")

        env = MonopolyEnv(num_players=4, seed=42, record_rollout=True)
        _run_episode(env, steps=5)
        rollout = env.current_rollout
        env.close()

        from scripts.replay_viewer import export_video
        out_path = str(tmp_path / "test_episode.mp4")
        export_video(rollout, out_path, fmt='mp4', fps=2)

        assert os.path.exists(out_path)
        assert os.path.getsize(out_path) > 0
