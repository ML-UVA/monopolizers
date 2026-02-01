"""
Minimal Test Suite for MonopolyEnv Research Invariants.

This file contains ONLY the tests required to validate:
1. Environment correctness (determinism, termination, action space)
2. H1 ablation invariants (dense vs sparse reward modes)
3. Trading integration correctness

Each test encodes a meaningful research invariant that would matter to reviewers.
"""

import pytest
import numpy as np
from Monopoly.envs.gym_env import MonopolyEnv
from Monopoly.state import PlayerStatus


# =============================================================================
# ENVIRONMENT DETERMINISM (Critical for reproducibility)
# =============================================================================

class TestDeterminism:
    """Verify that identical seeds produce identical trajectories."""

    def test_reset_determinism(self):
        """Same seed → same initial observation."""
        env1 = MonopolyEnv(num_players=2, seed=42)
        env2 = MonopolyEnv(num_players=2, seed=42)
        
        obs1, _ = env1.reset(seed=42)
        obs2, _ = env2.reset(seed=42)
        
        np.testing.assert_array_equal(obs1['cash'], obs2['cash'])
        np.testing.assert_array_equal(obs1['positions'], obs2['positions'])
        np.testing.assert_array_equal(obs1['legal_mask'], obs2['legal_mask'])

    def test_trajectory_determinism(self):
        """Same seed + same actions → identical game state progression."""
        env1 = MonopolyEnv(num_players=2, seed=42)
        env2 = MonopolyEnv(num_players=2, seed=42)
        
        obs1, _ = env1.reset(seed=42)
        obs2, _ = env2.reset(seed=42)
        
        # Track cash (deterministic) rather than rewards (may have float issues)
        cash1, cash2 = [], []
        for _ in range(10):
            # Take roll action (0) which is always legal at turn start
            obs1, r1, term1, trunc1, _ = env1.step(0)
            obs2, r2, term2, trunc2, _ = env2.step(0)
            
            cash1.append(obs1['cash'].copy())
            cash2.append(obs2['cash'].copy())
            
            if term1 or trunc1:
                break
        
        # Game states must be identical
        for i, (c1, c2) in enumerate(zip(cash1, cash2)):
            np.testing.assert_array_equal(c1, c2, err_msg=f"Step {i}: cash diverged")


# =============================================================================
# H1 ABLATION: REWARD MODE INVARIANTS (Core research hypothesis)
# =============================================================================

class TestRewardModes:
    """
    Tests for H1 ablation: dense_networth vs sparse_terminal.
    
    These tests validate the core research question:
    - Dense: non-zero during play, based on relative net worth
    - Sparse: zero during play, ±1 only at terminal
    """

    def test_invalid_reward_mode_raises(self):
        """Only valid reward modes are accepted."""
        with pytest.raises(ValueError, match="Invalid reward_mode"):
            MonopolyEnv(num_players=2, seed=42, reward_mode='invalid_mode')

    def test_dense_reward_nonzero_during_play(self):
        """Dense mode produces non-zero reward during gameplay."""
        env = MonopolyEnv(num_players=2, seed=42, reward_mode='dense_networth')
        env.reset(seed=42)
        
        _, reward, terminated, truncated, _ = env.step(0)
        
        if not (terminated or truncated):
            assert reward > 0, "Dense reward should be positive during play"

    def test_sparse_reward_zero_during_play(self):
        """Sparse mode produces exactly 0 reward during gameplay."""
        env = MonopolyEnv(num_players=2, seed=42, reward_mode='sparse_terminal')
        env.reset(seed=42)
        
        _, reward, terminated, truncated, _ = env.step(0)
        
        if not (terminated or truncated):
            assert reward == 0.0, f"Sparse reward should be 0 during play, got {reward}"

    def test_reward_modes_same_transitions(self):
        """
        Switching reward_mode changes ONLY the reward stream.
        Observations and termination conditions must be identical.
        """
        env_dense = MonopolyEnv(num_players=2, seed=42, reward_mode='dense_networth')
        env_sparse = MonopolyEnv(num_players=2, seed=42, reward_mode='sparse_terminal')
        
        obs_d, _ = env_dense.reset(seed=42)
        obs_s, _ = env_sparse.reset(seed=42)
        
        # Observations must match
        np.testing.assert_array_equal(obs_d['cash'], obs_s['cash'])
        np.testing.assert_array_equal(obs_d['positions'], obs_s['positions'])
        
        # Step with same action
        obs_d, r_d, term_d, trunc_d, _ = env_dense.step(0)
        obs_s, r_s, term_s, trunc_s, _ = env_sparse.step(0)
        
        # Transitions must match
        np.testing.assert_array_equal(obs_d['cash'], obs_s['cash'])
        np.testing.assert_array_equal(obs_d['positions'], obs_s['positions'])
        assert term_d == term_s
        assert trunc_d == trunc_s
        
        # But rewards must differ (unless terminal)
        if not (term_d or trunc_d):
            assert r_d != r_s, "Reward modes should produce different rewards"


# =============================================================================
# ACTION SPACE INVARIANTS
# =============================================================================

class TestActionSpace:
    """Verify action space correctness and legal action masking."""

    def test_action_space_size(self):
        """Action space is Discrete(174) = 90 base + 84 trade actions."""
        env = MonopolyEnv(num_players=4, seed=42)
        assert env.action_space.n == 174

    def test_legal_mask_at_turn_start(self):
        """At turn start, only roll (action 0) should be legal."""
        env = MonopolyEnv(num_players=2, seed=42)
        obs, _ = env.reset(seed=42)
        
        legal_mask = obs['legal_mask']
        
        # Roll must be legal
        assert legal_mask[0] == 1, "Roll should be legal at turn start"
        # End turn must NOT be legal before rolling
        assert legal_mask[89] == 0, "End turn should not be legal before rolling"

    def test_illegal_action_corrected(self):
        """Environment corrects illegal actions to legal ones."""
        env = MonopolyEnv(num_players=2, seed=42)
        obs, _ = env.reset(seed=42)
        
        # Try end_turn (89) before rolling - should be corrected to roll (0)
        obs, reward, terminated, truncated, info = env.step(89)
        
        # Should not crash and should have valid observation
        assert env.observation_space.contains(obs)


# =============================================================================
# TRADING INTEGRATION (Affects net worth → H1)
# =============================================================================

class TestTradingActions:
    """Verify trading actions are correctly integrated."""

    def test_trade_action_range(self):
        """Trade actions occupy indices 90-173."""
        env = MonopolyEnv(num_players=4, seed=42)
        
        # Decode action 90 (first trade)
        trade = env._decode_action(90)
        assert trade['type'] == 'trade'
        
        # Decode action 173 (last trade)
        trade = env._decode_action(173)
        assert trade['type'] == 'trade'

    def test_trade_legal_mask_empty_when_no_properties(self):
        """Trade actions are illegal when agent owns no tradeable properties."""
        env = MonopolyEnv(num_players=2, seed=42)
        obs, _ = env.reset(seed=42)
        
        # At game start, agent owns nothing
        legal_mask = obs['legal_mask']
        trade_mask = legal_mask[90:]  # Trade action range
        
        # All trade actions should be illegal
        assert np.sum(trade_mask) == 0, "No trade actions legal without properties"


# =============================================================================
# TERMINATION CONDITIONS
# =============================================================================

class TestTermination:
    """Verify correct game termination behavior."""

    def test_bankruptcy_terminates_game(self):
        """Game terminates when opponent goes bankrupt."""
        env = MonopolyEnv(num_players=2, seed=42)
        env.reset(seed=42)
        
        # Force opponent bankruptcy
        env.state.players[1].status = PlayerStatus.BANKRUPT
        env.state.players[1].cash = -100
        
        obs, reward, terminated, truncated, info = env.step(0)
        
        # Game should terminate or show single active player
        assert terminated or info['active_players'] == 1

    def test_max_turns_truncation(self):
        """Game truncates after max_turns."""
        env = MonopolyEnv(num_players=2, seed=42, max_turns=5)
        env.reset(seed=42)
        
        # Run until truncation
        for _ in range(100):  # More than enough steps
            obs, reward, terminated, truncated, info = env.step(0)
            if terminated or truncated:
                break
        
        # Should have truncated (or terminated for other reasons)
        assert terminated or truncated


# =============================================================================
# OBSERVATION SPACE COMPLIANCE
# =============================================================================

class TestObservationSpace:
    """Verify observations match declared space."""

    def test_observation_in_space(self):
        """All observations must be in the declared observation space."""
        env = MonopolyEnv(num_players=3, seed=42)
        obs, _ = env.reset(seed=42)
        
        assert env.observation_space.contains(obs)
        
        obs, _, _, _, _ = env.step(0)
        assert env.observation_space.contains(obs)

    def test_observation_structure(self):
        """Observation contains all required keys."""
        env = MonopolyEnv(num_players=2, seed=42)
        obs, _ = env.reset(seed=42)
        
        required_keys = [
            'player_id', 'cash', 'positions', 'property_owner',
            'houses', 'mortgaged', 'legal_mask', 'net_worth'
        ]
        for key in required_keys:
            assert key in obs, f"Missing key: {key}"


# =============================================================================
# INTEGRATION: SHORT EPISODE
# =============================================================================

class TestIntegration:
    """End-to-end integration test."""

    def test_short_episode_runs(self):
        """Environment can run a short episode without crashing."""
        env = MonopolyEnv(num_players=4, seed=42, max_turns=20)
        obs, _ = env.reset(seed=42)
        
        total_reward = 0.0
        steps = 0
        
        while True:
            # Pick first legal action
            legal_mask = obs['legal_mask']
            legal_actions = np.where(legal_mask == 1)[0]
            action = legal_actions[0] if len(legal_actions) > 0 else 0
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            if terminated or truncated:
                break
            
            if steps > 500:  # Safety limit
                break
        
        assert steps > 0
        assert np.isfinite(total_reward)

