"""Tests for the PettingZoo AEC Monopoly environment."""

import pytest
import numpy as np
from Monopoly.envs.pettingzoo_env import MonopolyAECEnv
from Monopoly.envs.wrappers import PettingZooToGymWrapper


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_env(**kwargs):
    """Create a default test environment."""
    defaults = {"seed": 42, "max_steps": 500}
    defaults.update(kwargs)
    env = MonopolyAECEnv(**defaults)
    env.reset()
    return env


def _get_action_mask(env, agent=None):
    """Get action mask from observation (handles PettingZoo convention)."""
    agent = agent or env.agent_selection
    obs = env.observe(agent)
    return obs["action_mask"]


def _take_first_legal(env):
    """Step with the first legal action for the current agent."""
    agent = env.agent_selection
    if env.terminations.get(agent, False) or env.truncations.get(agent, False):
        env.step(None)
        return None
    mask = _get_action_mask(env)
    legal = np.where(mask == 1)[0]
    action = int(legal[0]) if len(legal) > 0 else 89
    env.step(action)
    return action


# ---------------------------------------------------------------------------
# Initialization & Reset
# ---------------------------------------------------------------------------

class TestReset:

    def test_reset_initializes_all_agents(self):
        env = _make_env()
        assert len(env.agents) == 4
        assert all(not env.terminations[ag] for ag in env.agents)
        assert all(not env.truncations[ag] for ag in env.agents)
        assert env.agent_selection == "player_0"

    def test_possible_agents(self):
        env = _make_env()
        assert env.possible_agents == [
            "player_0", "player_1", "player_2", "player_3"
        ]


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class TestObservation:

    def test_observation_pettingzoo_convention(self):
        """Observation must have 'observation' and 'action_mask' keys."""
        env = _make_env()
        obs = env.observe("player_0")
        assert "observation" in obs
        assert "action_mask" in obs

    def test_observation_perspective(self):
        """Each agent's observation has their own player_id (first element)."""
        env = _make_env()
        obs0 = env.observe("player_0")
        assert obs0["observation"][0] == 0.0  # player_id is first element

        obs1 = env.observe("player_1")
        assert obs1["observation"][0] == 1.0

    def test_observation_is_flat_array(self):
        """Observation should be a flat float32 numpy array."""
        env = _make_env()
        obs = env.observe("player_0")
        assert isinstance(obs["observation"], np.ndarray)
        assert obs["observation"].dtype == np.float32
        assert obs["observation"].ndim == 1

    def test_observation_in_space(self):
        env = _make_env()
        obs = env.observe("player_0")
        space = env.observation_space("player_0")
        assert space.contains(obs)


# ---------------------------------------------------------------------------
# Action Space
# ---------------------------------------------------------------------------

class TestActionSpace:

    def test_action_space_all_agents(self):
        """All agents share the same action space size."""
        env = _make_env()
        for agent in env.possible_agents:
            assert env.action_space(agent).n == 174

    def test_legal_mask_at_start(self):
        """At game start, only roll (action 0) should be legal for player 0."""
        env = _make_env()
        mask = _get_action_mask(env, "player_0")
        legal_actions = np.where(mask == 1)[0]
        assert 0 in legal_actions  # roll must be legal

    def test_legal_mask_not_your_turn(self):
        """Players whose turn it isn't should have an all-zero legal mask."""
        env = _make_env()
        mask = _get_action_mask(env, "player_1")
        assert np.sum(mask) == 0


# ---------------------------------------------------------------------------
# Step / Turn Flow
# ---------------------------------------------------------------------------

class TestStepFlow:

    def test_step_advances_on_end_turn(self):
        """After roll + end_turn, agent_selection changes."""
        env = _make_env()
        assert env.agent_selection == "player_0"
        # Roll
        env.step(0)
        # End turn (or handle buy decision first)
        mask = _get_action_mask(env)
        if mask[89] == 1:
            env.step(89)  # end_turn
        else:
            legal = np.where(mask == 1)[0]
            env.step(int(legal[0]))
            mask = _get_action_mask(env)
            if mask[89] == 1:
                env.step(89)

        # Should have advanced to a different player
        assert env.agent_selection != "player_0" or env.terminations.get("player_0", False)

    def test_short_episode_no_crash(self):
        """Run a short episode without crashing."""
        env = _make_env(max_steps=100)
        steps = 0
        while env.agents and steps < 100:
            _take_first_legal(env)
            steps += 1


# ---------------------------------------------------------------------------
# Termination
# ---------------------------------------------------------------------------

class TestTermination:

    def test_max_steps_truncation(self):
        """Game should truncate after max_steps."""
        env = _make_env(max_steps=20)
        game_ended = False
        for _ in range(50):
            if not env.agents:
                game_ended = True
                break
            agent = env.agent_selection
            if env.terminations.get(agent, False) or env.truncations.get(agent, False):
                game_ended = True
                env.step(None)
                continue
            _take_first_legal(env)

        assert game_ended

    def test_stalemate_detection(self):
        """Stalemate with low threshold should cause truncation."""
        env = _make_env(stalemate_threshold=2, max_steps=5000)
        game_ended = False
        for _ in range(3000):
            if not env.agents:
                game_ended = True
                break
            agent = env.agent_selection
            if env.terminations.get(agent, False) or env.truncations.get(agent, False):
                game_ended = True
                env.step(None)
                continue
            _take_first_legal(env)

        assert game_ended


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:

    def test_determinism_two_envs(self):
        """Two envs with same seed + actions produce identical states."""
        env1 = _make_env(seed=123)
        env2 = _make_env(seed=123)

        for _ in range(30):
            if not env1.agents or not env2.agents:
                break

            assert env1.agent_selection == env2.agent_selection
            obs1 = env1.observe(env1.agent_selection)
            obs2 = env2.observe(env2.agent_selection)
            np.testing.assert_array_equal(
                obs1["observation"], obs2["observation"]
            )
            np.testing.assert_array_equal(
                obs1["action_mask"], obs2["action_mask"]
            )

            mask = obs1["action_mask"]
            legal = np.where(mask == 1)[0]
            action = int(legal[0]) if len(legal) > 0 else 89
            env1.step(action)
            env2.step(action)


# ---------------------------------------------------------------------------
# Fast Mode
# ---------------------------------------------------------------------------

class TestFastMode:

    def test_fast_mode_zeros_net_worth(self):
        """In fast mode, net_worth (last num_players elements) should be all zeros."""
        env = _make_env(fast_mode=True)
        obs = env.observe("player_0")
        # Net worth occupies the last num_players elements of the flat array
        net_worth_slice = obs["observation"][-4:]
        np.testing.assert_array_equal(
            net_worth_slice,
            np.zeros(4, dtype=np.float32),
        )


# ---------------------------------------------------------------------------
# Reward Modes
# ---------------------------------------------------------------------------

class TestRewardModes:

    def test_invalid_reward_mode_raises(self):
        with pytest.raises(ValueError):
            MonopolyAECEnv(reward_mode="invalid")

    def test_sparse_zero_during_play(self):
        """Sparse terminal mode should give 0 reward during play."""
        env = _make_env(reward_mode="sparse_terminal")
        _take_first_legal(env)
        for ag in env.agents:
            if not env.terminations[ag] and not env.truncations[ag]:
                assert env.rewards[ag] == 0.0


# ---------------------------------------------------------------------------
# Gym Wrapper
# ---------------------------------------------------------------------------

class TestGymWrapper:

    def test_gym_wrapper_basic(self):
        """PettingZooToGymWrapper should return valid gym step tuples."""
        aec = MonopolyAECEnv(seed=42, max_steps=200)
        env = PettingZooToGymWrapper(aec)
        obs, info = env.reset()

        assert "action_mask" in obs
        assert "observation" in obs
        assert isinstance(obs["observation"], np.ndarray)

        for _ in range(10):
            mask = obs["action_mask"]
            legal = np.where(mask == 1)[0]
            action = int(legal[0]) if len(legal) > 0 else 89
            obs, reward, terminated, truncated, info = env.step(action)
            assert isinstance(reward, (float, int, np.floating))
            if terminated or truncated:
                break

    def test_gym_wrapper_obs_in_space(self):
        aec = MonopolyAECEnv(seed=42)
        env = PettingZooToGymWrapper(aec)
        obs, _ = env.reset()
        assert env.observation_space.contains(obs)


# ---------------------------------------------------------------------------
# PettingZoo API Conformance
# ---------------------------------------------------------------------------

class TestAPIConformance:

    def test_api_test(self):
        """PettingZoo's built-in API test should pass."""
        from pettingzoo.test import api_test
        env = MonopolyAECEnv(seed=42)
        api_test(env, num_cycles=10, verbose_progress=False)
