import pytest
import numpy as np
from ..Monopoly.envs.gym_env import MonopolyEnv
from ..Monopoly.state import PlayerStatus


def test_env_creation():
    """Test that environment can be created with default parameters."""
    env = MonopolyEnv(num_players=4, agent_player_id=0)
    assert env.num_players == 4
    assert env.agent_player_id == 0
    assert env.action_space is not None
    assert env.observation_space is not None


def test_env_reset():
    """Test environment reset functionality."""
    env = MonopolyEnv(num_players=2, agent_player_id=0, seed=42)
    obs, info = env.reset(seed=42)
    
    # Check observation structure
    assert 'player_id' in obs
    assert 'cash' in obs
    assert 'positions' in obs
    assert 'property_owner' in obs
    assert 'legal_mask' in obs
    
    # Check initial values
    assert obs['player_id'][0] == 0
    assert len(obs['cash']) == 2
    assert all(obs['cash'] == 1500)  # Starting cash
    assert all(obs['positions'] == 0)  # Start at GO
    assert all(obs['property_owner'] == -1)  # No properties owned
    assert obs['turn_number'][0] == 0


def test_legal_action_mask():
    """Test that legal action mask is computed correctly."""
    env = MonopolyEnv(num_players=2, agent_player_id=0, seed=42)
    obs, info = env.reset(seed=42)
    
    legal_mask = obs['legal_mask']
    assert legal_mask.shape == (env.n_actions,)
    assert legal_mask.dtype == np.int32
    
    # Roll should always be legal at start of turn (before rolling)
    assert legal_mask[0] == 1
    # End turn should NOT be legal before rolling (must roll first)
    assert legal_mask[89] == 0
    
    # After rolling, end turn should be legal
    obs, _, _, _, _ = env.step(0)  # Roll
    legal_mask = obs['legal_mask']
    # After rolling, roll should not be legal (already rolled)
    # and end turn should be legal (unless awaiting buy decision)
    if not env.state.awaiting_buy_decision:
        assert legal_mask[89] == 1


def test_env_step():
    """Test basic environment step functionality."""
    env = MonopolyEnv(num_players=2, agent_player_id=0, seed=42)
    obs, info = env.reset(seed=42)
    
    # Take roll action
    action = 0  # roll
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Check that state changed
    assert 'cash' in obs
    assert 'positions' in obs
    assert not terminated  # Game shouldn't end after one turn
    assert isinstance(reward, float)
    
    # Check info
    assert 'turn_number' in info
    assert 'agent_cash' in info


def test_observation_space_compliance():
    """Test that observations match the declared observation space."""
    env = MonopolyEnv(num_players=3, agent_player_id=0, seed=42)
    obs, info = env.reset(seed=42)
    
    # Check that observation is in observation space
    assert env.observation_space.contains(obs)


def test_action_space_compliance():
    """Test that actions are within action space bounds."""
    env = MonopolyEnv(num_players=2, agent_player_id=0, seed=42)
    obs, info = env.reset(seed=42)
    
    # Sample random action
    action = env.action_space.sample()
    assert 0 <= action < env.n_actions
    
    # Test step with sampled action
    obs, reward, terminated, truncated, info = env.step(action)
    assert env.observation_space.contains(obs)


def test_game_termination():
    """Test that game terminates when only one player remains active."""
    env = MonopolyEnv(num_players=2, agent_player_id=0, seed=42)
    obs, info = env.reset(seed=42)
    
    # Manually bankrupt opponent to test termination
    env.state.players[1].status = PlayerStatus.BANKRUPT
    env.state.players[1].cash = -100
    
    # Take an action
    obs, reward, terminated, truncated, info = env.step(0)
    
    # Game should terminate
    assert terminated or info['active_players'] == 1


def test_multiple_episodes():
    """Test that environment can be reset and reused for multiple episodes."""
    env = MonopolyEnv(num_players=2, agent_player_id=0, seed=42)
    
    for episode in range(3):
        obs, info = env.reset(seed=42 + episode)
        assert obs['turn_number'][0] == 0
        
        # Take a few steps
        for _ in range(5):
            action = 0  # roll
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break


def test_render_methods():
    """Test that render methods don't crash."""
    env = MonopolyEnv(num_players=2, agent_player_id=0, render_mode='human', seed=42)
    obs, info = env.reset(seed=42)
    
    # Render should not crash
    env.render()
    
    obs, reward, terminated, truncated, info = env.step(0)
    env.render()
    
    env.close()


def test_different_player_counts():
    """Test environment with different numbers of players."""
    for n_players in [2, 3, 4]:
        env = MonopolyEnv(num_players=n_players, agent_player_id=0, seed=42)
        obs, info = env.reset(seed=42)
        
        assert len(obs['cash']) == n_players
        assert len(obs['positions']) == n_players
        assert obs['player_id'][0] == 0
        
        # Take one step
        obs, reward, terminated, truncated, info = env.step(0)
        assert env.observation_space.contains(obs)


def test_action_decoding():
    """Test that action decoding works correctly."""
    env = MonopolyEnv(num_players=2, agent_player_id=0, seed=42)
    obs, info = env.reset(seed=42)
    
    # Test different action types
    roll_action = env._decode_action(0)
    assert roll_action['type'] == 'roll'
    
    buy_action = env._decode_action(1)
    assert buy_action['type'] == 'buy'
    
    pass_action = env._decode_action(2)
    assert pass_action['type'] == 'pass'
    
    build_action = env._decode_action(3)
    assert build_action['type'] == 'build'
    assert build_action['property_idx'] == 0
    
    mortgage_action = env._decode_action(31)
    assert mortgage_action['type'] == 'mortgage'
    assert mortgage_action['property_idx'] == 0


def test_reward_shaping():
    """Test that reward includes shaped components."""
    env = MonopolyEnv(num_players=2, agent_player_id=0, seed=42)
    obs, info = env.reset(seed=42)
    
    initial_cash = obs['cash'][0]
    
    # Take action
    obs, reward, terminated, truncated, info = env.step(0)
    
    # Reward should be a finite number
    assert np.isfinite(reward)
    assert isinstance(reward, float)


def test_opponent_simulation():
    """Test that opponent turns are simulated automatically."""
    env = MonopolyEnv(num_players=3, agent_player_id=1, seed=42)  # Agent is player 1
    obs, info = env.reset(seed=42)
    
    # After reset, it should be agent's turn (player 1)
    assert env.state.current_player == 1
    
    # Take action
    obs, reward, terminated, truncated, info = env.step(0)
    
    # After step, should still be agent's turn (or game over)
    if not terminated:
        assert env.state.current_player == 1 or env.state.current_player == env.agent_player_id
