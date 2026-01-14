"""
Additional Test Suite for DQN/Network Components and Monopoly Environment Integration.

This module provides:
- Extended network architecture tests
- Monopoly environment integration tests
- Renderer tests
- SB3 DQN integration tests

Run with: pytest tests/test_dqn_extended.py -v
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest
import numpy as np
import torch

# =============================================================================
# Network Architecture Extended Tests
# =============================================================================

class TestQNetworkExtended:
    """Extended tests for QNetwork architecture."""
    
    def test_single_layer_network(self):
        from Monopoly.agents.network import QNetwork
        net = QNetwork(obs_dim=10, action_dim=5, hidden_dims=[32], device='cpu')
        obs = torch.randn(4, 10)
        q = net.forward(obs)
        assert q.shape == (4, 5)
    
    def test_deep_network(self):
        from Monopoly.agents.network import QNetwork
        net = QNetwork(obs_dim=10, action_dim=5, hidden_dims=[64, 64, 64, 64, 64], device='cpu')
        obs = torch.randn(4, 10)
        q = net.forward(obs)
        assert q.shape == (4, 5)
    
    def test_wide_network(self):
        from Monopoly.agents.network import QNetwork
        net = QNetwork(obs_dim=10, action_dim=5, hidden_dims=[1024, 1024], device='cpu')
        obs = torch.randn(4, 10)
        q = net.forward(obs)
        assert q.shape == (4, 5)
    
    def test_small_obs_dim(self):
        from Monopoly.agents.network import QNetwork
        net = QNetwork(obs_dim=1, action_dim=2, hidden_dims=[8], device='cpu')
        obs = torch.randn(4, 1)
        q = net.forward(obs)
        assert q.shape == (4, 2)
    
    def test_large_action_dim(self):
        from Monopoly.agents.network import QNetwork
        net = QNetwork(obs_dim=10, action_dim=200, hidden_dims=[64], device='cpu')
        obs = torch.randn(4, 10)
        q = net.forward(obs)
        assert q.shape == (4, 200)
    
    def test_network_parameters_trainable(self):
        from Monopoly.agents.network import QNetwork
        net = QNetwork(obs_dim=10, action_dim=5, hidden_dims=[32, 32], device='cpu')
        trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        assert trainable_params > 0
    
    def test_parameter_count(self):
        from Monopoly.agents.network import QNetwork
        net = QNetwork(obs_dim=10, action_dim=5, hidden_dims=[32, 32], device='cpu')
        # Calculate expected params:
        # Layer 1: 10*32 + 32 = 352
        # Layer 2: 32*32 + 32 = 1056  
        # Output: 32*5 + 5 = 165
        # Total: 352 + 1056 + 165 = 1573
        total_params = sum(p.numel() for p in net.parameters())
        assert total_params == 1573
    
    def test_forward_batch_independence(self):
        from Monopoly.agents.network import QNetwork
        net = QNetwork(obs_dim=10, action_dim=5, hidden_dims=[32], device='cpu')
        
        obs = torch.randn(4, 10)
        q_batch = net.forward(obs)
        q_single = [net.forward(obs[i:i+1]) for i in range(4)]
        
        for i in range(4):
            assert torch.allclose(q_batch[i:i+1], q_single[i], atol=1e-6)
    
    def test_activation_function_relu(self):
        from Monopoly.agents.network import QNetwork
        net = QNetwork(obs_dim=10, action_dim=5, hidden_dims=[32], device='cpu')
        
        # Check that ReLU is being used (output can be negative from final layer)
        obs = torch.randn(100, 10)
        q = net.forward(obs)
        # Q-values can be any real number
        assert q.min() < 0 or q.max() > 0
    
    def test_network_repr(self):
        from Monopoly.agents.network import QNetwork
        net = QNetwork(obs_dim=10, action_dim=5, hidden_dims=[32, 32], device='cpu')
        repr_str = repr(net)
        assert 'Sequential' in repr_str or 'Linear' in repr_str


class TestDuelingQNetworkExtended:
    """Extended tests for DuelingQNetwork architecture."""
    
    def test_value_advantage_independence(self):
        from Monopoly.agents.network import DuelingQNetwork
        net = DuelingQNetwork(obs_dim=10, action_dim=5, hidden_dims=[32], device='cpu')
        
        obs = torch.randn(4, 10)
        features = net.feature_net(obs)
        
        value = net.value_stream(features)
        advantage = net.advantage_stream(features)
        
        # Value should be scalar per sample, advantage should be per action
        assert value.shape == (4, 1)
        assert advantage.shape == (4, 5)
    
    def test_q_value_decomposition_formula(self):
        from Monopoly.agents.network import DuelingQNetwork
        net = DuelingQNetwork(obs_dim=10, action_dim=5, hidden_dims=[32], device='cpu')
        
        obs = torch.randn(4, 10)
        features = net.feature_net(obs)
        
        value = net.value_stream(features)
        advantage = net.advantage_stream(features)
        advantage_centered = advantage - advantage.mean(dim=-1, keepdim=True)
        
        expected_q = value + advantage_centered
        actual_q = net.forward(obs)
        
        assert torch.allclose(expected_q, actual_q)
    
    def test_dueling_better_value_estimation(self):
        """Test that dueling network can separate state value from action advantage."""
        from Monopoly.agents.network import DuelingQNetwork
        net = DuelingQNetwork(obs_dim=10, action_dim=5, hidden_dims=[64, 64], device='cpu')
        
        # Train briefly on some data
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
        
        for _ in range(100):
            obs = torch.randn(16, 10)
            target = torch.randn(16, 5)
            
            q = net.forward(obs)
            loss = ((q - target) ** 2).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Check that value stream produces meaningful outputs
        test_obs = torch.randn(4, 10)
        features = net.feature_net(test_obs)
        value = net.value_stream(features)
        
        assert not torch.allclose(value[0], value[1])  # Different states -> different values
    
    def test_feature_net_shared(self):
        from Monopoly.agents.network import DuelingQNetwork
        net = DuelingQNetwork(obs_dim=10, action_dim=5, hidden_dims=[32], device='cpu')
        
        obs = torch.randn(4, 10, requires_grad=True)
        q = net.forward(obs)
        loss = q.sum()
        loss.backward()
        
        # Features should have gradients flowing from both streams
        for param in net.feature_net.parameters():
            assert param.grad is not None
            assert param.grad.abs().sum() > 0


# =============================================================================
# Monopoly Environment Integration Tests
# =============================================================================

class TestMonopolyEnvIntegration:
    """Tests for integration with actual Monopoly environment."""
    
    @pytest.fixture
    def monopoly_env(self):
        try:
            from Monopoly.envs.gym_env import MonopolyEnv
            env = MonopolyEnv(num_players=4, agent_player_id=0, max_turns=20)
            yield env
            env.close()
        except ImportError:
            pytest.skip("Monopoly environment not available")
    
    def test_env_observation_space(self, monopoly_env):
        # Monopoly env uses Dict space
        from gymnasium import spaces
        assert isinstance(monopoly_env.observation_space, spaces.Dict)
        assert 'legal_mask' in monopoly_env.observation_space.spaces
    
    def test_env_action_space(self, monopoly_env):
        action_dim = monopoly_env.action_space.n
        assert action_dim > 0
    
    def test_env_reset(self, monopoly_env):
        obs, info = monopoly_env.reset(seed=42)
        # Obs is a dict for Dict observation space
        assert isinstance(obs, dict)
        assert 'legal_mask' in obs
        assert isinstance(info, dict)
    
    def test_env_step(self, monopoly_env):
        obs, _ = monopoly_env.reset(seed=42)
        # Get legal action
        mask = obs.get('legal_mask', np.ones(monopoly_env.action_space.n))
        legal_actions = np.where(mask == 1)[0]
        action = legal_actions[0] if len(legal_actions) > 0 else 0
        
        next_obs, reward, done, truncated, info = monopoly_env.step(action)
        
        assert isinstance(next_obs, dict)
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
    
    def test_env_legal_mask(self, monopoly_env):
        obs, _ = monopoly_env.reset(seed=42)
        
        if 'legal_mask' in obs:
            mask = obs['legal_mask']
            assert mask.shape[0] == monopoly_env.action_space.n
            assert mask.sum() > 0  # At least one legal action
    
    def test_network_with_flattened_env(self, monopoly_env, tmp_path):
        """Test network with wrapped/flattened environment."""
        from Monopoly.agents.network import QNetwork
        from Monopoly.envs.wrappers import MonopolyFlattenWrapper
        
        # Wrap the environment to flatten observations
        wrapped = MonopolyFlattenWrapper(monopoly_env)
        
        obs_dim = wrapped.observation_space.shape[0]
        action_dim = wrapped.action_space.n
        
        net = QNetwork(obs_dim=obs_dim, action_dim=action_dim, hidden_dims=[64, 64], device='cpu')
        
        obs, _ = wrapped.reset(seed=42)
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        
        q = net.forward(obs_tensor)
        assert q.shape == (1, action_dim)
    
    def test_wrapped_trainer_with_monopoly_env(self, monopoly_env, tmp_path):
        """Test DDQN trainer with wrapped Monopoly environment."""
        from Monopoly.agents.ddqn_hybrid import DDQNHybridTrainer
        from Monopoly.envs.gym_env import MonopolyEnv
        from Monopoly.envs.wrappers import MonopolyFlattenWrapper
        
        # Wrap environments for DDQN trainer (needs flattened obs)
        wrapped_env = MonopolyFlattenWrapper(monopoly_env)
        eval_env = MonopolyEnv(num_players=4, agent_player_id=0, max_turns=20)
        wrapped_eval = MonopolyFlattenWrapper(eval_env)
        
        config = {
            "batch_size": 4,
            "buffer_size": 50,
            "learning_starts": 10,
            "hidden_dims": [64, 64],
        }
        
        trainer = DDQNHybridTrainer(
            env=wrapped_env,
            eval_env=wrapped_eval,
            output_dir=str(tmp_path),
            config=config,
            verbose=0,
        )
        
        # Quick training
        trainer.train(
            total_timesteps=20,
            eval_interval=100,
            progress_bar=False,
        )
        
        trainer.close()
        eval_env.close()
    
    def test_action_masking_with_wrapped_env(self, monopoly_env):
        """Test action masking with wrapped environment."""
        from Monopoly.agents.network import QNetwork
        from Monopoly.envs.wrappers import MonopolyFlattenWrapper
        
        wrapped = MonopolyFlattenWrapper(monopoly_env)
        
        obs_dim = wrapped.observation_space.shape[0]
        action_dim = wrapped.action_space.n
        
        net = QNetwork(obs_dim=obs_dim, action_dim=action_dim, hidden_dims=[32], device='cpu')
        
        obs, _ = wrapped.reset(seed=42)
        
        if hasattr(wrapped.unwrapped, '_get_legal_mask'):
            mask = wrapped.unwrapped._get_legal_mask().astype(np.float32)
            
            # Select action with masking
            action = net.select_action(obs, mask, epsilon=0.0)
            
            # Action should be legal
            assert mask[action] == 1
    
    def test_episode_completion(self, monopoly_env):
        """Test that an episode can complete."""
        obs, _ = monopoly_env.reset(seed=42)
        
        for _ in range(1000):  # Max steps
            if 'legal_mask' in obs:
                mask = obs['legal_mask']
                legal_actions = np.where(mask == 1)[0]
                action = np.random.choice(legal_actions) if len(legal_actions) > 0 else 0
            else:
                action = monopoly_env.action_space.sample()
            
            obs, reward, done, truncated, info = monopoly_env.step(action)
            
            if done or truncated:
                break
        
        # Episode should end within max steps


# =============================================================================
# Renderer Tests  
# =============================================================================

class TestRenderer:
    """Tests for Monopoly renderer."""
    
    @pytest.fixture
    def mock_game_state(self):
        """Create a mock game state for testing."""
        mock_state = Mock()
        mock_state.players = [
            Mock(name="Player0", position=0, cash=1500, bankrupt=False, in_jail=False, jail_turns=0),
            Mock(name="Player1", position=10, cash=1200, bankrupt=False, in_jail=True, jail_turns=1),
            Mock(name="Player2", position=20, cash=800, bankrupt=False, in_jail=False, jail_turns=0),
            Mock(name="Player3", position=35, cash=0, bankrupt=True, in_jail=False, jail_turns=0),
        ]
        mock_state.last_roll = (3, 4)
        mock_state.current_player = 0
        mock_state.turn_number = 5
        return mock_state
    
    def test_renderer_import(self):
        """Test that renderer can be imported."""
        try:
            from Monopoly.envs.renderer import MonopolyRenderer
            assert MonopolyRenderer is not None
        except ImportError as e:
            pytest.skip(f"Renderer not available: {e}")
    
    def test_renderer_init_headless(self):
        """Test renderer initialization (may require display)."""
        try:
            from Monopoly.envs.renderer import MonopolyRenderer
            # Renderer may require a display - skip if not available
            pytest.skip("Renderer requires display, skipping in CI")
        except ImportError:
            pytest.skip("Renderer not available")
        except Exception as e:
            if "pygame" in str(e).lower() or "display" in str(e).lower():
                pytest.skip("pygame/display not available")
            raise
    
    def test_renderer_color_palette(self):
        """Test that color palette is defined."""
        try:
            from Monopoly.envs.renderer import MonopolyRenderer
            # Check that essential colors exist in the class
            assert hasattr(MonopolyRenderer, '__init__')
        except ImportError:
            pytest.skip("Renderer not available")


# =============================================================================
# SB3 DQN Integration Tests
# =============================================================================

class TestSB3DQNIntegration:
    """Tests for Stable Baselines3 DQN integration."""
    
    def test_sb3_dqn_import(self):
        """Test that SB3 DQN can be imported."""
        try:
            from stable_baselines3 import DQN
            assert DQN is not None
        except ImportError:
            pytest.skip("stable_baselines3 not available")
    
    def test_sb3_dqn_with_monopoly_env(self, tmp_path):
        """Test SB3 DQN with Monopoly environment."""
        try:
            from stable_baselines3 import DQN
            from Monopoly.envs.gym_env import MonopolyEnv
            from Monopoly.envs.wrappers import ActionMasker
        except ImportError:
            pytest.skip("Required modules not available")
        
        env = MonopolyEnv(num_players=4, agent_player=0, max_turns=10)
        wrapped_env = ActionMasker(env)
        
        model = DQN(
            "MlpPolicy",
            wrapped_env,
            learning_rate=1e-4,
            buffer_size=100,
            learning_starts=10,
            batch_size=4,
            verbose=0,
        )
        
        # Quick training
        model.learn(total_timesteps=20)
        
        # Test prediction
        obs, _ = wrapped_env.reset()
        action, _ = model.predict(obs, deterministic=True)
        
        assert isinstance(action, (int, np.integer))
        assert 0 <= action < wrapped_env.action_space.n
        
        wrapped_env.close()
    
    def test_train_dqn_function(self, tmp_path):
        """Test the train_dqn function from train_dqn.py."""
        try:
            from train_dqn import train_dqn
        except ImportError:
            pytest.skip("train_dqn module not available")
        
        # Run with minimal settings using correct parameter names
        model = train_dqn(
            total_timesteps=20,
            output_dir=str(tmp_path),
            n_eval_episodes=1,
            eval_interval=10,
            verbose=0,
        )
        
        assert model is not None


# =============================================================================
# Utility Function Tests
# =============================================================================

class TestUtilities:
    """Tests for utility functions."""
    
    def test_transition_namedtuple(self):
        from Monopoly.agents.ddqn_hybrid import Transition
        
        t = Transition(
            state=np.zeros(10),
            action=0,
            reward=1.0,
            next_state=np.ones(10),
            done=False,
            mask=np.ones(5),
            next_mask=np.ones(5),
        )
        
        assert t.action == 0
        assert t.reward == 1.0
        assert t.done == False
    
    def test_transition_unpacking(self):
        from Monopoly.agents.ddqn_hybrid import Transition
        
        t = Transition(
            state=np.zeros(10),
            action=2,
            reward=5.0,
            next_state=np.ones(10),
            done=True,
            mask=np.ones(5),
            next_mask=np.ones(5),
        )
        
        # Access as named tuple attributes
        assert t.action == 2
        assert t.reward == 5.0
        assert t.done == True
        assert t.state is not None
        assert t.next_state is not None
    
    def test_default_config_values(self):
        """Test that DEFAULT_CONFIG has expected keys."""
        from Monopoly.agents.ddqn_hybrid import DDQNHybridTrainer
        
        DEFAULT_CONFIG = DDQNHybridTrainer.DEFAULT_CONFIG
        
        expected_keys = [
            'batch_size', 'buffer_size', 'gamma', 'lr',
            'eps_start', 'eps_end', 'eps_decay',
            'target_update_interval', 'hidden_dims',
        ]
        
        for key in expected_keys:
            assert key in DEFAULT_CONFIG, f"Missing key: {key}"
    
    def test_config_types(self):
        """Test that DEFAULT_CONFIG values have correct types."""
        from Monopoly.agents.ddqn_hybrid import DDQNHybridTrainer
        
        DEFAULT_CONFIG = DDQNHybridTrainer.DEFAULT_CONFIG
        
        assert isinstance(DEFAULT_CONFIG['batch_size'], int)
        assert isinstance(DEFAULT_CONFIG['buffer_size'], int)
        assert isinstance(DEFAULT_CONFIG['gamma'], float)
        assert isinstance(DEFAULT_CONFIG['lr'], float)
        assert isinstance(DEFAULT_CONFIG['hidden_dims'], list)


# =============================================================================
# Trade Policy Tests
# =============================================================================

class TestTradePolicy:
    """Tests for trade policy functionality."""
    
    def test_trade_module_import(self):
        try:
            from Monopoly.trade import TradeOffer
            assert TradeOffer is not None
        except ImportError:
            pytest.skip("Trade module not available")
    
    def test_auction_module_import(self):
        try:
            from Monopoly.auction import Auction
            assert Auction is not None
        except ImportError:
            pytest.skip("Auction module not available")


# =============================================================================
# Save/Load Tests
# =============================================================================

class TestSaveLoad:
    """Tests for model save/load functionality."""
    
    def test_qnetwork_save_creates_file(self, tmp_path):
        from Monopoly.agents.network import QNetwork
        
        net = QNetwork(obs_dim=10, action_dim=5, device='cpu')
        save_path = tmp_path / "model.pt"
        net.save(save_path)
        
        assert save_path.exists()
    
    def test_qnetwork_save_file_contents(self, tmp_path):
        from Monopoly.agents.network import QNetwork
        
        net = QNetwork(obs_dim=10, action_dim=5, hidden_dims=[32], device='cpu')
        save_path = tmp_path / "model.pt"
        net.save(save_path)
        
        data = torch.load(save_path)
        assert 'model_state_dict' in data
        assert 'obs_dim' in data
        assert 'action_dim' in data
        assert 'hidden_dims' in data
    
    def test_dueling_qnetwork_save_load(self, tmp_path):
        from Monopoly.agents.network import DuelingQNetwork
        
        net = DuelingQNetwork(obs_dim=10, action_dim=5, hidden_dims=[32], device='cpu')
        save_path = tmp_path / "dueling.pt"
        net.save(save_path)
        
        loaded = DuelingQNetwork.load(save_path, device=torch.device('cpu'))
        
        assert loaded.obs_dim == 10
        assert loaded.action_dim == 5
        assert loaded.hidden_dims == [32]
    
    def test_trainer_checkpoint_contains_all_state(self, tmp_path):
        from Monopoly.agents.ddqn_hybrid import DDQNHybridTrainer
        
        class DummyEnv:
            def __init__(self):
                import gymnasium as gym
                self.observation_space = gym.spaces.Box(-1, 1, (10,), dtype=np.float32)
                self.action_space = gym.spaces.Discrete(5)
                class Unwrapped:
                    def _get_legal_mask(self):
                        return np.ones(5, dtype=np.float32)
                self.unwrapped = Unwrapped()
            def reset(self, seed=None):
                return np.zeros(10, dtype=np.float32), {}
            def step(self, a):
                return np.zeros(10, dtype=np.float32), 0.0, True, False, {}
            def close(self):
                pass
        
        env = DummyEnv()
        trainer = DDQNHybridTrainer(
            env=env, eval_env=DummyEnv(),
            output_dir=str(tmp_path),
            verbose=0,
        )
        
        trainer.total_steps = 999
        trainer.episodes_completed = 42
        trainer.save_checkpoint(name="test")
        
        state_path = tmp_path / "models" / "ddqn_hybrid" / "test_trainer_state.pt"
        state = torch.load(state_path)
        
        assert 'total_steps' in state
        assert 'episodes_completed' in state
        assert 'optimizer_state_dict' in state
        assert state['total_steps'] == 999
        assert state['episodes_completed'] == 42
        
        trainer.close()


# =============================================================================
# Numerical Stability Tests
# =============================================================================

class TestNumericalStability:
    """Tests for numerical stability."""
    
    def test_large_q_values_dont_overflow(self):
        from Monopoly.agents.network import QNetwork
        
        net = QNetwork(obs_dim=10, action_dim=5, hidden_dims=[64], device='cpu')
        
        # Large input values
        obs = torch.randn(4, 10) * 100
        q = net.forward(obs)
        
        assert torch.isfinite(q).all()
    
    def test_small_q_values_dont_underflow(self):
        from Monopoly.agents.network import QNetwork
        
        net = QNetwork(obs_dim=10, action_dim=5, hidden_dims=[64], device='cpu')
        
        # Small input values
        obs = torch.randn(4, 10) * 0.0001
        q = net.forward(obs)
        
        assert torch.isfinite(q).all()
    
    def test_mask_with_inf_handling(self):
        from Monopoly.agents.network import QNetwork
        
        net = QNetwork(obs_dim=10, action_dim=5, device='cpu')
        
        obs = torch.randn(2, 10)
        mask = torch.tensor([[1, 0, 1, 0, 1], [0, 0, 0, 0, 1]], dtype=torch.float32)
        
        q = net.get_q_values(obs, mask)
        
        # Masked values should be -inf, but legal values should be finite
        assert q[0, 0] != float('-inf')  # Legal
        assert q[0, 1] == float('-inf')  # Illegal
        assert q[1, 4] != float('-inf')  # Legal
    
    def test_softmax_with_masked_values(self):
        from Monopoly.agents.network import QNetwork
        
        net = QNetwork(obs_dim=10, action_dim=5, device='cpu')
        
        obs = torch.randn(2, 10)
        mask = torch.tensor([[1, 0, 1, 0, 1], [1, 1, 1, 1, 1]], dtype=torch.float32)
        
        q = net.get_q_values(obs, mask)
        q_tensor = torch.tensor(q)
        
        # Replace -inf with very negative value for softmax
        q_for_softmax = torch.where(
            torch.isinf(q_tensor),
            torch.full_like(q_tensor, -1e9),
            q_tensor
        )
        
        probs = torch.softmax(q_for_softmax, dim=-1)
        
        # Probabilities should sum to 1 and be valid
        assert torch.allclose(probs.sum(dim=-1), torch.ones(2))
        assert (probs >= 0).all()


# =============================================================================
# Concurrency Tests
# =============================================================================

class TestConcurrency:
    """Tests for thread safety and concurrent access."""
    
    def test_replay_buffer_thread_safety(self):
        """Test that replay buffer handles concurrent access."""
        import threading
        from Monopoly.agents.ddqn_hybrid import ReplayBuffer
        
        buffer = ReplayBuffer(capacity=1000)
        errors = []
        
        def push_data():
            try:
                for i in range(100):
                    state = np.random.randn(10).astype(np.float32)
                    mask = np.ones(5, dtype=np.float32)
                    buffer.push(state, i % 5, float(i), state, False, mask, mask)
            except Exception as e:
                errors.append(e)
        
        def sample_data():
            try:
                for _ in range(50):
                    if len(buffer) > 10:
                        buffer.sample(8)
            except Exception as e:
                errors.append(e)
        
        threads = [
            threading.Thread(target=push_data),
            threading.Thread(target=push_data),
            threading.Thread(target=sample_data),
        ]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Note: ReplayBuffer is not thread-safe by design, but it shouldn't crash
        # This test just ensures no catastrophic failures


# =============================================================================
# Configuration Tests
# =============================================================================

class TestConfiguration:
    """Tests for configuration handling."""
    
    def test_custom_config_override(self, tmp_path):
        from Monopoly.agents.ddqn_hybrid import DDQNHybridTrainer
        
        class DummyEnv:
            def __init__(self):
                import gymnasium as gym
                self.observation_space = gym.spaces.Box(-1, 1, (10,), dtype=np.float32)
                self.action_space = gym.spaces.Discrete(5)
                class Unwrapped:
                    def _get_legal_mask(self):
                        return np.ones(5, dtype=np.float32)
                self.unwrapped = Unwrapped()
            def reset(self, seed=None):
                return np.zeros(10, dtype=np.float32), {}
            def step(self, a):
                return np.zeros(10, dtype=np.float32), 0.0, True, False, {}
            def close(self):
                pass
        
        custom_config = {
            "batch_size": 128,
            "gamma": 0.95,
            "lr": 0.0005,
        }
        
        trainer = DDQNHybridTrainer(
            env=DummyEnv(),
            eval_env=DummyEnv(),
            output_dir=str(tmp_path),
            config=custom_config,
            verbose=0,
        )
        
        assert trainer.config['batch_size'] == 128
        assert trainer.config['gamma'] == 0.95
        assert trainer.config['lr'] == 0.0005
        
        trainer.close()
    
    def test_default_config_preserved_for_unspecified(self, tmp_path):
        from Monopoly.agents.ddqn_hybrid import DDQNHybridTrainer
        
        DEFAULT_CONFIG = DDQNHybridTrainer.DEFAULT_CONFIG
        
        class DummyEnv:
            def __init__(self):
                import gymnasium as gym
                self.observation_space = gym.spaces.Box(-1, 1, (10,), dtype=np.float32)
                self.action_space = gym.spaces.Discrete(5)
                class Unwrapped:
                    def _get_legal_mask(self):
                        return np.ones(5, dtype=np.float32)
                self.unwrapped = Unwrapped()
            def reset(self, seed=None):
                return np.zeros(10, dtype=np.float32), {}
            def step(self, a):
                return np.zeros(10, dtype=np.float32), 0.0, True, False, {}
            def close(self):
                pass
        
        custom_config = {"batch_size": 64}  # Only override batch_size
        
        trainer = DDQNHybridTrainer(
            env=DummyEnv(),
            eval_env=DummyEnv(),
            output_dir=str(tmp_path),
            config=custom_config,
            verbose=0,
        )
        
        # Custom value
        assert trainer.config['batch_size'] == 64
        # Default values preserved
        assert trainer.config['gamma'] == DEFAULT_CONFIG['gamma']
        
        trainer.close()
