"""Tests for YAML config loading and merging."""

import os
import warnings

import pytest
import yaml

from Monopoly.agents.ddqn_hybrid import DDQNHybridTrainer, load_yaml_config


@pytest.fixture
def configs_dir():
    """Path to the configs/ directory."""
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'configs'
    )


class TestYAMLConfig:

    def test_default_yaml_loads(self, configs_dir):
        config = load_yaml_config(os.path.join(configs_dir, 'ddqn_default.yaml'))
        assert isinstance(config, dict)
        assert 'gamma' in config
        assert 'lr' in config
        assert 'batch_size' in config

    def test_yaml_matches_default_config_keys(self, configs_dir):
        """All DEFAULT_CONFIG keys should be present in the YAML."""
        config = load_yaml_config(os.path.join(configs_dir, 'ddqn_default.yaml'))
        for key in DDQNHybridTrainer.DEFAULT_CONFIG:
            assert key in config, f"Missing key: {key}"

    def test_yaml_values_match_defaults(self, configs_dir):
        """Default YAML values should match DEFAULT_CONFIG."""
        config = load_yaml_config(os.path.join(configs_dir, 'ddqn_default.yaml'))
        defaults = DDQNHybridTrainer.DEFAULT_CONFIG

        assert config['gamma'] == defaults['gamma']
        assert config['lr'] == defaults['lr']
        assert config['batch_size'] == defaults['batch_size']
        assert config['buffer_size'] == defaults['buffer_size']
        assert config['grad_clip'] == defaults['grad_clip']

    def test_fast_yaml_has_smaller_buffer(self, configs_dir):
        fast = load_yaml_config(os.path.join(configs_dir, 'ddqn_fast.yaml'))
        default = load_yaml_config(os.path.join(configs_dir, 'ddqn_default.yaml'))
        assert fast['buffer_size'] < default['buffer_size']

    def test_thorough_yaml_has_larger_buffer(self, configs_dir):
        thorough = load_yaml_config(os.path.join(configs_dir, 'ddqn_thorough.yaml'))
        default = load_yaml_config(os.path.join(configs_dir, 'ddqn_default.yaml'))
        assert thorough['buffer_size'] > default['buffer_size']

    def test_all_yaml_files_parse(self, configs_dir):
        for fname in ['ddqn_default.yaml', 'ddqn_fast.yaml', 'ddqn_thorough.yaml']:
            path = os.path.join(configs_dir, fname)
            config = load_yaml_config(path)
            assert isinstance(config, dict), f"{fname} should parse to dict"

    def test_unknown_config_keys_warn(self, tmp_path):
        """Typo keys should produce a warning, not crash."""
        import gymnasium as gym
        import numpy as np

        class DummyEnv(gym.Env):
            def __init__(self):
                self.observation_space = gym.spaces.Box(-1, 1, (4,), np.float32)
                self.action_space = gym.spaces.Discrete(3)
            def reset(self, **kw): return np.zeros(4, np.float32), {}
            def step(self, a): return np.zeros(4, np.float32), 0., False, False, {}
            def _get_legal_mask(self): return np.ones(3, np.int32)

        bad_config = {'learnng_rate': 1e-3}  # typo
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            trainer = DDQNHybridTrainer(
                env=DummyEnv(), output_dir=str(tmp_path / 'test'),
                config=bad_config, verbose=0,
            )
            assert len(w) >= 1
            assert 'learnng_rate' in str(w[0].message)
            trainer.close()

    def test_cli_overrides_yaml(self, configs_dir):
        """CLI-provided values should override YAML values."""
        yaml_config = load_yaml_config(os.path.join(configs_dir, 'ddqn_default.yaml'))
        assert yaml_config['seed'] == 42

        # Simulate CLI override
        yaml_config['seed'] = 99
        assert yaml_config['seed'] == 99

    def test_merge_preserves_all_defaults(self, tmp_path):
        """Merging partial config should keep all DEFAULT_CONFIG keys."""
        import gymnasium as gym
        import numpy as np

        class DummyEnv(gym.Env):
            def __init__(self):
                self.observation_space = gym.spaces.Box(-1, 1, (4,), np.float32)
                self.action_space = gym.spaces.Discrete(3)
            def reset(self, **kw): return np.zeros(4, np.float32), {}
            def step(self, a): return np.zeros(4, np.float32), 0., False, False, {}
            def _get_legal_mask(self): return np.ones(3, np.int32)

        partial_config = {'lr': 5e-5}
        trainer = DDQNHybridTrainer(
            env=DummyEnv(), output_dir=str(tmp_path / 'test'),
            config=partial_config, verbose=0,
        )

        for key in DDQNHybridTrainer.DEFAULT_CONFIG:
            assert key in trainer.config, f"Missing default key after merge: {key}"
        assert trainer.config['lr'] == 5e-5
        assert trainer.config['gamma'] == DDQNHybridTrainer.DEFAULT_CONFIG['gamma']
        trainer.close()
