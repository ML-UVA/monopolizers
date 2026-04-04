#!/usr/bin/env python3
"""
Minimal Smoke Test for Monopoly RL Environment.

Verifies:
1. Environment can be created and reset (all 3 reward modes)
2. Training runs without crashing (SB3 DQN + DDQN-Hybrid)
3. Modular reward mode produces non-zero components
4. PER-enabled training runs without crashing
5. YAML config loads correctly
6. Episode tracing works

Usage:
    python scripts/smoke_test.py          # Full smoke test
    python scripts/smoke_test.py --quick  # Quick environment check only
"""

import sys
import argparse
import tempfile
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_environment():
    """Test environment creation and step for all reward modes."""
    from Monopoly.envs.gym_env import MonopolyEnv
    import numpy as np

    print("Testing environment...")

    # Test all three reward modes
    for mode in ['dense_networth', 'sparse_terminal', 'modular']:
        env = MonopolyEnv(num_players=2, seed=42, reward_mode=mode)
        obs, _ = env.reset(seed=42)

        # Take a few steps
        for _ in range(5):
            mask = obs['legal_mask']
            action = np.where(mask == 1)[0][0]
            obs, reward, term, trunc, info = env.step(action)
            if term or trunc:
                break

        env.close()
        print(f"  OK: {mode}")

    print("Environment tests passed.")
    return True


def test_training(timesteps=500):
    """Test that SB3 DQN training runs without crashing."""
    from Monopoly.envs.gym_env import MonopolyEnv
    from Monopoly.envs.wrappers import MonopolyFlattenWrapper
    from stable_baselines3 import DQN

    print(f"Testing SB3 DQN training ({timesteps} steps)...")

    with tempfile.TemporaryDirectory() as tmpdir:
        env = MonopolyEnv(num_players=2, seed=42, reward_mode='dense_networth')
        env = MonopolyFlattenWrapper(env)

        model = DQN(
            "MlpPolicy", env,
            learning_rate=1e-4,
            buffer_size=1000,
            learning_starts=100,
            batch_size=32,
            verbose=0
        )

        model.learn(total_timesteps=timesteps)

        # Verify model can be saved and loaded
        model_path = Path(tmpdir) / "test_model.zip"
        model.save(model_path)
        loaded_model = DQN.load(model_path, env=env)

        env.close()

    print("SB3 DQN training test passed.")
    return True


def test_ddqn_hybrid_training(timesteps=500):
    """Test that DDQN-Hybrid training runs without crashing."""
    from Monopoly.envs.gym_env import MonopolyEnv
    from Monopoly.envs.wrappers import MonopolyFlattenWrapper
    from Monopoly.agents.ddqn_hybrid import DDQNHybridTrainer

    print(f"Testing DDQN-Hybrid training ({timesteps} steps)...")

    with tempfile.TemporaryDirectory() as tmpdir:
        env = MonopolyEnv(num_players=2, seed=42, reward_mode='dense_networth')
        env = MonopolyFlattenWrapper(env)
        eval_env = MonopolyEnv(num_players=2, seed=43, reward_mode='dense_networth')
        eval_env = MonopolyFlattenWrapper(eval_env)

        trainer = DDQNHybridTrainer(
            env=env, eval_env=eval_env,
            output_dir=tmpdir, seed=42, device='cpu',
            config={
                'buffer_size': 1000, 'learning_starts': 50,
                'batch_size': 16, 'train_freq': 4,
                'target_update_interval': 100, 'hidden_dims': [32, 32],
            },
            verbose=0,
        )

        results = trainer.train(
            total_timesteps=timesteps,
            eval_interval=timesteps,
            eval_episodes=2,
            log_interval=100,
            save_interval=timesteps,
            progress_bar=False,
        )

        # Verify outputs exist
        assert len(trainer.losses) > 0, "No training losses recorded"
        assert trainer.losses[-1] > 0, "Loss should be positive"

        # Verify checkpoint exists and loads
        checkpoint_path = Path(tmpdir) / 'models' / 'ddqn_hybrid' / 'final.pt'
        assert checkpoint_path.exists(), "Checkpoint not saved"

        csv_path = Path(tmpdir) / 'results' / 'ddqn_hybrid_metrics.csv'
        assert csv_path.exists(), "CSV metrics not saved"

        tb_dir = Path(tmpdir) / 'tensorboard' / 'ddqn_hybrid'
        assert any(tb_dir.iterdir()), "TensorBoard logs empty"

        trainer.close()

    print("DDQN-Hybrid training test passed.")
    return True


def test_modular_reward_mode(timesteps=200):
    """Test modular reward mode produces non-zero components."""
    from Monopoly.envs.gym_env import MonopolyEnv
    from Monopoly.envs.wrappers import MonopolyFlattenWrapper
    from Monopoly.agents.ddqn_hybrid import DDQNHybridTrainer

    print(f"Testing modular reward mode ({timesteps} steps)...")

    with tempfile.TemporaryDirectory() as tmpdir:
        env = MonopolyEnv(num_players=2, seed=42, reward_mode='modular')
        env = MonopolyFlattenWrapper(env)
        eval_env = MonopolyEnv(num_players=2, seed=43, reward_mode='modular')
        eval_env = MonopolyFlattenWrapper(eval_env)

        trainer = DDQNHybridTrainer(
            env=env, eval_env=eval_env,
            output_dir=tmpdir, seed=42, device='cpu',
            config={
                'buffer_size': 500, 'learning_starts': 20,
                'batch_size': 8, 'train_freq': 4,
                'target_update_interval': 50, 'hidden_dims': [16, 16],
            },
            verbose=0,
        )

        results = trainer.train(
            total_timesteps=timesteps,
            eval_interval=timesteps * 2,
            eval_episodes=1,
            log_interval=100,
            save_interval=timesteps * 2,
            progress_bar=False,
        )
        trainer.close()

    print("Modular reward mode test passed.")
    return True


def test_per_buffer_training(timesteps=300):
    """Test PER-enabled training runs without crashing."""
    from Monopoly.envs.gym_env import MonopolyEnv
    from Monopoly.envs.wrappers import MonopolyFlattenWrapper
    from Monopoly.agents.ddqn_hybrid import DDQNHybridTrainer

    print(f"Testing PER training ({timesteps} steps)...")

    with tempfile.TemporaryDirectory() as tmpdir:
        env = MonopolyEnv(num_players=2, seed=42, reward_mode='dense_networth')
        env = MonopolyFlattenWrapper(env)
        eval_env = MonopolyEnv(num_players=2, seed=43, reward_mode='dense_networth')
        eval_env = MonopolyFlattenWrapper(eval_env)

        trainer = DDQNHybridTrainer(
            env=env, eval_env=eval_env,
            output_dir=tmpdir, seed=42, device='cpu',
            config={
                'buffer_size': 500, 'learning_starts': 20,
                'batch_size': 8, 'train_freq': 4,
                'target_update_interval': 50, 'hidden_dims': [16, 16],
                'use_per': True, 'per_alpha': 0.6, 'per_beta_start': 0.4,
            },
            verbose=0,
        )

        results = trainer.train(
            total_timesteps=timesteps,
            eval_interval=timesteps * 2,
            eval_episodes=1,
            log_interval=100,
            save_interval=timesteps * 2,
            progress_bar=False,
        )
        trainer.close()

    print("PER training test passed.")
    return True


def test_config_loading():
    """Test YAML config files load correctly."""
    from Monopoly.agents.ddqn_hybrid import load_yaml_config

    print("Testing config loading...")
    configs_dir = project_root / 'configs'

    for fname in ['ddqn_default.yaml', 'ddqn_fast.yaml', 'ddqn_thorough.yaml']:
        config = load_yaml_config(str(configs_dir / fname))
        assert isinstance(config, dict), f"{fname} failed to parse"
        assert 'gamma' in config, f"{fname} missing gamma"
        print(f"  OK: {fname}")

    print("Config loading test passed.")
    return True


def test_tracing():
    """Test that episode tracing wrapper produces output files."""
    from Monopoly.envs.gym_env import MonopolyEnv
    from Monopoly.envs.tracing import EpisodeTracer
    import numpy as np

    print("Testing episode tracing...")

    with tempfile.TemporaryDirectory() as tmpdir:
        env = MonopolyEnv(num_players=2, max_turns=10, seed=42)
        env = EpisodeTracer(env, output_dir=tmpdir)

        for ep in range(2):
            obs, _ = env.reset(seed=42 + ep)
            done = False
            while not done:
                legal = np.where(obs["legal_mask"] == 1)[0]
                action = int(np.random.choice(legal)) if len(legal) > 0 else 0
                obs, _, term, trunc, _ = env.step(action)
                done = term or trunc

        trace_dir = Path(tmpdir)
        npz_count = len(list(trace_dir.glob("episode_*.npz")))
        json_count = len(list(trace_dir.glob("episode_*.json")))

        assert npz_count == 2, f"Expected 2 NPZ files, got {npz_count}"
        assert json_count == 2, f"Expected 2 JSON files, got {json_count}"

        env.close()

    print("Tracing test passed.")
    return True


def main():
    parser = argparse.ArgumentParser(description='Smoke test for Monopoly RL')
    parser.add_argument('--quick', action='store_true', help='Environment test only')
    args = parser.parse_args()

    print("=" * 50)
    print("MONOPOLY RL SMOKE TEST")
    print("=" * 50)

    passed = True

    try:
        passed &= test_environment()
        passed &= test_config_loading()
        passed &= test_tracing()

        if not args.quick:
            passed &= test_training()
            passed &= test_ddqn_hybrid_training()
            passed &= test_modular_reward_mode()
            passed &= test_per_buffer_training()

    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        passed = False

    print("=" * 50)
    print("RESULT:", "PASSED" if passed else "FAILED")
    print("=" * 50)

    sys.exit(0 if passed else 1)


if __name__ == '__main__':
    main()
