#!/usr/bin/env python3
"""
Minimal Smoke Test for Monopoly RL Environment.

Verifies:
1. Environment can be created and reset
2. Both reward modes work
3. Training runs without crashing

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
    """Test environment creation and step."""
    from Monopoly.envs.gym_env import MonopolyEnv
    import numpy as np
    
    print("Testing environment...")
    
    # Test both reward modes
    for mode in ['dense_networth', 'sparse_terminal']:
        env = MonopolyEnv(num_players=2, seed=42, reward_mode=mode)
        obs, _ = env.reset(seed=42)
        
        # Take a few steps
        for _ in range(5):
            mask = obs['legal_mask']
            action = np.where(mask == 1)[0][0]
            obs, reward, term, trunc, _ = env.step(action)
            if term or trunc:
                break
        
        env.close()
        print(f"  OK: {mode}")
    
    print("Environment tests passed.")
    return True


def test_training(timesteps=500):
    """Test that training runs without crashing."""
    from Monopoly.envs.gym_env import MonopolyEnv
    from Monopoly.envs.wrappers import MonopolyFlattenWrapper
    from stable_baselines3 import DQN
    
    print(f"Testing training ({timesteps} steps)...")
    
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
    
    print("Training test passed.")
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
        
        if not args.quick:
            passed &= test_training()
    
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
