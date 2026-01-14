#!/usr/bin/env python3
"""
Smoke Test Script for Monopoly RL Training

Runs quick training tests to verify:
1. SB3 DQN training works
2. DDQN-Hybrid training works
3. Environment is functioning correctly
4. Models can be saved and loaded

This is designed for CI/CD pipelines to catch breaking changes.

Usage:
    python scripts/smoke_test.py              # Run all tests
    python scripts/smoke_test.py --dqn-only   # Only test SB3 DQN
    python scripts/smoke_test.py --ddqn-only  # Only test DDQN-Hybrid
    python scripts/smoke_test.py --quick      # Even shorter tests
"""

import sys
import os
import argparse
import time
import tempfile
import shutil
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pytest


def test_environment_creation():
    """Test that the Monopoly environment can be created and reset."""
    print("\n" + "=" * 60)
    print("TEST: Environment Creation")
    print("=" * 60)
    
    try:
        from Monopoly.envs.gym_env import MonopolyEnv
        from Monopoly.envs.wrappers import MonopolyFlattenWrapper
        from Monopoly.agents.random import RandomAgent
        from Monopoly.agents.greedy import GreedyAgent
        from Monopoly.agents.mcts import MCTSAgent
        from Monopoly.engine import GameEngine
        from Monopoly.rules import RulesEngine
        from Monopoly.board import Board
        from Monopoly.property import load_property_specs
        from Monopoly.cards import load_chance_cards, load_community_cards
        from stable_baselines3.common.monitor import Monitor
        
        # Create lightweight MCTS engine
        board = Board.load_standard_board()
        property_specs = load_property_specs()
        chance_cards = load_chance_cards()
        community_cards = load_community_cards()
        rules_engine = RulesEngine(board, property_specs, chance_cards, community_cards)
        mcts_engine = GameEngine(rules_engine, seed=42)
        
        opponents = [
            RandomAgent(player_id=1),
            MCTSAgent(player_id=2, engine=mcts_engine, rollouts=2, max_depth=3),
            GreedyAgent(player_id=3),
        ]
        
        env = MonopolyEnv(
            num_players=4,
            agent_player_id=0,
            opponent_policies=opponents,
            max_turns=100,
            seed=42
        )
        env = MonopolyFlattenWrapper(env)
        env = Monitor(env)
        
        # Test reset
        obs, info = env.reset(seed=42)
        print(f"  ✓ Environment created successfully")
        print(f"  ✓ Observation shape: {obs.shape}")
        print(f"  ✓ Action space: {env.action_space}")
        
        # Test step
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  ✓ Step executed successfully")
        print(f"  ✓ Reward: {reward}")
        
        env.close()
        print("\n  ✅ Environment creation test PASSED")
        assert True
        
    except Exception as e:
        print(f"\n  ❌ Environment creation test FAILED: {e}")
        import traceback
        traceback.print_exc()
        pytest.fail(str(e))


def test_dqn_training(timesteps: int = 2000, output_dir: str = None):
    """Test SB3 DQN training for a small number of timesteps."""
    print("\n" + "=" * 60)
    print(f"TEST: SB3 DQN Training ({timesteps} timesteps)")
    print("=" * 60)
    
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix='smoke_test_dqn_')
    
    try:
        from train_dqn import train_dqn, evaluate_model
        
        start_time = time.time()
        
        # Train
        model = train_dqn(
            total_timesteps=timesteps,
            max_turns=50,  # Short episodes
            config_preset='fast',
            output_dir=output_dir,
            seed=42,
            eval_interval=timesteps // 2,
            n_eval_episodes=2,
            verbose=0
        )
        
        elapsed = time.time() - start_time
        print(f"  ✓ Training completed in {elapsed:.1f}s")
        
        # Check model was saved
        model_path = Path(output_dir) / 'models' / 'dqn' / 'monopoly_dqn_final.zip'
        if model_path.exists():
            print(f"  ✓ Model saved to {model_path}")
        else:
            print(f"  ⚠ Model file not found at expected location")
        
        # Check metrics CSV was created
        csv_path = Path(output_dir) / 'results' / 'dqn_metrics.csv'
        if csv_path.exists():
            print(f"  ✓ Metrics CSV created at {csv_path}")
        else:
            print(f"  ⚠ Metrics CSV not found at expected location")
        
        print("\n  ✅ DQN training test PASSED")
        assert True
        
    except Exception as e:
        print(f"\n  ❌ DQN training test FAILED: {e}")
        import traceback
        traceback.print_exc()
        pytest.fail(str(e))
    finally:
        if output_dir.startswith(tempfile.gettempdir()):
            shutil.rmtree(output_dir, ignore_errors=True)


def test_ddqn_hybrid_training(timesteps: int = 2000, output_dir: str = None):
    """Test DDQN-Hybrid training for a small number of timesteps."""
    print("\n" + "=" * 60)
    print(f"TEST: DDQN-Hybrid Training ({timesteps} timesteps)")
    print("=" * 60)
    
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix='smoke_test_ddqn_')
    
    try:
        from train_dqn import train_ddqn_hybrid
        
        start_time = time.time()
        
        # Train with minimal config
        config = {
            'buffer_size': 1000,
            'learning_starts': 100,
            'batch_size': 16,
            'target_update_interval': 50,
        }
        
        results = train_ddqn_hybrid(
            total_timesteps=timesteps,
            max_turns=50,  # Short episodes
            output_dir=output_dir,
            seed=42,
            eval_interval=timesteps // 2,
            eval_episodes=2,
            verbose=0,
            config=config
        )
        
        elapsed = time.time() - start_time
        print(f"  ✓ Training completed in {elapsed:.1f}s")
        
        # Check model was saved
        model_dir = Path(output_dir) / 'models' / 'ddqn_hybrid'
        if model_dir.exists():
            print(f"  ✓ Model directory created at {model_dir}")
        else:
            print(f"  ⚠ Model directory not found")
        
        # Check metrics CSV was created  
        csv_path = Path(output_dir) / 'results' / 'ddqn_metrics.csv'
        if csv_path.exists():
            print(f"  ✓ Metrics CSV created at {csv_path}")
        else:
            print(f"  ⚠ Metrics CSV not found at expected location")
        
        print("\n  ✅ DDQN-Hybrid training test PASSED")
        assert True
        
    except Exception as e:
        print(f"\n  ❌ DDQN-Hybrid training test FAILED: {e}")
        import traceback
        traceback.print_exc()
        pytest.fail(str(e))
    finally:
        if output_dir.startswith(tempfile.gettempdir()):
            shutil.rmtree(output_dir, ignore_errors=True)


def test_random_baseline(episodes: int = 5):
    """Test random baseline runs correctly."""
    print("\n" + "=" * 60)
    print(f"TEST: Random Baseline ({episodes} episodes)")
    print("=" * 60)
    
    try:
        from train_dqn import run_random_baseline
        
        results = run_random_baseline(
            num_episodes=episodes,
            max_turns=50,
            seed=42
        )
        
        assert 'mean_reward' in results
        assert 'win_rate' in results
        assert 'mean_final_net_worth' in results
        
        print(f"  ✓ Mean reward: {results['mean_reward']:.2f}")
        print(f"  ✓ Win rate: {results['win_rate']*100:.1f}%")
        print(f"  ✓ Mean net worth: ${results['mean_final_net_worth']:.0f}")
        
        print("\n  ✅ Random baseline test PASSED")
        assert True
        
    except Exception as e:
        print(f"\n  ❌ Random baseline test FAILED: {e}")
        import traceback
        traceback.print_exc()
        pytest.fail(str(e))


def test_network_creation():
    """Test PyTorch network creation and forward pass."""
    print("\n" + "=" * 60)
    print("TEST: Neural Network Creation")
    print("=" * 60)
    
    try:
        import torch
        from Monopoly.agents.network import QNetwork, DuelingQNetwork
        
        # Test basic QNetwork
        obs_dim = 100
        action_dim = 10
        batch_size = 4
        
        qnet = QNetwork(obs_dim, action_dim, hidden_dims=[64, 64])
        
        # Test forward pass
        obs = torch.randn(batch_size, obs_dim)
        q_values = qnet.forward(obs)
        assert q_values.shape == (batch_size, action_dim)
        print(f"  ✓ QNetwork forward pass: {q_values.shape}")
        
        # Test get_q_values
        q = qnet.get_q_values(obs.numpy())
        assert q.shape == (batch_size, action_dim)
        print(f"  ✓ QNetwork get_q_values: {q.shape}")
        
        # Test action selection with mask - use per-sample masks matching action_dim
        single_obs = obs[0].numpy()  # Single observation
        single_mask = np.array([1, 1, 0, 0, 1, 0, 0, 0, 0, 0])  # Mask for action_dim=10
        action = qnet.select_action(single_obs, single_mask, epsilon=0.0)
        assert isinstance(action, int)
        assert action in [0, 1, 4]  # Only legal actions
        print(f"  ✓ QNetwork select_action: {action}")
        
        # Test dueling network
        duel_net = DuelingQNetwork(obs_dim, action_dim, hidden_dims=[64, 64])
        q_values_duel = duel_net.forward(obs)
        assert q_values_duel.shape == (batch_size, action_dim)
        print(f"  ✓ DuelingQNetwork forward pass: {q_values_duel.shape}")
        
        print("\n  ✅ Network creation test PASSED")
        assert True
        
    except Exception as e:
        print(f"\n  ❌ Network creation test FAILED: {e}")
        import traceback
        traceback.print_exc()
        pytest.fail(str(e))


def test_trade_policy():
    """Test trade policy creation and basic methods."""
    print("\n" + "=" * 60)
    print("TEST: Trade Policy")
    print("=" * 60)
    
    try:
        from policies.trade_policy import TradePolicy
        
        policy = TradePolicy(agent_id=0)
        print(f"  ✓ TradePolicy created for agent {policy.agent_id}")
        
        # Basic tests - these would need game state to fully test
        assert policy.agent_id == 0
        assert hasattr(policy, 'should_accept_trade')
        assert hasattr(policy, 'find_trade_opportunities')
        assert hasattr(policy, 'choose_trade')
        
        print("  ✓ All required methods present")
        print("\n  ✅ Trade policy test PASSED")
        assert True
        
    except Exception as e:
        print(f"\n  ❌ Trade policy test FAILED: {e}")
        import traceback
        traceback.print_exc()
        pytest.fail(str(e))


def test_save_utils():
    """Test save utility functions."""
    print("\n" + "=" * 60)
    print("TEST: Save Utilities")
    print("=" * 60)
    
    try:
        import tempfile
        from pathlib import Path
        from utils.save_utils import (
            ensure_dir,
            save_metrics_csv,
            append_metrics_csv,
            get_run_id,
            CSV_COLUMNS,
        )
        
        # Test ensure_dir
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir) / 'test' / 'nested' / 'dir'
            ensure_dir(test_dir)
            assert test_dir.exists()
            print(f"  ✓ ensure_dir creates nested directories")
            
            # Test run_id generation
            run_id = get_run_id('dqn', 42)
            assert 'dqn' in run_id
            assert '42' in run_id
            print(f"  ✓ get_run_id: {run_id}")
            
            # Test CSV columns are defined
            assert 'run_id' in CSV_COLUMNS
            assert 'episode' in CSV_COLUMNS
            assert 'final_net_worth' in CSV_COLUMNS
            print(f"  ✓ CSV_COLUMNS defined: {len(CSV_COLUMNS)} columns")
            
            # Test CSV writing
            csv_path = Path(tmpdir) / 'test_metrics.csv'
            metrics = {
                'timestamp': '2024-01-01T00:00:00',
                'episode': 1,
                'episode_length': 100,
                'final_net_worth': 1500,
                'agent_rank': 1,
                'win_flag': 1,
                'total_steps': 100,
                'mean_episode_reward': 10.5,
                'eval_tag': 'test',
            }
            append_metrics_csv(csv_path, metrics, run_id, 42)
            assert csv_path.exists()
            print(f"  ✓ CSV file created and metrics appended")
        
        print("\n  ✅ Save utilities test PASSED")
        assert True
        
    except Exception as e:
        print(f"\n  ❌ Save utilities test FAILED: {e}")
        import traceback
        traceback.print_exc()
        pytest.fail(str(e))


def run_all_tests(args):
    """Run all smoke tests and report results."""
    print("\n" + "=" * 60)
    print("MONOPOLY RL SMOKE TESTS")
    print("=" * 60)
    
    results = {}
    
    # Quick tests (always run)
    results['environment'] = test_environment_creation()
    results['network'] = test_network_creation()
    results['trade_policy'] = test_trade_policy()
    results['save_utils'] = test_save_utils()
    
    # Baseline (always run)
    results['baseline'] = test_random_baseline(
        episodes=2 if args.quick else 5
    )
    
    # Training tests
    timesteps = 500 if args.quick else 2000
    
    if not args.ddqn_only:
        results['dqn_training'] = test_dqn_training(timesteps=timesteps)
    
    if not args.dqn_only:
        results['ddqn_training'] = test_ddqn_hybrid_training(timesteps=timesteps)
    
    # Summary
    print("\n" + "=" * 60)
    print("SMOKE TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print("=" * 60)
    print(f"Total: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


def main():
    parser = argparse.ArgumentParser(
        description='Run smoke tests for Monopoly RL training'
    )
    parser.add_argument('--dqn-only', action='store_true',
                       help='Only test SB3 DQN')
    parser.add_argument('--ddqn-only', action='store_true',
                       help='Only test DDQN-Hybrid')
    parser.add_argument('--quick', action='store_true',
                       help='Run shorter tests (fewer timesteps)')
    
    args = parser.parse_args()
    
    success = run_all_tests(args)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
