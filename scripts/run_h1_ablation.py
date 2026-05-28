#!/usr/bin/env python3
"""
H1 Ablation Study: Dense Net-Worth vs Sparse Terminal Reward

This script runs the complete H1 ablation experiment comparing:
- dense_networth: r = nw_agent / Σ(nw_others) every step
- sparse_terminal: +1 win, -1 lose, 0 otherwise

Experimental Protocol:
- Algorithm: DQN (SB3) or DDQN-Hybrid
- Seeds: 5 independent runs per condition (42, 43, 44, 45, 46)
- Timesteps: 500k per run (adjustable)
- Metrics: Win rate, final net worth, episode length, learning curves

Usage:
    # Run full ablation (5 seeds × 2 conditions = 10 runs)
    python scripts/run_h1_ablation.py --agent dqn --timesteps 500000
    
    # Quick test (1 seed, 50k steps)
    python scripts/run_h1_ablation.py --agent dqn --timesteps 50000 --seeds 42
    
    # With DDQN-Hybrid
    python scripts/run_h1_ablation.py --agent ddqn_hybrid --timesteps 500000

Output Structure:
    runs/h1_ablation/
    ├── dqn_dense_networth_seed42/
    ├── dqn_dense_networth_seed43/
    ├── dqn_sparse_terminal_seed42/
    ├── dqn_sparse_terminal_seed43/
    └── ablation_summary.csv

See RESEARCH_IMPLEMENTATION_AUDIT_NOTES.txt for full experimental design.
"""

import sys
import os
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List
import pandas as pd

# Default experimental parameters
DEFAULT_SEEDS = [42, 43, 44, 45, 46]
DEFAULT_TIMESTEPS = 500000
REWARD_MODES = ['dense_networth', 'sparse_terminal']


def run_single_experiment(
    agent: str,
    reward_mode: str,
    seed: int,
    timesteps: int,
    output_base: str,
    max_turns: int = 500,
    verbose: bool = True
) -> dict:
    """Run a single training experiment.
    
    Returns:
        dict with run metadata and status
    """
    output_dir = f"{output_base}/{agent}_{reward_mode}_seed{seed}"
    
    cmd = [
        sys.executable, "train_dqn.py",
        "--train",
        "--agent", agent,
        "--reward_mode", reward_mode,
        "--seed", str(seed),
        "--total_timesteps", str(timesteps),
        "--max_turns", str(max_turns),
        "--output_dir", output_dir,
        "--verbose", "1" if verbose else "0"
    ]
    
    print(f"\n{'='*60}")
    print(f"Running: {agent} | {reward_mode} | seed={seed}")
    print(f"Output: {output_dir}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    start_time = datetime.now()
    
    try:
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).parent.parent,
            check=True,
            capture_output=not verbose
        )
        status = "success"
        error = None
    except subprocess.CalledProcessError as e:
        status = "failed"
        error = str(e)
        print(f"ERROR: Run failed - {error}")
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    return {
        "agent": agent,
        "reward_mode": reward_mode,
        "seed": seed,
        "timesteps": timesteps,
        "output_dir": output_dir,
        "status": status,
        "duration_seconds": duration,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "error": error
    }


def run_ablation_study(
    agent: str,
    seeds: List[int],
    timesteps: int,
    output_base: str,
    max_turns: int = 500,
    verbose: bool = True
) -> pd.DataFrame:
    """Run the complete H1 ablation study.
    
    Args:
        agent: 'dqn' or 'ddqn_hybrid'
        seeds: List of random seeds
        timesteps: Training timesteps per run
        output_base: Base directory for outputs
        max_turns: Max game turns per episode
        verbose: Print training output
        
    Returns:
        DataFrame with all run results
    """
    results = []
    total_runs = len(seeds) * len(REWARD_MODES)
    current_run = 0
    
    print(f"\n{'#'*60}")
    print(f"# H1 ABLATION STUDY")
    print(f"# Agent: {agent}")
    print(f"# Reward modes: {REWARD_MODES}")
    print(f"# Seeds: {seeds}")
    print(f"# Timesteps: {timesteps}")
    print(f"# Total runs: {total_runs}")
    print(f"{'#'*60}\n")
    
    for reward_mode in REWARD_MODES:
        for seed in seeds:
            current_run += 1
            print(f"\n[Run {current_run}/{total_runs}]")
            
            result = run_single_experiment(
                agent=agent,
                reward_mode=reward_mode,
                seed=seed,
                timesteps=timesteps,
                output_base=output_base,
                max_turns=max_turns,
                verbose=verbose
            )
            results.append(result)
    
    # Create summary DataFrame
    df = pd.DataFrame(results)
    
    # Save summary
    summary_path = Path(output_base) / "ablation_summary.csv"
    df.to_csv(summary_path, index=False)
    print(f"\n{'='*60}")
    print(f"ABLATION STUDY COMPLETE")
    print(f"Summary saved to: {summary_path}")
    print(f"{'='*60}")
    
    # Print summary statistics
    print("\nRun Summary:")
    print(df[["agent", "reward_mode", "seed", "status", "duration_seconds"]])
    
    success_count = (df["status"] == "success").sum()
    print(f"\nSuccess rate: {success_count}/{total_runs}")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Run H1 Ablation Study: Dense vs Sparse Reward",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full ablation (5 seeds × 2 conditions)
    python scripts/run_h1_ablation.py --agent dqn --timesteps 500000
    
    # Quick test
    python scripts/run_h1_ablation.py --agent dqn --timesteps 50000 --seeds 42
    
    # Multiple specific seeds
    python scripts/run_h1_ablation.py --agent ddqn_hybrid --seeds 42 43 44
"""
    )
    
    parser.add_argument(
        "--agent", type=str, default="dqn",
        choices=["dqn", "ddqn_hybrid"],
        help="Agent type (default: dqn)"
    )
    parser.add_argument(
        "--timesteps", type=int, default=DEFAULT_TIMESTEPS,
        help=f"Training timesteps per run (default: {DEFAULT_TIMESTEPS})"
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
        help=f"Random seeds (default: {DEFAULT_SEEDS})"
    )
    parser.add_argument(
        "--output_dir", type=str, default="runs/h1_ablation",
        help="Base output directory (default: runs/h1_ablation)"
    )
    parser.add_argument(
        "--max_turns", type=int, default=500,
        help="Max game turns per episode (default: 500)"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress training output"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Run ablation study
    results = run_ablation_study(
        agent=args.agent,
        seeds=args.seeds,
        timesteps=args.timesteps,
        output_base=args.output_dir,
        max_turns=args.max_turns,
        verbose=not args.quiet
    )
    
    print("\nDone! Next steps:")
    print("  1. tensorboard --logdir runs/h1_ablation")
    print("  2. Compare learning curves for dense_networth vs sparse_terminal")
    print("  3. Analyze ablation_summary.csv for win rates and final net worth")


if __name__ == "__main__":
    main()
