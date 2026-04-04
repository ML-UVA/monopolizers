#!/usr/bin/env python3
"""Batch evaluation of trained Monopoly RL agents with episode tracing.

Runs a model checkpoint for N episodes, recording per-turn traces (NPZ/JSON),
producing a summary CSV, plots, and TensorBoard scalars.

Usage:
    python scripts/evaluate_agent.py \
        --model runs/dqn_dense_networth_seed42/models/dqn/monopoly_dqn_final.zip \
        --agent-type dqn \
        --episodes 200 \
        --seed 42 \
        --output-dir runs/dqn_dense_networth_seed42/analysis

    python scripts/evaluate_agent.py \
        --model runs/ddqn_hybrid_.../models/ddqn_hybrid/checkpoint_online.pt \
        --agent-type ddqn_hybrid \
        --episodes 100
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from Monopoly.envs.gym_env import MonopolyEnv
from Monopoly.envs.tracing import EpisodeTracer
from Monopoly.envs.wrappers import MonopolyFlattenWrapper
from train_dqn import create_opponent_agents
from utils.trace_utils import (
    load_all_summaries,
    load_episode_trace,
    extract_all_features,
)


def create_eval_env(seed, max_turns, reward_mode, trace_dir, model_path=""):
    """Create evaluation environment with tracing enabled."""
    opponents = create_opponent_agents(seed=seed)
    env = MonopolyEnv(
        num_players=4,
        agent_player_id=0,
        opponent_policies=opponents,
        max_turns=max_turns,
        seed=seed,
        reward_mode=reward_mode,
    )
    env = EpisodeTracer(env, output_dir=trace_dir, model_path=model_path)
    env = MonopolyFlattenWrapper(env)
    return env


def load_model(model_path, agent_type):
    """Load a model checkpoint based on agent type."""
    if agent_type == "dqn":
        from stable_baselines3 import DQN
        return DQN.load(model_path), "sb3"
    elif agent_type == "ddqn_hybrid":
        from utils.trace_utils import load_ddqn_for_eval
        return load_ddqn_for_eval(model_path), "ddqn"
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def predict_action(model, model_type, obs, env):
    """Get action from model with legal action masking."""
    if model_type == "sb3":
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)
        # Check legality
        legal_mask = env.unwrapped._get_legal_mask()
        if legal_mask[action] == 0:
            legal_actions = np.where(legal_mask)[0]
            if len(legal_actions) > 0:
                action = int(np.random.choice(legal_actions))
    else:
        # DDQN - use QNetwork.select_action with masking
        legal_mask = env.unwrapped._get_legal_mask()
        action = model.select_action(obs, action_mask=legal_mask, epsilon=0.0)
    return action


def run_evaluation(model, model_type, env, num_episodes, seed):
    """Run evaluation episodes. Traces are saved by the EpisodeTracer wrapper."""
    for ep in range(num_episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        step_count = 0
        while not done:
            action = predict_action(model, model_type, obs, env)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step_count += 1

        if (ep + 1) % 25 == 0 or ep == 0:
            nw = info.get("agent_net_worth", 0)
            print(f"  Episode {ep + 1}/{num_episodes}: steps={step_count}, NW=${nw:.0f}")

    env.close()


def generate_plots(trace_dir, output_dir):
    """Generate analysis plots from trace data."""
    plots_dir = Path(output_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    summaries = load_all_summaries(trace_dir)
    if summaries.empty:
        print("No episode summaries found, skipping plots.")
        return

    # --- Win rate bar chart ---
    win_rate = summaries["agent_win"].mean()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Agent", "Random\nBaseline"], [win_rate, 0.25], color=["#2196F3", "#9E9E9E"])
    ax.set_ylabel("Win Rate")
    ax.set_title("Agent Win Rate vs Random Baseline (25%)")
    ax.set_ylim(0, 1)
    for i, v in enumerate([win_rate, 0.25]):
        ax.text(i, v + 0.02, f"{v:.1%}", ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(plots_dir / "win_rate.png", dpi=150)
    plt.close()

    # --- Episode length histogram ---
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(summaries["episode_length"], bins=30, color="#4CAF50", edgecolor="black", alpha=0.8)
    ax.set_xlabel("Episode Length (steps)")
    ax.set_ylabel("Count")
    ax.set_title("Episode Length Distribution")
    ax.axvline(summaries["episode_length"].mean(), color="red", linestyle="--",
               label=f"Mean: {summaries['episode_length'].mean():.0f}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "episode_length_hist.png", dpi=150)
    plt.close()

    # --- Final net worth box plot ---
    nw_data = summaries["final_net_worths"].tolist()
    if nw_data:
        n_players = len(nw_data[0])
        player_nws = {f"Player {i}": [ep[i] for ep in nw_data] for i in range(n_players)}
        labels = ["Agent (P0)", "Random (P1)", "MCTS (P2)", "Greedy (P3)"][:n_players]

        fig, ax = plt.subplots(figsize=(8, 5))
        bp = ax.boxplot(
            [player_nws[f"Player {i}"] for i in range(n_players)],
            labels=labels,
            patch_artist=True,
        )
        colors = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0"][:n_players]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_ylabel("Final Net Worth ($)")
        ax.set_title("Final Net Worth by Player")
        plt.tight_layout()
        plt.savefig(plots_dir / "final_net_worth_box.png", dpi=150)
        plt.close()

    # --- Net worth curves (sample up to 10 episodes) ---
    trace_path = Path(trace_dir)
    npz_files = sorted(trace_path.glob("episode_*.npz"))
    sample_files = npz_files[:min(10, len(npz_files))]

    if sample_files:
        fig, ax = plt.subplots(figsize=(10, 5))
        # Collect all agent NW curves for mean/std
        all_agent_nws = []
        max_len = 0
        for npz_file in npz_files:
            trace = load_episode_trace(str(npz_file))
            if "net_worth" in trace:
                agent_nw = trace["net_worth"][:, 0]
                all_agent_nws.append(agent_nw)
                max_len = max(max_len, len(agent_nw))

        if all_agent_nws:
            # Pad to same length for mean/std
            padded = np.full((len(all_agent_nws), max_len), np.nan)
            for i, nw in enumerate(all_agent_nws):
                padded[i, :len(nw)] = nw

            mean_nw = np.nanmean(padded, axis=0)
            std_nw = np.nanstd(padded, axis=0)
            turns = np.arange(max_len)

            ax.plot(turns, mean_nw, color="#2196F3", linewidth=2, label="Agent Mean NW")
            ax.fill_between(turns, mean_nw - std_nw, mean_nw + std_nw,
                            alpha=0.2, color="#2196F3")

            # Plot a few individual curves lightly
            for nw in all_agent_nws[:5]:
                ax.plot(range(len(nw)), nw, alpha=0.15, color="#2196F3", linewidth=0.5)

        ax.set_xlabel("Turn")
        ax.set_ylabel("Net Worth ($)")
        ax.set_title("Agent Net Worth Over Time")
        ax.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / "net_worth_curves.png", dpi=150)
        plt.close()

    print(f"Plots saved to {plots_dir}")


def write_tensorboard(summaries_df, output_dir):
    """Write evaluation metrics to TensorBoard."""
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        print("TensorBoard not available, skipping.")
        return

    tb_dir = Path(output_dir) / "tensorboard"
    writer = SummaryWriter(log_dir=str(tb_dir))

    for i, row in summaries_df.iterrows():
        writer.add_scalar("eval/net_worth", row["final_net_worths"][0], i)
        writer.add_scalar("eval/episode_length", row["episode_length"], i)
        writer.add_scalar("eval/win", int(row["agent_win"]), i)
        writer.add_scalar("eval/total_reward", row["total_reward"], i)

    win_rate = summaries_df["agent_win"].mean()
    writer.add_scalar("eval/win_rate", win_rate, 0)
    writer.add_scalar("eval/mean_net_worth",
                       summaries_df["final_net_worths"].apply(lambda x: x[0]).mean(), 0)
    writer.close()
    print(f"TensorBoard logs written to {tb_dir}")


def main():
    parser = argparse.ArgumentParser(description="Batch evaluate a Monopoly RL agent with tracing")
    parser.add_argument("--model", required=True, help="Path to model checkpoint")
    parser.add_argument("--agent-type", choices=["dqn", "ddqn_hybrid"], default="dqn",
                        help="Model type (default: dqn)")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes (default: 100)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--max-turns", type=int, default=500, help="Max turns per episode (default: 500)")
    parser.add_argument("--reward-mode", default="dense_networth",
                        choices=["dense_networth", "sparse_terminal"],
                        help="Reward mode (default: dense_networth)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: alongside model)")
    args = parser.parse_args()

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(args.model).parent.parent.parent / "analysis"
    trace_dir = output_dir / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Monopoly Agent Batch Evaluation")
    print("=" * 60)
    print(f"  Model:      {args.model}")
    print(f"  Agent type: {args.agent_type}")
    print(f"  Episodes:   {args.episodes}")
    print(f"  Seed:       {args.seed}")
    print(f"  Output:     {output_dir}")
    print("=" * 60)

    # Load model
    print("\nLoading model...")
    model, model_type = load_model(args.model, args.agent_type)

    # Create environment with tracing
    print("Creating evaluation environment with tracing...")
    env = create_eval_env(
        seed=args.seed,
        max_turns=args.max_turns,
        reward_mode=args.reward_mode,
        trace_dir=str(trace_dir),
        model_path=args.model,
    )

    # Run evaluation
    print(f"\nRunning {args.episodes} episodes...")
    run_evaluation(model, model_type, env, args.episodes, args.seed)

    # Load summaries and save CSV
    print("\nGenerating summary CSV...")
    summaries = load_all_summaries(str(trace_dir))
    csv_path = output_dir / "eval_summary.csv"
    summaries.to_csv(csv_path, index=False)
    print(f"Summary CSV saved to {csv_path}")

    # Extract features CSV
    print("Extracting strategy features...")
    features_df = extract_all_features(str(trace_dir))
    features_path = output_dir / "strategy_features.csv"
    features_df.to_csv(features_path, index=False)
    print(f"Features CSV saved to {features_path}")

    # Generate plots
    print("\nGenerating plots...")
    generate_plots(str(trace_dir), str(output_dir))

    # Write TensorBoard
    print("\nWriting TensorBoard logs...")
    write_tensorboard(summaries, str(output_dir))

    # Print summary
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    win_rate = summaries["agent_win"].mean()
    mean_nw = summaries["final_net_worths"].apply(lambda x: x[0]).mean()
    mean_len = summaries["episode_length"].mean()
    print(f"  Win Rate:           {win_rate:.1%}")
    print(f"  Mean Final NW:      ${mean_nw:.0f}")
    print(f"  Mean Episode Length: {mean_len:.0f}")
    print(f"  Episodes Traced:    {len(summaries)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
