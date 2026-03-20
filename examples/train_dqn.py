import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import numpy as np
import torch
import logging

from Monopoly.envs.gym_env import MonopolyEnv
from Monopoly.envs.wrappers import MonopolyFlattenWrapper
from Monopoly.agents.random import RandomAgent
from Monopoly.agents.greedy import GreedyAgent
from Monopoly.agents.dqn import DQNAgent


def train(
        total_episodes: int = 5000,
        training_stage: int = 1,
        save_path: str = 'checkpoints/dqn_stage1.pt',
        load_path: str = None,
        log_every: int = 100,
):

    os.makedirs('logs', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)

    logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log'),
        logging.StreamHandler()  # also prints to terminal
    ]
)


    env = MonopolyFlattenWrapper(MonopolyEnv(
        num_players=4,
        opponent_policies=[GreedyAgent(), RandomAgent(), RandomAgent()],
        training_stage=training_stage,
        seed=42,
        max_turns=2000,
    ))

    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"obs_dim: {obs_dim}, n_actions: {n_actions}")

    # Create agent
    agent = DQNAgent(
        obs_dim=obs_dim,
        n_actions=n_actions,
        device=device,
        lr=5e-5, # lower lr for fine-tuning
        epsilon_decay=2_000_000,
        buffer_capacity=200_000,
        batch_size=128,
        target_update_freq=2000,
    ) # everything else already set as default

    if load_path and os.path.exists(load_path):
        agent.load(load_path)
        agent.steps_done = 1_200_000
        print(f"Loaded checkpoint from {load_path}, epsilon reset to {agent.epsilon:.3f}")
    
    episode_rewards = []
    episode_lengths = []
    losses = []

    for episode in range(total_episodes):
        obs, info = env.reset(seed=episode)
        episode_reward = 0.0
        episode_length = 0
        done = False

        while not done:
            # Extract legal mask from flat obs
            # Need to unwrap env to access it
            legal_mask = env.unwrapped._get_legal_mask()

            action = agent.select_action(obs, legal_mask)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            next_legal_mask = env.unwrapped._get_legal_mask()

            agent.store_transition(
                obs=obs,
                action=action,
                reward=reward,
                next_obs=next_obs,
                done=float(done),
                next_legal_mask=next_legal_mask
            )

            loss = agent.update()
            if loss is not None:
                losses.append(loss)
            
            obs = next_obs

            episode_reward += reward
            episode_length += 1

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        # Logging
        if (episode + 1) % log_every == 0:
            avg_reward = np.mean(episode_rewards[-log_every:])
            avg_length = np.mean(episode_lengths[-log_every:])
            avg_loss = np.mean(losses[-100:]) if losses else 0.0
            logging.info(
                f"Episode {episode + 1}/{total_episodes} | "
                f"Avg Reward: {avg_reward:.2f} | "
                f"Avg Length: {avg_length:.0f} | "
                f"Avg Loss: {avg_loss:.4f} | "
                f"Epsilon: {agent.epsilon:.3f} | "
                f"Buffer: {len(agent.buffer)}"
            )

        # Save checkpoint periodically
        if (episode + 1) % 500 == 0:
            os.makedirs('checkpoints', exist_ok=True)
            agent.save(save_path)
            print(f"Saved checkpoint to {save_path}")
    
    env.close()
    return agent

if __name__ == '__main__':
    train(
        total_episodes=5000,
        training_stage=1,
        load_path='checkpoints/dqn_phase2.pt',
        save_path='checkpoints/dqn_phase3.pt',
        log_every=100,
    )
