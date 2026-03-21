import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
from Monopoly.envs.gym_env import MonopolyEnv
from Monopoly.envs.wrappers import MonopolyFlattenWrapper
from Monopoly.agents.greedy import GreedyAgent
from Monopoly.agents.random import RandomAgent
from Monopoly.agents.dqn import DQNAgent


def evaluate(
    checkpoint_path: str = 'examples/checkpoints/dqn_stage1.pt',
    num_episodes: int = 200,
    opponent_type: str = 'random'
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    match opponent_type:
        case 'random':
            opp_agent = RandomAgent()
        case 'greedy':
            opp_agent = GreedyAgent()

    env = MonopolyFlattenWrapper(MonopolyEnv(
        num_players=4,
        opponent_policies=[opp_agent, opp_agent, opp_agent],
        training_stage=1,
        seed=42,
    ))

    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent = DQNAgent(obs_dim=obs_dim, n_actions=n_actions, device=device)
    agent.load(checkpoint_path)

    # Force fully greedy policy
    agent.epsilon_end = 0.0
    agent.steps_done = 999_999_999

    wins = 0
    bankruptcies = 0
    total_rewards = []
    total_properties = []

    for ep in range(num_episodes):
        obs, info = env.reset(seed=20000 + ep)
        done = False
        while not done:
            legal_mask = env.unwrapped._get_legal_mask()
            action = agent.select_action(obs, legal_mask)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        
        state = env.unwrapped.state
        print(f"\nEpisode {ep+1} end state:")
        for i, p in enumerate(state.players):
            marker = "DQN" if i == 0 else f"{opponent_type}{i}"
            print(f"  {marker}: ${p.cash} | "
                f"Props: {len(p.properties_owned)} | "
                f"Status: {p.status.value}")
    

    # print(f"\n{'='*50}")
    # print(f"Evaluation over {num_episodes} episodes")
    # print(f"{'='*50}")
    # print(f"Win rate:        {wins}/{num_episodes} ({wins/num_episodes*100:.1f}%)")
    # print(f"Bankruptcy rate: {bankruptcies}/{num_episodes} ({bankruptcies/num_episodes*100:.1f}%)")
    # print(f"Avg reward:      {np.mean(total_rewards):.2f}")
    # print(f"Avg properties:  {np.mean(total_properties):.1f}")
    # print(f"{'='*50}\n")

    env.close()


if __name__ == '__main__':
    evaluate(
        checkpoint_path='examples/checkpoints/dqn_phase3.pt',
        num_episodes=5,
        opponent_type='random'
    )
