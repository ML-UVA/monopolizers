import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import numpy as np
import torch
import logging
from typing import List

from Monopoly.state import GameState
from Monopoly.property import PropertySpec
from Monopoly.rules import RulesEngine, PlayerStatus

from Monopoly.envs.gym_env import MonopolyEnv
from Monopoly.envs.wrappers import MonopolyFlattenWrapper
from Monopoly.agents.random import RandomAgent
from Monopoly.agents.greedy import GreedyAgent
from Monopoly.agents.dqn import DQNAgent

def compute_net_worth(
        state: GameState,
        player_id: int,
        property_specs: List[PropertySpec],
        rules_engine: RulesEngine
):
    """Compute net worth of player `player_id`"""
    player = state.players[player_id]

    net_worth = player.cash

    # Add asset values
    for prop_idx in player.properties_owned:
        prop = state.properties[prop_idx]
        spec = property_specs[prop_idx]

        # Don't include mortgage value if not mortgaged yet
        mv = spec.mortgage_value if prop.mortgaged else 0

        # Determine bonus constant based on monopoly status
        has_monopoly = rules_engine._has_monopoly(state, prop_idx, player_id)
        b = 2.0 if has_monopoly else 1.5

        # Count houses and hotels
        if prop.houses_count == 5:
            # Hotel
            n_houses = 0
            n_hotels = 1
            # One hotel is equivalent to 5 houses, since you give up the
            # 4 houses before to get 1 hotel
            house_investment = 5 * spec.house_cost
        else:
            n_houses = prop.houses_count
            n_hotels = 0
            house_investment = n_houses * spec.house_cost
        
        # Implement asset value formula from paper
        asset_value = (spec.price - mv) * b + house_investment
        net_worth += asset_value

    return max(net_worth, 0)

def compute_reward(
        state: GameState, 
        player_id: int, # agent's player_id 
        property_specs: List[PropertySpec], 
        rules_engine: RulesEngine, 
        terminated: bool, 
        won: bool, 
        c: int = 0.5
):
    """Compute reward to `player_id`"""
    if terminated:
        return c if won else -c
    
    agent_nw = compute_net_worth(state, player_id, property_specs, rules_engine)

    others_nw = sum(
        compute_net_worth(state, i, property_specs, rules_engine)
        for i in range(len(state.players))
        if i != player_id and state.players[i].status == PlayerStatus.ACTIVE
    )

    if others_nw <= 0:
        return c    # agent is the only one left, treat as a win
    
    rx = agent_nw / others_nw
    
    # Center around 0 so reward is positive when ahead, negative when behind
    # At equal wealth: rx = 1/3 in a 4-player game (agent vs 3 others)
    # Subtract fair share baseline to center the reward signal
    fair_share = 1.0 / (len(state.players) - 1)
    return rx - fair_share



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
        opponent_policies=[
            GreedyAgent(player_id=1), 
            RandomAgent(player_id=2), 
            RandomAgent(player_id=3)
        ],
        training_stage=training_stage,
        seed=42,
        max_turns=500,
    ))

    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    property_specs = env.unwrapped.property_specs
    rules_engine = env.unwrapped.rules_engine

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Training stage: {training_stage}")

    # Create agent
    agent = DQNAgent(
        obs_dim=obs_dim,
        n_actions=n_actions,
        device=device,
        gamma=0.99,
        lr=3e-5, # lower lr for fine-tuning
        epsilon_decay=2_000_000,
        buffer_capacity=200_000,
        batch_size=128,
        target_update_freq=2000,
    ) # everything else already set as default

    if load_path and os.path.exists(load_path):
        agent.load(load_path)
        agent.steps_done = 800_000
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

            next_obs, env_reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            state = env.unwrapped.state
            won = (
                state.players[env.unwrapped.agent_player_id].status == PlayerStatus.ACTIVE
                and sum(1 for p in state.players if p.status == PlayerStatus.ACTIVE) == 1
            )

            reward = compute_reward(
                state=state,
                player_id=env.unwrapped.agent_player_id,
                property_specs=property_specs,
                rules_engine=rules_engine,
                terminated=done,
                won=won,
                c=0.5   # can tune this
            )

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
            avg_reward = np.mean(episode_rewards[-log_every:]) / np.mean(episode_lengths[-log_every:])
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
        total_episodes=15000,
        training_stage=1,
        # load_path='checkpoints/dqn_phase4b.pt',
        save_path='checkpoints/dqn_v2_phase1.pt',
        log_every=100,
    )
