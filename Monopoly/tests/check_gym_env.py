import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Monopoly.envs.gym_env import MonopolyEnv
from Monopoly.agents.random import RandomAgent

env = MonopolyEnv(
    num_players=4,
    agent_player_id=0,
    opponent_policies=[RandomAgent(), RandomAgent(), RandomAgent()],
    training_stage=1,
    seed=42
)

obs, info = env.reset(seed=42)
print("Reset successful")
print(f"Obs keys: {list(obs.keys())}")
print(f"Legal actions: {obs['legal_mask'].sum()} available")

bought = 0
passed = 0
rolled = 0
end_turn = 0

done = False
steps = 0
while not done and steps < 50:
    legal = obs['legal_mask'].nonzero()[0]
    priority = [0, 1, 89, 87, 88]
    action = next((a for a in priority if a in legal), int(legal[0]))

    if action == 0:
        rolled += 1
    elif action == 1:
        bought += 1
    elif action == 2:
        passed += 1
    elif action == 89:
        end_turn += 1
    
    print(f"Step {steps}: action={action}, turn_number={info['turn_number']}, current_player={info['current_player']}, turn_phase={info['turn_phase']}")
    print(f"Agent owns: {env.state.players[0].properties_owned}")
    print(f"Mask bits set: {obs['legal_mask'].nonzero()[0]}")

    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    steps += 1

print(f"Rolled: {rolled}, Bought: {bought}, Passed: {passed}, Ended Turn: {end_turn}")
print(f"Total game turns elapsed: {info['turn_number']}")
print(f"Agent steps taken: {steps}")
print(f"Completed {steps} steps without crashing")
print(f"Final cash: ${info['agent_cash']}")
print(f"Final properties: {info['agent_properties']}")
print(f"Active players: {info['active_players']}")
env.close()
