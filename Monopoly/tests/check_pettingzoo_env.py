from Monopoly.envs.pettingzoo_env import PettingZooEnv

env = PettingZooEnv(seed=42)
env.reset()

priority = [0, 1, 89, 87, 88]

for agent in env.agent_iter():
    obs, reward, terminated, truncated, info = env.last()
    if terminated or truncated:
        env.step(None)
        continue
    legal = obs['legal_mask'].nonzero()[0]
    action = next((a for a in priority if a in legal), int(legal[0]))
    env.step(action)

    agent_id = int(agent.split('_')[1])

    print(f"Agent {agent}, Action: {action}")
    print(f"    Owns {env.state.players[agent_id].properties_owned}")
    
    if env.state.turn_number > 10:
        break

print("PettingZoo env check passed")
env.close()
