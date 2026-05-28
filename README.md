# Monopoly RL Environment

A complete 4-player Monopoly environment for reinforcement learning research, featuring:
- **Gymnasium** single-agent environment (train against Random, MCTS, Greedy bots)
- **PettingZoo** multi-agent environment (4 independent RL agents)
- **DDQN** with action masking and target networks
- **Strategy analysis** with episode tracing and clustering
- **Interactive renderer** with video export

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Verify environment
python scripts/smoke_test.py --quick

# Train an agent (without visualization for speed)
python train_dqn.py --train --agent dqn --total_timesteps 100000 --no-render

# Monitor training
tensorboard --logdir runs/
```

---

## Training Modes

### 1. Gymnasium: Single RL Agent vs Bots (Default)

Train one RL agent against three scripted opponents:

| Player | Type | Strategy |
|--------|------|----------|
| 0 | **RL Agent** | DQN / DDQN-Hybrid |
| 1 | Random | Uniform random |
| 2 | MCTS | 5 rollouts, depth 5 |
| 3 | Greedy | Buys everything, builds on monopolies |

```bash
# Train DQN against bots
python train_dqn.py --train --agent dqn --total_timesteps 500000 --seed 42 --no-render

# Train DDQN-Hybrid against bots
python train_dqn.py --train --agent ddqn_hybrid --total_timesteps 500000 --seed 42 --no-render

# Continue DDQN-Hybrid training from a .pt checkpoint
# (total_timesteps is interpreted as ADDITIONAL steps in resume mode)
python train_dqn.py --train --agent ddqn_hybrid \
    --resume_checkpoint runs/ddqn_hybrid_dense_networth_seed42/models/ddqn_hybrid/final.pt \
    --total_timesteps 2000000 --seed 42 --no-render

# Optional: restore replay buffer explicitly
python train_dqn.py --train --agent ddqn_hybrid \
    --resume_checkpoint runs/ddqn_hybrid_dense_networth_seed42/models/ddqn_hybrid/final.pt \
    --resume_replay_buffer runs/ddqn_hybrid_dense_networth_seed42/buffers/ddqn_hybrid_replay.pkl \
    --total_timesteps 2000000 --seed 42 --no-render

# Evaluate trained model
python train_dqn.py --evaluate --model runs/dqn_dense_networth_seed42/models/dqn/monopoly_dqn_final.zip --episodes 100

# Evaluate DDQN-Hybrid .pt model
python train_dqn.py --evaluate --agent ddqn_hybrid \
    --model runs/ddqn_hybrid_dense_networth_seed42/models/ddqn_hybrid/final.pt --episodes 100
```

### 2. PettingZoo: 4 Independent RL Agents (Self-Play)

All 4 players are controlled by RL policies:

```python
from Monopoly.envs.pettingzoo_env import MonopolyAECEnv
from stable_baselines3 import DQN

# Load a trained model (can be trained via Gymnasium mode first)
model = DQN.load("runs/dqn_dense_networth_seed42/models/dqn/monopoly_dqn_final.zip")

# Create PettingZoo environment
env = MonopolyAECEnv(seed=42, max_steps=2000)
env.reset()

# Run episode with same model for all 4 players
while env.agents:
    agent = env.agent_selection
    obs = env.observe(agent)
    
    if env.terminations[agent] or env.truncations[agent]:
        env.step(None)
        continue
    
    # Use model to select action (respecting action mask)
    action, _ = model.predict(obs["observation"], deterministic=True)
    
    # Validate action is legal
    if obs["action_mask"][action] == 0:
        legal = [i for i, m in enumerate(obs["action_mask"]) if m == 1]
        action = legal[0] if legal else 89
    
    env.step(action)

env.close()
```

Or use the wrapper for Gymnasium-style interface:

```python
from Monopoly.envs.pettingzoo_env import MonopolyAECEnv
from Monopoly.envs.wrappers import PettingZooToGymWrapper

# Wrap AEC env for single-agent training (opponents use first legal action)
aec_env = MonopolyAECEnv(seed=42)
gym_env = PettingZooToGymWrapper(aec_env, agent_id="player_0")

obs, info = gym_env.reset()
for _ in range(100):
    action = gym_env.action_space.sample()  # Replace with your policy
    obs, reward, terminated, truncated, info = gym_env.step(action)
    if terminated or truncated:
        break
```

---

## Rendering Control

```bash
# Disable rendering (faster training, default behavior)
python train_dqn.py --train --agent dqn --no-render

# Enable rendering (visualize training, slower)
python train_dqn.py --train --agent dqn --render

# Evaluate with visualization
python train_dqn.py --evaluate --model path/to/model.zip --render
```

---

## Environment Details

### Action Space: Discrete(174)

| Range | Action | Description |
|-------|--------|-------------|
| 0 | Roll | Start turn |
| 1 | Buy | Purchase property |
| 2 | Pass | Decline to buy |
| 3-30 | Build | Build house on property |
| 31-58 | Mortgage | Mortgage property |
| 59-86 | Unmortgage | Unmortgage property |
| 87 | Pay Fine | Exit jail ($50) |
| 88 | Use Card | Use get-out-of-jail-free |
| 89 | End Turn | End turn |
| 90-173 | Trade | Sell property to opponent |

### Observation Space

```python
{
    'player_id': (1,),           # Agent's index
    'cash': (4,),                # All players' cash
    'positions': (4,),           # Board positions
    'property_owner': (28,),     # -1=unowned, else player_id
    'houses': (28,),             # Houses per property (5=hotel)
    'mortgaged': (28,),          # Mortgage status
    'jail_turns': (4,),          # Jail status
    'get_out_cards': (4,),       # Jail-free cards
    'legal_mask': (174,),        # Valid actions
    'turn_number': (1,),         # Current turn
    'last_roll': (2,),           # Dice values
    'net_worth': (4,),           # Computed net worth
}
```

### Reward Modes

| Mode | Formula | Use Case |
|------|---------|----------|
| `dense_networth` | r = NW_agent / Σ(NW_others) | Faster learning (default) |
| `sparse_terminal` | +1 win, -1 lose, 0 otherwise | Sparse reward experiments |

```bash
# Dense reward (default)
python train_dqn.py --train --reward_mode dense_networth

# Sparse reward
python train_dqn.py --train --reward_mode sparse_terminal
```

---

## Algorithms

| Feature | SB3 DQN | DDQN-Hybrid |
|---------|---------|-------------|
| Library | Stable-Baselines3 | Custom PyTorch |
| Double DQN | No | Yes |
| Action Masking | Env correction | Explicit |
| Target Network | ✓ | ✓ (hard + soft) |

```bash
# SB3 DQN
python train_dqn.py --train --agent dqn --seed 42

# Custom DDQN
python train_dqn.py --train --agent ddqn_hybrid --seed 42
```

---

## Strategy Analysis

Analyze agent behavior with episode tracing:

```bash
# Evaluate with tracing enabled
python train_dqn.py --evaluate --model path/to/model.zip --episodes 100 --trace_eval

# Evaluate DDQN-Hybrid .pt with tracing enabled
python train_dqn.py --evaluate --agent ddqn_hybrid --model path/to/model.pt --episodes 100 --trace_eval

# Analyze strategy clusters
python scripts/analyze_strategies.py \
    --trace-dir runs/dqn_dense_networth_seed42/analysis/traces \
    --output-dir runs/dqn_dense_networth_seed42/analysis
```

Features extracted (9 dimensions):
- Build rate, trade frequency, cash conservation
- Aggression score, endgame dominance, peak net worth
- Mortgage rate, property diversity, time to first monopoly

---

## Replay Viewer

Record and replay games:

```bash
# Interactive playback
python scripts/replay_viewer.py --rollout path/to/episode.rollout.gz

# Export to video
python scripts/replay_viewer.py --rollout path/to/episode.rollout.gz --export mp4
```

---

## Project Structure

```
monopolizers/
├── train_dqn.py                    # Main training script
├── Monopoly/
│   ├── envs/
│   │   ├── gym_env.py              # Gymnasium environment
│   │   ├── pettingzoo_env.py       # PettingZoo AEC environment
│   │   ├── renderer.py             # Board visualization
│   │   ├── tracing.py              # Episode recording
│   │   └── wrappers.py             # Observation flattening
│   ├── agents/
│   │   ├── ddqn_hybrid.py          # DDQN trainer
│   │   ├── network.py              # Q-network architectures
│   │   ├── random.py               # Random opponent
│   │   ├── greedy.py               # Greedy opponent
│   │   └── mcts.py                 # MCTS opponent
│   ├── rules.py                    # Game rules
│   └── trade.py                    # Trading system
├── scripts/
│   ├── smoke_test.py               # Quick validation
│   ├── analyze_strategies.py       # Strategy clustering
│   └── replay_viewer.py            # Playback & export
└── tests/                          # Test suite
```

---

## Testing

```bash
# Run all tests
pytest tests/

# Quick smoke test
python scripts/smoke_test.py --quick

# With coverage
pytest --cov=Monopoly tests/
```

---

## Command Reference

```bash
# Training
python train_dqn.py --train --agent [dqn|ddqn_hybrid] [options]
    --total_timesteps N    # Training steps (default: 100000)
    --seed N               # Random seed (default: 42)
    --max_turns N          # Max turns per episode (default: 500)
    --reward_mode MODE     # dense_networth or sparse_terminal
    --no-render            # Disable visualization (faster)
    --render               # Enable visualization
    --output_dir PATH      # Output directory

# Evaluation
python train_dqn.py --evaluate --model PATH [options]
    --episodes N           # Number of games (default: 100)
    --trace_eval           # Enable episode tracing
    --render               # Enable visualization

# Baseline
python train_dqn.py --baseline --episodes N
```

usage: evaluate_agent.py [-h] --model MODEL
                         [--agent-type {auto,dqn,ddqn_hybrid}]
                         [--episodes EPISODES] [--seed SEED]
                         [--max-turns MAX_TURNS]
                         [--reward-mode {dense_networth,sparse_terminal}]
                         [--output-dir OUTPUT_DIR]

Batch evaluate a Monopoly RL agent with tracing

options:
  -h, --help            show this help message and exit
  --model MODEL         Path to model checkpoint
  --agent-type {auto,dqn,ddqn_hybrid}
                        Model type (default: auto; inferred from .zip/.pt)
  --episodes EPISODES   Number of episodes (default: 100)
  --seed SEED           Random seed (default: 42)
  --max-turns MAX_TURNS
                        Max turns per episode (default: 500)
  --reward-mode {dense_networth,sparse_terminal}
                        Reward mode (default: dense_networth)
  --output-dir OUTPUT_DIR
                        Output directory (default: alongside model)

---

## License

MIT License
