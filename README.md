# Monopoly RL Environment

A 4-player Monopoly game environment for reinforcement learning research. Train DQN agents to play Monopoly against mixed bot opponents (Random, MCTS, Greedy).

## What's This?

This project lets you:
1. **Train RL agents** to play Monopoly using DQN or custom Double-DQN
2. **Compare algorithms** with standardized metrics and TensorBoard logging  
3. **Visualize games** with a Pygame renderer
4. **Run experiments** with reproducible seeds and CSV exports

The RL agent (Player 0) learns to play against three bot opponents in a 4-player game.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run a quick test to verify everything works
python scripts/smoke_test.py --quick

# 3. Train a DQN agent (takes ~5-10 minutes for 100k steps)
python train_dqn.py --train --agent dqn --total_timesteps 100000 --seed 42

# 4. Watch training progress
tensorboard --logdir runs/
```

## Installation

```bash
git clone https://github.com/ML-UVA/monopolizers.git
cd monopolizers
pip install -r requirements.txt
```

**Requirements:** Python 3.8+, PyTorch, Stable-Baselines3, Gymnasium

## Training

### Basic Training Commands

```bash
# Train with Stable-Baselines3 DQN (recommended for beginners)
python train_dqn.py --train --agent dqn --total_timesteps 500000 --seed 42

# Train with custom Double-DQN (for research/customization)
python train_dqn.py --train --agent ddqn_hybrid --total_timesteps 500000 --seed 42

# Evaluate a trained model
python train_dqn.py --evaluate --model runs/dqn_seed42/models/dqn/monopoly_dqn_final.zip

# Run random baseline for comparison
python train_dqn.py --baseline --episodes 100
```

### What Gets Saved

```
runs/dqn_seed42/
├── models/dqn/
│   ├── monopoly_dqn_final.zip    # Final trained model
│   └── monopoly_dqn_*.zip        # Checkpoints
├── results/
│   └── dqn_metrics.csv           # Episode-level metrics
├── tensorboard/
│   └── DQN/                      # TensorBoard logs
└── logs/
    └── evaluations.npz           # Evaluation results
```

### Training Configuration

The training uses these settings (tuned for Monopoly):

| Parameter | DQN (SB3) | DDQN-Hybrid |
|-----------|-----------|-------------|
| Learning Rate | 1e-4 | 1e-4 |
| Discount (γ) | 0.99 | 0.99 |
| Batch Size | 64 | 64 |
| Buffer Size | 100k | 100k |
| Target Update | 1000 steps | 1000 steps |
| Exploration | ε: 1.0 → 0.05 | ε: 1.0 → 0.05 |

### Comparing Algorithms

Both training modes output CSV files with identical schemas, making comparison easy:

```bash
# Train both algorithms with same seed
python train_dqn.py --train --agent dqn --seed 42
python train_dqn.py --train --agent ddqn_hybrid --seed 42

# Compare in TensorBoard
tensorboard --logdir runs/

# Or load CSVs in pandas
import pandas as pd
dqn = pd.read_csv('runs/dqn_seed42/results/dqn_metrics.csv')
ddqn = pd.read_csv('runs/ddqn_hybrid_seed42/results/ddqn_hybrid_metrics.csv')
```

## How It Works

### Game Setup (4-Player Mode)

| Player | Type | Strategy |
|--------|------|----------|
| 0 | **RL Agent** | Learning via DQN/DDQN |
| 1 | Random Bot | Uniform random legal actions |
| 2 | MCTS Bot | 5 rollouts, depth 5 |
| 3 | Greedy Bot | Buy properties, build on monopolies |

### Reward Function

The agent receives a **net worth-based reward** after each action:

```
reward = agent_net_worth / sum(other_active_players_net_worth)
```

This encourages the agent to maximize its relative wealth. There's no bonus for winning—the agent learns that accumulating wealth leads to eventual victory.

### Action Space

The agent chooses from 90 discrete actions:
- Action 0: Roll dice
- Actions 1-2: Buy/pass on property
- Actions 3-30: Build houses
- Actions 31-58: Mortgage properties
- Actions 59-86: Unmortgage properties  
- Actions 87-88: Jail decisions
- Action 89: End turn

Illegal actions are masked—the agent only sees valid choices.

## Project Structure

```
monopolizers/
├── train_dqn.py              # Main training script
├── scripts/
│   └── smoke_test.py         # Verification tests
├── Monopoly/
│   ├── envs/
│   │   ├── gym_env.py        # Gymnasium environment
│   │   └── wrappers.py       # Observation flattening
│   └── agents/
│       ├── network.py        # PyTorch Q-networks
│       ├── ddqn_hybrid.py    # Custom DDQN trainer
│       ├── random.py         # Random bot
│       ├── greedy.py         # Greedy bot
│       └── mcts.py           # MCTS bot
├── policies/
│   └── trade_policy.py       # Deterministic trade heuristics
└── utils/
    └── save_utils.py         # CSV/model saving utilities
```

## Testing

```bash
# Run all tests
pytest tests/

# Run smoke tests (quick verification)
python scripts/smoke_test.py --quick

# Run full smoke tests (includes short training)
python scripts/smoke_test.py
```
python train_dqn.py --baseline --episodes 100
```

**Training Output Structure:**
```
runs/<agent>_seed<N>/
├── models/              # Saved models and checkpoints
│   └── dqn/ or ddqn_hybrid/
├── results/             # CSV metrics (comparable across agent types)
│   └── dqn_metrics.csv or ddqn_metrics.csv
├── tensorboard/         # TensorBoard logs
└── logs/                # Evaluation logs
```

**4-Player Training Mode:**
- **Player 0**: RL Agent (DQN or DDQN-Hybrid) - generates training data
- **Player 1**: Random Bot
- **Player 2**: MCTS Bot (lightweight, 5 rollouts)
- **Player 3**: Greedy Bot

**Reward System:**
- Net worth-based reward: `r = agent_net_worth / sum(other_active_players_net_worth)`
- No bonus for winning or penalty for losing
- Encourages maximizing relative wealth throughout the game

**CSV Metrics Schema** (same for both agent types):
## Visualization

Watch games with the Pygame renderer:

```bash
python examples/visualize_game.py --mode random --players 4
```

## Contributing

1. Run tests before submitting: `pytest tests/`
2. Run smoke tests: `python scripts/smoke_test.py`
3. Format code with black: `black .`

## License

MIT License

## References

- [DQN Paper (Mnih et al., 2015)](https://www.nature.com/articles/nature14236)
- [Double DQN (van Hasselt et al., 2016)](https://arxiv.org/abs/1509.06461)
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)

## Testing

Run the test suite:

```bash
# All tests
pytest

# Specific test file
pytest tests/test_engine.py

# With coverage
pytest --cov=Monopoly tests/
```

## Documentation

## Quick Commands

```bash
# Run tests
pytest tests/

# Smoke tests (CI/CD)
python scripts/smoke_test.py --quick

# Random baseline evaluation
python train_dqn.py --baseline --episodes 20

# Train SB3 DQN
python train_dqn.py --train --agent dqn --total_timesteps 100000

# Train DDQN-Hybrid (custom PyTorch)
python train_dqn.py --train --agent ddqn_hybrid --total_timesteps 100000

# Monitor training
tensorboard --logdir runs/
```

## Agent Types

### SB3 DQN (`--agent dqn`)
- Uses Stable-Baselines3 DQN implementation
- Well-tested, production-ready
- Supports MlpPolicy with configurable presets

### DDQN-Hybrid (`--agent ddqn_hybrid`)
- Custom PyTorch Double-DQN implementation
- Same environment, reward, and action masking as DQN
- Uses Double-DQN formula to reduce overestimation bias:
  - `a_max = argmax Q_online(s')`
  - `target = r + γ * Q_target(s', a_max)`
- Supports replay buffer persistence and resumption

See **[IMPLEMENTATION_ANALYSIS.md](IMPLEMENTATION_ANALYSIS.md)** for:

## Contributing

Contributions welcome! Priority areas:
1. Edge case handling (see IMPLEMENTATION_ANALYSIS.md)
2. Additional RL agents
3. Performance optimizations
4. Enhanced visualizations

## License

MIT License - See LICENSE file for details

## Acknowledgments

Built for ML-UVA Machine Learning Club research project on game-playing AI agents.
