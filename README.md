# Monopoly RL Environment

A research-grade Monopoly environment for studying reward shaping in long-horizon stochastic games.

---

## Research Focus: Hypothesis H1 — Reward Shaping Ablation

**H1**: Dense relative-net-worth reward improves learning speed and final performance compared to sparse terminal win/loss reward in stochastic Monopoly RL.

### Reward Modes (Controlled Ablation)

| Mode | Formula | Signal Type |
|------|---------|-------------|
| `dense_networth` | r = NW_agent / Σ(NW_others) | Dense, shaped |
| `sparse_terminal` | +1 win, -1 lose, 0 otherwise | Sparse |

Where **net worth** is computed as:

```
NW = cash + Σ(property_values)
property_value = 0 if mortgaged, else (price - mortgage) × β + houses × house_cost
β = 2.0 if monopoly, else 1.5
```

### Run the H1 Experiment

```bash
# Dense reward (default)
python train_dqn.py --train --agent dqn --reward_mode dense_networth --seed 42

# Sparse reward (ablation)
python train_dqn.py --train --agent dqn --reward_mode sparse_terminal --seed 42
```

**Key property**: Switching reward modes requires NO changes to agent or training code—only the `--reward_mode` flag.

---

## Environment Details

### Action Space: Discrete(174)

| Range | Action Type | Description |
|-------|-------------|-------------|
| 0 | Roll | Start turn by rolling dice |
| 1 | Buy | Purchase property (if awaiting decision) |
| 2 | Pass | Decline to buy |
| 3-30 | Build | Build house on property (idx = action - 3) |
| 31-58 | Mortgage | Mortgage property (idx = action - 31) |
| 59-86 | Unmortgage | Unmortgage property (idx = action - 59) |
| 87 | Pay Fine | Pay $50 to exit jail |
| 88 | Use Card | Use get-out-of-jail-free card |
| 89 | End Turn | End current turn |
| 90-173 | **Trade** | Sell property to opponent for fixed price |

**Trading** (actions 90-173): Agent can sell any unimproved, unmortgaged property to an opponent for `1.5 × mortgage_value`. Trades are deterministic (auto-accepted if buyer can afford).

Trade action encoding: `action = 90 + property_idx × 3 + opponent_offset`

### Four-Player Setup

| Player | Type | Strategy |
|--------|------|----------|
| 0 | **RL Agent** | DQN / DDQN-Hybrid |
| 1 | Random Bot | Uniform random (seeded) |
| 2 | MCTS Bot | 5 rollouts, depth 5 (seeded) |
| 3 | Greedy Bot | Buy properties, build on monopolies |

### Observation Space

```python
{
    'player_id': (1,),           # Agent's player index
    'cash': (4,),                # Cash for all players
    'positions': (4,),           # Board positions
    'property_owner': (28,),     # Property ownership (-1 = unowned)
    'houses': (28,),             # House count per property
    'mortgaged': (28,),          # Mortgage status
    'jail_turns': (4,),          # Jail status
    'get_out_cards': (4,),       # Jail-free cards held
    'legal_mask': (174,),        # Binary mask of legal actions
    'turn_number': (1,),         # Current turn
    'last_roll': (2,),           # Last dice roll
    'net_worth': (4,),           # Computed net worth (for observation)
}
```

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Verify environment works
python scripts/smoke_test.py --quick

# Train with dense reward (H1 baseline)
python train_dqn.py --train --agent dqn --total_timesteps 500000 --seed 42

# Monitor training
tensorboard --logdir runs/
```

---

## Algorithms

Both SB3 DQN and custom DDQN-Hybrid use the **same environment and reward**:

| Parameter | SB3 DQN | DDQN-Hybrid |
|-----------|---------|-------------|
| Learning Rate | 1e-4 | 1e-4 |
| Discount (γ) | 0.99 | 0.99 |
| Batch Size | 64 | 64 |
| Buffer Size | 100k | 100k |
| Target Update | 1000 steps | 1000 steps |
| Action Masking | Env correction | Explicit masking |

```bash
# Compare algorithms
python train_dqn.py --train --agent dqn --seed 42
python train_dqn.py --train --agent ddqn_hybrid --seed 42
```

---

## Project Structure

```
monopolizers/
├── train_dqn.py                    # Main training harness
├── scripts/
│   ├── smoke_test.py               # Quick verification
│   └── run_h1_ablation.py          # Full H1 experiment runner
├── Monopoly/
│   ├── envs/
│   │   ├── gym_env.py              # Gymnasium environment (reward lives here)
│   │   └── wrappers.py             # Observation flattening
│   ├── rules.py                    # Game rules (no reward logic)
│   ├── trade.py                    # Trading system (SimpleTrade)
│   └── agents/
│       ├── ddqn_hybrid.py          # Custom DDQN trainer
│       ├── random.py               # Random opponent
│       ├── greedy.py               # Greedy opponent
│       └── mcts.py                 # MCTS opponent
└── tests/
    ├── test_gym_env.py             # Environment + H1 invariant tests
    ├── test_trade.py               # Trading system tests
    └── test_rent.py                # Rent calculation tests
```

---

## Research Guarantees

- **Single source of truth**: All reward logic is in `MonopolyEnv._compute_reward()`
- **No confounders**: Dense vs sparse experiments differ ONLY by `reward_mode`
- **Reproducible**: All randomness is seed-controlled
- **Trading integrated**: Trading affects net worth and reward signal
- **Algorithm-agnostic**: Same env works with SB3 DQN, DDQN-Hybrid, or any Gymnasium-compatible algorithm

---

## Training Commands

### Single Run
```bash
python train_dqn.py --train --agent dqn --reward_mode dense_networth --seed 42 --total_timesteps 500000
```

### Evaluation
```bash
python train_dqn.py --evaluate --model runs/dqn_seed42/models/dqn/monopoly_dqn_final.zip --episodes 100
```

### Random Baseline
```bash
python train_dqn.py --baseline --episodes 100
```

### Output Structure
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

---

## Testing

```bash
# Run all tests
pytest tests/

# Run smoke tests (quick verification)
python scripts/smoke_test.py --quick

# Run with coverage
pytest --cov=Monopoly tests/
```

---

## Command-Line Cookbook (Presentation-Ready)

### 1) Quick sanity checks
```bash
# Fast environment validation
python scripts/smoke_test.py --quick

# Full smoke test (includes short DQN training)
python scripts/smoke_test.py
```

What this does:
- `--quick` runs only environment checks (reset/step/reward modes).
- Full smoke test also runs a very short training session to validate the pipeline.
- Expected duration: ~30–90 seconds depending on machine.

### 2) Train DQN (dense vs sparse)
```bash
# Dense reward (default)
python train_dqn.py --train --agent dqn --reward_mode dense_networth --seed 42 --total_timesteps 500000

# Sparse reward (ablation)
python train_dqn.py --train --agent dqn --reward_mode sparse_terminal --seed 42 --total_timesteps 500000
```

Key parameters:
- `--train`: run training mode (required for learning).
- `--agent dqn`: use SB3 DQN baseline.
- `--reward_mode`: `dense_networth` or `sparse_terminal`.
- `--seed`: RNG seed for reproducibility.
- `--total_timesteps`: total environment steps (each step is one agent decision).

Game speed and run length:
- One “step” is a single agent action; an episode may take dozens to hundreds of steps depending on game length.
- A 500k-step run typically takes minutes to hours depending on CPU speed.

Common variants:
```bash
# Short debug run (fast)
python train_dqn.py --train --agent dqn --reward_mode dense_networth --seed 42 --total_timesteps 5000

# Longer training for curves
python train_dqn.py --train --agent dqn --reward_mode dense_networth --seed 42 --total_timesteps 1000000
```

### 3) Train DDQN-Hybrid (custom implementation)
```bash
python train_dqn.py --train --agent ddqn_hybrid --reward_mode dense_networth --seed 42 --total_timesteps 500000
```

Notes:
- This uses the custom Double DQN trainer in Monopoly/agents/ddqn_hybrid.py.
- Same environment and reward modes as SB3 DQN.

Common variants:
```bash
# Sparse reward with DDQN
python train_dqn.py --train --agent ddqn_hybrid --reward_mode sparse_terminal --seed 42 --total_timesteps 500000
```

### 4) Evaluate a trained model
```bash
python train_dqn.py --evaluate --model runs/dqn_seed42/models/dqn/monopoly_dqn_final.zip --episodes 100
```

Parameters:
- `--evaluate`: evaluation mode.
- `--model`: path to a saved model.
- `--episodes`: number of evaluation episodes (more = smoother metrics, slower run).

Speed:
- Evaluation runs are typically faster than training because no backprop is performed.

### 5) Random baseline (for reference)
```bash
python train_dqn.py --baseline --episodes 100
```

Parameters:
- `--baseline`: runs a random agent for comparison.
- `--episodes`: number of games to run.

### 6) Multi-seed H1 ablation (paper-style)
```bash
# 5 seeds × 2 reward modes
python scripts/run_h1_ablation.py --agent dqn --timesteps 500000
```

What it does:
- Runs multiple seeds for both dense and sparse rewards.
- Produces per-run folders and an `ablation_summary.csv`.

Parameters:
- `--agent`: `dqn` or `ddqn_hybrid`.
- `--timesteps`: steps per run.
- `--seeds`: optional custom seed list (e.g., `--seeds 1 2 3`).

Speed:
- Total runtime is roughly (seeds × 2 reward modes × timesteps) / throughput.

### 7) Presentation demo (3–5 minutes)
```bash
# 1) Quick environment validation (30–60s)
python scripts/smoke_test.py --quick

# 2) Short training run (fast, visible learning signal)
python train_dqn.py --train --agent dqn --reward_mode dense_networth --seed 42 --total_timesteps 5000

# 3) Optional: evaluate the short model
python train_dqn.py --evaluate --model runs/dqn_seed42/models/dqn/monopoly_dqn_final.zip --episodes 10
```

Presentation timing notes:
- Step 2 produces a model quickly but is not “good”; it is for live demonstration only.
- Step 3 shows the evaluation loop and metrics without long training time.

------------------------------------------------------------
### 8) Parameter reference (common flags)
------------------------------------------------------------

Training / evaluation mode:
- `--train`: train a model.
- `--evaluate`: evaluate a saved model.
- `--baseline`: run random agent baseline.

Algorithm selection:
- `--agent dqn`: SB3 DQN baseline.
- `--agent ddqn_hybrid`: custom Double DQN.

Reward control:
- `--reward_mode dense_networth`: shaped reward (H1 baseline).
- `--reward_mode sparse_terminal`: terminal-only reward (H1 ablation).

Game and runtime control:
- `--total_timesteps`: number of environment steps for training.
- `--episodes`: number of episodes for evaluation/baseline.
- `--max_turns`: maximum turns per game before truncation (controls episode length).
- `--seed`: reproducibility seed.

Output paths:
- `--output_dir`: optional override for where training outputs are saved.

------------------------------------------------------------
### 9) Common parameter combinations
------------------------------------------------------------

Dense vs sparse comparison (single seed):
```bash
python train_dqn.py --train --agent dqn --reward_mode dense_networth --seed 42 --total_timesteps 200000
python train_dqn.py --train --agent dqn --reward_mode sparse_terminal --seed 42 --total_timesteps 200000
```

Short runs for debugging:
```bash
python train_dqn.py --train --agent dqn --reward_mode dense_networth --seed 1 --total_timesteps 2000 --max_turns 100
```

Long runs for paper plots:
```bash
python train_dqn.py --train --agent dqn --reward_mode dense_networth --seed 42 --total_timesteps 1000000 --max_turns 1000
```

DDQN vs DQN (same seed, same reward):
```bash
python train_dqn.py --train --agent dqn --reward_mode dense_networth --seed 7 --total_timesteps 500000
python train_dqn.py --train --agent ddqn_hybrid --reward_mode dense_networth --seed 7 --total_timesteps 500000
```

------------------------------------------------------------
### 10) How to reason about speed and run count
------------------------------------------------------------

- Total steps = `total_timesteps`.
- Episodes per run vary based on game length; shorter `max_turns` increases episode count.
- If you set `max_turns` low, training runs faster but gameplay is less complete.
- For consistent comparison, keep `total_timesteps`, `max_turns`, and opponent policies fixed across runs.

---

## License

MIT License
