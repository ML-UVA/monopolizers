# Monopoly Gymnasium Environment

A complete Gymnasium-compatible reinforcement learning environment for Monopoly board game training and evaluation.

## Features

✅ **Full Gymnasium API Compliance**
- Standard `reset()` and `step()` methods
- Dict observation space with structured game state
- Discrete action space with legal action masking
- Proper reward shaping for learning

✅ **Comprehensive Game State Encoding**
- Player cash, positions, and property ownership
- Houses/hotels and mortgage status
- Jail turns and get-out-of-jail cards
- Bank resources and turn tracking

✅ **Legal Action Masking**
- Dynamic legal action computation
- Prevents invalid actions
- Supports all game mechanics (buy, build, mortgage, jail actions)

✅ **Multi-Agent Support**
- Single learning agent vs. opponent policies
- Support for 2-4 players
- Pluggable opponent agents (Random, Greedy, custom)

✅ **Reward Shaping**
- Cash change rewards (normalized)
- Property acquisition bonuses
- Win/bankruptcy terminal rewards
- Configurable reward functions

## Installation

```bash
# Install dependencies
pip install gymnasium numpy

# Optional: for training with RL algorithms
pip install stable-baselines3
```

## Quick Start

### Basic Usage

```python
from monopolizers.Monopoly.envs.gym_env import MonopolyEnv
from monopolizers.Monopoly.agents.random import RandomAgent

# Create environment
env = MonopolyEnv(
    num_players=2,
    agent_player_id=0,
    opponent_policies=[RandomAgent()],
    max_turns=500,
    seed=42
)

# Reset environment
obs, info = env.reset(seed=42)

# Run episode
done = False
while not done:
    # Get legal actions
    legal_mask = obs['legal_mask']
    legal_actions = legal_mask.nonzero()[0]
    
    # Sample action (random agent)
    action = np.random.choice(legal_actions)
    
    # Step environment
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    
    # Optional: render
    env.render()

env.close()
```

### Training with Stable-Baselines3

```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from monopolizers.Monopoly.envs.gym_env import MonopolyEnv

# Create vectorized environment
env = DummyVecEnv([lambda: MonopolyEnv(num_players=2, seed=42)])

# Initialize PPO agent
model = PPO(
    "MultiInputPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
)

# Train
model.learn(total_timesteps=100000)

# Save model
model.save("monopoly_agent")
```

## Action Space

Discrete action space with 90 actions:

| Action Index | Action Type | Description |
|-------------|-------------|-------------|
| 0 | Roll | Roll dice and move |
| 1 | Buy | Buy property at current position |
| 2 | Pass | Pass on buying/end turn |
| 3-30 | Build | Build house on property (idx - 3) |
| 31-58 | Mortgage | Mortgage property (idx - 31) |
| 59-86 | Unmortgage | Unmortgage property (idx - 59) |
| 87 | Pay Fine | Pay $50 to get out of jail |
| 88 | Use Card | Use get-out-of-jail-free card |
| 89 | End Turn | Explicitly end turn |

## Observation Space

Dict observation space with the following keys:

```python
{
    'player_id': Box(0, n_players-1, shape=(1,)),
    'cash': Box(0, 100000, shape=(n_players,)),
    'positions': Box(0, 39, shape=(n_players,)),
    'property_owner': Box(-1, n_players-1, shape=(28,)),
    'houses': Box(0, 5, shape=(28,)),
    'mortgaged': Box(0, 1, shape=(28,)),
    'jail_turns': Box(0, 10, shape=(n_players,)),
    'get_out_cards': Box(0, 10, shape=(n_players,)),
    'legal_mask': Box(0, 1, shape=(n_actions,)),
    'turn_number': Box(0, max_turns, shape=(1,)),
    'last_roll': Box(1, 6, shape=(2,)),
    'bank_houses': Box(0, 32, shape=(1,)),
    'bank_hotels': Box(0, 12, shape=(1,)),
}
```

## Examples

The repository includes several example scripts:

### 1. Smoke Tests
```bash
python monopolizers/examples/run_smoke_games.py
```
Runs basic games to verify the engine works correctly.

### 2. Training vs. Greedy Agent
```bash
python monopolizers/examples/train_vs_greedy.py
```
Demonstrates training a learning agent against greedy opponents.

### 3. Gymnasium Training Example
```bash
python monopolizers/examples/train_gym_agent.py
```
Comprehensive example showing:
- Random agent baseline
- Environment speed benchmarking
- Stable-Baselines3 integration (optional)
- Multi-player configurations

## Architecture

### Key Components

1. **MonopolyEnv** (`gym_env.py`)
   - Main Gymnasium environment class
   - Handles reset, step, render, and observation generation
   - Manages opponent turn simulation

2. **Observation Utilities** (`observation.py`)
   - `encode_observation_compact()`: Flat vector encoding
   - `encode_observation_dict()`: Structured Dict encoding
   - `encode_observation_verbose()`: Human-readable encoding

3. **Action Utilities** (built into `MonopolyEnv`)
   - `_decode_action()`: Convert discrete actions to game actions
   - `_get_legal_mask()`: Compute legal action mask

4. **Game Engine** (`engine.py`, `rules.py`)
   - Core game logic and rules enforcement
   - State transitions and action application
   - Board, properties, cards, and auction mechanics

### Integration with Existing Codebase

The Gymnasium environment integrates seamlessly with:
- ✅ `GameEngine` for turn simulation
- ✅ `RulesEngine` for action validation and application
- ✅ `GameState` for state representation
- ✅ `Board`, `PropertySpec`, `Card` for game data
- ✅ `RandomAgent`, `GreedyAgent` for opponent policies

## Testing

Run the test suite:

```bash
# All tests
pytest

# Only Gymnasium environment tests
pytest monopolizers/tests/test_gym_env.py -v

# Specific test
pytest monopolizers/tests/test_gym_env.py::test_env_reset -v
```

Current test coverage:
- ✅ Environment creation and initialization
- ✅ Reset functionality
- ✅ Step mechanics
- ✅ Legal action masking
- ✅ Observation/action space compliance
- ✅ Multi-player support
- ✅ Game termination
- ✅ Render methods
- ✅ Multiple episodes

## Performance

Typical performance on modern hardware:
- **Step speed**: ~1000-2000 steps/second
- **Episode length**: 50-300 turns (depending on players)
- **Training speed**: ~10-20 episodes/minute (2-player)

## Customization

### Custom Opponent Policies

```python
class MyAgent:
    def act(self, observation, legal_mask):
        # Your policy here
        legal_actions = np.where(legal_mask)[0]
        return np.random.choice(legal_actions)
    
    def reset(self):
        pass

# Use in environment
env = MonopolyEnv(
    num_players=3,
    opponent_policies=[MyAgent(), GreedyAgent()]
)
```

### Custom Reward Function

Subclass `MonopolyEnv` and override the reward computation in `step()`:

```python
class CustomMonopolyEnv(MonopolyEnv):
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        
        # Custom reward logic
        reward = self._custom_reward(obs, info)
        
        return obs, reward, terminated, truncated, info
    
    def _custom_reward(self, obs, info):
        # Your reward shaping here
        return info['agent_cash'] / 1000.0
```

## Limitations and Future Work

Current limitations:
- ⚠️ Auctions not fully implemented
- ⚠️ Trading between players simplified
- ⚠️ House building rules (even building) partially enforced
- ⚠️ Some advanced Monopoly variants not supported

Planned improvements:
- 🔄 Full auction system
- 🔄 Advanced trading mechanisms
- 🔄 PettingZoo multi-agent environment
- 🔄 Curriculum learning support
- 🔄 Self-play training utilities

## Citation

If you use this environment in your research, please cite:

```bibtex
@software{monopoly_gym_env,
  title = {Monopoly Gymnasium Environment},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/monopolizers}
}
```

## License

[Specify your license here]

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## Support

For questions and issues:
- GitHub Issues: [Link to issues page]
- Documentation: [Link to docs]
- Discussions: [Link to discussions]
