# Monopoly RL Environment

A comprehensive Monopoly game implementation for Reinforcement Learning research, featuring:
- 🎮 **Gymnasium-compatible environment** for RL training
- 🤖 **Multiple AI agents**: Random, Greedy, and MCTS
- 🎨 **Pygame visualization** for real-time game watching
- 🏗️ **Complete game mechanics**: properties, houses, cards, jail, etc.
- 🧪 **Extensive test coverage** with pytest

## Project Structure

```
📦 monopolizers/
┣ 📂 Monopoly/
┃ ┣ 📜 board.py          # Game board (40 tiles)
┃ ┣ 📜 property.py       # Property specifications & rent tables
┃ ┣ 📜 cards.py          # Chance & Community Chest cards
┃ ┣ 📜 state.py          # Game state management
┃ ┣ 📜 rules.py          # Rules engine & legal actions
┃ ┣ 📜 engine.py         # Game loop & turn execution
┃ ┣ 📜 player.py         # Player utilities
┃ ┣ 📜 auction.py        # Auction system
┃ ┣ 📜 trade.py          # Property trading
┃ ┣ 📂 agents/
┃ ┃ ┣ 📜 random.py       # Random agent
┃ ┃ ┣ 📜 greedy.py       # Greedy heuristic agent
┃ ┃ ┗ 📜 mcts.py         # Monte Carlo Tree Search agent
┃ ┗ 📂 envs/
┃   ┣ 📜 gym_env.py      # Gymnasium environment wrapper
┃   ┣ 📜 renderer.py     # Pygame visualization (NEW!)
┃   └ 📜 observation.py  # Observation utilities
┣ 📂 examples/
┃ ┣ 📜 train_gym_agent.py      # Training examples with SB3
┃ ┣ 📜 train_vs_greedy.py      # Train against greedy opponents
┃ ┣ 📜 run_smoke_games.py      # Quick game tests
┃ ┗ 📜 visualize_game.py       # Pygame visualization demo (NEW!)
┣ 📂 tests/                     # Comprehensive test suite
┗ 📜 IMPLEMENTATION_ANALYSIS.md # Detailed codebase analysis (NEW!)
```

## Installation

```bash
# Clone the repository
git clone https://github.com/ML-UVA/monopolizers.git
cd monopolizers

# Install dependencies
pip install -r requirements.txt

# For visualization support
pip install pygame

# For RL training (optional)
pip install stable-baselines3
```

## Quick Start

### 1. Visualize a Game (NEW!)

Watch the game play out with real-time graphics:

```bash
# Random agents playing
python examples/visualize_game.py --mode random --players 4 --speed 2

# Random vs Greedy agents
python examples/visualize_game.py --mode greedy --players 3 --speed 1
```

**Visualization features:**
- 🎨 Classic Monopoly board layout
- 👥 Player tokens with positions
- 🏠 Houses and hotels display
- 💰 Real-time statistics panel
- 🔒 Jail indicators
- 📊 Property ownership tracking

### 2. Run a Quick Game

```bash
python examples/run_smoke_games.py
```

### 3. Train an RL Agent

```bash
# Basic training example
python examples/train_gym_agent.py

# Train against greedy opponents
python examples/train_vs_greedy.py
```

### 4. Use as a Gymnasium Environment

```python
from monopolizers.Monopoly.envs.gym_env import MonopolyEnv
from monopolizers.Monopoly.agents.random import RandomAgent

# Create environment
env = MonopolyEnv(
    num_players=4,
    agent_player_id=0,
    opponent_policies=[RandomAgent() for _ in range(3)],
    render_mode='human',  # Enable visualization!
    seed=42
)

# Run game loop
obs, info = env.reset()
done = False

while not done:
    # Get legal actions
    legal_mask = obs['legal_mask']
    legal_actions = [i for i, legal in enumerate(legal_mask) if legal]
    action = legal_actions[0]  # Choose first legal action
    
    # Take step
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    
    # Render (if render_mode='human')
    env.render()

env.close()
```

## Features

### Game Mechanics
- ✅ **Complete board**: All 40 tiles (properties, railroads, utilities, special tiles)
- ✅ **Property system**: Buying, mortgaging, unmortgaging
- ✅ **Building**: Houses and hotels with monopoly requirements
- ✅ **Rent calculation**: Accurate rent for properties, railroads, utilities
- ✅ **Cards**: 16 Chance + 16 Community Chest cards with full effects
- ✅ **Jail**: Multiple ways to get out (roll doubles, pay fine, use card)
- ✅ **Bankruptcy**: Asset liquidation and property transfer
- ✅ **Turn management**: Dice rolling, doubles, movement

### RL Environment
- 📊 **Observation Space**: Dict with:
  - Player cash, positions, properties
  - Property ownership, houses, mortgages
  - Legal action mask
  - Game state (turn number, dice rolls, bank resources)
  
- 🎯 **Action Space**: Discrete(90)
  - 0: Roll dice
  - 1-2: Buy/pass on property
  - 3-30: Build house on property
  - 31-58: Mortgage property
  - 59-86: Unmortgage property
  - 87-88: Jail actions
  - 89: End turn

- 🏆 **Rewards**: Shaped rewards for:
  - Cash changes
  - Property acquisition
  - Winning (+100) / Losing (-50)

### Agents
- 🎲 **RandomAgent**: Uniform random selection from legal actions
- 🧠 **GreedyAgent**: Heuristic-based (buy properties, build on monopolies)
- 🌳 **MCTSAgent**: Monte Carlo Tree Search with configurable simulations

### Visualization (NEW!)
- 🎮 **Pygame renderer**: Real-time graphical display
- 📈 **Statistics panel**: Player cash, positions, properties
- 🎨 **Color-coded**: Property groups, player tokens, ownership
- 🏠 **Visual indicators**: Houses, hotels, mortgages, jail status

## Training with Stable-Baselines3

```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from monopolizers.Monopoly.envs.gym_env import MonopolyEnv

# Create vectorized environment
env = DummyVecEnv([lambda: MonopolyEnv(num_players=2, render_mode=None)])

# Initialize PPO agent
model = PPO(
    "MultiInputPolicy",  # For Dict observation space
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
)

# Train
model.learn(total_timesteps=100000)

# Save model
model.save("monopoly_ppo")

# Evaluate with visualization
eval_env = MonopolyEnv(num_players=2, render_mode='human')
obs, _ = eval_env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = eval_env.step(action)
    eval_env.render()
    if done or truncated:
        break
eval_env.close()
```

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

See **[IMPLEMENTATION_ANALYSIS.md](IMPLEMENTATION_ANALYSIS.md)** for:
- 📋 Complete feature list
- ⚠️ Known limitations and edge cases
- 🔧 Recommended fixes
- 🎯 Priority improvements
- 📖 Detailed API documentation

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
