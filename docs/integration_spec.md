# Monopoly PettingZoo Environment — Integration Spec

## Environment ID
`monopoly_v0`

## API
PettingZoo AEC (Agent-Environment-Cycle) — turn-based, sequential actions.

## Agents
- IDs: `player_0`, `player_1`, `player_2`, `player_3`
- Max agents: 4 (configurable via `num_players`)
- Agents are removed when bankrupt

## Observation Space (per agent)
Type: `gymnasium.spaces.Dict` following PettingZoo convention:

```python
{
    "observation": Box(shape=(N,), dtype=float32),  # flat game state vector
    "action_mask": Box(shape=(174,), dtype=int8),   # legal action mask
}
```

### Observation Vector Layout (N = 1 + 5*num_players + 3*28 + 4 = 109 for 4 players)

| Offset           | Length           | Feature         | Range         |
|------------------|------------------|-----------------|---------------|
| 0                | 1                | player_id       | [0, 3]        |
| 1                | num_players      | cash            | [0, 100000]   |
| 1+n              | num_players      | positions       | [0, 39]       |
| 1+2n             | 28               | property_owner  | [-1, 3]       |
| 1+2n+28          | 28               | houses          | [0, 5]        |
| 1+2n+56          | 28               | mortgaged       | [0, 1]        |
| 1+2n+84          | num_players      | jail_turns      | [0, 10]       |
| 1+3n+84          | num_players      | get_out_cards   | [0, 10]       |
| 1+4n+84          | 1                | turn_number     | [0, max_steps]|
| 2+4n+84          | 2                | last_roll       | [1, 6]        |
| 4+4n+84          | 1                | bank_houses     | [0, 32]       |
| 5+4n+84          | 1                | bank_hotels     | [0, 12]       |
| 6+4n+84          | num_players      | net_worth       | [0, 100000]   |

For 4 players: flat dimension = **109**

The existing single-agent `MonopolyEnv` uses a Dict observation with the same features
as separate arrays (183 flat elements including the legal mask via `MonopolyFlattenWrapper`).

## Action Space (per agent)
Type: `gymnasium.spaces.Discrete(174)`

| Range   | Meaning                              |
|---------|--------------------------------------|
| 0       | Roll dice                            |
| 1       | Buy property at current position     |
| 2       | Pass on buying                       |
| 3-30    | Build house on property [0-27]       |
| 31-58   | Mortgage property [0-27]             |
| 59-86   | Unmortgage property [0-27]           |
| 87      | Pay jail fine ($50)                  |
| 88      | Use get-out-of-jail card             |
| 89      | End turn                             |
| 90-173  | Trade: 28 props x 3 opponents        |

## Config Keys

| Key                  | Type  | Default          | Description                          |
|----------------------|-------|------------------|--------------------------------------|
| num_players          | int   | 4                | Number of players                    |
| max_steps            | int   | 2000             | Max step() calls before truncation   |
| seed                 | int   | None (->42)      | Deterministic RNG seed               |
| reward_mode          | str   | "sparse_terminal"| "sparse_terminal" or "dense_networth"|
| stalemate_threshold  | int   | 50               | Stale rounds before early truncation |
| fast_mode            | bool  | False            | Skip net_worth in obs, minimal info  |
| render_mode          | str   | None             | "human", "ansi", or None             |

## Turn Flow (AEC)

Monopoly turns are multi-action. Within one turn, a player may:
1. Roll dice (action 0) — mandatory first action
2. Buy/pass on property (actions 1-2) — if landed on unowned property
3. Build/mortgage/unmortgage/trade (actions 3-86, 90+) — optional post-roll
4. End turn (action 89) — mandatory to finish turn

`agent_selection` stays on the same player until they execute `end_turn`,
then advances to the next active (non-bankrupt) player.

## Reward Modes

### sparse_terminal (default for self-play)
- During play: 0
- Game ends naturally: +1 winner, -1 losers
- Truncation (max_steps/stalemate): net-worth tiebreak (+1/-1)

### dense_networth
- Each step: `nw_agent / sum(nw_others)` for acting agent only
- Terminal: 0

## Gym Wrapper

`PettingZooToGymWrapper` provides a single-agent `gymnasium.Env` interface
over the AEC env, where one player learns and others use provided policies:

```python
from Monopoly.envs.pettingzoo_env import MonopolyAECEnv
from Monopoly.envs.wrappers import PettingZooToGymWrapper

aec = MonopolyAECEnv(seed=42)
env = PettingZooToGymWrapper(aec, agent_id="player_0")
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(action)
```

## Stalemate Detection

Enabled by default (`stalemate_threshold=50`). Tracks per-round snapshots
of property ownership, houses, active players, and net worths. If all remain
unchanged for `stalemate_threshold` consecutive rounds, the game truncates.
