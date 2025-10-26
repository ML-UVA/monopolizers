import random
from typing import Dict, Optional

import gym
from gym import spaces
import numpy as np


class SimpleMonopolyEnv(gym.Env):
    """A minimal Monopoly-like environment for RL prototyping.

    Simplifications and assumptions (initial scope):
    - Fixed board size 40 (standard Monopoly) with a subset of tiles as properties.
    - No Chance/Community Chest, no houses/hotels, no mortgages, no trading.
    - Dice roll is internal (two 6-sided dice). Agent action is only used when
      landing on an unowned property: 0=pass, 1=buy.
    - Properties have fixed price and rent; rent does not increase.
    - Bankruptcy: if a player cannot pay required amount (rent or buy), they are
      removed from the game and their properties become unowned.
    - Observations are public (this is a fully-observed simplified setting).

    Observation (Dict):
      - current_player: scalar int
      - cash: array of shape (n_players,)
      - positions: array of shape (n_players,) with values in [0, board_size-1]
      - property_owners: array of shape (n_properties,) with values in [0..n_players] where n_players means unowned

    Action space:
      - Discrete(2): 0=pass, 1=buy (only meaningful when landing on an unowned property)

    Reward:
      - For now, reward is change in the current player's cash after the step.
        This gives a simple and immediate signal. You can extend to net worth or
        shaped rewards later.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, n_players: int = 2, seed: Optional[int] = None, max_turns: int = 1000):
        super().__init__()
        assert 2 <= n_players <= 6, "n_players should be between 2 and 6"
        self.n_players = n_players
        self.board_size = 40
        # Define a simple set of properties: here we'll mark many board indices as properties
        # For simplicity, every 3rd tile is a property (except Go which is 0)
        self.property_indices = [i for i in range(1, self.board_size) if i % 3 == 0]
        self.n_properties = len(self.property_indices)

        # Property prices and rents (simple deterministic values)
        base_price = 100
        self.property_prices = np.array([base_price + 10 * (i % 5) for i in range(self.n_properties)], dtype=np.int32)
        self.property_rents = (self.property_prices // 4).astype(np.int32)

        # Spaces
        # current_player index
        # cash for each player
        # positions for each player
        # property owners: values 0..n_players (n_players = unowned)

        self.observation_space = spaces.Dict({
            "current_player": spaces.Discrete(self.n_players),
            "cash": spaces.Box(low=0, high=10_000_000, shape=(self.n_players,), dtype=np.int32),
            "positions": spaces.MultiDiscrete([self.board_size] * self.n_players),
            "property_owners": spaces.MultiDiscrete([self.n_players + 1] * self.n_properties),
        })

        # Action: buy or pass
        self.action_space = spaces.Discrete(2)

        self.max_turns = max_turns
        self.turn_count = 0

        self.seed(seed)
        self.reset()

    def seed(self, seed: Optional[int] = None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        random.seed(seed)
        return [seed]

    def reset(self):
        # Initialize player states
        self.cash = np.array([1500] * self.n_players, dtype=np.int32)  # starting cash
        self.positions = np.zeros(self.n_players, dtype=np.int32)
        # property_owners: n_players means unowned
        self.property_owners = np.array([self.n_players] * self.n_properties, dtype=np.int32)
        self.active = [True] * self.n_players
        self.current_player = 0
        self.turn_count = 0
        # compute initial observation
        return self._get_obs()

    def _get_obs(self) -> Dict:
        return {
            "current_player": int(self.current_player),
            "cash": self.cash.copy(),
            "positions": self.positions.copy(),
            "property_owners": self.property_owners.copy(),
        }

    def _index_to_property(self, board_index: int) -> Optional[int]:
        try:
            return self.property_indices.index(board_index)
        except ValueError:
            return None

    def step(self, action: int):
        assert self.action_space.contains(action), f"Invalid action {action}"
        if not self.active[self.current_player]:
            # Skip inactive players
            self._advance_player()
            return self._get_obs(), 0.0, False, {}

        self.turn_count += 1
        prev_cash = int(self.cash[self.current_player])

        # Roll dice and move
        d1 = self.np_random.integers(1, 7)
        d2 = self.np_random.integers(1, 7)
        move = int(d1 + d2)
        self.positions[self.current_player] = (self.positions[self.current_player] + move) % self.board_size

        board_pos = int(self.positions[self.current_player])
        prop_idx = self._index_to_property(board_pos)

        # Resolve landing
        if prop_idx is not None:
            owner = int(self.property_owners[prop_idx])
            if owner == self.n_players:
                # Unowned property -> action decides
                if action == 1:
                    price = int(self.property_prices[prop_idx])
                    if self.cash[self.current_player] >= price:
                        self.cash[self.current_player] -= price
                        self.property_owners[prop_idx] = int(self.current_player)
                    else:
                        # Can't afford: treated as pass
                        pass
                else:
                    # pass
                    pass
            elif owner != self.current_player:
                # Pay rent
                rent = int(self.property_rents[prop_idx])
                payer = self.current_player
                payee = owner
                if self.cash[payer] >= rent:
                    self.cash[payer] -= rent
                    self.cash[payee] += rent
                else:
                    # Bankruptcy: remove player, unassign properties, give remaining cash to owner
                    remaining = int(self.cash[payer])
                    self.cash[payer] = 0
                    if payee < self.n_players:
                        self.cash[payee] += remaining
                    # set all properties owned by payer to unowned
                    self.property_owners[self.property_owners == payer] = self.n_players
                    self.active[payer] = False
        # else: non-property tile - do nothing for now

        reward = float(self.cash[self.current_player] - prev_cash)

        done = self._check_done()

        info = {}

        # advance to next active player
        self._advance_player()

        # safety: max turns
        if self.turn_count >= self.max_turns:
            done = True

        return self._get_obs(), reward, done, info

    def _advance_player(self):
        # Move current_player to next active
        for _ in range(self.n_players):
            self.current_player = (self.current_player + 1) % self.n_players
            if self.active[self.current_player]:
                return
        # No active players left

    def _check_done(self) -> bool:
        active_count = sum(1 for a in self.active if a)
        return active_count <= 1

    def render(self, mode: str = "human"):
        lines = [f"Turn {self.turn_count}, current_player={self.current_player}"]
        for i in range(self.n_players):
            status = "active" if self.active[i] else "bankrupt"
            lines.append(f"P{i}: pos={self.positions[i]}, cash={self.cash[i]} ({status})")
        lines.append("Properties:")
        for idx, board_idx in enumerate(self.property_indices):
            owner = self.property_owners[idx]
            owner_str = f"P{owner}" if owner < self.n_players else "unowned"
            lines.append(f"  idx={board_idx}: price={self.property_prices[idx]}, rent={self.property_rents[idx]}, owner={owner_str}")
        print("\n".join(lines))


if __name__ == "__main__":
    # Quick smoke test: run a few episodes with a random policy
    env = SimpleMonopolyEnv(n_players=3, seed=42, max_turns=200)
    episodes = 3
    for ep in range(episodes):
        obs = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        while not done:
            # Simple policy: buy if we land on unowned property and we have > price
            # But the env expects an action even when not relevant; we'll choose random
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
            if steps % 20 == 0:
                env.render()
        print(f"Episode {ep} finished in {steps} steps, total_reward={total_reward}")
