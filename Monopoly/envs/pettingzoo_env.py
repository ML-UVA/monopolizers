"""PettingZoo AEC environment for 4-player Monopoly self-play.

Exposes all players as independent agents using the AEC (Agent-Environment-Cycle)
API. Each step() call advances one agent's action. A player's "turn" may consist
of multiple step() calls (roll, buy/pass, build, end_turn).

This environment reuses the same GameEngine/RulesEngine/GameState stack as the
single-agent MonopolyEnv but removes the restriction to a single learning agent.
"""

from typing import Optional, Dict, Any, List, Literal
import functools
import logging
import numpy as np

logger = logging.getLogger(__name__)

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from gymnasium import spaces

from ..engine import GameEngine
from ..rules import RulesEngine
from ..state import GameState, PlayerState, PropertyState, DeckState, PlayerStatus
from ..board import Board
from ..property import load_property_specs
from ..cards import load_chance_cards, load_community_cards
from ..trade import decode_trade_action, encode_trade_action, compute_trade_price
from .stalemate import StalemateDetector, _compute_net_worth


RewardMode = Literal['dense_networth', 'sparse_terminal']


class MonopolyAECEnv(AECEnv):
    """PettingZoo AEC environment for 4-player Monopoly.

    Each step() applies one action for the current agent_selection.
    Monopoly turns are multi-action: roll -> buy/pass -> build/mortgage/trade -> end_turn.
    agent_selection stays on the same player until they execute end_turn (action 89),
    at which point it advances to the next active player.
    """

    metadata = {
        "render_modes": ["human", "ansi"],
        "name": "monopoly_v0",
        "is_parallelizable": False,
        "render_fps": 1,
    }

    N_BASE_ACTIONS = 90
    N_PROPERTIES = 28

    def __init__(
        self,
        num_players: int = 4,
        max_steps: int = 2000,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
        reward_mode: RewardMode = "sparse_terminal",
        stalemate_threshold: int = 50,
        fast_mode: bool = False,
    ):
        super().__init__()

        if reward_mode not in ("dense_networth", "sparse_terminal"):
            raise ValueError(
                f"Invalid reward_mode '{reward_mode}'. "
                f"Must be 'dense_networth' or 'sparse_terminal'."
            )

        self.num_players = num_players
        self.max_steps = max_steps
        self.render_mode = render_mode
        self._seed = seed or 42
        self.reward_mode: RewardMode = reward_mode
        self._fast_mode = fast_mode

        # Agent identifiers
        self.possible_agents: List[str] = [
            f"player_{i}" for i in range(num_players)
        ]
        self.agent_name_mapping: Dict[str, int] = {
            f"player_{i}": i for i in range(num_players)
        }

        # Game components
        self.board = Board.load_standard_board()
        self.property_specs = load_property_specs()
        self.chance_cards = load_chance_cards()
        self.community_cards = load_community_cards()
        self.rules_engine = RulesEngine(
            self.board, self.property_specs,
            self.chance_cards, self.community_cards,
        )

        # Action/observation space dimensions
        self.n_trade_actions = self.N_PROPERTIES * (num_players - 1)
        self.n_actions = self.N_BASE_ACTIONS + self.n_trade_actions

        # Stalemate detection
        self._stalemate_detector = StalemateDetector(
            threshold_rounds=stalemate_threshold,
        )

        # Track which player started the current round (for stalemate detection)
        self._round_start_player: int = 0

        # Game state (set in reset)
        self.state: Optional[GameState] = None
        self.engine: Optional[GameEngine] = None
        self._step_count: int = 0
        self._net_worth_cache: Optional[np.ndarray] = None

        # Renderer
        self.renderer = None
        if self.render_mode == "human":
            from .renderer import MonopolyRenderer
            self.renderer = MonopolyRenderer(self.board, self.property_specs)

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> spaces.Dict:
        """Return observation space for agent (identical for all).

        Uses PettingZoo convention: {"observation": <flat array>, "action_mask": <array>}.
        The observation is a flat float32 vector concatenating all game state features.
        """
        obs_dim = self._obs_flat_dim()
        return spaces.Dict({
            "observation": spaces.Box(
                low=-1.0, high=np.inf, shape=(obs_dim,), dtype=np.float32,
            ),
            "action_mask": spaces.Box(0, 1, shape=(self.n_actions,), dtype=np.int8),
        })

    def _obs_flat_dim(self) -> int:
        """Compute the flat observation dimension."""
        n = self.num_players
        # player_id(1) + cash(n) + positions(n) + property_owner(28)
        # + houses(28) + mortgaged(28) + jail_turns(n) + get_out_cards(n)
        # + turn_number(1) + last_roll(2) + bank_houses(1) + bank_hotels(1)
        # + net_worth(n)
        return 1 + n + n + 28 + 28 + 28 + n + n + 1 + 2 + 1 + 1 + n

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str) -> spaces.Discrete:
        """Return action space for agent (identical for all)."""
        return spaces.Discrete(self.n_actions)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> None:
        """Reset environment to initial state."""
        if seed is not None:
            self._seed = seed

        # Create initial game state
        players = [
            PlayerState(
                id=i, cash=1500, position=0,
                properties_owned=set(), houses_on_property={},
                mortgaged_properties=set(), jail_turns=0,
                get_out_of_jail_cards=0, status=PlayerStatus.ACTIVE,
            )
            for i in range(self.num_players)
        ]
        properties = [
            PropertyState(owner=None, houses_count=0, mortgaged=False)
            for _ in range(28)
        ]

        self.state = GameState(
            players=players,
            properties=properties,
            chance_deck=DeckState(pointer=0, seed=self._seed),
            community_deck=DeckState(pointer=0, seed=self._seed + 1),
            bank_houses_left=32,
            bank_hotels_left=12,
            current_player=0,
            last_roll=None,
            doubles_count=0,
            turn_number=0,
            seed=self._seed,
            has_rolled=False,
            awaiting_buy_decision=False,
            pending_rent=0,
            rent_creditor=None,
        )

        self.engine = GameEngine(self.rules_engine, seed=self._seed)
        self._step_count = 0
        self._net_worth_cache = None
        self._stalemate_detector.reset()
        self._round_start_player = 0

        # PettingZoo required state
        self.agents = list(self.possible_agents)
        self.rewards = {agent: 0.0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        # Agent selection
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

    def step(self, action) -> None:
        """Apply action for the current agent_selection."""
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            # PettingZoo requires None action for dead agents
            self._was_dead_step(action)
            return

        agent = self.agent_selection
        player_id = self.agent_name_mapping[agent]

        # Decode and validate action
        legal_mask = self._get_legal_mask(player_id)
        game_action = self._decode_action(action, player_id)

        if legal_mask[action] == 0:
            # Invalid action — force safe fallback
            if self.state.has_rolled:
                game_action = {"type": "end_turn"}
            else:
                game_action = {"type": "roll"}

        # Apply action through rules engine
        try:
            self.state, _action_reward, done, _log = self.rules_engine.apply_action(
                self.state, game_action, self.engine.rng, engine=self.engine,
            )
        except Exception as e:
            logger.warning("apply_action failed for %s action=%s: %s", agent, game_action, e)
            # Force end turn to prevent stuck state
            if self.state.has_rolled:
                self.state, _, _, _ = self.rules_engine.apply_action(
                    self.state, {"type": "end_turn"},
                    self.engine.rng, engine=self.engine,
                )

        self._step_count += 1
        self._net_worth_cache = None  # Invalidate cache

        # Check termination conditions
        game_over = self._is_game_over()
        step_limit = self._step_count >= self.max_steps

        # Check stalemate: when current_player cycles back to round start
        stalemate = False
        if (
            not game_over
            and not step_limit
            and self.state.current_player == self._round_start_player
            and game_action.get("type") == "end_turn"
        ):
            stalemate = self._stalemate_detector.on_round_complete(
                self.state, self.property_specs, self.rules_engine,
            )

        # Compute rewards
        self._assign_rewards(game_over, step_limit or stalemate, player_id)

        # Update terminations/truncations
        if game_over:
            for ag in self.agents:
                self.terminations[ag] = True
        if step_limit or stalemate:
            for ag in self.agents:
                self.truncations[ag] = True

        # Remove bankrupt players from agents list
        for ag in list(self.agents):
            pid = self.agent_name_mapping[ag]
            if self.state.players[pid].status == PlayerStatus.BANKRUPT:
                self.terminations[ag] = True

        # Update infos
        if not self._fast_mode:
            self.infos[agent] = self._get_info(player_id)

        # Advance agent selection.
        # The rules engine advances current_player on end_turn.
        # We follow the game state's current_player to determine next agent.
        if not (game_over or step_limit or stalemate):
            # Skip bankrupt players (the rules engine should already do this,
            # but be safe)
            for _ in range(self.num_players):
                cp = self.state.current_player
                if self.state.players[cp].status == PlayerStatus.ACTIVE:
                    break
                self.state.current_player = (cp + 1) % self.num_players

        # Set agent_selection directly from game state
        target = f"player_{self.state.current_player}"
        if target in self.agents:
            self.agent_selection = target
        elif self.agents:
            # Fallback: first remaining agent (handles edge cases during cleanup)
            self.agent_selection = self.agents[0]

        # Accumulate rewards
        self._accumulate_rewards()

    def observe(self, agent: str) -> Dict[str, np.ndarray]:
        """Return observation for the given agent.

        Uses PettingZoo convention: {"observation": <flat array>, "action_mask": <array>}.
        The observation is a flat float32 vector concatenating all game state features.
        """
        player_id = self.agent_name_mapping[agent]

        # Net worth computation
        if self._fast_mode:
            net_worths = np.zeros(self.num_players, dtype=np.float32)
        else:
            net_worths = self._get_net_worths()

        action_mask = self._get_legal_mask(player_id)

        # Build flat observation vector
        observation = np.concatenate([
            np.array([player_id], dtype=np.float32),
            np.array([p.cash for p in self.state.players], dtype=np.float32),
            np.array([p.position for p in self.state.players], dtype=np.float32),
            np.array(
                [p.owner if p.owner is not None else -1 for p in self.state.properties],
                dtype=np.float32,
            ),
            np.array([p.houses_count for p in self.state.properties], dtype=np.float32),
            np.array([int(p.mortgaged) for p in self.state.properties], dtype=np.float32),
            np.array([p.jail_turns for p in self.state.players], dtype=np.float32),
            np.array(
                [p.get_out_of_jail_cards for p in self.state.players], dtype=np.float32,
            ),
            np.array([self.state.turn_number], dtype=np.float32),
            np.array(
                self.state.last_roll if self.state.last_roll else [1, 1],
                dtype=np.float32,
            ),
            np.array([self.state.bank_houses_left], dtype=np.float32),
            np.array([self.state.bank_hotels_left], dtype=np.float32),
            net_worths,
        ])

        return {"observation": observation, "action_mask": action_mask}

    def _get_legal_mask(self, player_id: int) -> np.ndarray:
        """Compute binary mask of legal actions for the given player."""
        mask = np.zeros(self.n_actions, dtype=np.int8)
        player = self.state.players[player_id]

        # Not this player's turn — no legal actions
        if self.state.current_player != player_id:
            return mask

        if player.status == PlayerStatus.BANKRUPT:
            return mask

        # Jail situation
        if player.jail_turns > 0:
            if not self.state.has_rolled:
                mask[0] = 1  # roll (try for doubles)
                if player.cash >= 50:
                    mask[87] = 1  # pay_fine
                if player.get_out_of_jail_cards > 0:
                    mask[88] = 1  # use_jail_card
            else:
                mask[89] = 1  # end_turn
            return mask

        # Must roll first
        if not self.state.has_rolled:
            mask[0] = 1
            return mask

        # Awaiting buy decision
        if self.state.awaiting_buy_decision:
            pos = player.position
            tile = self.board.get_tile(pos)
            if tile.property_idx is not None:
                prop = self.state.properties[tile.property_idx]
                spec = self.property_specs[tile.property_idx]
                if prop.owner is None and player.cash >= spec.price:
                    mask[1] = 1  # buy
            mask[2] = 1  # pass
            return mask

        # Post-roll phase
        # Building
        for prop_idx in player.properties_owned:
            if self.rules_engine._has_monopoly(self.state, prop_idx, player_id):
                spec = self.property_specs[prop_idx]
                prop = self.state.properties[prop_idx]
                if (
                    prop.houses_count < 5
                    and player.cash >= spec.house_cost
                    and not prop.mortgaged
                    and self.state.bank_houses_left > 0
                ):
                    mask[3 + prop_idx] = 1

        # Mortgage / unmortgage
        for prop_idx in player.properties_owned:
            prop = self.state.properties[prop_idx]
            if not prop.mortgaged and prop.houses_count == 0:
                mask[31 + prop_idx] = 1
            elif prop.mortgaged:
                spec = self.property_specs[prop_idx]
                unmortgage_cost = int(spec.mortgage_value * 1.1)
                if player.cash >= unmortgage_cost:
                    mask[59 + prop_idx] = 1

        # Trade actions
        for prop_idx in player.properties_owned:
            prop = self.state.properties[prop_idx]
            if not prop.mortgaged and prop.houses_count == 0:
                price = compute_trade_price(self.property_specs, prop_idx)
                for opponent_id in range(self.num_players):
                    if opponent_id == player_id:
                        continue
                    opponent = self.state.players[opponent_id]
                    if (
                        opponent.status == PlayerStatus.ACTIVE
                        and opponent.cash >= price
                    ):
                        action_idx = encode_trade_action(
                            prop_idx, opponent_id, player_id, self.num_players,
                        )
                        if action_idx < self.n_actions:
                            mask[action_idx] = 1

        # End turn always legal after rolling
        mask[89] = 1
        return mask

    def _decode_action(self, action: int, player_id: int) -> Dict[str, Any]:
        """Convert discrete action index to game action dict for given player."""
        if action == 0:
            return {"type": "roll"}
        elif action == 1:
            pos = self.state.players[player_id].position
            tile = self.board.get_tile(pos)
            return {"type": "buy", "property_idx": tile.property_idx}
        elif action == 2:
            return {"type": "pass"}
        elif 3 <= action <= 30:
            return {"type": "build", "property_idx": action - 3}
        elif 31 <= action <= 58:
            return {"type": "mortgage", "property_idx": action - 31}
        elif 59 <= action <= 86:
            return {"type": "unmortgage", "property_idx": action - 59}
        elif action == 87:
            return {"type": "pay_fine"}
        elif action == 88:
            return {"type": "use_jail_card"}
        elif action == 89:
            return {"type": "end_turn"}
        elif action >= self.N_BASE_ACTIONS:
            result = decode_trade_action(action, player_id, self.num_players)
            if result is not None:
                prop_idx, buyer_id = result
                price = compute_trade_price(self.property_specs, prop_idx)
                return {
                    "type": "trade",
                    "property_idx": prop_idx,
                    "buyer_id": buyer_id,
                    "price": price,
                }
            return {"type": "pass"}
        else:
            return {"type": "pass"}

    def _is_game_over(self) -> bool:
        """Check if <= 1 active player."""
        active = sum(
            1 for p in self.state.players if p.status == PlayerStatus.ACTIVE
        )
        return active <= 1

    def _get_net_worths(self) -> np.ndarray:
        """Compute net worths for all players (cached per step)."""
        if self._net_worth_cache is None:
            self._net_worth_cache = np.array(
                [
                    _compute_net_worth(
                        self.state, i, self.property_specs, self.rules_engine,
                    )
                    for i in range(self.num_players)
                ],
                dtype=np.float32,
            )
        return self._net_worth_cache

    def _assign_rewards(
        self, terminated: bool, truncated: bool, acting_player_id: int,
    ) -> None:
        """Assign rewards based on reward_mode."""
        # Reset rewards each step
        for ag in self.agents:
            self.rewards[ag] = 0.0

        if self.reward_mode == "dense_networth":
            if not terminated and not truncated:
                # Only the acting agent gets a reward signal this step
                agent_name = f"player_{acting_player_id}"
                if agent_name in self.agents:
                    net_worths = self._get_net_worths()
                    agent_nw = float(net_worths[acting_player_id])
                    other_nw = sum(
                        float(net_worths[i])
                        for i in range(self.num_players)
                        if i != acting_player_id
                        and self.state.players[i].status == PlayerStatus.ACTIVE
                    )
                    if other_nw > 0:
                        self.rewards[agent_name] = agent_nw / other_nw
                    elif agent_nw > 0:
                        self.rewards[agent_name] = 1.0

        elif self.reward_mode == "sparse_terminal":
            if terminated:
                for ag in self.agents:
                    pid = self.agent_name_mapping[ag]
                    if self.state.players[pid].status == PlayerStatus.ACTIVE:
                        self.rewards[ag] = 1.0  # Winner
                    else:
                        self.rewards[ag] = -1.0  # Loser
            elif truncated:
                # Tie-break by net worth
                net_worths = self._get_net_worths()
                max_nw = max(
                    float(net_worths[i])
                    for i in range(self.num_players)
                    if self.state.players[i].status == PlayerStatus.ACTIVE
                )
                for ag in self.agents:
                    pid = self.agent_name_mapping[ag]
                    if self.state.players[pid].status != PlayerStatus.ACTIVE:
                        self.rewards[ag] = -1.0
                    elif float(net_worths[pid]) >= max_nw:
                        self.rewards[ag] = 1.0
                    else:
                        self.rewards[ag] = -1.0

    def _get_info(self, player_id: int) -> Dict[str, Any]:
        """Get info dict for a player."""
        return {
            "turn_number": self.state.turn_number,
            "step_count": self._step_count,
            "current_player": self.state.current_player,
            "active_players": sum(
                1 for p in self.state.players if p.status == PlayerStatus.ACTIVE
            ),
        }

    def render(self) -> None:
        """Render the current game state."""
        if self._fast_mode:
            return

        if self.render_mode == "human":
            if self.renderer and self.state:
                self.renderer.render(self.state, show_stats=True)
        elif self.render_mode == "ansi":
            print(f"\n{'=' * 60}")
            print(f"Turn {self.state.turn_number} | Current Player: {self.state.current_player}")
            print(f"{'=' * 60}")
            for i, player in enumerate(self.state.players):
                status = "BANKRUPT" if player.status == PlayerStatus.BANKRUPT else "ACTIVE"
                nw = _compute_net_worth(
                    self.state, i, self.property_specs, self.rules_engine,
                )
                print(
                    f"  Player {i}: ${player.cash} | NW: ${nw:.0f} | "
                    f"Pos: {player.position} | Props: {len(player.properties_owned)} | {status}"
                )
            print(f"{'=' * 60}\n")

    def close(self) -> None:
        """Clean up resources."""
        if self.renderer:
            self.renderer.close()
            self.renderer = None
