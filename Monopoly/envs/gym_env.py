"""Gymnasium wrapper for the Monopoly game engine.

================================================================================
RESEARCH FOCUS (H1): Reward Shaping in Long-Horizon Stochastic Games
================================================================================

Hypothesis: Dense relative-net-worth reward improves learning speed and final
performance compared to sparse terminal win/loss reward in stochastic Monopoly RL.

This environment provides a single-agent interface where one player (the RL agent)
learns against a fixed mixture of opponent policies (Random, MCTS, Greedy).

================================================================================
REWARD MODES (H1 Ablation Variable)
================================================================================

1. reward_mode='dense_networth' (Default)
   - Every step: r_t = NW_agent / Σ(NW_opponents)
   - At termination: r_T = 0
   - Rationale: Provides continuous learning signal based on relative wealth

2. reward_mode='sparse_terminal'
   - During play: r_t = 0
   - At termination: r_T = +1 (win) or -1 (lose)
   - Rationale: Standard game-theoretic outcome signal

Net Worth Computation:
   NW_x = cash_x + Σ(property_value(p) for p in owned_properties)
   
   property_value(p) = {
       0                                    if mortgaged
       (price - mortgage_value) * bonus     if unmortgaged, no houses
       + houses * house_cost                if has houses
       + house_cost                         if hotel (houses=5)
   }
   where bonus = 2.0 if monopoly else 1.5

================================================================================
ACTION SPACE (Discrete, with Trading)
================================================================================

Actions 0-89: Base actions
   0: Roll dice
   1: Buy property (if awaiting buy decision)
   2: Pass on buying
   3-30: Build house on property idx (3 + prop_idx)
   31-58: Mortgage property idx (31 + prop_idx)
   59-86: Unmortgage property idx (59 + prop_idx)
   87: Pay jail fine ($50)
   88: Use get-out-of-jail card
   89: End turn

Actions 90+: Trade actions (sell property to opponent for fixed price)
   Encoding: action = 90 + prop_idx * (num_players - 1) + opponent_offset
   
   For 4 players: 84 trade actions (28 properties × 3 opponents)
   Total action space: 174 actions

Trading is deterministic: if the opponent can afford the fixed price
(1.5 × mortgage_value), the trade is automatically accepted.

================================================================================
FOUR-PLAYER MODE (Default)
================================================================================

Player 0: RL Agent (generates training data)
Player 1: Random Bot (seeded for reproducibility)
Player 2: MCTS Bot (seeded, rollouts=5, depth=5)
Player 3: Greedy Bot (deterministic)

See RESEARCH_IMPLEMENTATION_AUDIT_NOTES.txt for full experimental protocol.
"""

from typing import Callable, Literal, Optional, Tuple, Dict, Any, List
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from ..engine import GameEngine
from ..rules import RulesEngine, ActionType
from ..state import GameState, PlayerState, PropertyState, DeckState, PlayerStatus
from ..board import Board
from ..property import load_property_specs
from ..cards import load_chance_cards, load_community_cards
from ..trade import decode_trade_action, encode_trade_action, compute_trade_price
from .renderer import MonopolyRenderer
from .rollout import RolloutRecorder, save_rollout as _save_rollout


# Type alias for reward mode configuration (H1 ablation variable)
RewardMode = Literal['dense_networth', 'sparse_terminal', 'modular']


class MonopolyEnv(gym.Env):
    """
    Monopoly Gymnasium Environment with Trading Support
    
    This environment implements a research-grade Monopoly simulator designed
    for studying reward shaping in long-horizon stochastic games (H1).
    
    Key Features:
    - Configurable reward modes for ablation study
    - Trading integrated into discrete action space
    - Legal action masking for all action types
    - Reproducible via explicit seeding
    
    Action Space: Discrete(n_actions) where:
        - Actions 0-89: Base Monopoly actions (roll, buy, build, mortgage, etc.)
        - Actions 90+: Trade actions (sell property to opponent)
        
    Observation Space: Dict with game state, legal mask, and net worth
    """
    
    metadata = {'render_modes': ['human', 'ansi'], 'render_fps': 1}
    
    # Action space constants
    N_BASE_ACTIONS = 90
    N_PROPERTIES = 28
    
    def __init__(
        self,
        num_players: int = 4,
        agent_player_id: int = 0,
        opponent_policies: Optional[List[Callable]] = None,
        max_turns: int = 1000,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
        reward_mode: RewardMode = 'dense_networth',
        record_rollout: bool = False,
        render_every_n: int = 1,
        reward_weights: Optional[Dict[str, float]] = None,
        normalize_rewards: bool = False,
        reward_clip: Optional[float] = 10.0,
    ):
        """Initialize the Monopoly environment.

        Args:
            num_players: Number of players (default 4)
            agent_player_id: Which player index the RL agent controls (default 0)
            opponent_policies: List of Agent instances for opponents
            max_turns: Maximum game turns before truncation (default 1000)
            render_mode: 'human', 'ansi', or None
            seed: Random seed for reproducibility
            reward_mode: Reward strategy for H1 ablation study
                - 'dense_networth': r = NW_agent / Σ(NW_others) every step
                - 'sparse_terminal': +1 win, -1 lose, 0 otherwise
                - 'modular': weighted sum of interpretable components
            record_rollout: If True, record full game states each step for replay
            render_every_n: Only render every Nth call (frame skip for training)
            reward_weights: Component weights for modular reward mode
            normalize_rewards: Apply online normalization to modular rewards
            reward_clip: Clip range for reward normalizer
        """
        super().__init__()

        self.num_players = num_players
        self.agent_player_id = agent_player_id
        self.max_turns = max_turns
        self.render_mode = render_mode
        self._seed = seed or 42
        self.reward_mode: RewardMode = reward_mode

        # Validate reward_mode
        if reward_mode not in ('dense_networth', 'sparse_terminal', 'modular'):
            raise ValueError(
                f"Invalid reward_mode '{reward_mode}'. "
                f"Must be 'dense_networth', 'sparse_terminal', or 'modular'."
            )

        # Modular reward state (lazy-initialized on first use)
        self._reward_weights = reward_weights
        self._normalize_rewards = normalize_rewards
        self._reward_clip = reward_clip
        self._modular_reward = None
        self._last_reward_components: Optional[Dict[str, float]] = None
        self._last_action: int = 0
        
        # Initialize game components
        self.board = Board.load_standard_board()
        self.property_specs = load_property_specs()
        self.chance_cards = load_chance_cards()
        self.community_cards = load_community_cards()
        self.rules_engine = RulesEngine(
            self.board, self.property_specs, self.chance_cards, self.community_cards
        )
        
        # Opponent policies (default to random if not provided)
        self.opponent_policies = opponent_policies or []
        
        # Fill missing opponents with seeded RandomAgent and assign player IDs
        current_opp_idx = 0
        for i in range(num_players):
            if i == agent_player_id:
                continue
            
            if current_opp_idx < len(self.opponent_policies):
                # Use provided policy, ensure player_id is set
                agent = self.opponent_policies[current_opp_idx]
                if agent.player_id == -1:
                    agent.player_id = i
            else:
                # Add default RandomAgent with deterministic seed
                from ..agents.random import RandomAgent
                opp_seed = self._seed + 1000 + i if self._seed is not None else None
                self.opponent_policies.append(RandomAgent(player_id=i, seed=opp_seed))
            current_opp_idx += 1

        # Compute action space size
        # Base actions: 90 (roll, buy, pass, build×28, mortgage×28, unmortgage×28, pay_fine, use_jail_card, end_turn)
        # Trade actions: 28 properties × (num_players - 1) opponents
        self.n_trade_actions = self.N_PROPERTIES * (num_players - 1)
        self.n_actions = self.N_BASE_ACTIONS + self.n_trade_actions
        self.action_space = spaces.Discrete(self.n_actions)
        
        # Observation space (includes net_worth for reward computation visibility)
        self.observation_space = spaces.Dict({
            'player_id': spaces.Box(0, num_players-1, shape=(1,), dtype=np.int32),
            'cash': spaces.Box(0, 100000, shape=(num_players,), dtype=np.float32),
            'positions': spaces.Box(0, 39, shape=(num_players,), dtype=np.int32),
            'property_owner': spaces.Box(-1, num_players-1, shape=(28,), dtype=np.int32),
            'houses': spaces.Box(0, 5, shape=(28,), dtype=np.int32),
            'mortgaged': spaces.Box(0, 1, shape=(28,), dtype=np.int32),
            'jail_turns': spaces.Box(0, 10, shape=(num_players,), dtype=np.int32),
            'get_out_cards': spaces.Box(0, 10, shape=(num_players,), dtype=np.int32),
            'legal_mask': spaces.Box(0, 1, shape=(self.n_actions,), dtype=np.int32),
            'turn_number': spaces.Box(0, max_turns, shape=(1,), dtype=np.int32),
            'last_roll': spaces.Box(1, 6, shape=(2,), dtype=np.int32),
            'bank_houses': spaces.Box(0, 32, shape=(1,), dtype=np.int32),
            'bank_hotels': spaces.Box(0, 12, shape=(1,), dtype=np.int32),
            'net_worth': spaces.Box(0, 100000, shape=(num_players,), dtype=np.float32),
        })
        
        # Game state
        self.state: Optional[GameState] = None
        self.engine: Optional[GameEngine] = None
        self._episode_step = 0
        
        # Net worth cache (invalidated each step to avoid stale data)
        self._net_worth_cache: Optional[np.ndarray] = None
        
        # Pygame renderer (lazy init on first render() call)
        self.renderer: Optional[MonopolyRenderer] = None

        # Rollout recording
        self.record_rollout = record_rollout
        self._rollout_recorder: Optional[RolloutRecorder] = None
        self._net_worth_history: List[List[float]] = []
        self._last_action_label: str = ""

        # Frame skip
        self.render_every_n = max(1, render_every_n)
        self._render_counter = 0

    def _compute_property_value(self, prop_idx: int, player_id: int) -> float:
        """
        Compute the value p_a for a property using the formula:
        p_a = (bp - mv) * b + nh * ph + nH * pH
        
        Where:
        - bp = base price
        - mv = mortgage value  
        - b = bonus multiplier (1.5 if not monopoly, 2 if monopoly)
        - nh = number of houses (0-4)
        - ph = price per house
        - nH = number of hotels (0 or 1, represented as houses=5)
        - pH = price per hotel (same as house cost in standard rules)
        
        If property is mortgaged, return 0.
        """
        prop = self.state.properties[prop_idx]
        spec = self.property_specs[prop_idx]
        
        # If mortgaged, value is 0
        if prop.mortgaged:
            return 0.0
        
        bp = spec.price  # base price
        mv = spec.mortgage_value  # mortgage value
        
        # Check if player has monopoly on this property
        has_monopoly = self.rules_engine._has_monopoly(self.state, prop_idx, player_id)
        b = 2.0 if has_monopoly else 1.5  # bonus multiplier
        
        # Houses and hotels
        houses_count = prop.houses_count
        ph = spec.house_cost  # price per house
        pH = spec.house_cost  # price per hotel (same as house cost)
        
        if houses_count == 5:
            # Has a hotel
            nh = 0
            nH = 1
        else:
            nh = houses_count
            nH = 0
        
        # Calculate property value
        p_a = (bp - mv) * b + nh * ph + nH * pH
        
        return p_a

    def _compute_net_worth(self, player_id: int) -> float:
        """
        Compute net worth for a player:
        nw_x = c_x + sum(p_a for all owned properties)
        
        Where c_x is cash and p_a is computed property value.
        """
        player = self.state.players[player_id]
        
        # If bankrupt, net worth is 0
        if player.status == PlayerStatus.BANKRUPT:
            return 0.0
        
        # Cash
        nw = float(player.cash)
        
        # Sum property values
        for prop_idx in player.properties_owned:
            nw += self._compute_property_value(prop_idx, player_id)
        
        return nw

    def _refresh_net_worth_cache(self) -> np.ndarray:
        """
        Compute and cache net worth for all players.
        Called once per step to avoid redundant computation.
        """
        self._net_worth_cache = np.array(
            [self._compute_net_worth(i) for i in range(self.num_players)], 
            dtype=np.float32
        )
        return self._net_worth_cache

    def _get_cached_net_worth(self, player_id: int) -> float:
        """Get net worth from cache (must call _refresh_net_worth_cache first)."""
        if self._net_worth_cache is None:
            self._refresh_net_worth_cache()
        return float(self._net_worth_cache[player_id])

    def _compute_net_worth_reward(self, player_id: int) -> float:
        """
        Compute the relative net worth reward:
        r_x = nw_x / sum(nw_other_active_players)
        
        Returns value between 0 and 1 (can be > 1 if agent dominates).
        Returns 0 if no other active players (shouldn't happen during game).
        
        Uses cached net worth values for efficiency.
        """
        agent_nw = self._get_cached_net_worth(player_id)
        
        # Sum net worth of all OTHER active players
        other_nw_sum = 0.0
        for i, p in enumerate(self.state.players):
            if i != player_id and p.status == PlayerStatus.ACTIVE:
                other_nw_sum += self._get_cached_net_worth(i)
        
        # Avoid division by zero
        if other_nw_sum <= 0:
            return 1.0 if agent_nw > 0 else 0.0
        
        reward = agent_nw / other_nw_sum
        return reward

    def _compute_reward(self, terminated: bool, truncated: bool) -> float:
        """
        Compute reward based on configured reward_mode (H1 ablation variable).
        
        Reward modes:
        - 'dense_networth': r = nw_agent / sum(nw_others) every step, 0 at terminal
        - 'sparse_terminal': +1 win, -1 lose, 0 otherwise
        
        Args:
            terminated: Whether the game ended (one player left)
            truncated: Whether max_turns was exceeded
            
        Returns:
            Reward scalar for this step
        """
        agent_status = self.state.players[self.agent_player_id].status
        
        if self.reward_mode == 'dense_networth':
            # Dense relative net-worth reward during play, 0 at terminal
            if terminated or truncated:
                return 0.0
            return self._compute_net_worth_reward(self.agent_player_id)
        
        elif self.reward_mode == 'sparse_terminal':
            # Sparse terminal reward only
            if terminated:
                if agent_status == PlayerStatus.ACTIVE:
                    # Agent is last player standing → win
                    return 1.0
                else:
                    # Agent went bankrupt → lose
                    return -1.0
            elif truncated:
                # Game timed out - use net worth comparison as tie-breaker
                agent_nw = self._get_cached_net_worth(self.agent_player_id)
                max_other_nw = max(
                    (self._get_cached_net_worth(i) for i in range(self.num_players) 
                     if i != self.agent_player_id and self.state.players[i].status == PlayerStatus.ACTIVE),
                    default=0.0
                )
                if agent_nw > max_other_nw:
                    return 1.0  # Win on net worth
                elif agent_nw < max_other_nw:
                    return -1.0  # Lose on net worth
                else:
                    return 0.0  # Tie
            else:
                return 0.0  # No reward during play
        
        elif self.reward_mode == 'modular':
            # Modular reward: weighted sum of interpretable components
            if self._modular_reward is None:
                from ..agents.reward import ModularRewardCalculator
                self._modular_reward = ModularRewardCalculator(
                    weights=self._reward_weights,
                    normalize=self._normalize_rewards,
                    clip=self._reward_clip,
                )
            total, components = self._modular_reward.compute(
                info=self._get_info(),
                action=self._last_action,
                terminated=terminated,
                truncated=truncated,
            )
            self._last_reward_components = components
            return total

        else:
            raise ValueError(f"Unknown reward_mode: {self.reward_mode}")

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)

        if seed is not None:
            self._seed = seed

        # Reset modular reward calculator if active
        if self._modular_reward is not None:
            self._modular_reward.reset()
        self._last_reward_components = None
        
        # Create initial game state
        players = []
        for i in range(self.num_players):
            players.append(PlayerState(
                id=i,
                cash=1500,
                position=0,
                properties_owned=set(),
                houses_on_property={},
                mortgaged_properties=set(),
                jail_turns=0,
                get_out_of_jail_cards=0,
                status=PlayerStatus.ACTIVE
            ))
        
        properties = [PropertyState(owner=None, houses_count=0, mortgaged=False) for _ in range(28)]
        
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
            rent_creditor=None
        )
        
        self.engine = GameEngine(self.rules_engine, seed=self._seed)
        self._episode_step = 0
        self._net_worth_cache = None  # Invalidate cache for new episode
        self._render_counter = 0
        self._net_worth_history = []
        self._last_action_label = ""
        
        # Reset opponent agents' RNG for reproducibility
        for i, agent in enumerate(self.opponent_policies):
            if hasattr(agent, 'reset'):
                opp_seed = self._seed + 1000 + agent.player_id if self._seed is not None else None
                agent.reset(seed=opp_seed)
        
        # If agent is not first player, simulate opponent turns until agent's turn
        while self.state.current_player != self.agent_player_id and not self._is_game_over():
            self._simulate_opponent_turn()

        # Initialize rollout recording if enabled
        if self.record_rollout:
            self._refresh_net_worth_cache()
            nw_list = [float(self._net_worth_cache[i]) for i in range(self.num_players)]
            self._net_worth_history = [nw_list]
            self._rollout_recorder = RolloutRecorder(metadata={
                'seed': self._seed,
                'reward_mode': self.reward_mode,
                'num_players': self.num_players,
                'max_turns': self.max_turns,
            })
            self._rollout_recorder.record_initial(self.state, nw_list)

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment.
        
        Only the RL agent's actions generate rewards and training data.
        Opponents take turns but don't contribute to training.
        
        Reward system (controlled by self.reward_mode - H1 ablation):
        - 'dense_networth': r_x = nw_x / sum(nw_other_active_players) every step
        - 'sparse_terminal': +1 for win, -1 for lose, 0 otherwise
        """
        if self.state is None:
            raise RuntimeError("Must call reset() before step()")
        
        # Ensure it's the agent's turn
        if self.state.current_player != self.agent_player_id:
            raise RuntimeError(f"Not agent's turn (current: {self.state.current_player}, agent: {self.agent_player_id})")
        
        # Convert discrete action to game action
        game_action = self._decode_action(action)
        
        # Validate action is legal
        legal_mask = self._get_legal_mask()
        if legal_mask[action] == 0:
            # Invalid action - force pass/end_turn (no penalty in new reward system)
            if self.state.has_rolled:
                game_action = {'type': 'end_turn'}
            else:
                game_action = {'type': 'roll'}
        
        # Apply action through rules engine
        try:
            self.state, action_reward, done, log = self.rules_engine.apply_action(
                self.state, game_action, self.engine.rng, engine=self.engine
            )
        except Exception as e:
            # If action fails, force end turn to prevent stuck state
            if self.state.has_rolled:
                self.state, _, _, _ = self.rules_engine.apply_action(
                    self.state, {'type': 'end_turn'}, self.engine.rng, engine=self.engine
                )
        
        self._episode_step += 1
        
        # Refresh net worth cache once per step (used by reward, obs, info)
        self._refresh_net_worth_cache()
        
        # Compute reward based on configured reward_mode (H1 ablation variable)
        terminated = self._is_game_over()
        truncated = self._episode_step >= self.max_turns

        self._last_action = action
        reward = self._compute_reward(terminated, truncated)
        
        # Simulate opponent turns only if it's no longer the agent's turn
        while (self.state.current_player != self.agent_player_id and 
               not self._is_game_over()):
            self._simulate_opponent_turn()
            
            # Recompute termination after opponent turns
            terminated = self._is_game_over()
        
        # Refresh cache again after opponent turns (state may have changed)
        self._refresh_net_worth_cache()

        # Record rollout step
        self._last_action_label = self._action_to_label(action)
        nw_list = [float(self._net_worth_cache[i]) for i in range(self.num_players)]
        self._net_worth_history.append(nw_list)
        if self._rollout_recorder is not None:
            self._rollout_recorder.record_step(
                state=self.state,
                action=action,
                action_label=self._last_action_label,
                reward=reward,
                net_worths=nw_list,
                terminated=terminated,
                truncated=truncated,
            )

        obs = self._get_observation()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def _decode_action(self, action: int) -> Dict[str, Any]:
        """Convert discrete action index to game action dict.
        
        Action Encoding:
            0: Roll dice
            1: Buy property at current position
            2: Pass on buying
            3-30: Build house on property (action - 3)
            31-58: Mortgage property (action - 31)
            59-86: Unmortgage property (action - 59)
            87: Pay jail fine
            88: Use get-out-of-jail card
            89: End turn
            90+: Trade actions (sell property to opponent)
        """
        if action == 0:
            return {'type': 'roll'}
        elif action == 1:
            # Buy property at current position
            pos = self.state.players[self.agent_player_id].position
            tile = self.board.get_tile(pos)
            return {'type': 'buy', 'property_idx': tile.property_idx}
        elif action == 2:
            return {'type': 'pass'}
        elif 3 <= action <= 30:
            prop_idx = action - 3
            return {'type': 'build', 'property_idx': prop_idx}
        elif 31 <= action <= 58:
            prop_idx = action - 31
            return {'type': 'mortgage', 'property_idx': prop_idx}
        elif 59 <= action <= 86:
            prop_idx = action - 59
            return {'type': 'unmortgage', 'property_idx': prop_idx}
        elif action == 87:
            return {'type': 'pay_fine'}
        elif action == 88:
            return {'type': 'use_jail_card'}
        elif action == 89:
            return {'type': 'end_turn'}
        elif action >= self.N_BASE_ACTIONS:
            # Trade action: decode to (property_idx, buyer_id)
            result = decode_trade_action(action, self.agent_player_id, self.num_players)
            if result is not None:
                prop_idx, buyer_id = result
                price = compute_trade_price(self.property_specs, prop_idx)
                return {
                    'type': 'trade',
                    'property_idx': prop_idx,
                    'buyer_id': buyer_id,
                    'price': price
                }
            return {'type': 'pass'}
        else:
            return {'type': 'pass'}

    def _action_to_label(self, action: int) -> str:
        """Convert a discrete action index to a human-readable label."""
        decoded = self._decode_action(action)
        action_type = decoded['type']
        if action_type == 'roll':
            return "Roll Dice"
        elif action_type == 'buy':
            idx = decoded.get('property_idx')
            name = self.property_specs[idx].name if idx is not None else "?"
            return f"Buy {name}"
        elif action_type == 'pass':
            return "Pass"
        elif action_type == 'build':
            return f"Build on {self.property_specs[decoded['property_idx']].name}"
        elif action_type == 'mortgage':
            return f"Mortgage {self.property_specs[decoded['property_idx']].name}"
        elif action_type == 'unmortgage':
            return f"Unmortgage {self.property_specs[decoded['property_idx']].name}"
        elif action_type == 'pay_fine':
            return "Pay Jail Fine ($50)"
        elif action_type == 'use_jail_card':
            return "Use Get Out of Jail Card"
        elif action_type == 'end_turn':
            return "End Turn"
        elif action_type == 'trade':
            return f"Sell {self.property_specs[decoded['property_idx']].name} to P{decoded['buyer_id']}"
        return f"Action {action}"

    def _get_observation(self) -> Dict[str, Any]:
        """Generate observation from current game state."""
        # Use cached net worth (refreshed at start of step)
        net_worths = self._net_worth_cache if self._net_worth_cache is not None else self._refresh_net_worth_cache()
        
        obs = {
            'player_id': np.array([self.agent_player_id], dtype=np.int32),
            'cash': np.array([p.cash for p in self.state.players], dtype=np.float32),
            'positions': np.array([p.position for p in self.state.players], dtype=np.int32),
            'property_owner': np.array([
                p.owner if p.owner is not None else -1 
                for p in self.state.properties
            ], dtype=np.int32),
            'houses': np.array([p.houses_count for p in self.state.properties], dtype=np.int32),
            'mortgaged': np.array([int(p.mortgaged) for p in self.state.properties], dtype=np.int32),
            'jail_turns': np.array([p.jail_turns for p in self.state.players], dtype=np.int32),
            'get_out_cards': np.array([p.get_out_of_jail_cards for p in self.state.players], dtype=np.int32),
            'legal_mask': self._get_legal_mask(),
            'turn_number': np.array([self.state.turn_number], dtype=np.int32),
            'last_roll': np.array(self.state.last_roll if self.state.last_roll else [1, 1], dtype=np.int32),
            'bank_houses': np.array([self.state.bank_houses_left], dtype=np.int32),
            'bank_hotels': np.array([self.state.bank_hotels_left], dtype=np.int32),
            'net_worth': net_worths,
        }
        return obs

    def _get_legal_mask(self) -> np.ndarray:
        """Compute binary mask of legal actions based on current game state.
        
        This mask is critical for:
        1. Training with action masking (DDQN-Hybrid)
        2. Observation (included in obs dict)
        3. Invalid action correction
        
        Returns:
            Binary array of shape (n_actions,) where 1 = legal, 0 = illegal
        """
        mask = np.zeros(self.n_actions, dtype=np.int32)
        player = self.state.players[self.agent_player_id]
        
        # If player is bankrupt, no actions
        if player.status == PlayerStatus.BANKRUPT:
            return mask
        
        # Handle jail situation
        if player.jail_turns > 0:
            if not self.state.has_rolled:
                mask[0] = 1  # roll (try for doubles)
                if player.cash >= 50:
                    mask[87] = 1  # pay_fine (get out and can roll next)
                if player.get_out_of_jail_cards > 0:
                    mask[88] = 1  # use_jail_card (get out and can roll next)
            else:
                # Already rolled (and didn't get doubles), must end turn
                mask[89] = 1  # end_turn
            return mask
        
        # Normal turn flow
        if not self.state.has_rolled:
            # Must roll first
            mask[0] = 1  # roll
            return mask
        
        # After rolling - check if awaiting buy decision
        if self.state.awaiting_buy_decision:
            pos = player.position
            tile = self.board.get_tile(pos)
            if tile.property_idx is not None:
                prop = self.state.properties[tile.property_idx]
                spec = self.property_specs[tile.property_idx]
                if prop.owner is None and player.cash >= spec.price:
                    mask[1] = 1  # buy
            mask[2] = 1  # pass (decline to buy)
            return mask
        
        # Post-roll phase - can build/mortgage/unmortgage/trade and must end turn
        
        # Building houses (on monopolies)
        for prop_idx in player.properties_owned:
            if self.rules_engine._has_monopoly(self.state, prop_idx, self.agent_player_id):
                spec = self.property_specs[prop_idx]
                prop = self.state.properties[prop_idx]
                if (prop.houses_count < 5 and 
                    player.cash >= spec.house_cost and 
                    not prop.mortgaged and
                    self.state.bank_houses_left > 0):
                    mask[3 + prop_idx] = 1  # build
        
        # Mortgage actions
        for prop_idx in player.properties_owned:
            prop = self.state.properties[prop_idx]
            if not prop.mortgaged and prop.houses_count == 0:
                mask[31 + prop_idx] = 1  # mortgage
            elif prop.mortgaged:
                spec = self.property_specs[prop_idx]
                unmortgage_cost = int(spec.mortgage_value * 1.1)
                if player.cash >= unmortgage_cost:
                    mask[59 + prop_idx] = 1  # unmortgage
        
        # Trade actions: sell property to opponent for fixed price
        # Only available for unimproved, unmortgaged properties
        for prop_idx in player.properties_owned:
            prop = self.state.properties[prop_idx]
            if not prop.mortgaged and prop.houses_count == 0:
                price = compute_trade_price(self.property_specs, prop_idx)
                # Check each opponent
                for opponent_id in range(self.num_players):
                    if opponent_id == self.agent_player_id:
                        continue
                    opponent = self.state.players[opponent_id]
                    if opponent.status == PlayerStatus.ACTIVE and opponent.cash >= price:
                        # Compute action index for this trade
                        action_idx = encode_trade_action(
                            prop_idx, opponent_id, 
                            self.agent_player_id, self.num_players
                        )
                        if action_idx < self.n_actions:
                            mask[action_idx] = 1
        
        # End turn (always legal after rolling and resolving buy decision)
        mask[89] = 1
        
        return mask

    def _simulate_opponent_turn(self):
        """Simulate one complete turn for an opponent using their policy."""
        if self._is_game_over():
            return
        
        current_player = self.state.current_player
        if current_player == self.agent_player_id:
            return
        
        # Check if current player is bankrupt
        if self.state.players[current_player].status == PlayerStatus.BANKRUPT:
            # Skip to next player
            self.state.has_rolled = False
            self.state.awaiting_buy_decision = False
            self.state.current_player = (self.state.current_player + 1) % self.num_players
            return
        
        # Find the agent for the current player
        agent = next((a for a in self.opponent_policies if a.player_id == current_player), None)
        
        if agent:
            # Full turn simulation loop for opponent
            turn_ended = False
            steps = 0
            max_steps = 30  # Prevent infinite loops
            
            while not turn_ended and steps < max_steps and not self._is_game_over():
                legal_actions = self.rules_engine.legal_actions(self.state, current_player)
                if not legal_actions:
                    # No legal actions - force end turn
                    self.state, _, _, _ = self.rules_engine.apply_action(
                        self.state, {'type': 'end_turn'}, self.engine.rng, engine=self.engine
                    )
                    turn_ended = True
                    break
                
                # Agent selects action
                action = agent.select_action(self.state, legal_actions)
                
                # Apply action through rules engine
                self.state, _, done, _ = self.rules_engine.apply_action(
                    self.state, action, self.engine.rng, engine=self.engine
                )
                
                if action['type'] == 'end_turn':
                    turn_ended = True
                
                if done:
                    turn_ended = True
                
                steps += 1
                
            # Force end turn if loop stuck
            if not turn_ended and not self._is_game_over():
                self.state.has_rolled = False
                self.state.awaiting_buy_decision = False
                self.state.last_roll = None
                self.state.doubles_count = 0
                self.state.current_player = (self.state.current_player + 1) % self.num_players
                self.state.turn_number += 1
        else:
            # Fallback if no agent found - skip turn
            self.state.has_rolled = False
            self.state.awaiting_buy_decision = False
            self.state.last_roll = None
            self.state.current_player = (self.state.current_player + 1) % self.num_players
            self.state.turn_number += 1


    def _is_game_over(self) -> bool:
        """Check if game has ended."""
        active_players = [p for p in self.state.players if p.status == PlayerStatus.ACTIVE]
        return len(active_players) <= 1

    def _get_info(self) -> Dict[str, Any]:
        """Get additional info dict."""
        agent_player = self.state.players[self.agent_player_id]
        
        # Use cached net worths
        if self._net_worth_cache is None:
            self._refresh_net_worth_cache()
        net_worths = {i: float(self._net_worth_cache[i]) for i in range(self.num_players)}
        
        return {
            'turn_number': self.state.turn_number,
            'episode_step': self._episode_step,
            'current_player': self.state.current_player,
            'agent_cash': agent_player.cash,
            'agent_properties': len(agent_player.properties_owned),
            'agent_status': agent_player.status.value,
            'agent_net_worth': net_worths[self.agent_player_id],
            'all_net_worths': net_worths,
            'active_players': sum(1 for p in self.state.players if p.status == PlayerStatus.ACTIVE),
            'has_rolled': self.state.has_rolled,
            'awaiting_buy_decision': self.state.awaiting_buy_decision,
            'reward_components': self._last_reward_components,
        }

    def save_rollout(self, path: str) -> None:
        """Save the recorded rollout to a gzip-compressed pickle file."""
        if self._rollout_recorder is None:
            raise RuntimeError("Rollout recording not enabled (set record_rollout=True)")
        from pathlib import Path as _Path
        _save_rollout(self._rollout_recorder.finalize(), _Path(path))

    @property
    def current_rollout(self):
        """Return the current rollout (finalized copy)."""
        if self._rollout_recorder is None:
            return None
        return self._rollout_recorder.finalize()

    def render(self):
        """Render the current game state."""
        if self.render_mode == 'human':
            # Frame skip
            self._render_counter += 1
            if self._render_counter % self.render_every_n != 0:
                return
            # Lazy init
            if self.renderer is None:
                self.renderer = MonopolyRenderer(self.board, self.property_specs)
            if self.state:
                overlay = {
                    'net_worths': [float(self._net_worth_cache[i]) for i in range(self.num_players)]
                        if self._net_worth_cache is not None else None,
                    'action_label': self._last_action_label,
                    'net_worth_history': self._net_worth_history,
                    'turn_number': self.state.turn_number,
                    'total_turns': self.max_turns,
                }
                self.renderer.render(self.state, show_stats=True, overlay_data=overlay)
        elif self.render_mode == 'ansi':
            # Text-based rendering with net worth
            print(f"\n{'='*60}")
            print(f"Turn {self.state.turn_number} | Current Player: {self.state.current_player}")
            print(f"{'='*60}")
            for i, player in enumerate(self.state.players):
                marker = "🤖" if i == self.agent_player_id else "🎮"
                status = "💀" if player.status == PlayerStatus.BANKRUPT else "✓"
                nw = self._get_cached_net_worth(i)
                print(f"{marker} Player {i}: ${player.cash} | NW: ${nw:.0f} | Pos: {player.position} | "
                      f"Props: {len(player.properties_owned)} | Jail: {player.jail_turns} | {status}")
            print(f"{'='*60}\n")

    def close(self):
        """Clean up resources."""
        if self.renderer:
            self.renderer.close()
            self.renderer = None