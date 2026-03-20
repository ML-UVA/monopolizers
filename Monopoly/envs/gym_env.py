"""Gymnasium wrapper for the Monopoly game engine.

This environment provides a single-agent interface where one player is controlled
by an RL agent and opponents follow fixed policies (e.g., random, greedy).
"""

from typing import Callable, Optional, Tuple, Dict, Any, List
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from ..engine import GameEngine, EngineConfig
from ..rules import RulesEngine, RulesConfig, ActionType
from ..state import GameState, PlayerState, PropertyState, DeckState, PlayerStatus, TurnPhase
from ..board import Board
from ..property import load_property_specs
from ..cards import load_chance_cards, load_community_cards
from .renderer import MonopolyRenderer



class MonopolyEnv(gym.Env):
    """
    Monopoly Gymnasium Environment
    
    Action Space: Discrete(n_actions) where actions are:
        0: Roll dice (start turn)
        1: Buy property (if landed on unowned)
        2: Pass on buying
        3-30: Build house on property idx (3 + property_idx)
        31-58: Mortgage property idx (31 + property_idx)
        59-86: Unmortgage property idx (59 + property_idx)
        87: Pay jail fine
        88: Use get-out-of-jail card
        89: End turn / Pass
        TODO: Update with added actions
        
    Observation Space: Dict with:
        - player_id: Box (agent's player id)
        - cash: Box (n_players,) - cash for each player
        - positions: Box (n_players,) - board position for each player
        - property_owner: Box (28,) - owner id or -1 for unowned
        - houses: Box (28,) - house count (0-5, 5=hotel)
        - mortgaged: Box (28,) - binary mortgage status
        - jail_turns: Box (n_players,) - turns in jail
        - get_out_cards: Box (n_players,) - jail cards held
        - legal_mask: Box (n_actions,) - binary legal action mask
        - turn_number: Box - current turn
        - last_roll: Box (2,) - last dice roll
    """
    
    metadata = {'render_modes': ['human', 'ansi'], 'render_fps': 1}
    
    def __init__(
        self,
        num_players: int = 4,
        agent_player_id: int = 0,
        opponent_policies: Optional[List[Callable]] = None,
        max_turns: int = 1000,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
        training_stage: int = 1,
        rules_config: RulesConfig = None,
        engine_config: EngineConfig = None,
    ):
        super().__init__()

        if num_players != 4:
            raise ValueError(
                f"MonopolyEnv requires exactly 4 players, got {num_players}."
            )
        
        self.num_players = num_players
        self.agent_player_id = agent_player_id
        self.max_turns = max_turns
        self.render_mode = render_mode
        self._seed = seed or 42
        self.turn_phase = TurnPhase.ROLL
        self._rolled_doubles = False
        
        # Initialize game components
        self.board = Board.load_standard_board()
        self.property_specs = load_property_specs()
        self.chance_cards = load_chance_cards()
        self.community_cards = load_community_cards()
        self.rules_engine = RulesEngine(
            self.board, 
            self.property_specs, 
            self.chance_cards, 
            self.community_cards,
            config=rules_config if rules_config is not None else RulesConfig()
        )
        
        self.engine_config = engine_config or EngineConfig()
        
        # Opponent policies (default to random if not provided)
        self.opponent_policies = opponent_policies or []
        
        # Fill missing opponents with RandomAgent and assign player IDs
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
                # Add default RandomAgent
                from ..agents.random import RandomAgent
                self.opponent_policies.append(RandomAgent(player_id=i))
            current_opp_idx += 1

        
        # Action space: simplified discrete actions
        # 0: roll
        # 1: buy
        # 2: pass
        # 3-30: Build house on property (action = 3 + property_idx)
        # 31-58: Mortgage house on property (action = 31 + property_idx)
        # 59-86: Unmortgage house on property (action = 59 + property_idx) 
        # 87: pay jail
        # 88: use jail card
        # 89: end turn
        # 90-117: Sell house on property (action = 90 + property_idx)
        # 118: Auction pass (don't bid)
        # 119-128: Bid amounts (10, 20, 50, 100, 150, 200, 250, 300, 400, 500)
        # 129: Initiate trade 
        # 130-157: Toggle property in offer (130 + property_idx)
        # 158-185: Toggle property in ask (158 + property_idx)
        # 186-188: Offer 50, 100, 200 cash
        # 189-191: Ask 50, 100, 200 cash
        # 192: Select player 1 as trade target
        # 193: Select player 2 as trade target
        # 194: Select player 3 as trade target
        # 195: Confirm and send trade proposal
        # 196: Accept incoming trade
        # 197: Decline incoming trade 
        self.n_actions = 198
        self.action_space = spaces.Discrete(self.n_actions)
        self.training_stage = training_stage
        
        # Observation space
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
            'auction_active': spaces.Box(0, 1, shape=(1,), dtype=np.int32),
            'auction_current_bid': spaces.Box(0, 100000, shape=(1,), dtype=np.float32),
            'trade_phase': spaces.Box(0, 1, shape=(1, ), dtype=np.int32),
            'trade_offer_properties': spaces.Box(0, 1, shape=(28,), dtype=np.int32),
            'trade_ask_properties': spaces.Box(0, 1, shape=(28,), dtype=np.int32),
            'trade_offer_cash': spaces.Box(0, 100000, shape=(1,), dtype=np.float32),
            'trade_ask_cash': spaces.Box(0, 100000, shape=(1,), dtype=np.float32),
            'trade_target': spaces.Box(-1, num_players-1, shape=(1,), dtype=np.int32),
            'pending_trade_offer_properties': spaces.Box(0, 1, shape=(28,), dtype=np.int32),
            'pending_trade_ask_properties': spaces.Box(0, 1, shape=(28,), dtype=np.int32),
            'pending_trade_offer_cash': spaces.Box(0, 100000, shape=(1,), dtype=np.float32),
            'pending_trade_ask_cash': spaces.Box(0, 100000, shape=(1,), dtype=np.float32),
            'pending_trade_proposer': spaces.Box(-1, num_players-1, shape=(1,), dtype=np.int32),
            'turn_phase': spaces.Box(0, 1, shape=(1,), dtype=np.int32),
        })
        
        # Game state
        self.state: Optional[GameState] = None
        self.engine: Optional[GameEngine] = None
        self._episode_step = 0
        
        # Pygame renderer
        self.renderer: Optional[MonopolyRenderer] = None
        if self.render_mode == 'human':
            self.renderer = MonopolyRenderer(self.board, self.property_specs)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        if seed is not None:
            self._seed = seed
        
        # Create initial game state
        players = []
        for i in range(self.num_players):
            players.append(PlayerState(
                id=i,
                cash=self.rules_engine.config.starting_cash,
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
            bank_houses_left=self.rules_engine.config.bank_houses,
            bank_hotels_left=self.rules_engine.config.bank_hotels,
            current_player=0,
            last_roll=None,
            doubles_count=0,
            turn_number=0,
            seed=self._seed
        )

        groups = set(spec.group for spec in self.property_specs)
        self.state.monopoly_status = {group: None for group in groups}
        
        self.engine = GameEngine(self.rules_engine, config=self.engine_config, seed=self._seed)
        self._episode_step = 0
        self.turn_phase = TurnPhase.ROLL
        self._rolled_doubles = False
        
        # If agent is not first player, simulate opponent turns until agent's turn
        while self.state.current_player != self.agent_player_id:
            self._simulate_opponent_turn()
        
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment."""
        if self.state is None:
            raise RuntimeError("Must call reset() before step()")
        
        # Ensure it's the agent's turn
        if self.state.current_player != self.agent_player_id:
            raise RuntimeError(f"Not agent's turn (current: {self.state.current_player}, agent: {self.agent_player_id})")
        
        # Convert discrete action to game action
        game_action = self._decode_action(action)
        
        # Apply action through rules engine
        reward = 0.0
        try:
            old_cash = self.state.players[self.agent_player_id].cash
            old_properties = len(self.state.players[self.agent_player_id].properties_owned)
            
            # Execute the action
            if game_action['type'] == ActionType.ROLL.value:
                # Roll dice and move, but don't advance current_player
                roll = self.rules_engine.roll_dice(self.engine.rng)
                self.state.last_roll = roll

                self.state = self.rules_engine.handle_doubles_and_jail(
                    self.state, self.agent_player_id, roll
                )
                player_after_roll = self.state.players[self.agent_player_id]

                # jail_turns > 0 after roll means either:
                # - player was in jail and failed to roll doubles (jail_turns incremented)
                # - player was sent to jail by triple doubles (jail_turns set to 1)
                # In both cases there is nothing left to decide this turn
                if player_after_roll.jail_turns > 0:
                    # Failed jail roll - nothing to decide, auto advance
                    self.state.current_player = (self.state.current_player + 1) % self.num_players
                    self.state.turn_number += 1
                    self.state.last_roll = None
                    self.turn_phase = TurnPhase.ROLL
                    while (self.state.current_player != self.agent_player_id
                        and not self._is_game_over()):
                        self._simulate_opponent_turn()
                else:
                    steps = sum(roll)
                    self.state = self.rules_engine.move_player(
                        self.state, self.agent_player_id, steps
                    )
                    self.state = self.rules_engine.handle_landing(
                        self.state, self.agent_player_id, self.engine.rng, engine=self.engine
                    )
                    # Track whether doubles were rolled
                    self._rolled_doubles = (
                        roll[0] == roll[1] and
                        self.state.players[self.agent_player_id].jail_turns == 0 and
                        self.state.doubles_count > 0
                    )
                    # Stay in post_roll phase - agent still needs to act
                    self.turn_phase = TurnPhase.POST_ROLL
            elif game_action['type'] == ActionType.END_TURN.value:
                if self._rolled_doubles:
                    self._rolled_doubles = False
                    self.state.last_roll = None
                    self.turn_phase = TurnPhase.ROLL
                else:
                    # Agent explicitly ends their turn, advance to next player
                    self.state.current_player = (self.state.current_player + 1) % self.num_players
                    self.state.turn_number += 1
                    self.state.last_roll = None
                    self.turn_phase = TurnPhase.ROLL

                    # Simulate opponent turns until it's the agent's turn again
                    while (self.state.current_player != self.agent_player_id
                        and not self._is_game_over()):
                        self._simulate_opponent_turn()
            else:
                self.state, action_reward, _, log = self.rules_engine.apply_action(
                    self.state, game_action, self.engine.rng
                )
                reward += action_reward
            
            # Calculate shaped reward
            new_cash = self.state.players[self.agent_player_id].cash
            new_properties = len(self.state.players[self.agent_player_id].properties_owned)
            reward += (new_cash - old_cash) / 1000.0  # Normalize cash changes
            # reward += (new_properties - old_properties) * 0.5  # Reward property acquisition
            
        except Exception as e:
            # If action fails, give negative reward and end episode
            reward = -10.0
            print(f"Action failed: {e}")
        
        self._episode_step += 1
        
        # Check termination conditions
        terminated = self._is_game_over()
        truncated = self._episode_step >= self.max_turns
        
        # Final reward if game ends
        if terminated:
            if self.state.players[self.agent_player_id].status == PlayerStatus.ACTIVE:
                # Agent won or is last standing
                active_count = sum(1 for p in self.state.players if p.status == PlayerStatus.ACTIVE)
                if active_count == 1:
                    reward += 100.0  # Win bonus
            else:
                reward -= 50.0  # Bankruptcy penalty
        
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, reward, terminated, truncated, info

    def _decode_action(self, action: int) -> Dict[str, Any]:
        """Convert discrete action index to game action dict."""
        if action == 0:
            return {'type': ActionType.ROLL.value}
        elif action == 1:
            pos = self.state.players[self.agent_player_id].position
            tile = self.board.get_tile(pos)
            return {'type': ActionType.BUY.value, 'property_idx': tile.property_idx}
        elif action == 2:
            return {'type': ActionType.PASS.value}
        elif 3 <= action <= 30:
            return {'type': ActionType.BUILD.value, 'property_idx': action - 3}
        elif 31 <= action <= 58:
            return {'type': ActionType.MORTGAGE.value, 'property_idx': action - 31}
        elif 59 <= action <= 86:
            return {'type': ActionType.UNMORTGAGE.value, 'property_idx': action - 59}
        elif action == 87:
            return {'type': ActionType.PAY_FINE.value}
        elif action == 88:
            return {'type': ActionType.USE_JAIL_CARD.value}
        elif action == 89:
            return {'type': ActionType.END_TURN.value}
        elif 90 <= action <= 117:
            return {'type': ActionType.SELL.value, 'property_idx': action - 90}
        elif action == 118:
            return {'type': ActionType.AUCTION_PASS.value}
        elif 119 <= action <= 128:
            bid_amounts = [10, 20, 50, 100, 150, 200, 250, 300, 400, 500]
            return {'type': ActionType.AUCTION_BID.value, 'amount': bid_amounts[action - 119]}
        elif action == 129:
            return {'type': ActionType.INITIATE_TRADE.value}
        elif 130 <= action <= 157:
            return {'type': ActionType.TOGGLE_OFFER_PROPERTY.value, 'property_idx': action - 130}
        elif 158 <= action <= 185:
            return {'type': ActionType.TOGGLE_ASK_PROPERTY.value, 'property_idx': action - 158}
        elif 186 <= action <= 188:
            return {'type': ActionType.OFFER_CASH.value, 'amount': [50, 100, 200][action - 186]}
        elif 189 <= action <= 191:
            return {'type': ActionType.ASK_CASH.value, 'amount': [50, 100, 200][action - 189]}
        elif 192 <= action <= 194:
            opponent_idx = action - 192
            opponents = [pid for pid in range(self.num_players) if pid != self.agent_player_id]
            if opponent_idx < len(opponents):
                return {'type': ActionType.SELECT_TRADE_TARGET.value, 'target_player_id': opponents[opponent_idx]}
            return {'type': ActionType.PASS.value}
        elif action == 195:
            return {'type': ActionType.CONFIRM_TRADE.value}
        elif action == 196:
            return {'type': ActionType.ACCEPT_TRADE.value}
        elif action == 197:
            return {'type': ActionType.DECLINE_TRADE.value}
        else:
            return {'type': ActionType.PASS.value}

    def _get_observation(self) -> Dict[str, Any]:
        """Generate observation from current game state."""
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
            'auction_active': np.array([int(self.state.auction_active)], dtype=np.int32),
            'auction_current_bid': np.array([self.state.auction_current_bid], dtype=np.float32),
            'trade_phase': np.array([int(self.state.trade_phase)], dtype=np.int32),
            'trade_offer_properties': np.array([
                1 if i in self.state.trade_offer_properties else 0
                for i in range(28)
            ], dtype=np.int32),
            'trade_ask_properties': np.array([
                1 if i in self.state.trade_ask_properties else 0
                for i in range(28)
            ], dtype=np.int32),
            'trade_offer_cash': np.array([float(self.state.trade_offer_cash)], dtype=np.float32),
            'trade_ask_cash': np.array([float(self.state.trade_ask_cash)], dtype=np.float32),
            'trade_target': np.array(
                [self.state.trade_target if self.state.trade_target is not None else -1], 
                dtype=np.int32
            ),
            'turn_phase': np.array(
                [1 if self.turn_phase == TurnPhase.POST_ROLL else 0],
                dtype=np.int32
            ),
        }

        if self.state.pending_trade is not None:
            pt = self.state.pending_trade
            offer_props = np.zeros(28, dtype=np.int32)
            ask_props = np.zeros(28, dtype=np.int32)
            for idx in pt.offer.get('properties', []):
                offer_props[idx] = 1
            for idx in pt.ask.get('properties', []):
                ask_props[idx] = 1
            obs['pending_trade_offer_properties'] = offer_props
            obs['pending_trade_ask_properties'] = ask_props
            obs['pending_trade_offer_cash'] = np.array([pt.offer.get('cash', 0)], dtype=np.float32)
            obs['pending_trade_ask_cash'] = np.array([pt.ask.get('cash', 0)], dtype=np.float32)
            obs['pending_trade_proposer'] = np.array([pt.proposer], dtype=np.int32)
        else:
            obs['pending_trade_offer_properties'] = np.zeros(28, dtype=np.int32)
            obs['pending_trade_ask_properties'] = np.zeros(28, dtype=np.int32)
            obs['pending_trade_offer_cash'] = np.array([0.0], dtype=np.float32)
            obs['pending_trade_ask_cash'] = np.array([0.0], dtype=np.float32)
            obs['pending_trade_proposer'] = np.array([-1], dtype=np.int32)

        return obs

    def _get_legal_mask(self) -> np.ndarray:
        """Compute binary mask of legal actions."""
        mask = np.zeros(self.n_actions, dtype=np.int32)
        player = self.state.players[self.agent_player_id]
        
        # Stage 1+: Always legal

        # Roll only available at start of turn
        if self.turn_phase == TurnPhase.ROLL:
            if player.jail_turns == 0:
                mask[0] = 1     # roll normally
            else:
                # In jail - can roll for doubles, pay fine, or use card
                mask[0] = 1
                if player.cash >= self.rules_engine.config.jail_fine:
                    mask[87] = 1
                if player.get_out_of_jail_cards > 0:
                    mask[88] = 1
            # Nothing else is legal before rolling
            return mask
        
        # post_roll phase - agent has rolled and landed somewhere
        # End turn (always legal as fallback)
        mask[89] = 1

        # Check if on buyable property
        pos = player.position
        tile = self.board.get_tile(pos)
        if tile.property_idx is not None:
            prop = self.state.properties[tile.property_idx]
            spec = self.property_specs[tile.property_idx]
            
            if prop.owner is None and player.cash >= spec.price:
                mask[1] = 1  # buy
        
        # Building houses (simplified: check monopoly and cash)
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
            # elif prop.mortgaged:
            #     spec = self.property_specs[prop_idx]
            #     unmortgage_cost = int(spec.mortgage_value * 1.1)
            #     if player.cash >= unmortgage_cost:
            #         mask[59 + prop_idx] = 1  # unmortgage

        # Stage 2+: selling houses
        # Sell house actions
        if self.training_stage >= 2:
            for prop_idx in player.properties_owned:
                prop = self.state.properties[prop_idx]
                if prop.houses_count > 0 and not prop.mortgaged:
                    mask[90 + prop_idx] = 1
        
        # Stage 3+
        # Auction actions (only when auction is active)
        if self.training_stage >= 3:
            if self.state.auction_active:
                mask[118] = 1 # pass on auction
                current_bid = self.state.auction_current_bid
                bid_amounts = [10, 20, 50, 100, 150, 200, 250, 300, 400, 500]
                for i, amount in enumerate(bid_amounts):
                    if amount > current_bid and player.cash >= amount:
                        mask[119 + i] = 1
                return mask # nothing else legal during auction

        # Stage 4+
        # Trade actions
        if self.training_stage >= 4:
            if self.state.trade_phase:
                # In trade construction phase
                for prop_idx in player.properties_owned:
                    mask[130 + prop_idx] = 1 # can toggle any owned property as offer
                
                if self.state.trade_target is not None:
                    for prop_idx in range(28):
                        if self.state.properties[prop_idx].owner == self.state.trade_target:
                            mask[158 + prop_idx] = 1
                mask[186] = mask[187] = mask[188] = 1 
                mask[189] = mask[190] = mask[191] = 1 
                
                opponents = [pid for pid in range(self.num_players) if pid != self.agent_player_id]
                for opponent_idx, pid in enumerate(opponents):
                    if self.state.players[pid].status == PlayerStatus.ACTIVE:
                        mask[192 + opponent_idx] = 1

                mask[195] = 1 # confirm
            else:
                mask[129] = 1 # can always initiate a trade on your turn
        
            # Incoming trade response
            if self.state.pending_trade is not None:
                if self.state.pending_trade.receiver == self.agent_player_id:
                    mask[196] = 1 # accept
                    mask[197] = 1 # decline
        
        return mask

    def _simulate_opponent_turn(self):
        """Simulate one turn for an opponent using their policy."""
        if self._is_game_over():
            return
        
        current_player = self.state.current_player
        if current_player == self.agent_player_id:
            return
        
        # Find the agent for the current player
        # self.opponent_policies is a list of agents. We need to find the one with player_id == current_player
        agent = next((a for a in self.opponent_policies if a.player_id == current_player), None)
        
        if agent is None:
            # Fallback if no agent found (shouldn't happen)
            self.state.current_player = (self.state.current_player + 1) % self.num_players
            self.state.turn_number += 1
            self.state.last_roll = None
            return

        # Full turn simulation loop for opponent
        turn_ended = False
        steps = 0
        max_steps = 20  # Prevent infinite loops
        
        while not turn_ended and steps < max_steps:
            legal_actions = self.rules_engine.legal_actions(self.state, current_player)
            if not legal_actions:
                turn_ended = True
                break
            
            # Agent selects action
            action = agent.select_action(self.state, legal_actions)

            # Apply action
            self.state, _, _, _ = self.rules_engine.apply_action(self.state, action, self.engine.rng, engine=self.engine)
            
            if action['type'] == ActionType.END_TURN.value:
                turn_ended = True
            
            # Check if game over during turn
            if self._is_game_over():
                turn_ended = True
            
            steps += 1
            
        # Force end turn if loop stuck
        if not turn_ended:
            self.state.current_player = (self.state.current_player + 1) % self.num_players
            self.state.turn_number += 1


    def _is_game_over(self) -> bool:
        """Check if game has ended."""
        active_players = [p for p in self.state.players if p.status == PlayerStatus.ACTIVE]
        return len(active_players) <= 1

    def _get_info(self) -> Dict[str, Any]:
        """Get additional info dict."""
        return {
            'turn_number': self.state.turn_number,
            'episode_step': self._episode_step,
            'current_player': self.state.current_player,
            'agent_cash': self.state.players[self.agent_player_id].cash,
            'agent_properties': len(self.state.players[self.agent_player_id].properties_owned),
            'active_players': sum(1 for p in self.state.players if p.status == PlayerStatus.ACTIVE),
            'turn_phase': self.turn_phase.value,
            'rolled_doubles': self._rolled_doubles,
        }

    def render(self):
        """Render the current game state."""
        if self.render_mode == 'human':
            # Use pygame renderer
            if self.renderer and self.state:
                self.renderer.render(self.state, show_stats=True)
        elif self.render_mode == 'ansi':
            # Text-based rendering
            print(f"\n{'='*60}")
            print(f"Turn {self.state.turn_number} | Current Player: {self.state.current_player}")
            print(f"{'='*60}")
            for i, player in enumerate(self.state.players):
                marker = "🤖" if i == self.agent_player_id else "🎮"
                status = "💀" if player.status == PlayerStatus.BANKRUPT else "✓"
                print(f"{marker} Player {i}: ${player.cash} | Pos: {player.position} | "
                      f"Props: {len(player.properties_owned)} | Jail: {player.jail_turns} | {status}")
            print(f"{'='*60}\n")

    def close(self):
        """Clean up resources."""
        if self.renderer:
            self.renderer.close()
            self.renderer = None
