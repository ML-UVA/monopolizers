from typing import Callable, Optional, Tuple, Dict, Any, List
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from ..engine import GameEngine
from ..rules import RulesEngine, ActionType
from ..state import GameState, PlayerState, PropertyState, DeckState, PlayerStatus
from ..board import Board
from ..property import load_property_specs
from ..cards import load_chance_cards, load_community_cards
from .renderer import MonopolyRenderer


"""Gymnasium wrapper for the Monopoly game engine.

This environment provides a single-agent interface where one player is controlled
by an RL agent and opponents follow fixed policies (e.g., random, greedy).
"""


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
        seed: Optional[int] = None
    ):
        super().__init__()
        
        self.num_players = num_players
        self.agent_player_id = agent_player_id
        self.max_turns = max_turns
        self.render_mode = render_mode
        self._seed = seed or 42
        
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
        # 0: roll, 1: buy, 2: pass, 3-30: build, 31-58: mortgage, 59-86: unmortgage, 87: pay jail, 88: use jail card, 89: end turn
        self.n_actions = 90
        self.action_space = spaces.Discrete(self.n_actions)
        
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
        
        # If agent is not first player, simulate opponent turns until agent's turn
        while self.state.current_player != self.agent_player_id and not self._is_game_over():
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
        
        # Validate action is legal
        legal_mask = self._get_legal_mask()
        if legal_mask[action] == 0:
            # Invalid action - give small penalty and force pass/end_turn
            reward = -1.0
            if self.state.has_rolled:
                game_action = {'type': 'end_turn'}
            else:
                game_action = {'type': 'roll'}
        else:
            reward = 0.0
        
        # Track state for reward shaping
        old_cash = self.state.players[self.agent_player_id].cash
        old_properties = len(self.state.players[self.agent_player_id].properties_owned)
        
        # Apply action through rules engine
        try:
            self.state, action_reward, done, log = self.rules_engine.apply_action(
                self.state, game_action, self.engine.rng, engine=self.engine
            )
            reward += action_reward
            
        except Exception as e:
            # If action fails, give negative reward
            reward = -5.0
            # Force end turn to prevent stuck state
            if self.state.has_rolled:
                self.state, _, _, _ = self.rules_engine.apply_action(
                    self.state, {'type': 'end_turn'}, self.engine.rng, engine=self.engine
                )
        
        # Calculate shaped reward
        new_cash = self.state.players[self.agent_player_id].cash
        new_properties = len(self.state.players[self.agent_player_id].properties_owned)
        reward += (new_cash - old_cash) / 1000.0  # Normalize cash changes
        reward += (new_properties - old_properties) * 0.5  # Reward property acquisition
        
        self._episode_step += 1
        
        # Simulate opponent turns only if it's no longer the agent's turn
        while (self.state.current_player != self.agent_player_id and 
               not self._is_game_over()):
            self._simulate_opponent_turn()
        
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
        else:
            return {'type': 'pass'}

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
        }
        return obs

    def _get_legal_mask(self) -> np.ndarray:
        """Compute binary mask of legal actions based on current game state."""
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
                    mask[87] = 1  # pay_fine
                if player.get_out_of_jail_cards > 0:
                    mask[88] = 1  # use_jail_card
            else:
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
        
        # Post-roll phase - can build/mortgage/unmortgage and must end turn
        
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
        return {
            'turn_number': self.state.turn_number,
            'episode_step': self._episode_step,
            'current_player': self.state.current_player,
            'agent_cash': agent_player.cash,
            'agent_properties': len(agent_player.properties_owned),
            'agent_status': agent_player.status.value,
            'active_players': sum(1 for p in self.state.players if p.status == PlayerStatus.ACTIVE),
            'has_rolled': self.state.has_rolled,
            'awaiting_buy_decision': self.state.awaiting_buy_decision,
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