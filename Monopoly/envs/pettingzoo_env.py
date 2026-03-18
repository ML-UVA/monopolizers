from typing import Optional, Dict, Any
import numpy as np
from gymnasium import spaces
from ..engine import GameEngine, EngineConfig
from ..rules import RulesEngine, RulesConfig, ActionType
from ..state import GameState, PlayerState, PropertyState, DeckState, PlayerStatus, TurnPhase
from ..board import Board
from ..property import load_property_specs
from ..cards import load_chance_cards, load_community_cards
from .renderer import MonopolyRenderer
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector

class PettingZooEnv(AECEnv):
    metadata = {'render_modes': ['human'], 'name': 'monopoly_v0'}

    def __init__(
            self,
            num_players: int = 4,
            training_stage: int = 1,
            render_mode: Optional[str] = None,
            rules_config: RulesConfig = None,
            engine_config: EngineConfig = None,
            seed: Optional[int] = None,
    ):
        super().__init__()

        self.agents = ['player_0', 'player_1', 'player_2', 'player_3']
        self.possible_agents = self.agents[:]

        self.engine_config = engine_config or EngineConfig()

        self.num_players = num_players
        self.max_turns = self.engine_config.max_turns
        self._seed = seed or 42
        self.turn_phase = TurnPhase.ROLL
        self._rolled_doubles = False

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

        self.n_actions = 198
        self.action_space = spaces.Discrete(self.n_actions)
        self.training_stage = training_stage

        self.state: Optional[GameState] = None
        self.engine: Optional[GameEngine] = None

        self.render_mode = render_mode
        self.renderer: Optional[MonopolyRenderer] = None
        if self.render_mode == 'human':
            self.renderer = MonopolyRenderer(self.board, self.property_specs)

        self.observation_spaces = {
            agent: spaces.Dict({
                'player_id': spaces.Box(0, num_players-1, shape=(1,), dtype=np.int32),
                'cash': spaces.Box(0, 100000, shape=(num_players,), dtype=np.float32),
                'positions': spaces.Box(0, 39, shape=(num_players,), dtype=np.int32),
                'property_owner': spaces.Box(-1, num_players-1, shape=(28,), dtype=np.int32),
                'houses': spaces.Box(0, 5, shape=(28,), dtype=np.int32),
                'mortgaged': spaces.Box(0, 1, shape=(28,), dtype=np.int32),
                'jail_turns': spaces.Box(0, 10, shape=(num_players,), dtype=np.int32),
                'get_out_cards': spaces.Box(0, 10, shape=(num_players,), dtype=np.int32),
                'legal_mask': spaces.Box(0, 1, shape=(self.n_actions,), dtype=np.int32),
                'turn_number': spaces.Box(0, self.max_turns, shape=(1,), dtype=np.int32),
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
            for agent in self.agents
        }
        self.action_spaces = {
            agent: spaces.Discrete(self.n_actions)
            for agent in self.agents
        }

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

    def reset(self, seed: int | None = None, options: dict | None = None) -> None:
        
        if seed is not None:
            self._seed = seed

        self.agents = self.possible_agents[:]
        
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

        self.engine = GameEngine(self.rules_engine, config=self.engine_config, seed=self._seed)
        
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

        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.next()

        self.turn_phase = TurnPhase.ROLL
        self._rolled_doubles = False
    
    def observe(self, agent: str) -> dict:
        player_id = int(agent.split('_')[1])
        return self._get_observation(player_id)

    def step(self, action: int) -> None:
        agent = self.agent_selection
        player_id = int(agent.split('_')[1])

        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(action)
            return
        
        self.rewards = {a: 0.0 for a in self.agents}
        
        game_action = self._decode_action(action, player_id)

        # Apply action through rules engine
        reward = 0.0
        try:
            old_cash = self.state.players[player_id].cash
            old_properties = len(self.state.players[player_id].properties_owned)
            
            # Execute the action
            if game_action['type'] == ActionType.ROLL.value:
                # Roll dice and move, but don't advance current_player
                roll = self.rules_engine.roll_dice(self.engine.rng)
                self.state.last_roll = roll

                self.state = self.rules_engine.handle_doubles_and_jail(
                    self.state, player_id, roll
                )
                player_after_roll = self.state.players[player_id]

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
                else:
                    steps = sum(roll)
                    self.state = self.rules_engine.move_player(
                        self.state, player_id, steps
                    )
                    self.state = self.rules_engine.handle_landing(
                        self.state, player_id, self.engine.rng, engine=self.engine
                    )
                    # Track whether doubles were rolled
                    self._rolled_doubles = (
                        roll[0] == roll[1] and
                        self.state.players[player_id].jail_turns == 0 and
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
            else:
                self.state, action_reward, _, log = self.rules_engine.apply_action(
                    self.state, game_action, self.engine.rng
                )
                reward += action_reward
            
            # Calculate shaped reward
            new_cash = self.state.players[player_id].cash
            new_properties = len(self.state.players[player_id].properties_owned)
            reward += (new_cash - old_cash) / 1000.0  # Normalize cash changes
            reward += (new_properties - old_properties) * 0.5  # Reward property acquisition
            
        except Exception as e:
            # If action fails, give negative reward and end episode
            reward = -10.0
            print(f"Action failed: {e}")

        self.rewards[agent] = reward

        # After applying action, check for new bankruptcies
        for i, player in enumerate(self.state.players):
            if player.status == PlayerStatus.BANKRUPT:
                a = f'player_{i}'
                if not self.terminations[a]:
                    self.terminations[a] = True
                    self.rewards[a] = -50.0

        active = [p for p in self.state.players if p.status == PlayerStatus.ACTIVE]
        if len(active) == 1:
            winner = f'player_{active[0].id}'
            self.rewards[winner] += 100.0
            for agent in self.agents:
                self.terminations[agent] = True
        
        # Advance to next agent
        current = self.state.current_player
        self.agent_selection = f'player_{current}'

        if self.state.turn_number >= self.max_turns:
            for agent in self.agents:
                self.truncations[agent] = True

        self._accumulate_rewards()

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
                marker = "🤖" if i == self.state.current_player else "🎮"
                status = "💀" if player.status == PlayerStatus.BANKRUPT else "✓"
                print(f"{marker} Player {i}: ${player.cash} | Pos: {player.position} | "
                      f"Props: {len(player.properties_owned)} | Jail: {player.jail_turns} | {status}")
            print(f"{'='*60}\n")

    def close(self):
        if self.renderer:
            self.renderer.close()
            self.renderer = None

    def _decode_action(self, action: int, player_id: int) -> Dict[str, Any]:
        """Convert discrete action index to game action dict."""
        if action == 0:
            return {'type': ActionType.ROLL.value}
        elif action == 1:
            pos = self.state.players[player_id].position
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
            opponents = [pid for pid in range(self.num_players) if pid != player_id]
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

    def _get_observation(self, player_id: int) -> Dict[str, Any]:
        """Generate observation from current game state."""
        obs = {
            'player_id': np.array([player_id], dtype=np.int32),
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
            'legal_mask': self._get_legal_mask(player_id),
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

    def _get_legal_mask(self, player_id) -> np.ndarray:
        """Compute binary mask of legal actions."""
        mask = np.zeros(self.n_actions, dtype=np.int32)
        player = self.state.players[player_id]
        
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
            if self.rules_engine._has_monopoly(self.state, prop_idx, player_id):
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
                
                opponents = [pid for pid in range(self.num_players) if pid != player_id]
                for opponent_idx, pid in enumerate(opponents):
                    if self.state.players[pid].status == PlayerStatus.ACTIVE:
                        mask[192 + opponent_idx] = 1

                mask[195] = 1 # confirm
            else:
                mask[129] = 1 # can always initiate a trade on your turn
        
            # Incoming trade response
            if self.state.pending_trade is not None:
                if self.state.pending_trade.receiver == player_id:
                    mask[196] = 1 # accept
                    mask[197] = 1 # decline
        
        return mask
