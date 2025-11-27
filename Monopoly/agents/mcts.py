from typing import List, Dict, Any, Optional
from ..state import GameState, PlayerStatus
from .agent import Agent
import random

class MCTSAgent(Agent):
    """
    MCTS Agent that uses a forward model (cloned engine) to simulate rollouts.
    Currently implements Flat Monte Carlo Search (1-ply lookahead with rollouts).
    """
    
    def __init__(self, player_id: int, engine: Any, rollouts: int = 20, max_depth: int = 10):
        super().__init__(player_id)
        self.engine = engine # Reference to the main game engine (to be cloned)
        self.rollouts = rollouts
        self.max_depth = max_depth
        # Store property specs for evaluation
        self.property_specs = engine.property_specs if hasattr(engine, 'property_specs') else None

    def select_action(self, state: GameState, legal_actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not legal_actions:
            return {'type': 'pass'}
        if len(legal_actions) == 1:
            return legal_actions[0]

        action_scores = []
        
        for action in legal_actions:
            total_score = 0
            for _ in range(self.rollouts):
                try:
                    # Clone state and engine
                    sim_state = state.clone()
                    sim_engine = self.engine.clone()
                    
                    # Apply the candidate action (pass engine for card effects)
                    sim_state, reward, done, _ = sim_engine.rules_engine.apply_action(
                        sim_state, action, sim_engine.rng, engine=sim_engine
                    )
                    
                    # Rollout
                    depth = 0
                    cumulative_reward = reward
                    
                    while not done and depth < self.max_depth:
                        current_player = sim_state.current_player
                        acts = sim_engine.rules_engine.legal_actions(sim_state, current_player)
                        if not acts:
                            break
                        
                        # Random policy for rollout
                        act = random.choice(acts)
                        sim_state, r, done, _ = sim_engine.rules_engine.apply_action(
                            sim_state, act, sim_engine.rng, engine=sim_engine
                        )
                        
                        if current_player == self.player_id:
                            cumulative_reward += r
                        
                        depth += 1
                    
                    # Final evaluation
                    final_value = self._evaluate_state(sim_state)
                    total_score += cumulative_reward + final_value
                except Exception:
                    # If simulation fails, use neutral score
                    total_score += 0
            
            avg_score = total_score / self.rollouts
            action_scores.append((avg_score, action))
        
        # Pick best action
        action_scores.sort(key=lambda x: x[0], reverse=True)
        return action_scores[0][1]

    def _evaluate_state(self, state: GameState) -> float:
        # Simple heuristic: Cash + Asset Value
        p = state.players[self.player_id]
        if p.status != PlayerStatus.ACTIVE:
            return -10000.0
            
        asset_value = p.cash
        
        if self.property_specs:
            for prop_idx in p.properties_owned:
                spec = self.property_specs[prop_idx]
                asset_value += spec.mortgage_value
                # Add house values
                houses = state.properties[prop_idx].houses_count
                asset_value += houses * spec.house_cost * 0.5 # Sell back value
            
        return float(asset_value)
