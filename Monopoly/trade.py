from dataclasses import dataclass
from typing import Dict, Tuple, Optional
from .state import GameState

@dataclass
class TradeProposal:
    proposer: int
    receiver: int
    offer: Dict[str, any]  # e.g., {'cash': 200, 'properties': [1, 3], 'jail_cards': 1}
    ask: Dict[str, any]

    def validate(self, state: GameState) -> bool:
        proposer = state.players[self.proposer]
        receiver = state.players[self.receiver]
        
        # Check offer
        if 'cash' in self.offer and proposer.cash < self.offer['cash']:
            return False
        if 'properties' in self.offer:
            for prop_idx in self.offer['properties']:
                if prop_idx not in proposer.properties_owned:
                    return False
        if 'jail_cards' in self.offer and proposer.get_out_of_jail_cards < self.offer.get('jail_cards', 0):
            return False
        
        # Check ask
        if 'cash' in self.ask and receiver.cash < self.ask['cash']:
            return False
        if 'properties' in self.ask:
            for prop_idx in self.ask['properties']:
                if prop_idx not in receiver.properties_owned:
                    return False
        if 'jail_cards' in self.ask and receiver.get_out_of_jail_cards < self.ask.get('jail_cards', 0):
            return False
        
        return True

    def execute(self, state: GameState) -> GameState:
        proposer = state.players[self.proposer]
        receiver = state.players[self.receiver]
        
        # Transfer offer from proposer to receiver
        if 'cash' in self.offer:
            proposer.cash -= self.offer['cash']
            receiver.cash += self.offer['cash']
        if 'properties' in self.offer:
            for prop_idx in self.offer['properties']:
                proposer.properties_owned.remove(prop_idx)
                receiver.properties_owned.add(prop_idx)
                state.properties[prop_idx].owner = self.receiver
        if 'jail_cards' in self.offer:
            proposer.get_out_of_jail_cards -= self.offer['jail_cards']
            receiver.get_out_of_jail_cards += self.offer['jail_cards']
        
        # Transfer ask from receiver to proposer
        if 'cash' in self.ask:
            receiver.cash -= self.ask['cash']
            proposer.cash += self.ask['cash']
        if 'properties' in self.ask:
            for prop_idx in self.ask['properties']:
                receiver.properties_owned.remove(prop_idx)
                proposer.properties_owned.add(prop_idx)
                state.properties[prop_idx].owner = self.proposer
        if 'jail_cards' in self.ask:
            receiver.get_out_of_jail_cards -= self.ask['jail_cards']
            proposer.get_out_of_jail_cards += self.ask['jail_cards']
        
        return state

class TradeManager:
    def propose_trade(self, state: GameState, proposal: TradeProposal) -> Dict[str, any]:
        if proposal.validate(state):
            return {'valid': True, 'message': 'Trade proposal is valid'}
        else:
            return {'valid': False, 'message': 'Trade proposal is invalid'}

    def accept_trade(self, state: GameState, proposal: TradeProposal) -> GameState:
        if proposal.validate(state):
            return proposal.execute(state)
        return state

    def decline_trade(self, state: GameState, proposal: TradeProposal) -> GameState:
        # No change to state
        return state