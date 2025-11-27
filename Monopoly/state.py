from dataclasses import dataclass, field
from typing import List, Optional, Dict, Set
from enum import Enum
import copy

class PlayerStatus(Enum):
    ACTIVE = 'ACTIVE'
    BANKRUPT = 'BANKRUPT'

@dataclass
class PlayerState:
    id: int
    cash: int
    position: int
    properties_owned: Set[int]  # set of property indices
    houses_on_property: Dict[int, int]  # property_idx -> houses_count (0-4, 5=hotel)
    mortgaged_properties: Set[int]  # set of mortgaged property indices
    jail_turns: int  # 0 if not in jail
    get_out_of_jail_cards: int
    status: PlayerStatus = PlayerStatus.ACTIVE

    def clone(self) -> 'PlayerState':
        return PlayerState(
            id=self.id,
            cash=self.cash,
            position=self.position,
            properties_owned=self.properties_owned.copy(),
            houses_on_property=self.houses_on_property.copy(),
            mortgaged_properties=self.mortgaged_properties.copy(),
            jail_turns=self.jail_turns,
            get_out_of_jail_cards=self.get_out_of_jail_cards,
            status=self.status
        )

@dataclass
class PropertyState:
    owner: Optional[int]  # player id or None
    houses_count: int = 0  # 0-4, 5=hotel
    mortgaged: bool = False

    def clone(self) -> 'PropertyState':
        return PropertyState(
            owner=self.owner,
            houses_count=self.houses_count,
            mortgaged=self.mortgaged
        )

@dataclass
class DeckState:
    pointer: int  # current card index
    seed: int  # for reproducibility

    def clone(self) -> 'DeckState':
        return DeckState(pointer=self.pointer, seed=self.seed)

@dataclass
class GameState:
    players: List[PlayerState]
    properties: List[PropertyState]  # list of property states, indexed by property_idx
    chance_deck: DeckState
    community_deck: DeckState
    bank_houses_left: int = 32
    bank_hotels_left: int = 12
    current_player: int = 0
    last_roll: Optional[tuple[int, int]] = None
    doubles_count: int = 0
    turn_number: int = 0
    seed: int = 0  # global seed for reproducibility
    monopoly_status: Dict[str, Optional[int]] = field(default_factory=dict) # Group -> Owner ID

    def clone(self) -> 'GameState':
        return GameState(
            players=[p.clone() for p in self.players],
            properties=[p.clone() for p in self.properties],
            chance_deck=self.chance_deck.clone(),
            community_deck=self.community_deck.clone(),
            bank_houses_left=self.bank_houses_left,
            bank_hotels_left=self.bank_hotels_left,
            current_player=self.current_player,
            last_roll=self.last_roll,
            doubles_count=self.doubles_count,
            turn_number=self.turn_number,
            seed=self.seed,
            monopoly_status=self.monopoly_status.copy()
        )