from dataclasses import dataclass, field
from typing import List, Optional, Dict, Set
from enum import Enum

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

@dataclass
class PropertyState:
    owner: Optional[int]  # player id or None
    houses_count: int = 0  # 0-4, 5=hotel
    mortgaged: bool = False

@dataclass
class DeckState:
    pointer: int  # current card index
    seed: int  # for reproducibility

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