from dataclasses import dataclass
from typing import List, Dict, Set, Sequence, Tuple, Any, Optional

# Forward reference for PropertySpec
class PropertySpec:
    pass

@dataclass
class PlayerState:
    id: int
    name: str
    cash: int
    position: int
    properties: List[int]  # or Set[int]
    houses: Dict[int, int]  # property_idx -> houses
    mortgaged: Set[int]
    in_jail: bool
    jail_turns: int
    get_out_cards: int
    active: bool

    def to_dict(self) -> Dict:
        pass

    @classmethod
    def from_dict(cls, d: Dict) -> 'PlayerState':
        pass

    def net_worth(self, property_specs: Sequence[PropertySpec]) -> int:
        pass

@dataclass
class PropertyState:
    idx: int
    owner: Optional[int]  # player id or None
    houses: int
    mortgaged: bool

    def to_dict(self) -> Dict:
        pass

    @classmethod
    def from_dict(cls, d: Dict) -> 'PropertyState':
        pass

@dataclass
class GameState:
    players: List[PlayerState]
    properties: List[PropertyState]
    bank_houses: int
    bank_hotels: int
    current_player: int
    last_roll: Optional[Tuple[int, int]]
    doubles_count: int
    turn_number: int
    rng_state: Any  # portable RNG state or seed
    history: List[Dict]  # optional action/state log

    def copy(self) -> 'GameState':
        pass

    def to_dict(self) -> Dict:
        pass

    @classmethod
    def from_dict(cls, d: Dict) -> 'GameState':
        pass

    def snapshot(self) -> str:
        pass