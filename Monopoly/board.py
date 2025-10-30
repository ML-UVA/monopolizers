from dataclasses import dataclass
from typing import List, Optional
from enum import Enum

class TileKind(Enum):
    GO = 'GO'
    PROPERTY = 'PROPERTY'
    RAILROAD = 'RAILROAD'
    UTILITY = 'UTILITY'
    TAX = 'TAX'
    CHANCE = 'CHANCE'
    COMMUNITY = 'COMMUNITY'
    JAIL = 'JAIL'
    GO_TO_JAIL = 'GO_TO_JAIL'
    FREE_PARKING = 'FREE_PARKING'

@dataclass
class TileSpec:
    idx: int
    kind: str  # or TileKind
    name: str
    property_idx: Optional[int]  # index into properties list if applicable

    def is_property(self) -> bool:
        pass

    def is_tax(self) -> bool:
        pass

    # etc. (simple bool helpers)

class Board:
    tiles: List[TileSpec]
    board_size: int

    def get_tile(self, idx: int) -> TileSpec:
        pass

    def next_tile(self, position: int, steps: int) -> int:
        pass

    @classmethod
    def load_standard_board(cls) -> 'Board':
        pass

    def property_index_for_board_idx(self, board_idx: int) -> Optional[int]:
        pass