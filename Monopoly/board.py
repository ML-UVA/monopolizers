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
    kind: TileKind
    name: str
    property_idx: Optional[int]  # index into properties list if applicable

    def is_property(self) -> bool:
        return self.kind == TileKind.PROPERTY

    def is_tax(self) -> bool:
        return self.kind == TileKind.TAX

    def is_railroad(self) -> bool:
        return self.kind == TileKind.RAILROAD

    def is_utility(self) -> bool:
        return self.kind == TileKind.UTILITY

    def is_chance(self) -> bool:
        return self.kind == TileKind.CHANCE

    def is_community(self) -> bool:
        return self.kind == TileKind.COMMUNITY

    def is_go_to_jail(self) -> bool:
        return self.kind == TileKind.GO_TO_JAIL

    def is_free_parking(self) -> bool:
        return self.kind == TileKind.FREE_PARKING

@dataclass
class Board:
    tiles: List[TileSpec]
    board_size: int

    def get_tile(self, idx: int) -> TileSpec:
        return self.tiles[idx]

    def next_tile(self, position: int, steps: int) -> int:
        return (position + steps) % self.board_size

    @classmethod
    def load_standard_board(cls) -> 'Board':
        tiles = [
            TileSpec(idx=0, kind=TileKind.GO, name="Go", property_idx=None),
            TileSpec(idx=1, kind=TileKind.PROPERTY, name="Mediterranean Avenue", property_idx=0),
            TileSpec(idx=2, kind=TileKind.COMMUNITY, name="Community Chest", property_idx=None),
            TileSpec(idx=3, kind=TileKind.PROPERTY, name="Baltic Avenue", property_idx=1),
            TileSpec(idx=4, kind=TileKind.TAX, name="Income Tax", property_idx=None),
            TileSpec(idx=5, kind=TileKind.RAILROAD, name="Reading Railroad", property_idx=2),
            TileSpec(idx=6, kind=TileKind.PROPERTY, name="Oriental Avenue", property_idx=3),
            TileSpec(idx=7, kind=TileKind.CHANCE, name="Chance", property_idx=None),
            TileSpec(idx=8, kind=TileKind.PROPERTY, name="Vermont Avenue", property_idx=4),
            TileSpec(idx=9, kind=TileKind.PROPERTY, name="Connecticut Avenue", property_idx=5),
            TileSpec(idx=10, kind=TileKind.JAIL, name="Jail / Just Visiting", property_idx=None),
            TileSpec(idx=11, kind=TileKind.PROPERTY, name="St. Charles Place", property_idx=6),
            TileSpec(idx=12, kind=TileKind.UTILITY, name="Electric Company", property_idx=7),
            TileSpec(idx=13, kind=TileKind.PROPERTY, name="States Avenue", property_idx=8),
            TileSpec(idx=14, kind=TileKind.PROPERTY, name="Virginia Avenue", property_idx=9),
            TileSpec(idx=15, kind=TileKind.RAILROAD, name="Pennsylvania Railroad", property_idx=10),
            TileSpec(idx=16, kind=TileKind.PROPERTY, name="St. James Place", property_idx=11),
            TileSpec(idx=17, kind=TileKind.COMMUNITY, name="Community Chest", property_idx=None),
            TileSpec(idx=18, kind=TileKind.PROPERTY, name="Tennessee Avenue", property_idx=12),
            TileSpec(idx=19, kind=TileKind.PROPERTY, name="New York Avenue", property_idx=13),
            TileSpec(idx=20, kind=TileKind.FREE_PARKING, name="Free Parking", property_idx=None),
            TileSpec(idx=21, kind=TileKind.PROPERTY, name="Kentucky Avenue", property_idx=14),
            TileSpec(idx=22, kind=TileKind.CHANCE, name="Chance", property_idx=None),
            TileSpec(idx=23, kind=TileKind.PROPERTY, name="Indiana Avenue", property_idx=15),
            TileSpec(idx=24, kind=TileKind.PROPERTY, name="Illinois Avenue", property_idx=16),
            TileSpec(idx=25, kind=TileKind.RAILROAD, name="B&O Railroad", property_idx=17),
            TileSpec(idx=26, kind=TileKind.PROPERTY, name="Atlantic Avenue", property_idx=18),
            TileSpec(idx=27, kind=TileKind.PROPERTY, name="Ventnor Avenue", property_idx=19),
            TileSpec(idx=28, kind=TileKind.UTILITY, name="Water Works", property_idx=20),
            TileSpec(idx=29, kind=TileKind.PROPERTY, name="Marvin Gardens", property_idx=21),
            TileSpec(idx=30, kind=TileKind.GO_TO_JAIL, name="Go to Jail", property_idx=None),
            TileSpec(idx=31, kind=TileKind.PROPERTY, name="Pacific Avenue", property_idx=22),
            TileSpec(idx=32, kind=TileKind.PROPERTY, name="North Carolina Avenue", property_idx=23),
            TileSpec(idx=33, kind=TileKind.COMMUNITY, name="Community Chest", property_idx=None),
            TileSpec(idx=34, kind=TileKind.PROPERTY, name="Pennsylvania Avenue", property_idx=24),
            TileSpec(idx=35, kind=TileKind.RAILROAD, name="Short Line Railroad", property_idx=25),
            TileSpec(idx=36, kind=TileKind.CHANCE, name="Chance", property_idx=None),
            TileSpec(idx=37, kind=TileKind.PROPERTY, name="Park Place", property_idx=26),
            TileSpec(idx=38, kind=TileKind.TAX, name="Luxury Tax", property_idx=None),
            TileSpec(idx=39, kind=TileKind.PROPERTY, name="Boardwalk", property_idx=27),
        ]
        return cls(tiles=tiles, board_size=len(tiles))

    def property_index_for_board_idx(self, board_idx: int) -> Optional[int]:
        tile = self.get_tile(board_idx)
        if tile.is_property():
            return tile.property_idx
        return None