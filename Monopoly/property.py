from dataclasses import dataclass
from typing import List, Optional

@dataclass
class PropertySpec:
    idx: int
    name: str
    group: str
    price: int
    house_cost: int
    mortgage_value: int
    rent_table: List[int]  # rents: [base, 1house, 2house, 3house, 4house, hotel]

    def rent_for(self, houses: int, monopoly: bool, dice_roll: Optional[int] = None) -> int:
        pass

def load_property_specs() -> List[PropertySpec]:
    pass