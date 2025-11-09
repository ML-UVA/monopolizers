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
    rent_table: List[int]  # rents: [base, 1house, 2house, 3house, 4house, hotel] for properties; [25,50,100,200] for railroads; [4,10] for utilities

    def rent_for(self, houses: int, monopoly: bool, dice_roll: Optional[int] = None) -> int:
        if self.group in ["Purple", "Light Blue", "Pink", "Orange", "Red", "Yellow", "Green", "Dark Blue"]:
            # If monopoly (owner owns all in group) and there are no houses, base rent is doubled
            if monopoly and houses == 0:
                return self.rent_table[0] * 2
            return self.rent_table[houses]
        elif self.group == "Railroad":
            if 1 <= houses <= 4:
                return self.rent_table[houses - 1]
            return self.rent_table[0]
        elif self.group == "Utility":
            if dice_roll is None:
                return 0
            if houses == 1:
                return self.rent_table[0] * dice_roll
            elif houses == 2:
                return self.rent_table[1] * dice_roll
            return 0
        return 0

def load_property_specs() -> List[PropertySpec]:
    return [
        PropertySpec(idx=0, name="Mediterranean Avenue", group="Purple", price=60, house_cost=50, mortgage_value=30, rent_table=[2, 10, 30, 90, 160, 250]),
        PropertySpec(idx=1, name="Baltic Avenue", group="Purple", price=60, house_cost=50, mortgage_value=30, rent_table=[4, 20, 60, 180, 320, 450]),
        PropertySpec(idx=2, name="Reading Railroad", group="Railroad", price=200, house_cost=0, mortgage_value=100, rent_table=[25, 50, 100, 200]),
        PropertySpec(idx=3, name="Oriental Avenue", group="Light Blue", price=100, house_cost=50, mortgage_value=50, rent_table=[6, 30, 90, 270, 400, 550]),
        PropertySpec(idx=4, name="Vermont Avenue", group="Light Blue", price=100, house_cost=50, mortgage_value=50, rent_table=[6, 30, 90, 270, 400, 550]),
        PropertySpec(idx=5, name="Connecticut Avenue", group="Light Blue", price=120, house_cost=50, mortgage_value=60, rent_table=[8, 40, 100, 300, 450, 600]),
        PropertySpec(idx=6, name="St. Charles Place", group="Pink", price=140, house_cost=100, mortgage_value=70, rent_table=[10, 50, 150, 450, 625, 750]),
        PropertySpec(idx=7, name="Electric Company", group="Utility", price=150, house_cost=0, mortgage_value=75, rent_table=[4, 10]),
        PropertySpec(idx=8, name="States Avenue", group="Pink", price=140, house_cost=100, mortgage_value=70, rent_table=[10, 50, 150, 450, 625, 750]),
        PropertySpec(idx=9, name="Virginia Avenue", group="Pink", price=160, house_cost=100, mortgage_value=80, rent_table=[12, 60, 180, 500, 700, 900]),
        PropertySpec(idx=10, name="Pennsylvania Railroad", group="Railroad", price=200, house_cost=0, mortgage_value=100, rent_table=[25, 50, 100, 200]),
        PropertySpec(idx=11, name="St. James Place", group="Orange", price=180, house_cost=100, mortgage_value=90, rent_table=[14, 70, 200, 550, 750, 950]),
        PropertySpec(idx=12, name="Tennessee Avenue", group="Orange", price=180, house_cost=100, mortgage_value=90, rent_table=[14, 70, 200, 550, 750, 950]),
        PropertySpec(idx=13, name="New York Avenue", group="Orange", price=200, house_cost=100, mortgage_value=100, rent_table=[16, 80, 220, 600, 800, 1000]),
        PropertySpec(idx=14, name="Kentucky Avenue", group="Red", price=220, house_cost=150, mortgage_value=110, rent_table=[18, 90, 250, 700, 875, 1050]),
        PropertySpec(idx=15, name="Indiana Avenue", group="Red", price=220, house_cost=150, mortgage_value=110, rent_table=[18, 90, 250, 700, 875, 1050]),
        PropertySpec(idx=16, name="Illinois Avenue", group="Red", price=240, house_cost=150, mortgage_value=120, rent_table=[20, 100, 300, 750, 925, 1100]),
        PropertySpec(idx=17, name="B&O Railroad", group="Railroad", price=200, house_cost=0, mortgage_value=100, rent_table=[25, 50, 100, 200]),
        PropertySpec(idx=18, name="Atlantic Avenue", group="Yellow", price=260, house_cost=150, mortgage_value=130, rent_table=[22, 110, 330, 800, 975, 1150]),
        PropertySpec(idx=19, name="Ventnor Avenue", group="Yellow", price=260, house_cost=150, mortgage_value=130, rent_table=[22, 110, 330, 800, 975, 1150]),
        PropertySpec(idx=20, name="Water Works", group="Utility", price=150, house_cost=0, mortgage_value=75, rent_table=[4, 10]),
        PropertySpec(idx=21, name="Marvin Gardens", group="Yellow", price=280, house_cost=150, mortgage_value=140, rent_table=[24, 120, 360, 850, 1025, 1200]),
        PropertySpec(idx=22, name="Pacific Avenue", group="Green", price=300, house_cost=200, mortgage_value=150, rent_table=[26, 130, 390, 900, 1100, 1275]),
        PropertySpec(idx=23, name="North Carolina Avenue", group="Green", price=300, house_cost=200, mortgage_value=150, rent_table=[26, 130, 390, 900, 1100, 1275]),
        PropertySpec(idx=24, name="Pennsylvania Avenue", group="Green", price=320, house_cost=200, mortgage_value=160, rent_table=[28, 150, 450, 1000, 1200, 1400]),
        PropertySpec(idx=25, name="Short Line Railroad", group="Railroad", price=200, house_cost=0, mortgage_value=100, rent_table=[25, 50, 100, 200]),
        PropertySpec(idx=26, name="Park Place", group="Dark Blue", price=350, house_cost=200, mortgage_value=175, rent_table=[35, 175, 500, 1100, 1300, 1500]),
        PropertySpec(idx=27, name="Boardwalk", group="Dark Blue", price=400, house_cost=200, mortgage_value=200, rent_table=[50, 200, 600, 1400, 1700, 2000]),
    ]