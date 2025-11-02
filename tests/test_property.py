import pytest
from ..Monopoly.property import load_property_specs, PropertySpec

def test_load_property_specs():
    specs = load_property_specs()
    assert len(specs) == 28
    assert specs[0].name == "Mediterranean Avenue"
    assert specs[0].price == 60

def test_rent_for_property():
    spec = PropertySpec(idx=0, name="Test", group="Purple", price=60, house_cost=50, mortgage_value=30, rent_table=[2, 10, 30, 90, 160, 250])
    assert spec.rent_for(0, False) == 2  # Base
    assert spec.rent_for(0, True) == 4  # Monopoly double
    assert spec.rent_for(2, False) == 30  # 2 houses

def test_rent_for_railroad():
    spec = PropertySpec(idx=2, name="Reading Railroad", group="Railroad", price=200, house_cost=0, mortgage_value=100, rent_table=[25, 50, 100, 200])
    assert spec.rent_for(1, False) == 50  # 2 railroads

def test_rent_for_utility():
    spec = PropertySpec(idx=7, name="Electric Company", group="Utility", price=150, house_cost=0, mortgage_value=75, rent_table=[4, 10])
    assert spec.rent_for(1, False, dice_roll=6) == 24  # 4x6
    assert spec.rent_for(2, False, dice_roll=6) == 60  # 10x6
