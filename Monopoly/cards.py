from dataclasses import dataclass
from typing import Callable, Tuple, Dict, List
import numpy as np

# Forward references
class GameState:
    pass

class GameEngine:
    pass

@dataclass
class Card:
    id: str
    text: str
    effect: Callable[[GameState, GameEngine, np.random.Generator, int], Tuple[GameState, Dict]]  # effect function

class CardDeck:
    def __init__(self, cards: List[Card]):
        self.cards = cards
        self.pointer = 0

    def shuffle(self, rng: np.random.Generator) -> None:
        self.cards = list(rng.permutation(self.cards))

    def draw(self) -> Card:
        card = self.cards[self.pointer]
        self.pointer = (self.pointer + 1) % len(self.cards)
        return card

    def reset(self, rng: np.random.Generator) -> None:
        self.shuffle(rng)
        self.pointer = 0

    def apply_card(self, card: Card, state: GameState, engine: GameEngine, rng: np.random.Generator) -> Tuple[GameState, Dict]:
        # Apply the card effect for the current player
        return card.effect(state, engine, rng, state.current_player)

def load_chance_cards() -> List[Card]:
    def advance_to_go(state: GameState, engine: GameEngine, rng: np.random.Generator, player_id: int) -> Tuple[GameState, Dict]:
        # Move to GO, collect $200
        state.players[player_id].position = 0
        state.players[player_id].cash += 200
        return state, {"collected": 200}

    def advance_to_illinois(state: GameState, engine: GameEngine, rng: np.random.Generator, player_id: int) -> Tuple[GameState, Dict]:
        # Advance to Illinois Avenue (pos 24), collect $200 if pass GO
        old_pos = state.players[player_id].position
        state.players[player_id].position = 24
        if state.players[player_id].position < old_pos:
            state.players[player_id].cash += 200
        return state, {"passed_go": state.players[player_id].position < old_pos}

    def advance_to_st_charles(state: GameState, engine: GameEngine, rng: np.random.Generator, player_id: int) -> Tuple[GameState, Dict]:
        old_pos = state.players[player_id].position
        state.players[player_id].position = 11
        if state.players[player_id].position < old_pos:
            state.players[player_id].cash += 200
        return state, {"passed_go": state.players[player_id].position < old_pos}

    def advance_to_reading(state: GameState, engine: GameEngine, rng: np.random.Generator, player_id: int) -> Tuple[GameState, Dict]:
        old_pos = state.players[player_id].position
        state.players[player_id].position = 5
        if state.players[player_id].position < old_pos:
            state.players[player_id].cash += 200
        return state, {"passed_go": state.players[player_id].position < old_pos}

    def advance_to_boardwalk(state: GameState, engine: GameEngine, rng: np.random.Generator, player_id: int) -> Tuple[GameState, Dict]:
        state.players[player_id].position = 39
        return state, {}

    def bank_dividend(state: GameState, engine: GameEngine, rng: np.random.Generator, player_id: int) -> Tuple[GameState, Dict]:
        state.players[player_id].cash += 50
        return state, {"collected": 50}

    def get_out_of_jail_free(state: GameState, engine: GameEngine, rng: np.random.Generator, player_id: int) -> Tuple[GameState, Dict]:
        state.players[player_id].get_out_of_jail_cards += 1
        return state, {"cards": 1}

    def go_back_3(state: GameState, engine: GameEngine, rng: np.random.Generator, player_id: int) -> Tuple[GameState, Dict]:
        state.players[player_id].position -= 3
        return state, {}

    def go_to_jail(state: GameState, engine: GameEngine, rng: np.random.Generator, player_id: int) -> Tuple[GameState, Dict]:
        state.players[player_id].position = 10
        state.players[player_id].jail_turns = 1
        return state, {}

    def general_repairs(state: GameState, engine: GameEngine, rng: np.random.Generator, player_id: int) -> Tuple[GameState, Dict]:
        houses = sum(state.properties[p].houses_count for p in state.players[player_id].properties_owned if state.properties[p].houses_count < 5)
        hotels = sum(1 for p in state.players[player_id].properties_owned if state.properties[p].houses_count == 5)
        cost = houses * 25 + hotels * 100
        state.players[player_id].cash -= cost
        return state, {"paid": cost}

    def advance_nearest_railroad(state: GameState, engine: GameEngine, rng: np.random.Generator, player_id: int) -> Tuple[GameState, Dict]:
        pos = state.players[player_id].position
        railroads = [5, 15, 25, 35]
        nearest = min(railroads, key=lambda x: (x - pos) % 40)
        old_pos = pos
        state.players[player_id].position = nearest
        if nearest < old_pos:
            state.players[player_id].cash += 200
        # Pay double rent if owned
        prop_idx = engine.board.get_tile(nearest).property_idx
        if prop_idx is not None and state.properties[prop_idx].owner is not None:
            rent = engine.calculate_rent(state, prop_idx) * 2
            state.players[player_id].cash -= rent
            state.players[state.properties[prop_idx].owner].cash += rent
        return state, {"moved_to": nearest, "passed_go": nearest < old_pos}

    def pay_15(state: GameState, engine: GameEngine, rng: np.random.Generator, player_id: int) -> Tuple[GameState, Dict]:
        state.players[player_id].cash -= 15
        return state, {"paid": 15}

    def ride_reading(state: GameState, engine: GameEngine, rng: np.random.Generator, player_id: int) -> Tuple[GameState, Dict]:
        old_pos = state.players[player_id].position
        state.players[player_id].position = 5
        if state.players[player_id].position < old_pos:
            state.players[player_id].cash += 200
        return state, {"passed_go": state.players[player_id].position < old_pos}

    def chairman_board(state: GameState, engine: GameEngine, rng: np.random.Generator, player_id: int) -> Tuple[GameState, Dict]:
        for i, p in enumerate(state.players):
            if i != player_id:
                p.cash -= 50
                state.players[player_id].cash += 50
        return state, {"collected_per": 50}

    def building_loan(state: GameState, engine: GameEngine, rng: np.random.Generator, player_id: int) -> Tuple[GameState, Dict]:
        state.players[player_id].cash += 150
        return state, {"collected": 150}

    def crossword(state: GameState, engine: GameEngine, rng: np.random.Generator, player_id: int) -> Tuple[GameState, Dict]:
        state.players[player_id].cash += 100
        return state, {"collected": 100}

    return [
        Card("Advance to GO", "Advance to GO (Collect $200)", advance_to_go),
        Card("Advance to Illinois Avenue", "Advance to Illinois Avenue. If you pass GO, collect $200.", advance_to_illinois),
        Card("Advance to St. Charles Place", "Advance to St. Charles Place. If you pass GO, collect $200.", advance_to_st_charles),
        Card("Advance to Reading Railroad", "Advance token to Reading Railroad. If you pass GO, collect $200.", advance_to_reading),
        Card("Advance to Boardwalk", "Advance to Boardwalk.", advance_to_boardwalk),
        Card("Bank pays dividend", "Bank pays you dividend of $50.", bank_dividend),
        Card("Get out of Jail Free", "Get out of Jail Free.", get_out_of_jail_free),
        Card("Go back 3 spaces", "Go back 3 spaces.", go_back_3),
        Card("Go to Jail", "Go to Jail. Go directly to Jail. Do not pass GO. Do not collect $200.", go_to_jail),
        Card("General repairs", "Make general repairs on all your property. For each house pay $25. For each hotel $100.", general_repairs),
        Card("Speeding fine", "Speeding fine $15.", pay_15),
        Card("Ride on Reading Railroad", "Take a ride on the Reading Railroad. Advance token and if you pass GO collect $200.", ride_reading),
        Card("Chairman of the Board", "Chairman of the Board. Pay each player $50.", chairman_board),
        Card("Building and loan matures", "Building and loan matures. Collect $150.", building_loan),
        Card("Crossword competition", "You have won a crossword competition. Collect $100.", crossword),
        Card("Advance to nearest Railroad", "Advance to nearest Railroad. Pay owner twice the rental to which he is otherwise entitled. If Railroad is unowned, you may buy it from the Bank.", advance_nearest_railroad),
    ]

def load_community_cards() -> List[Card]:
    def advance_to_go_comm(state: GameState, engine: GameEngine, rng: np.random.Generator, player_id: int) -> Tuple[GameState, Dict]:
        state.players[player_id].position = 0
        state.players[player_id].cash += 200
        return state, {"collected": 200}

    def bank_error(state: GameState, engine: GameEngine, rng: np.random.Generator, player_id: int) -> Tuple[GameState, Dict]:
        state.players[player_id].cash += 200
        return state, {"collected": 200}

    def pay_50(state: GameState, engine: GameEngine, rng: np.random.Generator, player_id: int) -> Tuple[GameState, Dict]:
        state.players[player_id].cash -= 50
        return state, {"paid": 50}

    def sale_stock(state: GameState, engine: GameEngine, rng: np.random.Generator, player_id: int) -> Tuple[GameState, Dict]:
        state.players[player_id].cash += 50
        return state, {"collected": 50}

    def get_out_jail_comm(state: GameState, engine: GameEngine, rng: np.random.Generator, player_id: int) -> Tuple[GameState, Dict]:
        state.players[player_id].get_out_of_jail_cards += 1
        return state, {"cards": 1}

    def go_to_jail_comm(state: GameState, engine: GameEngine, rng: np.random.Generator, player_id: int) -> Tuple[GameState, Dict]:
        state.players[player_id].position = 10
        state.players[player_id].jail_turns = 1
        return state, {}

    def holiday_fund(state: GameState, engine: GameEngine, rng: np.random.Generator, player_id: int) -> Tuple[GameState, Dict]:
        state.players[player_id].cash += 100
        return state, {"collected": 100}

    def income_refund(state: GameState, engine: GameEngine, rng: np.random.Generator, player_id: int) -> Tuple[GameState, Dict]:
        state.players[player_id].cash += 20
        return state, {"collected": 20}

    def birthday(state: GameState, engine: GameEngine, rng: np.random.Generator, player_id: int) -> Tuple[GameState, Dict]:
        for i, p in enumerate(state.players):
            if i != player_id:
                p.cash -= 10
                state.players[player_id].cash += 10
        return state, {"collected_per": 10}

    def life_insurance(state: GameState, engine: GameEngine, rng: np.random.Generator, player_id: int) -> Tuple[GameState, Dict]:
        state.players[player_id].cash += 100
        return state, {"collected": 100}

    def hospital_fees(state: GameState, engine: GameEngine, rng: np.random.Generator, player_id: int) -> Tuple[GameState, Dict]:
        state.players[player_id].cash -= 100
        return state, {"paid": 100}

    def school_fees(state: GameState, engine: GameEngine, rng: np.random.Generator, player_id: int) -> Tuple[GameState, Dict]:
        state.players[player_id].cash -= 50
        return state, {"paid": 50}

    def consultancy_fee(state: GameState, engine: GameEngine, rng: np.random.Generator, player_id: int) -> Tuple[GameState, Dict]:
        state.players[player_id].cash += 25
        return state, {"collected": 25}

    def street_repairs(state: GameState, engine: GameEngine, rng: np.random.Generator, player_id: int) -> Tuple[GameState, Dict]:
        # Calculate based on houses/hotels - $40 per house, $115 per hotel
        houses = sum(state.properties[p].houses_count for p in state.players[player_id].properties_owned if state.properties[p].houses_count < 5)
        hotels = sum(1 for p in state.players[player_id].properties_owned if state.properties[p].houses_count == 5)
        cost = houses * 40 + hotels * 115
        state.players[player_id].cash -= cost
        return state, {"paid": cost}

    def beauty_contest(state: GameState, engine: GameEngine, rng: np.random.Generator, player_id: int) -> Tuple[GameState, Dict]:
        state.players[player_id].cash += 10
        return state, {"collected": 10}

    def inherit(state: GameState, engine: GameEngine, rng: np.random.Generator, player_id: int) -> Tuple[GameState, Dict]:
        state.players[player_id].cash += 100
        return state, {"collected": 100}

    return [
        Card("Advance to GO", "Advance to GO (Collect $200)", advance_to_go_comm),
        Card("Bank error", "Bank error in your favor. Collect $200.", bank_error),
        Card("Doctor's fee", "Doctor's fee. Pay $50.", pay_50),
        Card("Sale of stock", "From sale of stock you get $50.", sale_stock),
        Card("Get out of Jail Free", "Get out of Jail Free.", get_out_jail_comm),
        Card("Go to Jail", "Go to Jail. Go directly to Jail. Do not pass GO. Do not collect $200.", go_to_jail_comm),
        Card("Holiday fund", "Holiday fund matures. Receive $100.", holiday_fund),
        Card("Income tax refund", "Income tax refund. Collect $20.", income_refund),
        Card("It's your birthday", "It's your birthday. Collect $10 from every player.", birthday),
        Card("Life insurance", "Life insurance matures. Collect $100.", life_insurance),
        Card("Hospital fees", "Pay hospital fees of $100.", hospital_fees),
        Card("School fees", "Pay school fees of $50.", school_fees),
        Card("Consultancy fee", "Receive $25 consultancy fee.", consultancy_fee),
        Card("Street repairs", "You are assessed for street repairs. $40 per house. $115 per hotel.", street_repairs),
        Card("Beauty contest", "You have won second prize in a beauty contest. Collect $10.", beauty_contest),
        Card("You inherit", "You inherit $100.", inherit),
    ]