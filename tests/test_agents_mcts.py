import pytest
from ..Monopoly.agents.mcts import MCTSAgent
from ..Monopoly.state import GameState, PlayerState, PropertyState

@pytest.fixture
def sample_state():
    players = [PlayerState(id=0, cash=1500, position=0, properties_owned=set(), houses_on_property={}, mortgaged_properties=set(), jail_turns=0, get_out_of_jail_cards=0)]
    properties = [PropertyState(owner=None) for _ in range(28)]
    return GameState(players=players, properties=properties, chance_deck=None, community_deck=None)

def test_mcts_agent_act(sample_state):
    agent = MCTSAgent(rollouts=10)
    action = agent.act(sample_state)
    assert isinstance(action, int)  # Assuming action is index
