"""Tests for the modular reward function and running normalizer."""

import numpy as np
import pytest

from Monopoly.agents.reward import (
    DEFAULT_REWARD_WEIGHTS,
    ModularRewardCalculator,
    RunningNormalizer,
)


class TestRunningNormalizer:

    def test_single_value(self):
        norm = RunningNormalizer(clip=10.0)
        norm.update(5.0)
        # With one sample, mean=5, var=0, std defaults to 1
        result = norm.normalize(5.0)
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_converges_to_correct_mean_and_var(self):
        norm = RunningNormalizer(clip=100.0)
        rng = np.random.RandomState(42)
        samples = rng.normal(loc=5.0, scale=2.0, size=10000)
        for x in samples:
            norm.update(float(x))

        assert norm.mean == pytest.approx(5.0, abs=0.1)
        assert norm.var == pytest.approx(4.0, abs=0.5)

    def test_clip_bounds_output(self):
        norm = RunningNormalizer(clip=2.0)
        # Feed many zeros to set mean=0, var≈0
        for _ in range(100):
            norm.update(0.0)
        # A very large value should be clipped
        result = norm.normalize(1000.0)
        assert result <= 2.0

    def test_state_dict_roundtrip(self):
        norm = RunningNormalizer(clip=5.0)
        for x in [1.0, 2.0, 3.0, 4.0, 5.0]:
            norm.update(x)

        state = norm.state_dict()
        norm2 = RunningNormalizer(clip=5.0)
        norm2.load_state_dict(state)

        assert norm2.mean == pytest.approx(norm.mean)
        assert norm2.var == pytest.approx(norm.var)
        assert norm2.count == norm.count


class TestModularRewardCalculator:

    def test_delta_networth_positive_when_gaining(self):
        calc = ModularRewardCalculator()
        calc.reset()

        # First step: delta is 0 (no previous)
        total1, comp1 = calc.compute(
            info={'agent_net_worth': 1500.0, 'agent_status': 'ACTIVE'},
            action=0, terminated=False, truncated=False,
        )
        assert comp1['delta_networth'] == 0.0

        # Second step: gained 200
        total2, comp2 = calc.compute(
            info={'agent_net_worth': 1700.0, 'agent_status': 'ACTIVE'},
            action=89, terminated=False, truncated=False,
        )
        assert comp2['delta_networth'] == pytest.approx(200.0 * DEFAULT_REWARD_WEIGHTS['delta_networth'])

    def test_delta_networth_negative_when_losing(self):
        calc = ModularRewardCalculator()
        calc.reset()

        calc.compute(
            info={'agent_net_worth': 2000.0, 'agent_status': 'ACTIVE'},
            action=0, terminated=False, truncated=False,
        )
        _, comp = calc.compute(
            info={'agent_net_worth': 1500.0, 'agent_status': 'ACTIVE'},
            action=89, terminated=False, truncated=False,
        )
        assert comp['delta_networth'] < 0

    def test_buy_property_reward(self):
        calc = ModularRewardCalculator()
        calc.reset()
        calc.compute(
            info={'agent_net_worth': 1500.0, 'agent_status': 'ACTIVE'},
            action=0, terminated=False, truncated=False,
        )

        # Buy action = 1
        _, comp = calc.compute(
            info={'agent_net_worth': 1500.0, 'agent_status': 'ACTIVE'},
            action=1, terminated=False, truncated=False,
        )
        assert comp['buy_property'] == pytest.approx(DEFAULT_REWARD_WEIGHTS['buy_property'])

        # Non-buy action
        _, comp2 = calc.compute(
            info={'agent_net_worth': 1500.0, 'agent_status': 'ACTIVE'},
            action=89, terminated=False, truncated=False,
        )
        assert comp2['buy_property'] == 0.0

    def test_build_reward(self):
        calc = ModularRewardCalculator()
        calc.reset()
        calc.compute(
            info={'agent_net_worth': 1500.0, 'agent_status': 'ACTIVE'},
            action=0, terminated=False, truncated=False,
        )

        # Build action = 3-30
        _, comp = calc.compute(
            info={'agent_net_worth': 1500.0, 'agent_status': 'ACTIVE'},
            action=5, terminated=False, truncated=False,
        )
        assert comp['build'] == pytest.approx(DEFAULT_REWARD_WEIGHTS['build'])

        # Non-build action
        _, comp2 = calc.compute(
            info={'agent_net_worth': 1500.0, 'agent_status': 'ACTIVE'},
            action=89, terminated=False, truncated=False,
        )
        assert comp2['build'] == 0.0

    def test_bankrupt_only_at_terminal(self):
        calc = ModularRewardCalculator()
        calc.reset()

        # Non-terminal: no bankrupt penalty
        _, comp = calc.compute(
            info={'agent_net_worth': 0.0, 'agent_status': 'BANKRUPT'},
            action=0, terminated=False, truncated=False,
        )
        assert comp['bankrupt'] == 0.0

        # Terminal + bankrupt
        _, comp2 = calc.compute(
            info={'agent_net_worth': 0.0, 'agent_status': 'BANKRUPT'},
            action=0, terminated=True, truncated=False,
        )
        assert comp2['bankrupt'] == pytest.approx(DEFAULT_REWARD_WEIGHTS['bankrupt'])

    def test_time_penalty_always_applied(self):
        calc = ModularRewardCalculator()
        calc.reset()

        _, comp = calc.compute(
            info={'agent_net_worth': 1500.0, 'agent_status': 'ACTIVE'},
            action=0, terminated=False, truncated=False,
        )
        assert comp['time_penalty'] == pytest.approx(DEFAULT_REWARD_WEIGHTS['time_penalty'])

    def test_component_keys_match_weights(self):
        weights = {
            'delta_networth': 2.0,
            'buy_property': 0.5,
            'build': 0.3,
            'bankrupt': -2.0,
            'time_penalty': -0.01,
        }
        calc = ModularRewardCalculator(weights=weights)
        calc.reset()

        _, comp = calc.compute(
            info={'agent_net_worth': 1500.0, 'agent_status': 'ACTIVE'},
            action=0, terminated=False, truncated=False,
        )
        assert set(comp.keys()) == set(weights.keys())

    def test_total_is_sum_of_components(self):
        calc = ModularRewardCalculator(normalize=False)
        calc.reset()

        total, comp = calc.compute(
            info={'agent_net_worth': 1500.0, 'agent_status': 'ACTIVE'},
            action=1, terminated=False, truncated=False,
        )
        expected = sum(comp.values())
        assert total == pytest.approx(expected)

    def test_custom_weights_override_defaults(self):
        calc = ModularRewardCalculator(weights={'buy_property': 999.0})
        calc.reset()

        _, comp = calc.compute(
            info={'agent_net_worth': 1500.0, 'agent_status': 'ACTIVE'},
            action=1, terminated=False, truncated=False,
        )
        assert comp['buy_property'] == pytest.approx(999.0)

    def test_normalization_does_not_produce_nan(self):
        calc = ModularRewardCalculator(normalize=True, clip=10.0)
        calc.reset()

        for i in range(100):
            total, _ = calc.compute(
                info={'agent_net_worth': float(i * 100), 'agent_status': 'ACTIVE'},
                action=0, terminated=False, truncated=False,
            )
            assert not np.isnan(total)
            assert not np.isinf(total)

    def test_state_dict_roundtrip(self):
        calc = ModularRewardCalculator(normalize=True)
        calc.reset()
        for i in range(10):
            calc.compute(
                info={'agent_net_worth': float(i * 100), 'agent_status': 'ACTIVE'},
                action=0, terminated=False, truncated=False,
            )

        state = calc.state_dict()
        calc2 = ModularRewardCalculator(normalize=True)
        calc2.load_state_dict(state)

        assert calc2._prev_net_worth == calc._prev_net_worth

    def test_reset_clears_state(self):
        calc = ModularRewardCalculator()
        calc.reset()
        calc.compute(
            info={'agent_net_worth': 2000.0, 'agent_status': 'ACTIVE'},
            action=0, terminated=False, truncated=False,
        )
        assert calc._prev_net_worth is not None

        calc.reset()
        assert calc._prev_net_worth is None
