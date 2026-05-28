"""Stalemate detection for early game truncation.

Tracks game state snapshots each full round. If no meaningful changes occur
for a configurable number of consecutive rounds, signals stalemate to avoid
wasted evaluation runs with no reward signal.
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple
import numpy as np

from ..state import GameState, PlayerStatus


@dataclass(frozen=True)
class _Snapshot:
    """Immutable snapshot of game state for stalemate comparison."""
    property_owners: Tuple[Optional[int], ...]
    houses: Tuple[int, ...]
    active_player_ids: frozenset
    net_worths: Tuple[float, ...]


class StalemateDetector:
    """Detects game stalemates to enable early truncation.

    Tracks a snapshot of game state each full round (when play cycles back
    to the first active player). A round is "stale" if all of:
    - No property ownership changes
    - No house/hotel construction changes
    - No bankruptcies
    - Max per-player net worth delta < net_worth_epsilon

    If threshold_rounds consecutive stale rounds occur, signals stalemate.
    """

    def __init__(self, threshold_rounds: int = 50, net_worth_epsilon: float = 100.0):
        self.threshold_rounds = threshold_rounds
        self.net_worth_epsilon = net_worth_epsilon
        self._consecutive_stale_rounds: int = 0
        self._last_snapshot: Optional[_Snapshot] = None

    def reset(self) -> None:
        """Reset detector for a new episode."""
        self._consecutive_stale_rounds = 0
        self._last_snapshot = None

    def on_round_complete(
        self,
        state: GameState,
        property_specs,
        rules_engine=None,
    ) -> bool:
        """Check for stalemate after a full round completes.

        Args:
            state: Current game state.
            property_specs: List of PropertySpec for net worth computation.
            rules_engine: Optional RulesEngine for monopoly checking in net worth.

        Returns:
            True if stalemate detected (threshold consecutive stale rounds).
        """
        snapshot = self._take_snapshot(state, property_specs, rules_engine)

        if self._last_snapshot is None:
            self._last_snapshot = snapshot
            return False

        if self._is_stale(self._last_snapshot, snapshot):
            self._consecutive_stale_rounds += 1
        else:
            self._consecutive_stale_rounds = 0

        self._last_snapshot = snapshot
        return self._consecutive_stale_rounds >= self.threshold_rounds

    def _take_snapshot(
        self, state: GameState, property_specs, rules_engine=None
    ) -> _Snapshot:
        """Capture current game state for comparison."""
        property_owners = tuple(p.owner for p in state.properties)
        houses = tuple(p.houses_count for p in state.properties)
        active_ids = frozenset(
            p.id for p in state.players if p.status == PlayerStatus.ACTIVE
        )
        net_worths = tuple(
            _compute_net_worth(state, i, property_specs, rules_engine)
            for i in range(len(state.players))
        )
        return _Snapshot(
            property_owners=property_owners,
            houses=houses,
            active_player_ids=active_ids,
            net_worths=net_worths,
        )

    def _is_stale(self, prev: _Snapshot, curr: _Snapshot) -> bool:
        """Compare two snapshots to determine if the round was stale."""
        if prev.property_owners != curr.property_owners:
            return False
        if prev.houses != curr.houses:
            return False
        if prev.active_player_ids != curr.active_player_ids:
            return False
        # Check net worth stability
        for prev_nw, curr_nw in zip(prev.net_worths, curr.net_worths):
            if abs(prev_nw - curr_nw) >= self.net_worth_epsilon:
                return False
        return True


def _compute_net_worth(
    state: GameState,
    player_id: int,
    property_specs,
    rules_engine=None,
) -> float:
    """Compute net worth for a player (standalone, no env dependency).

    Uses the same formula as MonopolyEnv._compute_net_worth:
    nw_x = cash_x + sum(property_value(p) for p in owned_properties)
    """
    player = state.players[player_id]
    if player.status == PlayerStatus.BANKRUPT:
        return 0.0

    nw = float(player.cash)
    for prop_idx in player.properties_owned:
        nw += _compute_property_value(
            state, prop_idx, player_id, property_specs, rules_engine
        )
    return nw


def _compute_property_value(
    state: GameState,
    prop_idx: int,
    player_id: int,
    property_specs,
    rules_engine=None,
) -> float:
    """Compute property value using the research formula.

    p_a = (bp - mv) * b + nh * ph + nH * pH
    """
    prop = state.properties[prop_idx]
    spec = property_specs[prop_idx]

    if prop.mortgaged:
        return 0.0

    bp = spec.price
    mv = spec.mortgage_value

    has_monopoly = False
    if rules_engine is not None:
        has_monopoly = rules_engine._has_monopoly(state, prop_idx, player_id)
    b = 2.0 if has_monopoly else 1.5

    houses_count = prop.houses_count
    ph = spec.house_cost
    pH = spec.house_cost

    if houses_count == 5:
        nh, nH = 0, 1
    else:
        nh, nH = houses_count, 0

    return (bp - mv) * b + nh * ph + nH * pH
