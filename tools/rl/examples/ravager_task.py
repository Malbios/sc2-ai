"""TrackingKiteTask for ravagers, plus corrosive bile: 60 damage in a small circle that lands about
1.6 s after the cast, so a moving target has to be led.

The model picks which of the 2 closest enemies to bile and how far ahead of it: its position plus
its movement over the bile's flight time, times a lead factor (0 = where it is now, 1 = where it
will be if it keeps moving straight).
"""

from collections.abc import Callable

import numpy as np
from gymnasium import spaces
from sc2.ids.ability_id import AbilityId
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units

from tools.rl.examples.free_kite_task import (
    DIRECTIONS,
    DISTANCE_SCALE,
    FIRST_ENEMY,
    FIRST_MOVE,
    FIRST_MOVEMENT,
    SEEN_ENEMIES,
    TrackingKiteTask,
    _by_distance,
)

BILE = AbilityId.EFFECT_CORROSIVEBILE
BILE_RANGE = 9.0
BILE_FLIGHT_SECONDS = 1.6
DECISION_SECONDS = 3 / 22.4  # decision interval 3, as TrackingKiteTask assumes
LEADS = (0.0, 0.5, 1.0, 1.5)

CLOSEST_DISTANCE = FIRST_ENEMY + 3  # present, x, y, distance, life
BILE_READY = FIRST_MOVEMENT + 2 * SEEN_ENEMIES
FIRST_BILE = FIRST_MOVE + len(DIRECTIONS)
NOW, STRAIGHT_LINE = LEADS.index(0.0), LEADS.index(1.0)


def bile_action(enemy_index: int, lead_index: int) -> int:
    return FIRST_BILE + enemy_index * len(LEADS) + lead_index


class RavagerTask(TrackingKiteTask):
    observation_space = spaces.Box(-np.inf, np.inf, shape=(BILE_READY + 1,), dtype=np.float32)
    action_space = spaces.Discrete(bile_action(SEEN_ENEMIES - 1, len(LEADS) - 1) + 1)
    wants_abilities = True

    def __init__(self):
        super().__init__()
        self._abilities: dict[int, set[AbilityId]] = {}
        self._movement: dict[tuple[int, int], Point2] = {}

    def see_abilities(self, available: dict[int, set[AbilityId]]) -> None:
        self._abilities = available

    def observe(self, unit: Unit, allies: Units, enemies: Units) -> np.ndarray:
        """TrackingKiteTask's inputs, then 1 if bile is ready, else 0."""
        for enemy in enemies:
            last = self._last_positions.get((unit.tag, enemy.tag))
            self._movement[(unit.tag, enemy.tag)] = enemy.position - last if last is not None else Point2((0, 0))
        observation = super().observe(unit, allies, enemies)
        observation[BILE_READY] = 1.0 if BILE in self._abilities.get(unit.tag, set()) else 0.0
        return observation

    def apply(self, unit: Unit, action: int, allies: Units, enemies: Units) -> None:
        if action < FIRST_BILE or not enemies:
            super().apply(unit, action, allies, enemies)
            return
        enemy_index, lead_index = divmod(action - FIRST_BILE, len(LEADS))
        targets = _by_distance(unit, enemies)
        target = targets[min(enemy_index, len(targets) - 1)]
        unit(BILE, self.bile_point(unit, target, LEADS[lead_index]))

    def bile_point(self, unit: Unit, target: Unit, lead: float) -> Point2:
        movement = self._movement.get((unit.tag, target.tag), Point2((0, 0)))
        return target.position + movement * (lead * BILE_FLIGHT_SECONDS / DECISION_SECONDS)

    def baseline_policies(self) -> dict[str, Callable[[np.ndarray], int]]:
        policies = super().baseline_policies()
        smart = policies["smart"]
        return {
            **policies,
            "bile_now": lambda observation: _bile_rule(observation, NOW, smart),
            "bile_lead": lambda observation: _bile_rule(observation, STRAIGHT_LINE, smart),
        }


def _bile_rule(observation: np.ndarray, lead_index: int, otherwise: Callable[[np.ndarray], int]) -> int:
    """Bile the closest enemy whenever bile is ready and it is in range; otherwise `otherwise`."""
    closest_present = observation[FIRST_ENEMY]
    closest_distance = observation[CLOSEST_DISTANCE] * DISTANCE_SCALE
    if observation[BILE_READY] and closest_present and closest_distance <= BILE_RANGE:
        return bile_action(0, lead_index)
    return otherwise(observation)
