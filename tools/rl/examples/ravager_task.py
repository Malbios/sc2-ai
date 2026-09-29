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
    FreeKiteTask,
    TrackingKiteTask,
    _by_distance,
)
from tools.rl.task import FightSnapshot

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
            "bile_now": lambda observation: _bile_rule(observation, bile_action(0, NOW), smart),
            "bile_lead": lambda observation: _bile_rule(observation, bile_action(0, STRAIGHT_LINE), smart),
        }


class RavagerBileTask(RavagerTask):
    """RavagerTask with a reward that counts bile: paid for the share of the enemies' life taken
    (by attacks and bile alike) instead of for the attack reloading, which made every cast cost
    reward. Bile actions are masked while bile isn't ready, so exploring doesn't waste picks on
    them. Trains with sb3-contrib's MaskablePPO."""

    uses_action_masks = True
    damage_reward_scale = 100.0

    def action_mask(self, unit: Unit, allies: Units, enemies: Units) -> np.ndarray:
        mask = np.ones(self.action_space.n, dtype=bool)
        mask[FIRST_BILE:] = BILE in self._abilities.get(unit.tag, set())
        return mask

    def reward(self, before: FightSnapshot, after: FightSnapshot) -> float:
        return self.damage_reward_scale * (before.enemy_life - after.enemy_life) / max(after.enemy_start, 1.0)

    def unit_reward(self, unit: Unit, allies: Units, enemies: Units) -> float:
        return 0.0


class RavagerHybridTask(RavagerBileTask):
    """RavagerBileTask where the lead rule aims: the model no longer picks a lead, only whether to
    bile now and at which of the 2 closest enemies, always with the straight-line lead. The
    earlier runs had to learn the aiming too, which a formula already gets right; this asks
    whether a model adds anything on top of that rule. 12 actions: TrackingKiteTask's 10, then
    bile the closest, bile the second closest."""

    action_space = spaces.Discrete(FIRST_BILE + SEEN_ENEMIES)

    def apply(self, unit: Unit, action: int, allies: Units, enemies: Units) -> None:
        if action >= FIRST_BILE:
            action = bile_action(action - FIRST_BILE, STRAIGHT_LINE)
        super().apply(unit, action, allies, enemies)

    def baseline_policies(self) -> dict[str, Callable[[np.ndarray], int]]:
        policies = FreeKiteTask.baseline_policies(self)
        smart = policies["smart"]
        return {**policies, "bile_lead": lambda observation: _bile_rule(observation, FIRST_BILE, smart)}


def _bile_rule(observation: np.ndarray, bile: int, otherwise: Callable[[np.ndarray], int]) -> int:
    """`bile` (an action biling the closest enemy) whenever bile is ready and the closest enemy is
    in range; otherwise `otherwise`."""
    closest_present = observation[FIRST_ENEMY]
    closest_distance = observation[CLOSEST_DISTANCE] * DISTANCE_SCALE
    if observation[BILE_READY] and closest_present and closest_distance <= BILE_RANGE:
        return bile
    return otherwise(observation)
