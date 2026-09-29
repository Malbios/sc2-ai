"""Roaches with ravagers against marines and marauders (GroupFightTask), for the headroom survey.

Ravagers can also cast corrosive bile (60 damage in a small circle, landing about 1.6 s after
the cast), which enemies can step out of. Bile makes the enemy move and breaks its formation,
so when and at whom to bile, and how far to lead it, is micro that attack-move lacks.
Roaches and ravagers share the model; a roach choosing a bile action attacks the closest instead.
"""

import numpy as np
from gymnasium import spaces
from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units

from tools.rl.examples.free_kite_task import _by_distance
from tools.rl.examples.group_fight_task import (
    ATTACK_CLOSEST,
    ATTACK_DANGEROUS,
    FIRST_EXTRA_ACTION,
    INPUTS,
    GroupFightTask,
    Rule,
    closest_enemy_distance,
    most_dangerous,
)
from tools.rl.examples.ravager_task import BILE, BILE_FLIGHT_SECONDS, BILE_RANGE, DECISION_SECONDS

IS_RAVAGER, BILE_READY = INPUTS, INPUTS + 1

# The bile actions: whom to bile (the closest, or the most dangerous within bile range) and the
# lead (0: where it is now, 1: where it will be if it keeps moving straight).
BILES = (("closest", 1.0), ("dangerous", 1.0), ("closest", 0.5), ("closest", 1.5))
BILE_CLOSEST, BILE_DANGEROUS, BILE_CLOSEST_LEAD_HALF, BILE_CLOSEST_LEAD_ONE_AND_HALF = (
    FIRST_EXTRA_ACTION + index for index in range(len(BILES)))


class RoachRavagerGroupTask(GroupFightTask):
    observation_space = spaces.Box(-np.inf, np.inf, shape=(BILE_READY + 1,), dtype=np.float32)
    action_space = spaces.Discrete(FIRST_EXTRA_ACTION + len(BILES))
    wants_abilities = True
    cooldown_scale = 32.0  # a roach's; a ravager's is 26

    def __init__(self):
        super().__init__()
        self._abilities: dict[int, set[AbilityId]] = {}
        self._movement: dict[tuple[int, int], Point2] = {}

    def see_abilities(self, available: dict[int, set[AbilityId]]) -> None:
        self._abilities = available

    def observe(self, unit: Unit, allies: Units, enemies: Units) -> np.ndarray:
        """GroupFightTask's inputs, then 1 if this unit is a ravager, and 1 if its bile is ready."""
        for enemy in enemies:
            last = self._last_positions.get((unit.tag, enemy.tag))
            self._movement[(unit.tag, enemy.tag)] = enemy.position - last if last is not None else Point2((0, 0))
        observation = super().observe(unit, allies, enemies)
        observation[IS_RAVAGER] = 1.0 if unit.type_id == UnitTypeId.RAVAGER else 0.0
        observation[BILE_READY] = 1.0 if BILE in self._abilities.get(unit.tag, set()) else 0.0
        return observation

    def apply(self, unit: Unit, action: int, allies: Units, enemies: Units) -> None:
        if action < FIRST_EXTRA_ACTION or not enemies:
            super().apply(unit, action, allies, enemies)
        elif unit.type_id != UnitTypeId.RAVAGER:
            super().apply(unit, ATTACK_CLOSEST, allies, enemies)
        else:
            whom, lead = BILES[action - FIRST_EXTRA_ACTION]
            unit(BILE, self.bile_point(unit, self._bile_target(unit, whom, enemies), lead))

    def bile_point(self, unit: Unit, target: Unit, lead: float) -> Point2:
        movement = self._movement.get((unit.tag, target.tag), Point2((0, 0)))
        return target.position + movement * (lead * BILE_FLIGHT_SECONDS / DECISION_SECONDS)

    @staticmethod
    def _bile_target(unit: Unit, whom: str, enemies: Units) -> Unit:
        by_distance = _by_distance(unit, enemies)
        in_bile_range = [enemy for enemy in by_distance if unit.position.distance_to(enemy.position) <= BILE_RANGE]
        if whom == "dangerous" and in_bile_range:
            return most_dangerous(unit, in_bile_range)
        return by_distance[0]

    def baseline_policies(self) -> dict[str, Rule]:
        rules = {"attack": lambda observation: ATTACK_CLOSEST}
        biles = {"": BILE_CLOSEST, "_dangerous": BILE_DANGEROUS,
                 "_lead_0.5": BILE_CLOSEST_LEAD_HALF, "_lead_1.5": BILE_CLOSEST_LEAD_ONE_AND_HALF}
        for suffix, bile in biles.items():
            rules[f"bile{suffix}"] = lambda observation, b=bile: bile_rule(observation, b, ATTACK_CLOSEST)
            rules[f"bile_threat{suffix}"] = lambda observation, b=bile: bile_rule(observation, b, ATTACK_DANGEROUS)
        return rules


def bile_rule(observation: np.ndarray, bile: int, otherwise: int) -> int:
    """`bile` when bile is ready and the closest enemy is within bile range, else `otherwise`.
    Roaches never have bile ready."""
    if observation[BILE_READY] and closest_enemy_distance(observation) <= BILE_RANGE:
        return bile
    return otherwise
