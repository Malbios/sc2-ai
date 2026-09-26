"""A port of the stalker kiting task from github.com/sharknice/SharkyRLMatrixTraining.

The model only chooses between attacking and retreating, sees three raw numbers, and is paid for
every step its weapon is cooling down (i.e. for firing as often as possible) plus the life it has
left when it wins. Nothing is ever subtracted.
"""

import numpy as np
from gymnasium import spaces
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units

from tools.rl.task import FightSnapshot, MicroTask

ATTACK_CLOSEST, RETREAT_FROM_CLOSEST = range(2)
RETREAT_MARGIN = 4.0


class CooldownKiteTask(MicroTask):
    observation_space = spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32)
    action_space = spaces.Discrete(2)
    win_reward_scale = 1.0

    def __init__(self):
        self._previous_distance: dict[tuple[int, int], float] = {}

    def start_episode(self) -> None:
        self._previous_distance.clear()

    def observe(self, unit: Unit, allies: Units, enemies: Units) -> np.ndarray:
        """Weapon cooldown, distance to the closest enemy, and how much that distance grew since
        the last decision (negative when closing in)."""
        cooldown = max(unit.weapon_cooldown, 0.0)
        if not enemies:
            return np.array([cooldown, 0.0, 0.0], dtype=np.float32)

        enemy = _closest(unit, enemies)
        distance = unit.position.distance_to(enemy.position)
        previous = self._previous_distance.get((unit.tag, enemy.tag))
        self._previous_distance[(unit.tag, enemy.tag)] = distance
        change = distance - previous if previous is not None else 0.0
        return np.array([cooldown, distance, change], dtype=np.float32)

    def apply(self, unit: Unit, action: int, allies: Units, enemies: Units) -> None:
        if not enemies:
            unit.stop()
            return
        enemy = _closest(unit, enemies)
        if action == ATTACK_CLOSEST:
            unit.attack(enemy)
        else:
            unit.move(_retreat_point(unit, enemy))

    def reward(self, before: FightSnapshot, after: FightSnapshot) -> float:
        return 0.0

    def unit_reward(self, unit: Unit, allies: Units, enemies: Units) -> float:
        return 1.0 if unit.weapon_cooldown > 0 else 0.0

    def terminal_reward(self, outcome: str, survivors: Units) -> float:
        if outcome != "win" or not survivors:
            return 0.0
        return self.win_reward_scale * sum(_life_left(unit) for unit in survivors) / len(survivors)


class SurvivalCooldownKiteTask(CooldownKiteTask):
    """The same task, but surviving with more life matters: firing adds up to roughly +200 per
    fight, which drowned out the original win reward of at most +1.5."""

    win_reward_scale = 100.0


def _closest(unit: Unit, enemies: Units) -> Unit:
    return min(enemies, key=lambda enemy: unit.position.distance_to(enemy.position))


def _retreat_point(unit: Unit, enemy: Unit) -> Point2:
    """Just outside the enemy's reach, on the far side from it (as in sharknice's RetreatFromClosest)."""
    away = unit.position - enemy.position
    if away.length < 0.01:
        away = Point2((1.0, 0.0))
    reach = enemy.ground_range + unit.radius + enemy.radius + RETREAT_MARGIN
    return enemy.position + away.normalized * reach


def _life_left(unit: Unit) -> float:
    health = unit.health / unit.health_max if unit.health_max else 0.0
    shield = unit.shield / unit.shield_max if unit.shield_max else 0.0
    return health + 0.5 * shield
