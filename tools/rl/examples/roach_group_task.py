"""A roach group against marines and marauders: every roach decides for itself each step, with a
choice of whom to shoot (the closest, the weakest in reach, or the most dangerous in reach) and
eight directions to move in.

Built for the headroom check: the attack, focus, threat and smart rules differ only in their
micro, so how far their results spread shows how much micro matters in this fight. Paid like
ZerglingSurroundTask, so it can be trained on as well.
"""

from collections.abc import Callable

import numpy as np
from gymnasium import spaces
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units

from tools.rl.examples.free_kite_task import (
    DIRECTIONS,
    DISTANCE_SCALE,
    MOVE_DISTANCE,
    MOVEMENT_SCALE,
    _by_distance,
    _dot,
    _fraction,
)
from tools.rl.examples.zergling_surround_task import ZerglingSurroundTask

ATTACK_CLOSEST, ATTACK_WEAKEST, ATTACK_DANGEROUS = range(3)
FIRST_MOVE = 3

SEEN_ENEMIES = 4
ENEMY_INPUTS = 8  # present, x, y, distance, life, movement x, movement y, threat
SEEN_ALLIES = 4
ALLY_INPUTS = 4  # present, x, y, life
COOLDOWN_SCALE = 32.0  # a roach's cooldown in game loops
THREAT_SCALE = 20.0  # damage per second
REACH_BONUS = 1.0  # "in reach": within weapon range plus this

COOLDOWN, LIFE = 0, 1
FIRST_ENEMY = 2
THREAT = 7  # within an enemy's inputs
FIRST_ALLY = FIRST_ENEMY + SEEN_ENEMIES * ENEMY_INPUTS
FIRST_RAY = FIRST_ALLY + SEEN_ALLIES * ALLY_INPUTS
INPUTS = FIRST_RAY + len(DIRECTIONS)

# The smart rule pulls a roach back below this share of its life while an enemy is this close.
PULL_BACK_BELOW = 0.35
PULL_BACK_DISTANCE = 7.0


class RoachGroupTask(ZerglingSurroundTask):
    observation_space = spaces.Box(-np.inf, np.inf, shape=(INPUTS,), dtype=np.float32)
    action_space = spaces.Discrete(FIRST_MOVE + len(DIRECTIONS))
    group_slots = 12

    def observe(self, unit: Unit, allies: Units, enemies: Units) -> np.ndarray:
        """Own weapon cooldown and life; the 4 closest enemies (closest first, zeros when missing):
        present, relative x and y, distance, life, x and y movement since this unit's last
        decision, and threat (its damage per second against this unit); the 4 closest allies:
        present, relative x and y, life; then the free distance in each of the 8 directions.
        Distances are divided by 10, movement by MOVEMENT_SCALE, threat by THREAT_SCALE."""
        self._group_size = max(self._group_size, len(allies) + 1)
        observation = np.zeros(INPUTS, dtype=np.float32)
        observation[COOLDOWN] = max(unit.weapon_cooldown, 0.0) / COOLDOWN_SCALE
        observation[LIFE] = _fraction(unit.health, unit.health_max)
        for index, enemy in enumerate(_by_distance(unit, enemies)[:SEEN_ENEMIES]):
            offset = (enemy.position - unit.position) / DISTANCE_SCALE
            last = self._last_positions.get((unit.tag, enemy.tag))
            movement = (enemy.position - last) / MOVEMENT_SCALE if last is not None else Point2((0, 0))
            start = FIRST_ENEMY + index * ENEMY_INPUTS
            observation[start:start + ENEMY_INPUTS] = [
                1.0, offset.x, offset.y, offset.length,
                _fraction(enemy.health + enemy.shield, enemy.health_max + enemy.shield_max),
                movement.x, movement.y,
                enemy.calculate_dps_vs_target(unit) / THREAT_SCALE,
            ]
        for enemy in enemies:
            self._last_positions[(unit.tag, enemy.tag)] = enemy.position
        for index, ally in enumerate(_by_distance(unit, allies)[:SEEN_ALLIES]):
            offset = (ally.position - unit.position) / DISTANCE_SCALE
            start = FIRST_ALLY + index * ALLY_INPUTS
            observation[start:start + ALLY_INPUTS] = [1.0, offset.x, offset.y, _fraction(ally.health, ally.health_max)]
        for index, direction in enumerate(DIRECTIONS):
            observation[FIRST_RAY + index] = self._free_distance(unit.position, direction) / DISTANCE_SCALE
        return observation

    def apply(self, unit: Unit, action: int, allies: Units, enemies: Units) -> None:
        if action >= FIRST_MOVE:
            unit.move(unit.position + DIRECTIONS[action - FIRST_MOVE] * MOVE_DISTANCE)
        elif not enemies:
            unit.stop()
        else:
            unit.attack(_target(unit, action, enemies))

    def baseline_policies(self) -> dict[str, Callable[[np.ndarray], int]]:
        return {
            "attack": lambda observation: ATTACK_CLOSEST,
            "focus": lambda observation: ATTACK_WEAKEST,
            "threat": lambda observation: ATTACK_DANGEROUS,
            "smart": _smart_rule,
        }


def _target(unit: Unit, action: int, enemies: Units) -> Unit:
    """The enemy an attack action shoots at. The weakest and most dangerous are picked among the
    enemies in reach; with none in reach, the closest."""
    closest = _by_distance(unit, enemies)[0]
    in_reach = [enemy for enemy in enemies if unit.target_in_range(enemy, bonus_distance=REACH_BONUS)]
    if action == ATTACK_CLOSEST or not in_reach:
        return closest
    life_left = lambda enemy: max(enemy.health + enemy.shield, 1.0)
    if action == ATTACK_WEAKEST:
        return min(in_reach, key=life_left)
    return max(in_reach, key=lambda enemy: enemy.calculate_dps_vs_target(unit) / life_left(enemy))


def _smart_rule(observation: np.ndarray) -> int:
    """Shoot the most dangerous enemy in reach; but a roach below PULL_BACK_BELOW life with an
    enemy within PULL_BACK_DISTANCE steps back, away from the enemies (closer ones count more)."""
    closest_distance = observation[FIRST_ENEMY + 3] * DISTANCE_SCALE
    hurt = observation[LIFE] < PULL_BACK_BELOW
    if not (hurt and observation[FIRST_ENEMY] and closest_distance < PULL_BACK_DISTANCE):
        return ATTACK_DANGEROUS
    away = Point2((0.0, 0.0))
    for index in range(SEEN_ENEMIES):
        start = FIRST_ENEMY + index * ENEMY_INPUTS
        if observation[start]:
            offset = Point2(observation[start + 1:start + 3])
            away -= offset / max(offset.length ** 2, 1e-6)
    return FIRST_MOVE + max(range(len(DIRECTIONS)), key=lambda k: _dot(DIRECTIONS[k], away))
