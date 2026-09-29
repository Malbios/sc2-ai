"""The shared part of the group fight tasks: every unit decides for itself each step, with a choice
of whom to shoot (the closest, the weakest in range, or the most dangerous in range) and eight
directions to move in. Paid like ZerglingSurroundTask, so it can be trained on as well.

Subclasses set the unit's cooldown scale and add inputs and actions after the shared ones; the
rule helpers here take parameters, so rule variants for the headroom survey are one-liners.
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
FIRST_EXTRA_ACTION = FIRST_MOVE + len(DIRECTIONS)

SEEN_ENEMIES = 4
ENEMY_INPUTS = 8  # present, x, y, distance, life, movement x, movement y, threat
SEEN_ALLIES = 4
ALLY_INPUTS = 4  # present, x, y, life
THREAT_SCALE = 20.0  # damage per second

COOLDOWN, LIFE = 0, 1
FIRST_ENEMY = 2
CLOSEST_DISTANCE = FIRST_ENEMY + 3
THREAT = 7  # within an enemy's inputs
FIRST_ALLY = FIRST_ENEMY + SEEN_ENEMIES * ENEMY_INPUTS
FIRST_RAY = FIRST_ALLY + SEEN_ALLIES * ALLY_INPUTS
INPUTS = FIRST_RAY + len(DIRECTIONS)  # subclasses' extra inputs start here

Rule = Callable[[np.ndarray], int]


class GroupFightTask(ZerglingSurroundTask):
    observation_space = spaces.Box(-np.inf, np.inf, shape=(INPUTS,), dtype=np.float32)
    action_space = spaces.Discrete(FIRST_EXTRA_ACTION)
    group_slots = 12
    cooldown_scale = 32.0  # the unit's weapon cooldown in game loops

    def observe(self, unit: Unit, allies: Units, enemies: Units) -> np.ndarray:
        """Own weapon cooldown and life; the 4 closest enemies (closest first, zeros when missing):
        present, relative x and y, distance, life, x and y movement since this unit's last
        decision, and threat (its damage per second against this unit); the 4 closest allies:
        present, relative x and y, life; then the free distance in each of the 8 directions.
        Distances are divided by 10, movement by MOVEMENT_SCALE, threat by THREAT_SCALE.
        Sized by the subclass's observation space; its extra inputs are left at zero."""
        self._group_size = max(self._group_size, len(allies) + 1)
        observation = np.zeros(self.observation_space.shape, dtype=np.float32)
        observation[COOLDOWN] = max(unit.weapon_cooldown, 0.0) / self.cooldown_scale
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
        if FIRST_MOVE <= action < FIRST_EXTRA_ACTION:
            unit.move(unit.position + DIRECTIONS[action - FIRST_MOVE] * MOVE_DISTANCE)
        elif not enemies:
            unit.stop()
        else:
            unit.attack(pick_target(unit, action, enemies))

    def baseline_policies(self) -> dict[str, Rule]:
        return {
            "attack": lambda observation: ATTACK_CLOSEST,
            "focus": lambda observation: ATTACK_WEAKEST,
            "threat": lambda observation: ATTACK_DANGEROUS,
        }


def pick_target(unit: Unit, action: int, enemies: Units) -> Unit:
    """The enemy an attack action shoots at. The weakest and most dangerous are picked among the
    enemies in weapon range; with none in range, the closest. Ties (e.g. several unhurt marines)
    go to the current target, then to the closest: switching targets restarts the attack, so a
    unit that switched every decision would never fire."""
    by_distance = _by_distance(unit, enemies)
    in_range = [enemy for enemy in by_distance if unit.target_in_range(enemy)]
    if action == ATTACK_CLOSEST or not in_range:
        return by_distance[0]
    return most_dangerous(unit, in_range) if action == ATTACK_DANGEROUS else weakest(unit, in_range)


def weakest(unit: Unit, candidates: list[Unit]) -> Unit:
    """The candidate with the least life left; ties as in pick_target (candidates closest first)."""
    return min(candidates, key=lambda enemy: (_life_left(enemy), enemy.tag != unit.order_target))


def most_dangerous(unit: Unit, candidates: list[Unit]) -> Unit:
    """The candidate with the most damage per second against `unit` per life left; ties as in
    pick_target (candidates closest first)."""
    return min(candidates, key=lambda enemy: (-enemy.calculate_dps_vs_target(unit) / _life_left(enemy),
                                              enemy.tag != unit.order_target))


def _life_left(enemy: Unit) -> float:
    return max(enemy.health + enemy.shield, 1.0)


def closest_enemy_distance(observation: np.ndarray) -> float:
    """Game distance to the closest enemy; infinite with none in sight."""
    return observation[CLOSEST_DISTANCE] * DISTANCE_SCALE if observation[FIRST_ENEMY] else float("inf")


def step_away(observation: np.ndarray) -> int:
    """The move pointing most directly away from the seen enemies, closer ones counting more."""
    away = Point2((0.0, 0.0))
    for index in range(SEEN_ENEMIES):
        start = FIRST_ENEMY + index * ENEMY_INPUTS
        if observation[start]:
            offset = Point2(observation[start + 1:start + 3])
            away -= offset / max(offset.length ** 2, 1e-6)
    return move_toward(away)


def move_toward(direction: Point2) -> int:
    return FIRST_MOVE + max(range(len(DIRECTIONS)), key=lambda k: _dot(DIRECTIONS[k], direction))


def allies_center(observation: np.ndarray) -> Point2 | None:
    """Relative position (game distance) of the seen allies' center; None with none in sight."""
    positions = [Point2(observation[start + 1:start + 3]) * DISTANCE_SCALE
                 for start in range(FIRST_ALLY, FIRST_RAY, ALLY_INPUTS) if observation[start]]
    if not positions:
        return None
    return Point2((sum(p.x for p in positions) / len(positions), sum(p.y for p in positions) / len(positions)))


def kite_rule(observation: np.ndarray, distance: float, attack: int = ATTACK_CLOSEST) -> int:
    """`attack` when the weapon is ready or no enemy is within `distance`; else step away."""
    if observation[COOLDOWN] == 0 or closest_enemy_distance(observation) > distance:
        return attack
    return step_away(observation)
