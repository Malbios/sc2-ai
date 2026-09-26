"""Kiting with everything the cooldown task leaves out: both closest enemies, the walls around the
unit, a choice of target, and eight directions to move in.

The cooldown task's model only sees the closest enemy and can only run straight away from it.
Against 2 roaches it learned the hand-written kite rule, which is the best that task allows. This
task keeps that task's reward (paid for firing often, and for the life left when winning) so any
gain comes from the extra inputs and actions.
"""

import math
from collections.abc import Callable

import numpy as np
from gymnasium import spaces
from sc2.game_info import GameInfo
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units

from tools.rl.examples.cooldown_kite_task import SurvivalCooldownKiteTask

ATTACK_CLOSEST, ATTACK_WEAKEST = range(2)
FIRST_MOVE = 2
DIRECTIONS = [Point2((math.cos(k * math.pi / 4), math.sin(k * math.pi / 4))) for k in range(8)]  # east, then counterclockwise

SEEN_ENEMIES = 2
ENEMY_INPUTS = 5  # present, x, y, distance, life
DISTANCE_SCALE = 10.0
COOLDOWN_SCALE = 30.0  # a stalker's cooldown in game loops
MOVE_DISTANCE = 3.0
RAY_LENGTH = 10.0
RAY_STEP = 0.5
NEAR_WALL = 3.0

COOLDOWN = 0
FIRST_ENEMY = 3
FIRST_RAY = FIRST_ENEMY + SEEN_ENEMIES * ENEMY_INPUTS
FIRST_MOVEMENT = FIRST_RAY + len(DIRECTIONS)  # TrackingKiteTask's extra inputs start here
MOVEMENT_SCALE = 0.42  # how far a roach moves in one decision at decision interval 3


class FreeKiteTask(SurvivalCooldownKiteTask):
    observation_space = spaces.Box(-np.inf, np.inf, shape=(FIRST_RAY + len(DIRECTIONS),), dtype=np.float32)
    action_space = spaces.Discrete(FIRST_MOVE + len(DIRECTIONS))

    def __init__(self):
        super().__init__()
        self._pathing_grid = None

    def start_game(self, game_info: GameInfo) -> None:
        self._pathing_grid = game_info.pathing_grid

    def observe(self, unit: Unit, allies: Units, enemies: Units) -> np.ndarray:
        """Own cooldown, health and shield; for the 2 closest enemies (closest first, zeros when
        missing): present, relative x and y, distance and life; then the free distance in each of
        the 8 directions. Distances are divided by 10."""
        observation = np.zeros(self.observation_space.shape, dtype=np.float32)
        observation[COOLDOWN] = max(unit.weapon_cooldown, 0.0) / COOLDOWN_SCALE
        observation[1] = _fraction(unit.health, unit.health_max)
        observation[2] = _fraction(unit.shield, unit.shield_max)
        for index, enemy in enumerate(_by_distance(unit, enemies)[:SEEN_ENEMIES]):
            offset = (enemy.position - unit.position) / DISTANCE_SCALE
            start = FIRST_ENEMY + index * ENEMY_INPUTS
            observation[start:start + ENEMY_INPUTS] = [
                1.0, offset.x, offset.y, offset.length,
                _fraction(enemy.health + enemy.shield, enemy.health_max + enemy.shield_max),
            ]
        for index, direction in enumerate(DIRECTIONS):
            observation[FIRST_RAY + index] = self._free_distance(unit.position, direction) / DISTANCE_SCALE
        return observation

    def apply(self, unit: Unit, action: int, allies: Units, enemies: Units) -> None:
        if action >= FIRST_MOVE:
            unit.move(unit.position + DIRECTIONS[action - FIRST_MOVE] * MOVE_DISTANCE)
        elif not enemies:
            unit.stop()
        elif action == ATTACK_WEAKEST:
            unit.attack(min(_by_distance(unit, enemies), key=lambda enemy: enemy.health + enemy.shield))
        else:
            unit.attack(_by_distance(unit, enemies)[0])

    def baseline_policies(self) -> dict[str, Callable[[np.ndarray], int]]:
        return {
            "attack": lambda observation: ATTACK_CLOSEST,
            "kite": _kite_rule,
            "smart": _smart_rule,
        }

    def situation(self, observation: np.ndarray) -> str:
        weapon = "weapon ready" if observation[COOLDOWN] == 0 else "weapon cooling"
        near_wall = min(observation[FIRST_RAY:FIRST_MOVEMENT]) * DISTANCE_SCALE < NEAR_WALL
        return f"{weapon}, {'near a wall' if near_wall else 'open ground'}"

    def _free_distance(self, start: Point2, direction: Point2) -> float:
        """How far a ground unit can walk from `start` in `direction`, up to RAY_LENGTH."""
        grid = self._pathing_grid
        travelled = 0.0
        while travelled < RAY_LENGTH:
            x, y = start + direction * (travelled + RAY_STEP)
            cell = (int(x), int(y))
            if not (0 <= cell[0] < grid.width and 0 <= cell[1] < grid.height) or not grid[cell]:
                break
            travelled += RAY_STEP
        return travelled


class TrackingKiteTask(FreeKiteTask):
    """FreeKiteTask plus how each of the 2 seen enemies moved since the last decision, so the model
    can tell a roach that keeps chasing from one that has turned back."""

    observation_space = spaces.Box(-np.inf, np.inf, shape=(FIRST_MOVEMENT + 2 * SEEN_ENEMIES,), dtype=np.float32)

    def __init__(self):
        super().__init__()
        self._last_positions: dict[tuple[int, int], Point2] = {}

    def start_episode(self) -> None:
        super().start_episode()
        self._last_positions.clear()

    def observe(self, unit: Unit, allies: Units, enemies: Units) -> np.ndarray:
        """FreeKiteTask's inputs, then each seen enemy's x and y movement since this unit's last
        decision (zero on its first), in the same closest-first order, divided by MOVEMENT_SCALE."""
        observation = super().observe(unit, allies, enemies)  # sized by this task's space, movement zero
        for index, enemy in enumerate(_by_distance(unit, enemies)[:SEEN_ENEMIES]):
            last = self._last_positions.get((unit.tag, enemy.tag))
            if last is not None:
                start = FIRST_MOVEMENT + 2 * index
                observation[start:start + 2] = (enemy.position - last) / MOVEMENT_SCALE
        for enemy in enemies:
            self._last_positions[(unit.tag, enemy.tag)] = enemy.position
        return observation


def _kite_rule(observation: np.ndarray) -> int:
    """The cooldown task's kite rule: attack the closest enemy when the weapon is ready, else run
    in the direction that points most directly away from it."""
    if observation[COOLDOWN] == 0 or not observation[FIRST_ENEMY]:
        return ATTACK_CLOSEST
    away = -Point2(observation[FIRST_ENEMY + 1:FIRST_ENEMY + 3])
    return FIRST_MOVE + max(range(len(DIRECTIONS)), key=lambda k: _dot(DIRECTIONS[k], away))


def _smart_rule(observation: np.ndarray) -> int:
    """Attack the weakest enemy when the weapon is ready. Otherwise run where it points away from
    all enemies (closer ones count more) and there is room to keep running."""
    if observation[COOLDOWN] == 0 or not observation[FIRST_ENEMY]:
        return ATTACK_WEAKEST
    away = Point2((0.0, 0.0))
    for index in range(SEEN_ENEMIES):
        start = FIRST_ENEMY + index * ENEMY_INPUTS
        if observation[start]:
            offset = Point2(observation[start + 1:start + 3])
            away -= offset / max(offset.length ** 2, 1e-6)
    away = away / max(away.length, 1e-6)
    free = observation[FIRST_RAY:]
    return FIRST_MOVE + max(range(len(DIRECTIONS)), key=lambda k: _dot(DIRECTIONS[k], away) + free[k])


def _by_distance(unit: Unit, enemies: Units) -> list[Unit]:
    return sorted(enemies, key=lambda enemy: unit.position.distance_to(enemy.position))


def _dot(a: Point2, b: Point2) -> float:
    return a.x * b.x + a.y * b.y


def _fraction(value: float, maximum: float) -> float:
    return value / maximum if maximum else 0.0
