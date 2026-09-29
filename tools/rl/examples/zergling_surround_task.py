"""A zergling group against marines: every zergling decides for itself each step, seeing the
closest enemies and allies, so running around the marines to surround them has to be learned
together (see MicroTask.group_slots).

Paid like RavagerBileTask: for the share of the enemies' life taken, plus the share of the
group's life left on a win. No penalty for damage taken (it taught the stalker to run away). The
damage reward goes to the zerglings that attacked, the win reward to each survivor.
"""

import math
from collections.abc import Callable

import numpy as np
from gymnasium import spaces
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units

from tools.rl.examples.free_kite_task import (
    ATTACK_CLOSEST,
    DIRECTIONS,
    DISTANCE_SCALE,
    FIRST_MOVE,
    MOVEMENT_SCALE,
    FreeKiteTask,
    _by_distance,
    _dot,
    _fraction,
)
from tools.rl.task import FightSnapshot

SEEN_ENEMIES = 4
ENEMY_INPUTS = 7  # present, x, y, distance, life, movement x, movement y
SEEN_ALLIES = 4
ALLY_INPUTS = 4  # present, x, y, life
COOLDOWN_SCALE = 11.0  # a zergling's cooldown in game loops

COOLDOWN, LIFE = 0, 1
FIRST_ENEMY = 2
FIRST_ALLY = FIRST_ENEMY + SEEN_ENEMIES * ENEMY_INPUTS
FIRST_RAY = FIRST_ALLY + SEEN_ALLIES * ALLY_INPUTS
INPUTS = FIRST_RAY + len(DIRECTIONS)

# The flank rule attacks once the closest enemy is this close (center to center) or once the
# unit is past the enemies' center; until then it runs for a point this far beside and behind them.
FLANK_ENGAGE = 1.5
FLANK_RADIUS = 3.0


class ZerglingSurroundTask(FreeKiteTask):
    observation_space = spaces.Box(-np.inf, np.inf, shape=(INPUTS,), dtype=np.float32)
    group_slots = 16
    # A melee attack only pays after many decisions of running in, which random first picks
    # almost never allow; so new models start out mostly attacking.
    favored_action = ATTACK_CLOSEST
    damage_reward_scale = 100.0
    win_reward_scale = 100.0

    def __init__(self):
        super().__init__()
        self._last_positions: dict[tuple[int, int], Point2] = {}
        self._group_size = 0
        self._last_cooldowns: dict[int, float] = {}

    def start_episode(self) -> None:
        super().start_episode()
        self._last_positions.clear()
        self._group_size = 0
        self._last_cooldowns.clear()

    def share_team_reward(self, team: float, units: Units) -> dict[int, float]:
        """The damage dealt this game step goes to the zerglings that attacked in it (their weapon
        cooldown went up since the last step), in equal parts; to all of them if none did."""
        attackers = [unit for unit in units if self._attacked(unit)]
        self._last_cooldowns = {unit.tag: unit.weapon_cooldown for unit in units}
        sharing = attackers or list(units)
        return {unit.tag: team / len(sharing) for unit in sharing}

    def _attacked(self, unit: Unit) -> bool:
        """Whether `unit` attacked in the last game step: its weapon cooldown went up."""
        return unit.weapon_cooldown > self._last_cooldowns.get(unit.tag, math.inf)

    def observe(self, unit: Unit, allies: Units, enemies: Units) -> np.ndarray:
        """Own weapon cooldown and life; the 4 closest enemies (closest first, zeros when missing):
        present, relative x and y, distance, life, and x and y movement since this unit's last
        decision; the 4 closest allies: present, relative x and y, life; then the free distance in
        each of the 8 directions. Distances are divided by 10, movement by MOVEMENT_SCALE."""
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

    def reward(self, before: FightSnapshot, after: FightSnapshot) -> float:
        return self.damage_reward_scale * (before.enemy_life - after.enemy_life) / max(after.enemy_start, 1.0)

    def unit_reward(self, unit: Unit, allies: Units, enemies: Units) -> float:
        return 0.0

    def terminal_reward(self, outcome: str, survivors: Units) -> float:
        """On a win: the share of the group's life left, counting dead units as 0."""
        if outcome != "win" or not self._group_size:
            return 0.0
        left = sum(_fraction(unit.health, unit.health_max) for unit in survivors)
        return self.win_reward_scale * left / self._group_size

    def baseline_policies(self) -> dict[str, Callable[[np.ndarray], int]]:
        return {"attack": lambda observation: ATTACK_CLOSEST, "flank": _flank_rule}

    def situation(self, observation: np.ndarray) -> str:
        return "all"


def _flank_rule(observation: np.ndarray) -> int:
    """Run around the enemies to their far side, on the side of the line from the allies to the
    enemies that the unit is already on, then attack the closest. Attacks right away once the
    closest enemy is within FLANK_ENGAGE or the unit is past the enemies' center."""
    enemies = _present(observation, FIRST_ENEMY, ENEMY_INPUTS, SEEN_ENEMIES)
    if not enemies:
        return ATTACK_CLOSEST
    closest_distance = observation[FIRST_ENEMY + 3] * DISTANCE_SCALE
    enemy_center = _mean(enemies)
    group_center = _mean(_present(observation, FIRST_ALLY, ALLY_INPUTS, SEEN_ALLIES) + [Point2((0, 0))])
    approach = enemy_center - group_center
    if closest_distance <= FLANK_ENGAGE or approach.length < 0.01:
        return ATTACK_CLOSEST
    forward = approach.normalized
    if _dot(-enemy_center, forward) >= 0:  # the unit (at 0, 0) is past the enemies' center
        return ATTACK_CLOSEST
    beside = Point2((-forward.y, forward.x))
    side = 1.0 if _dot(-group_center, beside) >= 0 else -1.0
    goal = enemy_center + (beside * side + forward) * FLANK_RADIUS
    return FIRST_MOVE + max(range(len(DIRECTIONS)), key=lambda k: _dot(DIRECTIONS[k], goal))


def _present(observation: np.ndarray, first: int, size: int, count: int) -> list[Point2]:
    """Relative positions (in game distance) of the listed units that are present."""
    positions = []
    for index in range(count):
        start = first + index * size
        if observation[start]:
            positions.append(Point2(observation[start + 1:start + 3]) * DISTANCE_SCALE)
    return positions


def _mean(points: list[Point2]) -> Point2:
    return Point2((sum(p.x for p in points) / len(points), sum(p.y for p in points) / len(points)))
