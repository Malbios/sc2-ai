"""A mutalisk group against marines (GroupFightTask), for the headroom survey.

Mutalisks fly at 5.6, marines walk at 3.15 and outrange them (5 against 3), so shooting and
backing off while the weapon cools down should matter; bunched mutalisks also hit harder, their
attack bouncing to units near the target. Flying, they see the edge of the playable area as
their only walls.
"""

import numpy as np
from sc2.game_info import GameInfo
from sc2.position import Point2

from tools.rl.examples.free_kite_task import RAY_LENGTH
from tools.rl.examples.group_fight_task import (
    ATTACK_CLOSEST,
    LIFE,
    GroupFightTask,
    Rule,
    allies_center,
    closest_enemy_distance,
    kite_rule,
    life_kite_rule,
    move_toward,
)

KITE_DISTANCE = 6.0  # a marine's range plus a margin
SHORT_KITE_DISTANCE = 4.5  # the best kite distance in the headroom survey
HURT_LIFE_SHARES = (0.5, 0.7, 0.9)
CLUMP_RADIUS = 1.5
CLUMP_SAFE_DISTANCE = 7.0  # regrouping only while no marine is this close


class MutaliskGroupTask(GroupFightTask):
    cooldown_scale = 24.0

    def __init__(self):
        super().__init__()
        self._playable_area = None

    def start_game(self, game_info: GameInfo) -> None:
        super().start_game(game_info)
        self._playable_area = game_info.playable_area

    def _free_distance(self, start: Point2, direction: Point2) -> float:
        """How far a flying unit can go from `start` in `direction` before leaving the playable
        area, up to RAY_LENGTH."""
        area = self._playable_area
        limits = [RAY_LENGTH]
        for position, step, low, high in ((start.x, direction.x, area.x, area.x + area.width),
                                          (start.y, direction.y, area.y, area.y + area.height)):
            if step > 1e-9:
                limits.append((high - position) / step)
            elif step < -1e-9:
                limits.append((low - position) / step)
        return max(min(limits), 0.0)

    def baseline_policies(self) -> dict[str, Rule]:
        return {
            "attack": lambda observation: ATTACK_CLOSEST,
            "kite_4.5": lambda observation: kite_rule(observation, SHORT_KITE_DISTANCE),
            "kite_6": lambda observation: kite_rule(observation, KITE_DISTANCE),
            "kite_8": lambda observation: kite_rule(observation, 8.0),
            "clump_kite_6": lambda observation: clump_kite_rule(observation, KITE_DISTANCE, CLUMP_RADIUS),
            "clump_kite_6_r1": lambda observation: clump_kite_rule(observation, KITE_DISTANCE, 1.0),
            "clump_kite_6_r3": lambda observation: clump_kite_rule(observation, KITE_DISTANCE, 3.0),
            **{f"hurt_kite_{life:g}": lambda observation, life=life: hurt_kite_rule(observation, life)
               for life in HURT_LIFE_SHARES},
            "life_kite": lambda observation: life_kite_rule(observation, SHORT_KITE_DISTANCE),
        }


def hurt_kite_rule(observation: np.ndarray, hurt_below: float) -> int:
    """kite_4.5 below `hurt_below` of full life, else always attack the closest: a hurt mutalisk
    stepping back hands the marines' fire (at the closest target) to a healthy one."""
    if observation[LIFE] < hurt_below:
        return kite_rule(observation, SHORT_KITE_DISTANCE)
    return ATTACK_CLOSEST


def clump_kite_rule(observation: np.ndarray, distance: float, radius: float) -> int:
    """Kite (see kite_rule), but first fly to the seen allies' center while it is more than
    `radius` away and no enemy is within CLUMP_SAFE_DISTANCE."""
    center = allies_center(observation)
    if center is not None and center.length > radius and closest_enemy_distance(observation) > CLUMP_SAFE_DISTANCE:
        return move_toward(center)
    return kite_rule(observation, distance)
