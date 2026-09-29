"""A roach group that can burrow, against marines and marauders with a few scans (GroupFightTask),
for the headroom survey.

A burrowed roach heals 7 life per second, moves with Tunneling Claws, and can't be shot or shoot.
Against an enemy without detection that makes burrowing a free heal (the single-roach lessons);
here the enemy scans a unit that stays burrowed, a limited number of times per fight, so when to
burrow and when to come back up is a real trade.

Needs the learner upgrades BURROW and TUNNELINGCLAWS in the config.
"""

import numpy as np
from gymnasium import spaces
from sc2.ids.ability_id import AbilityId
from sc2.unit import Unit
from sc2.units import Units

from tools.rl.examples.group_fight_task import (
    ATTACK_CLOSEST,
    ATTACK_DANGEROUS,
    FIRST_EXTRA_ACTION,
    INPUTS,
    LIFE,
    GroupFightTask,
    Rule,
)
from tools.rl.examples.free_kite_task import DISTANCE_SCALE
from tools.rl.examples.roach_kite_task import BURROW_BELOW, UNBURROW_ABOVE, detection_margin

BURROWED, DETECTOR_PRESENT, DETECTION_MARGIN = INPUTS, INPUTS + 1, INPUTS + 2
BURROW, UNBURROW = FIRST_EXTRA_ACTION, FIRST_EXTRA_ACTION + 1

# Burrow below the first share of life, come back up above the second.
THRESHOLDS = ((0.3, 0.6), (BURROW_BELOW, UNBURROW_ABOVE), (0.5, 0.8))


class RoachBurrowGroupTask(GroupFightTask):
    observation_space = spaces.Box(-np.inf, np.inf, shape=(DETECTION_MARGIN + 1,), dtype=np.float32)
    action_space = spaces.Discrete(UNBURROW + 1)
    cooldown_scale = 32.0

    def __init__(self):
        super().__init__()
        self._detectors: list = []

    def see_detectors(self, detectors: list) -> None:
        self._detectors = detectors

    def observe(self, unit: Unit, allies: Units, enemies: Units) -> np.ndarray:
        """GroupFightTask's inputs, then: 1 if this roach is burrowed; 1 if the enemy detects
        anywhere (a detector or a scan); and how far this roach is outside the closest detection
        circle, divided by 10 (negative inside it; 0 without detection)."""
        observation = super().observe(unit, allies, enemies)
        observation[BURROWED] = 1.0 if unit.is_burrowed else 0.0
        if self._detectors:
            observation[DETECTOR_PRESENT] = 1.0
            observation[DETECTION_MARGIN] = detection_margin(unit.position, self._detectors) / DISTANCE_SCALE
        return observation

    def apply(self, unit: Unit, action: int, allies: Units, enemies: Units) -> None:
        if action == BURROW:
            unit(AbilityId.BURROWDOWN_ROACH)
        elif action == UNBURROW:
            unit(AbilityId.BURROWUP_ROACH)
        else:
            super().apply(unit, action, allies, enemies)

    def baseline_policies(self) -> dict[str, Rule]:
        rules = {
            "attack": lambda observation: ATTACK_CLOSEST,
            "threat": lambda observation: ATTACK_DANGEROUS,
        }
        for below, above in THRESHOLDS:
            suffix = "" if (below, above) == (BURROW_BELOW, UNBURROW_ABOVE) else f"_{below:g}_{above:g}"
            rules[f"burrow{suffix}"] = lambda observation, b=below, a=above: burrow_rule(observation, b, a)
            rules[f"careful_burrow{suffix}"] = lambda observation, b=below, a=above: careful_burrow_rule(observation, b, a)
        return rules


def burrow_rule(observation: np.ndarray, below: float, above: float) -> int:
    """Burrow below `below` of life and stay burrowed until above `above`; otherwise shoot the
    most dangerous enemy in range."""
    life = observation[LIFE]
    if observation[BURROWED]:
        return UNBURROW if life > above else BURROW
    return BURROW if life < below else ATTACK_DANGEROUS


def careful_burrow_rule(observation: np.ndarray, below: float, above: float) -> int:
    """The burrow rule, but not where the enemy detects: there it doesn't burrow, and a burrowed
    roach that gets detected comes back up to fight."""
    detected_here = observation[DETECTOR_PRESENT] and observation[DETECTION_MARGIN] < 0
    if not detected_here:
        return burrow_rule(observation, below, above)
    return UNBURROW if observation[BURROWED] else ATTACK_DANGEROUS
