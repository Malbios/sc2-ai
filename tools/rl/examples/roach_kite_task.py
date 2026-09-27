"""TrackingKiteTask for roaches, plus burrowing: a burrowed roach heals 7 life per second, can move
with Tunneling Claws, and can't be attacked unless an enemy detector (e.g. an Overseer) is within
its detection range. It can't attack either.

Needs the learner upgrades BURROW and TUNNELINGCLAWS in the config.
"""

from collections.abc import Callable

import numpy as np
from gymnasium import spaces
from sc2.ids.ability_id import AbilityId
from sc2.unit import Unit
from sc2.units import Units

from tools.rl.examples.free_kite_task import (
    DIRECTIONS,
    DISTANCE_SCALE,
    FIRST_MOVE,
    FIRST_MOVEMENT,
    SEEN_ENEMIES,
    TrackingKiteTask,
)

LIFE = 1
BURROWED = FIRST_MOVEMENT + 2 * SEEN_ENEMIES
DETECTOR_PRESENT = BURROWED + 1
DETECTION_MARGIN = BURROWED + 2
BURROW, UNBURROW = FIRST_MOVE + len(DIRECTIONS), FIRST_MOVE + len(DIRECTIONS) + 1

# sharpy-sc2's roach rule
BURROW_BELOW = 0.4
UNBURROW_ABOVE = 0.7


class RoachKiteTask(TrackingKiteTask):
    observation_space = spaces.Box(-np.inf, np.inf, shape=(DETECTION_MARGIN + 1,), dtype=np.float32)
    action_space = spaces.Discrete(UNBURROW + 1)

    def __init__(self):
        super().__init__()
        self._detectors: Units | list = []

    def see_detectors(self, detectors: Units) -> None:
        self._detectors = detectors

    def observe(self, unit: Unit, allies: Units, enemies: Units) -> np.ndarray:
        """TrackingKiteTask's inputs, then: 1 if this roach is burrowed; 1 if an enemy detector is
        in sight; and how far this roach is outside the closest detector's range, divided by 10
        (negative inside it, where burrowing doesn't hide; 0 without a detector)."""
        observation = super().observe(unit, allies, enemies)
        observation[BURROWED] = 1.0 if unit.is_burrowed else 0.0
        if self._detectors:
            margin = min(unit.position.distance_to(detector.position) - detector.detect_range for detector in self._detectors)
            observation[DETECTOR_PRESENT] = 1.0
            observation[DETECTION_MARGIN] = margin / DISTANCE_SCALE
        return observation

    def apply(self, unit: Unit, action: int, allies: Units, enemies: Units) -> None:
        if action == BURROW:
            unit(AbilityId.BURROWDOWN_ROACH)
        elif action == UNBURROW:
            unit(AbilityId.BURROWUP_ROACH)
        else:
            super().apply(unit, action, allies, enemies)

    def baseline_policies(self) -> dict[str, Callable[[np.ndarray], int]]:
        policies = super().baseline_policies()
        smart = policies["smart"]
        return {
            **policies,
            "burrow": lambda observation: _burrow_rule(observation, smart),
            "careful_burrow": lambda observation: _careful_burrow_rule(observation, smart),
        }


def _burrow_rule(observation: np.ndarray, otherwise: Callable[[np.ndarray], int]) -> int:
    """Burrow below 40% life and stay burrowed until above 70%; otherwise `otherwise`."""
    life, burrowed = observation[LIFE], observation[BURROWED]
    if burrowed:
        return UNBURROW if life > UNBURROW_ABOVE else BURROW
    return BURROW if life < BURROW_BELOW else otherwise(observation)


def _careful_burrow_rule(observation: np.ndarray, otherwise: Callable[[np.ndarray], int]) -> int:
    """The burrow rule, but only where no detector sees it: inside detection it doesn't burrow,
    and a burrowed roach that gets detected comes back up."""
    detected_here = observation[DETECTOR_PRESENT] and observation[DETECTION_MARGIN] < 0
    if not detected_here:
        return _burrow_rule(observation, otherwise)
    return UNBURROW if observation[BURROWED] else otherwise(observation)
