"""TrackingKiteTask for roaches, plus burrowing: a burrowed roach heals 7 life per second, can move
with Tunneling Claws, and can't be attacked by an enemy without detection. It can't attack either.

Needs the learner upgrades BURROW and TUNNELINGCLAWS in the config.
"""

from collections.abc import Callable

import numpy as np
from gymnasium import spaces
from sc2.ids.ability_id import AbilityId
from sc2.unit import Unit
from sc2.units import Units

from tools.rl.examples.free_kite_task import FIRST_MOVE, FIRST_MOVEMENT, SEEN_ENEMIES, DIRECTIONS, TrackingKiteTask

LIFE = 1
BURROWED = FIRST_MOVEMENT + 2 * SEEN_ENEMIES
BURROW, UNBURROW = FIRST_MOVE + len(DIRECTIONS), FIRST_MOVE + len(DIRECTIONS) + 1

# sharpy-sc2's roach rule
BURROW_BELOW = 0.4
UNBURROW_ABOVE = 0.7


class RoachKiteTask(TrackingKiteTask):
    observation_space = spaces.Box(-np.inf, np.inf, shape=(BURROWED + 1,), dtype=np.float32)
    action_space = spaces.Discrete(UNBURROW + 1)

    def observe(self, unit: Unit, allies: Units, enemies: Units) -> np.ndarray:
        """TrackingKiteTask's inputs, then 1 if this roach is burrowed, else 0."""
        observation = super().observe(unit, allies, enemies)
        observation[BURROWED] = 1.0 if unit.is_burrowed else 0.0
        return observation

    def apply(self, unit: Unit, action: int, allies: Units, enemies: Units) -> None:
        if action == BURROW:
            unit(AbilityId.BURROWDOWN_ROACH)
        elif action == UNBURROW:
            unit(AbilityId.BURROWUP_ROACH)
        else:
            super().apply(unit, action, allies, enemies)

    def baseline_policies(self) -> dict[str, Callable[[np.ndarray], int]]:
        smart = super().baseline_policies()["smart"]
        return {**super().baseline_policies(), "burrow": lambda observation: _burrow_rule(observation, smart)}


def _burrow_rule(observation: np.ndarray, otherwise: Callable[[np.ndarray], int]) -> int:
    """Burrow below 40% life and stay burrowed until above 70%; otherwise `otherwise`."""
    life, burrowed = observation[LIFE], observation[BURROWED]
    if burrowed:
        return UNBURROW if life > UNBURROW_ABOVE else BURROW
    return BURROW if life < BURROW_BELOW else otherwise(observation)
