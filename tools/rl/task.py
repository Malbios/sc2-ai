"""The plug-in a trained model is built around: what it sees, what it can do, and what counts as
doing well. Training and live play use the same task, so the model never sees different inputs
in a real game than it was trained on."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from gymnasium import spaces
from sc2.unit import Unit
from sc2.units import Units


@dataclass(frozen=True)
class FightSnapshot:
    """Remaining life (HP + shields) of both sides of one fight at one moment."""

    own_life: float
    enemy_life: float
    own_start: float
    enemy_start: float


def life(units: Units) -> float:
    return sum(unit.health + unit.shield for unit in units)


class MicroTask:
    """Base class for tasks. A subclass sets the two spaces and implements observe and apply;
    the reward methods and start_episode have sensible defaults."""

    observation_space: spaces.Box
    action_space: spaces.Discrete

    def observe(self, unit: Unit, allies: Units, enemies: Units) -> np.ndarray:
        """What the model sees when deciding for `unit`. `allies` excludes the unit itself;
        `enemies` are the enemy units of this fight that are still alive."""
        raise NotImplementedError

    def apply(self, unit: Unit, action: int, allies: Units, enemies: Units) -> None:
        """Turn the model's choice into python-sc2 commands for `unit`."""
        raise NotImplementedError

    def start_episode(self) -> None:
        """Called when a fight starts, for tasks that keep state across decisions."""

    def baseline_policies(self) -> dict[str, Callable[[np.ndarray], int]]:
        """Hand-written policies (observation -> action) a trained model should be compared with."""
        return {}

    def reward(self, before: FightSnapshot, after: FightSnapshot) -> float:
        """Shared team reward for one game step: the share of the enemy's starting life taken
        away, minus the share of our own starting life lost."""
        dealt = (before.enemy_life - after.enemy_life) / max(after.enemy_start, 1.0)
        taken = (before.own_life - after.own_life) / max(after.own_start, 1.0)
        return dealt - taken

    def unit_reward(self, unit: Unit, allies: Units, enemies: Units) -> float:
        """Extra reward for one learner unit's state after a game step, added to reward()."""
        return 0.0

    def terminal_reward(self, outcome: str, survivors: Units) -> float:
        """Extra reward when a fight ends: 'win', 'loss', 'tie' (both sides died) or 'timeout'
        (the scenario's time limit ran out). `survivors` are the learner's units still alive."""
        return {"win": 1.0, "loss": -1.0}.get(outcome, 0.0)
