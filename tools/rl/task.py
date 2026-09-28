"""The plug-in a trained model is built around: what it sees, what it can do, and what counts as
doing well. Training and live play use the same task, so the model never sees different inputs
in a real game than it was trained on."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from gymnasium import spaces
from sc2.game_info import GameInfo
from sc2.ids.ability_id import AbilityId
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
    wants_abilities = False  # see_abilities costs one extra request to SC2 per game step
    uses_action_masks = False  # trains with sb3-contrib's MaskablePPO, see action_mask
    # Group tasks: all units decide together, each in its own slot and training stream (see
    # SC2GroupEnv). The number of slots caps the learner units per fight. 0: one unit at a time.
    group_slots = 0
    # A new model picks this action with this probability at first (the others equally), so that
    # exploring starts from a sensible default instead of from random picks. None: all equal.
    favored_action: int | None = None
    favored_action_probability = 0.85

    def observe(self, unit: Unit, allies: Units, enemies: Units) -> np.ndarray:
        """What the model sees when deciding for `unit`. `allies` excludes the unit itself;
        `enemies` are the enemy units of this fight that are still alive."""
        raise NotImplementedError

    def apply(self, unit: Unit, action: int, allies: Units, enemies: Units) -> None:
        """Turn the model's choice into python-sc2 commands for `unit`."""
        raise NotImplementedError

    def action_mask(self, unit: Unit, allies: Units, enemies: Units) -> np.ndarray:
        """Which actions `unit` may pick right now (True = allowed). Only used by tasks that set
        `uses_action_masks`; called right after observe for the same decision."""
        return np.ones(self.action_space.n, dtype=bool)

    def start_game(self, game_info: GameInfo) -> None:
        """Called once per game before any fight, for tasks that need the map (e.g. its pathing grid)."""

    def start_episode(self) -> None:
        """Called when a fight starts, for tasks that keep state across decisions."""

    def see_detectors(self, detectors: Units) -> None:
        """Called every game step before the units decide, with the visible enemy units and
        structures that detect burrowed or cloaked units, for tasks that care."""

    def see_abilities(self, available: dict[int, set[AbilityId]]) -> None:
        """Called every game step before the units decide, with the abilities each learner unit
        (by tag) can use right now, e.g. whether a spell is off cooldown. Only for tasks that set
        `wants_abilities`."""

    def baseline_policies(self) -> dict[str, Callable[[np.ndarray], int]]:
        """Hand-written policies (observation -> action) a trained model should be compared with."""
        return {}

    def situation(self, observation: np.ndarray) -> str:
        """A short label grouping observations when comparing a model with a baseline."""
        return "all"

    def reward(self, before: FightSnapshot, after: FightSnapshot) -> float:
        """Shared team reward for one game step: the share of the enemy's starting life taken
        away, minus the share of our own starting life lost."""
        dealt = (before.enemy_life - after.enemy_life) / max(after.enemy_start, 1.0)
        taken = (before.own_life - after.own_life) / max(after.own_start, 1.0)
        return dealt - taken

    def share_team_reward(self, team: float, units: Units) -> dict[int, float]:
        """Group tasks: how one game step's team reward is split among the living learner units
        (by tag). By default every unit gets all of it."""
        return {unit.tag: team for unit in units}

    def unit_reward(self, unit: Unit, allies: Units, enemies: Units) -> float:
        """Extra reward for one learner unit's state after a game step, added to reward()."""
        return 0.0

    def terminal_reward(self, outcome: str, survivors: Units) -> float:
        """Extra reward when a fight ends: 'win', 'loss', 'tie' (both sides died) or 'timeout'
        (the scenario's time limit ran out). `survivors` are the learner's units still alive."""
        return {"win": 1.0, "loss": -1.0}.get(outcome, 0.0)
