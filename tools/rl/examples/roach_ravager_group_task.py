"""Roaches with ravagers against marines and marauders (GroupFightTask): the fight the headroom
survey picked as the RL target.

Ravagers can also cast corrosive bile (60 damage in a small circle, landing about 1.6 s after
the cast), which enemies can step out of. Bile makes the enemy move and breaks its formation,
so when and at whom to bile, and how far to lead it, is micro that attack-move lacks.
Roaches and ravagers share the model; a roach choosing a bile action attacks the closest instead.
"""

import numpy as np
from gymnasium import spaces
from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units

from tools.rl.examples.free_kite_task import DISTANCE_SCALE, _by_distance
from tools.rl.examples.group_fight_task import (
    ATTACK_CLOSEST,
    ATTACK_DANGEROUS,
    ENEMY_INPUTS,
    FIRST_ALLY,
    FIRST_ENEMY,
    FIRST_EXTRA_ACTION,
    INPUTS,
    SEEN_ENEMIES,
    GroupFightTask,
    Rule,
    closest_enemy_distance,
    life_kite_rule,
    most_dangerous,
)
from tools.rl.examples.ravager_task import BILE, BILE_FLIGHT_SECONDS, BILE_RANGE, DECISION_SECONDS

IS_RAVAGER, BILE_READY = INPUTS, INPUTS + 1

# The bile actions: whom to bile (the closest, or the most dangerous within bile range) times the
# lead (0: where it is now, 1: where it will be if it keeps moving straight).
BILE_TARGETS = ("closest", "dangerous")
BILE_LEADS = (0.0, 0.125, 0.25, 0.5, 0.75, 1.0, 1.5)
# The clump condition: other seen enemies within this distance of the bile's target.
CLUMP_RADIUS = 2.0
CLUMP_SIZES = (0, 1, 2)
# Game steps after a bile cast in which damage may be the bile's: its 1.6 s flight is 12 steps
# at decision interval 3, plus a margin.
BILE_CREDIT_STEPS = 16
# The tuned bile rules (bile target, clump size; lead 0) that also get a version which rotates
# hurt units out between shots (life_kite_rule), within this distance of an enemy.
ROTATING_BILE_RULES = {"bile_dangerous_lead_0": ("dangerous", 0), "bile_closest_lead_0_clump_2": ("closest", 2)}
ROTATION_DISTANCE = 5.0


def bile_action(target: str, lead: float) -> int:
    return FIRST_EXTRA_ACTION + BILE_TARGETS.index(target) * len(BILE_LEADS) + BILE_LEADS.index(lead)


# Then one more bile action per seen enemy: bile the i-th closest where it is now.
FIRST_BILE_SEEN = bile_action(BILE_TARGETS[-1], BILE_LEADS[-1]) + 1


def bile_seen_action(index: int) -> int:
    return FIRST_BILE_SEEN + index


class RoachRavagerGroupTask(GroupFightTask):
    observation_space = spaces.Box(-np.inf, np.inf, shape=(BILE_READY + 1,), dtype=np.float32)
    action_space = spaces.Discrete(FIRST_BILE_SEEN + SEEN_ENEMIES)
    wants_abilities = True
    cooldown_scale = 32.0  # a roach's; a ravager's is 26

    def __init__(self):
        super().__init__()
        self._abilities: dict[int, set[AbilityId]] = {}
        self._movement: dict[tuple[int, int], Point2] = {}
        self._game_steps = 0
        self._bile_casts: dict[int, int] = {}  # ravager tag -> game step of its last bile

    def start_episode(self) -> None:
        super().start_episode()
        self._game_steps = 0
        self._bile_casts.clear()

    def share_team_reward(self, team: float, units: Units) -> dict[int, float]:
        """Like ZerglingSurroundTask's, but a ravager whose bile may be landing (cast within the
        last BILE_CREDIT_STEPS game steps) counts as attacking too."""
        self._game_steps += 1
        return super().share_team_reward(team, units)

    def _attacked(self, unit: Unit) -> bool:
        cast = self._bile_casts.get(unit.tag)
        return super()._attacked(unit) or (cast is not None and self._game_steps - cast <= BILE_CREDIT_STEPS)

    def see_abilities(self, available: dict[int, set[AbilityId]]) -> None:
        self._abilities = available

    def observe(self, unit: Unit, allies: Units, enemies: Units) -> np.ndarray:
        """GroupFightTask's inputs, then 1 if this unit is a ravager, and 1 if its bile is ready."""
        for enemy in enemies:
            last = self._last_positions.get((unit.tag, enemy.tag))
            self._movement[(unit.tag, enemy.tag)] = enemy.position - last if last is not None else Point2((0, 0))
        observation = super().observe(unit, allies, enemies)
        observation[IS_RAVAGER] = 1.0 if unit.type_id == UnitTypeId.RAVAGER else 0.0
        observation[BILE_READY] = 1.0 if BILE in self._abilities.get(unit.tag, set()) else 0.0
        return observation

    def apply(self, unit: Unit, action: int, allies: Units, enemies: Units) -> None:
        if action < FIRST_EXTRA_ACTION or not enemies:
            super().apply(unit, action, allies, enemies)
        elif unit.type_id != UnitTypeId.RAVAGER:
            super().apply(unit, ATTACK_CLOSEST, allies, enemies)
        elif action >= FIRST_BILE_SEEN:
            by_distance = _by_distance(unit, enemies)
            self._cast_bile(unit, by_distance[min(action - FIRST_BILE_SEEN, len(by_distance) - 1)], 0.0)
        else:
            target_index, lead_index = divmod(action - FIRST_EXTRA_ACTION, len(BILE_LEADS))
            whom, lead = BILE_TARGETS[target_index], BILE_LEADS[lead_index]
            self._cast_bile(unit, self._bile_target(unit, whom, enemies), lead)

    def _cast_bile(self, unit: Unit, target: Unit, lead: float) -> None:
        self._bile_casts[unit.tag] = self._game_steps
        unit(BILE, self.bile_point(unit, target, lead))

    def bile_point(self, unit: Unit, target: Unit, lead: float) -> Point2:
        movement = self._movement.get((unit.tag, target.tag), Point2((0, 0)))
        return target.position + movement * (lead * BILE_FLIGHT_SECONDS / DECISION_SECONDS)

    @staticmethod
    def _bile_target(unit: Unit, whom: str, enemies: Units) -> Unit:
        by_distance = _by_distance(unit, enemies)
        in_bile_range = [enemy for enemy in by_distance if unit.position.distance_to(enemy.position) <= BILE_RANGE]
        if whom == "dangerous" and in_bile_range:
            return most_dangerous(unit, in_bile_range)
        return by_distance[0]

    def baseline_policies(self) -> dict[str, Rule]:
        """attack, and for every bile target, lead and clump size k: bile_{target}_lead_{lead}
        (k = 0) or bile_{target}_lead_{lead}_clump_{k}, shooting the most dangerous otherwise."""
        rules = {"attack": lambda observation: ATTACK_CLOSEST}
        for target in BILE_TARGETS:
            for lead in BILE_LEADS:
                for clump in CLUMP_SIZES:
                    name = f"bile_{target}_lead_{lead:g}" + (f"_clump_{clump}" if clump else "")
                    rules[name] = lambda observation, b=bile_action(target, lead), k=clump: bile_rule(
                        observation, b, ATTACK_DANGEROUS, k)
        for name, (target, clump) in ROTATING_BILE_RULES.items():
            rules[f"{name}_life_kite_{ROTATION_DISTANCE:g}"] = (
                lambda observation, b=bile_action(target, 0.0), k=clump: bile_rule(
                    observation, b, life_kite_rule(observation, ROTATION_DISTANCE, ATTACK_DANGEROUS), k))
        return rules


def bile_rule(observation: np.ndarray, bile: int, otherwise: int, clump: int = 0) -> int:
    """`bile` when bile is ready, the closest enemy is within bile range and has at least `clump`
    other seen enemies within CLUMP_RADIUS; else `otherwise`. Roaches never have bile ready."""
    ready = observation[BILE_READY] and closest_enemy_distance(observation) <= BILE_RANGE
    return bile if ready and clump_size(observation) >= clump else otherwise


def clump_size(observation: np.ndarray) -> int:
    """How many other seen enemies are within CLUMP_RADIUS of the closest one."""
    sizes = clump_sizes(observation)
    return sizes[0] if sizes else 0


def clump_sizes(observation: np.ndarray) -> list[int]:
    """For each seen enemy (closest first), how many other seen enemies are within CLUMP_RADIUS."""
    positions = [Point2(observation[start + 1:start + 3]) * DISTANCE_SCALE
                 for start in range(FIRST_ENEMY, FIRST_ALLY, ENEMY_INPUTS) if observation[start]]
    return [sum(1 for other in positions if other is not position and other.distance_to(position) <= CLUMP_RADIUS)
            for position in positions]
