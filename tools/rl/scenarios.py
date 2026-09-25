"""Training scenarios: which units fight which, how far apart they start, and how often each
scenario is picked."""

import math
import random
from dataclasses import dataclass

from sc2.ids.unit_typeid import UnitTypeId


@dataclass(frozen=True)
class Scenario:
    name: str
    learner: dict[UnitTypeId, int]  # units the model controls
    enemy: dict[UnitTypeId, int]
    weight: float = 1.0
    distance: tuple[float, float] = (8.0, 12.0)  # between the two groups at spawn
    time_limit: float = 45.0  # game seconds before the fight counts as a tie

    def swapped(self) -> "Scenario":
        """The same fight from the other side, for training the enemy's model in self-play."""
        return Scenario(self.name, self.enemy, self.learner, self.weight, self.distance, self.time_limit)


def _parse_units(units: dict, where: str) -> dict[UnitTypeId, int]:
    if not units:
        raise ValueError(f"{where}: needs at least one unit")
    parsed = {}
    for name, count in units.items():
        key = name.upper()
        if key not in UnitTypeId.__members__:
            raise ValueError(f"{where}: unknown unit type '{name}'")
        if int(count) < 1:
            raise ValueError(f"{where}: count for '{name}' must be at least 1")
        parsed[UnitTypeId[key]] = int(count)
    return parsed


def parse_scenarios(items: list[dict]) -> list[Scenario]:
    if not items:
        raise ValueError("'scenarios' must be a non-empty list")
    scenarios = []
    for item in items:
        name = item["name"]
        low, high = item.get("distance", (8.0, 12.0))
        scenario = Scenario(
            name=name,
            learner=_parse_units(item.get("learner"), f"scenario '{name}' learner"),
            enemy=_parse_units(item.get("enemy"), f"scenario '{name}' enemy"),
            weight=float(item.get("weight", 1.0)),
            distance=(float(low), float(high)),
            time_limit=float(item.get("time_limit", 45.0)),
        )
        if scenario.weight <= 0 or not 0 < low <= high:
            raise ValueError(f"scenario '{name}': weight must be positive and 0 < distance low <= high")
        scenarios.append(scenario)
    if len({s.name for s in scenarios}) != len(scenarios):
        raise ValueError("scenario names must be unique")
    return scenarios


class ScenarioSampler:
    def __init__(self, scenarios: list[Scenario], rng: random.Random):
        self.scenarios = scenarios
        self.rng = rng

    def sample(self) -> Scenario:
        return self.rng.choices(self.scenarios, weights=[s.weight for s in self.scenarios])[0]

    def by_name(self, name: str) -> Scenario:
        for scenario in self.scenarios:
            if scenario.name == name:
                return scenario
        raise KeyError(name)


def spawn_centers(
    center: tuple[float, float], scenario: Scenario, rng: random.Random
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Learner and enemy group centers: a random direction, a random distance within the
    scenario's range, centered on the given point."""
    angle = rng.uniform(0, 2 * math.pi)
    half = rng.uniform(*scenario.distance) / 2
    dx, dy = math.cos(angle) * half, math.sin(angle) * half
    return (center[0] - dx, center[1] - dy), (center[0] + dx, center[1] + dy)
