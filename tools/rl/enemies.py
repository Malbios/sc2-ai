"""Who controls the enemy side of a training fight: the built-in AI, a simple script, or a frozen
copy of a trained model."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from sc2.bot_ai import BotAI
from sc2.data import Difficulty, Race
from sc2.ids.effect_id import EffectId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.player import Bot, Computer
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units

from tools.rl.config import EnemyConfig, load_class

# Units that never take part in a fight, even though a player may own them.
NON_FIGHTERS = {
    UnitTypeId.SCV, UnitTypeId.PROBE, UnitTypeId.DRONE, UnitTypeId.MULE,
    UnitTypeId.LARVA, UnitTypeId.EGG, UnitTypeId.OVERLORD,
}
# Units that stay with their army without fighting (see Scenario.enemy_support).
SUPPORT = {UnitTypeId.OVERSEER}


def fighters(units: Units) -> Units:
    return units.filter(lambda unit: unit.type_id not in NON_FIGHTERS | SUPPORT)


@dataclass(frozen=True)
class EnemyBriefing:
    """How the scripted enemy fights the current fight."""

    behavior: str  # "chase", "leash" or "kite", see Scenario.enemy_behavior
    home: Point2  # where the enemy group spawned
    leash: float
    bile_dodge_reaction: float | None = None  # seconds before stepping out of a bile; None: never


# A leashed unit walking home fights again once it is this close to home.
HOME_RADIUS = 1.5
# A dodging unit keeps this far outside a bile's edge, so it also doesn't walk into one.
DODGE_MARGIN = 0.5
# A kiting unit steps back this far while its weapon cools down with a target this close.
KITE_TRIGGER = 3.0
KITE_STEP = 2.0


def dodge_point(position: Point2, radius: float, biles: list[tuple[Point2, float]]) -> Point2 | None:
    """Where a unit at `position` steps to stay clear of the first of `biles` (center, radius) it
    is in or next to: straight away from its center. None if it is clear of all of them."""
    for center, bile_radius in biles:
        reach = bile_radius + radius
        offset = position - center
        if offset.length < reach + DODGE_MARGIN:
            away = offset.normalized if offset.length > 0.01 else Point2((1.0, 0.0))
            return center + away * (reach + 2 * DODGE_MARGIN)
    return None


def leash_order(unit: Unit, targets: Units, briefing: EnemyBriefing, returning: bool) -> tuple[Unit | Point2, bool]:
    """A leashed unit's order: the target to attack or the point to walk to, and whether it is now
    walking home. Past the leash it walks home and ignores everything until it gets there."""
    distance_home = unit.position.distance_to(briefing.home)
    if distance_home > briefing.leash:
        returning = True
    elif distance_home < HOME_RADIUS:
        returning = False
    if returning:
        return briefing.home, True
    return min(targets, key=lambda target: unit.position.distance_to(target.position)), False


def kite_order(unit: Unit, targets: Units) -> Unit | Point2:
    """A kiting unit's order: the closest target to attack, or, while its weapon cools down with a
    target within KITE_TRIGGER, the point to step back to, away from the targets' center."""
    closest = min(targets, key=lambda target: unit.position.distance_to(target.position))
    if unit.weapon_cooldown <= 0 or unit.position.distance_to(closest.position) > KITE_TRIGGER:
        return closest
    away = unit.position - Point2.center([target.position for target in targets])
    if away.length < 0.01:
        return closest
    return unit.position + away.normalized * KITE_STEP


class ScriptedEnemyBot(BotAI):
    """Every fighter attacks the closest enemy fighter. Predictable on purpose. With a leash
    briefing, it gives up and walks home past the leash distance instead of chasing forever. With
    a kite briefing, it steps back from close targets while its weapon cools down.
    With a bile dodge reaction, a unit steps out of a ravager's bile once it has seen the bile for
    that long. Support units (an Overseer) stay at the center of the fighters."""

    def __init__(self, decision_interval: int, briefing: Callable[[], EnemyBriefing | None] = lambda: None):
        super().__init__()
        self.decision_interval = decision_interval
        self.briefing = briefing
        self.returning: set[int] = set()
        self.biles_seen: dict[Point2, int] = {}  # bile center -> game loop it was first seen

    def _noticed_biles(self, reaction: float) -> list[tuple[Point2, float]]:
        loop = self.state.game_loop
        biles = {Point2(position): effect.radius for effect in self.state.effects
                 if effect.id == EffectId.RAVAGERCORROSIVEBILECP for position in effect.positions}
        self.biles_seen = {center: self.biles_seen.get(center, loop) for center in biles}
        return [(center, radius) for center, radius in biles.items()
                if loop - self.biles_seen[center] >= reaction * 22.4]

    async def on_start(self):
        # No debug_show_map here: the learner already turned on full vision for the whole game,
        # and a second call would turn it off again.
        self.client.game_step = self.decision_interval

    async def on_step(self, iteration: int):
        own = fighters(self.units)
        if own:
            for unit in self.units.of_type(SUPPORT):
                unit.move(own.center)
        targets = fighters(self.enemy_units).visible  # a burrowed unit is listed but can't be attacked
        if not targets:
            return
        briefing = self.briefing()
        dodging = briefing is not None and briefing.bile_dodge_reaction is not None
        biles = self._noticed_biles(briefing.bile_dodge_reaction) if dodging else []
        for unit in own:
            escape = dodge_point(unit.position, unit.radius, biles)
            if escape is not None:
                unit.move(escape)
                continue
            if briefing is not None and briefing.behavior == "kite":
                order = kite_order(unit, targets)
                if isinstance(order, Point2):
                    unit.move(order)
                else:
                    unit.attack(order)
                continue
            if briefing is None or briefing.behavior != "leash":
                unit.attack(targets.closest_to(unit))
                continue
            order, returning = leash_order(unit, targets, briefing, unit.tag in self.returning)
            if returning:
                self.returning.add(unit.tag)
                unit.move(order)
            else:
                self.returning.discard(unit.tag)
                unit.attack(order)


class FrozenEnemyBot(BotAI):
    """Every fighter is controlled by a trained model that doesn't learn during the game."""

    def __init__(self, decision_interval: int, task_path: str, model_path: str):
        super().__init__()
        from stable_baselines3 import PPO  # imported here so the other modes don't need torch

        self.decision_interval = decision_interval
        self.task = load_class(task_path)()
        self.model = PPO.load(model_path, device="cpu")

    async def on_start(self):
        self.client.game_step = self.decision_interval  # no debug_show_map, see ScriptedEnemyBot

    async def on_step(self, iteration: int):
        own = fighters(self.units)
        enemies = fighters(self.enemy_units)
        if not own or not enemies:
            return
        observations = np.stack([self.task.observe(unit, own.tags_not_in({unit.tag}), enemies) for unit in own])
        actions, _ = self.model.predict(observations, deterministic=False)
        for unit, action in zip(own, actions):
            self.task.apply(unit, int(action), own.tags_not_in({unit.tag}), enemies)


def make_enemy_player(
    config: EnemyConfig,
    decision_interval: int,
    model_path: str | None = None,
    briefing: Callable[[], EnemyBriefing | None] = lambda: None,
):
    """The enemy player for a game. `model_path` overrides config.model (used by self-play);
    `briefing` tells the scripted enemy how to fight the current fight."""
    race = Race[config.race]
    if config.mode == "builtin":
        return Computer(race, Difficulty[config.difficulty])
    if config.mode == "scripted":
        return Bot(race, ScriptedEnemyBot(decision_interval, briefing), name="ScriptedEnemy")
    return Bot(race, FrozenEnemyBot(decision_interval, config.task, model_path or config.model), name="FrozenEnemy")
