"""Who controls the enemy side of a training fight: the built-in AI, a simple script, or a frozen
copy of a trained model."""

import numpy as np
from sc2.bot_ai import BotAI
from sc2.data import Difficulty, Race
from sc2.ids.unit_typeid import UnitTypeId
from sc2.player import Bot, Computer
from sc2.units import Units

from tools.rl.config import EnemyConfig, load_class

# Units that never take part in a fight, even though a player may own them.
NON_FIGHTERS = {
    UnitTypeId.SCV, UnitTypeId.PROBE, UnitTypeId.DRONE, UnitTypeId.MULE,
    UnitTypeId.LARVA, UnitTypeId.EGG, UnitTypeId.OVERLORD,
}


def fighters(units: Units) -> Units:
    return units.filter(lambda unit: unit.type_id not in NON_FIGHTERS)


class ScriptedEnemyBot(BotAI):
    """Every fighter attacks the closest enemy fighter. Predictable on purpose."""

    def __init__(self, decision_interval: int):
        super().__init__()
        self.decision_interval = decision_interval

    async def on_start(self):
        self.client.game_step = self.decision_interval
        await self.client.debug_show_map()

    async def on_step(self, iteration: int):
        targets = fighters(self.enemy_units)
        if not targets:
            return
        for unit in fighters(self.units):
            unit.attack(targets.closest_to(unit))


class FrozenEnemyBot(BotAI):
    """Every fighter is controlled by a trained model that doesn't learn during the game."""

    def __init__(self, decision_interval: int, task_path: str, model_path: str):
        super().__init__()
        from stable_baselines3 import PPO  # imported here so the other modes don't need torch

        self.decision_interval = decision_interval
        self.task = load_class(task_path)()
        self.model = PPO.load(model_path, device="cpu")

    async def on_start(self):
        self.client.game_step = self.decision_interval
        await self.client.debug_show_map()

    async def on_step(self, iteration: int):
        own = fighters(self.units)
        enemies = fighters(self.enemy_units)
        if not own or not enemies:
            return
        observations = np.stack([self.task.observe(unit, own.tags_not_in({unit.tag}), enemies) for unit in own])
        actions, _ = self.model.predict(observations, deterministic=False)
        for unit, action in zip(own, actions):
            self.task.apply(unit, int(action), own.tags_not_in({unit.tag}), enemies)


def make_enemy_player(config: EnemyConfig, decision_interval: int, model_path: str | None = None):
    """The enemy player for a game. `model_path` overrides config.model (used by self-play)."""
    race = Race[config.race]
    if config.mode == "builtin":
        return Computer(race, Difficulty[config.difficulty])
    if config.mode == "scripted":
        return Bot(race, ScriptedEnemyBot(decision_interval), name="ScriptedEnemy")
    return Bot(race, FrozenEnemyBot(decision_interval, config.task, model_path or config.model), name="FrozenEnemy")
