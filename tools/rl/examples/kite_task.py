"""A toy micro task for any ground unit: attack, retreat, advance or hold, based on the nearest
units around it. Used to prove the training tooling works ("1 stalker vs 2 roaches"); real tasks
will live with the bot and read its World Model."""

import numpy as np
from gymnasium import spaces
from sc2.unit import Unit
from sc2.units import Units

from tools.rl.task import MicroTask

NEAREST_ENEMIES = 4
NEAREST_ALLIES = 2
ENEMY_FEATURES = 10
ALLY_FEATURES = 4
DISTANCE_SCALE = 15.0  # distances are divided by this, then clipped to [-1, 1]
RETREAT_DISTANCE = 3.0

ATTACK_NEAREST, ATTACK_WEAKEST, RETREAT, ADVANCE, HOLD = range(5)


def _fraction(value: float, maximum: float) -> float:
    return value / maximum if maximum else 0.0


class KiteTask(MicroTask):
    observation_space = spaces.Box(
        -1.0, 1.0, shape=(4 + NEAREST_ENEMIES * ENEMY_FEATURES + NEAREST_ALLIES * ALLY_FEATURES,), dtype=np.float32
    )
    action_space = spaces.Discrete(5)

    def observe(self, unit: Unit, allies: Units, enemies: Units) -> np.ndarray:
        features = [
            _fraction(unit.health, unit.health_max),
            _fraction(unit.shield, unit.shield_max),
            min(max(unit.weapon_cooldown, 0) / 30, 1.0),  # game frames until it can shoot again
            min(unit.ground_range / 10, 1.0),
        ]

        nearest = sorted(enemies, key=unit.distance_to)[:NEAREST_ENEMIES]
        for slot in range(NEAREST_ENEMIES):
            if slot >= len(nearest):
                features += [0.0] * ENEMY_FEATURES
                continue
            enemy = nearest[slot]
            offset = enemy.position - unit.position
            distance = unit.distance_to(enemy)
            reach = unit.radius + enemy.radius
            features += [
                1.0,  # slot is filled
                offset.x / DISTANCE_SCALE,
                offset.y / DISTANCE_SCALE,
                distance / DISTANCE_SCALE,
                _fraction(enemy.health, enemy.health_max),
                _fraction(enemy.shield, enemy.shield_max),
                min(enemy.ground_range / 10, 1.0),
                min(enemy.ground_dps / 30, 1.0),
                float(distance <= unit.ground_range + reach),  # I can hit it
                float(distance <= enemy.ground_range + reach),  # it can hit me
            ]

        nearest_allies = sorted(allies, key=unit.distance_to)[:NEAREST_ALLIES]
        for slot in range(NEAREST_ALLIES):
            if slot >= len(nearest_allies):
                features += [0.0] * ALLY_FEATURES
                continue
            ally = nearest_allies[slot]
            offset = ally.position - unit.position
            features += [1.0, offset.x / DISTANCE_SCALE, offset.y / DISTANCE_SCALE, _fraction(ally.health, ally.health_max)]

        return np.clip(np.asarray(features, dtype=np.float32), -1.0, 1.0)

    def terminal_reward(self, outcome: str, survivors: Units) -> float:
        # A time-out costs as much as a loss. With a free time-out, running away forever is the
        # easiest way to avoid damage, and the model learns that instead of kiting.
        return {"win": 1.0, "loss": -1.0, "timeout": -1.0}.get(outcome, 0.0)

    def apply(self, unit: Unit, action: int, allies: Units, enemies: Units) -> None:
        if not enemies:
            unit.stop()
            return
        nearest = enemies.closest_to(unit)

        if action == ATTACK_NEAREST:
            unit.attack(nearest)
        elif action == ATTACK_WEAKEST:
            in_range = enemies.filter(unit.target_in_range)
            pool = in_range or enemies
            unit.attack(min(pool, key=lambda enemy: enemy.health + enemy.shield))
        elif action == RETREAT:
            away = unit.position - enemies.center
            if away.length < 0.1:
                away = unit.position - nearest.position
            unit.move(unit.position + away.normalized * RETREAT_DISTANCE)
        elif action == ADVANCE:
            unit.move(nearest.position)
        else:
            unit.hold_position()
