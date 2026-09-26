import math
import random
import unittest
from collections import Counter

from sc2.ids.unit_typeid import UnitTypeId

from tools.rl.config import parse_config, self_play_side
from tools.rl.scenarios import ScenarioSampler, parse_scenarios, spawn_centers
from tools.rl.task import FightSnapshot, MicroTask

SCENARIOS = [
    {"name": "one", "learner": {"Stalker": 1}, "enemy": {"Roach": 2}, "weight": 3, "distance": [9, 11]},
    {"name": "two", "learner": {"stalker": 2}, "enemy": {"Zergling": 4}},
]
CONFIG = {
    "learner": {"race": "Protoss", "task": "some.module:TaskA"},
    "enemy": {"mode": "scripted", "race": "Zerg", "task": "some.module:TaskB"},
    "scenarios": SCENARIOS,
}


class ScenarioTest(unittest.TestCase):
    def test_parse(self):
        one, two = parse_scenarios(SCENARIOS)
        self.assertEqual(one.learner, {UnitTypeId.STALKER: 1})
        self.assertEqual(two.learner, {UnitTypeId.STALKER: 2})  # names are case-insensitive
        self.assertEqual(one.distance, (9.0, 11.0))
        self.assertEqual(two.time_limit, 45.0)

    def test_rejects_bad_scenarios(self):
        bad = [
            [{**SCENARIOS[0], "learner": {"Dragoon": 1}}],
            [{**SCENARIOS[0], "enemy": {}}],
            [{**SCENARIOS[0], "weight": 0}],
            [{**SCENARIOS[0], "distance": [5, 2]}],
            [SCENARIOS[0], SCENARIOS[0]],
            [],
        ]
        for items in bad:
            with self.subTest(items=items), self.assertRaises(ValueError):
                parse_scenarios(items)

    def test_sampler_follows_weights(self):
        sampler = ScenarioSampler(parse_scenarios(SCENARIOS), random.Random(1))
        counts = Counter(sampler.sample().name for _ in range(4000))
        self.assertAlmostEqual(counts["one"] / 4000, 0.75, delta=0.03)
        self.assertEqual(sampler.by_name("two").name, "two")

    def test_spawn_centers_are_in_range_and_centered(self):
        scenario = parse_scenarios(SCENARIOS)[0]
        rng = random.Random(2)
        for _ in range(200):
            (ax, ay), (bx, by) = spawn_centers((32.0, 32.0), scenario, rng)
            self.assertTrue(9.0 <= math.dist((ax, ay), (bx, by)) <= 11.0 + 1e-9)
            self.assertAlmostEqual((ax + bx) / 2, 32.0)
            self.assertAlmostEqual((ay + by) / 2, 32.0)

    def test_swapped(self):
        one = parse_scenarios(SCENARIOS)[0]
        self.assertEqual(one.swapped().learner, one.enemy)
        self.assertEqual(one.swapped().swapped(), one)


class ConfigTest(unittest.TestCase):
    def test_defaults(self):
        config = parse_config(CONFIG)
        self.assertEqual(config.map, "MicroTraining")
        self.assertEqual(config.decision_interval, 4)
        self.assertIsNone(config.self_play)

    def test_validation(self):
        bad = [
            {**CONFIG, "learner": {"race": "Protoss"}},
            {**CONFIG, "enemy": {**CONFIG["enemy"], "mode": "smart"}},
            {**CONFIG, "enemy": {"mode": "frozen", "race": "Zerg", "task": "m:T"}},  # no model
            {**CONFIG, "enemy": {"mode": "scripted", "race": "Zerg"}, "self_play": {"rounds": 2, "steps_per_round": 10}},
            {**CONFIG, "decision_interval": 0},
        ]
        for data in bad:
            with self.subTest(data=data), self.assertRaises(ValueError):
                parse_config(data)

    def test_ppo_settings(self):
        self.assertEqual(parse_config(CONFIG).ppo.learning_rate, 3e-4)
        ppo = parse_config({**CONFIG, "ppo": {"learning_rate": 0.01, "net_arch": [128, 128]}}).ppo
        kwargs = ppo.as_kwargs()
        self.assertEqual(kwargs["learning_rate"], 0.01)
        self.assertEqual(kwargs["policy_kwargs"], {"net_arch": [128, 128]})
        self.assertNotIn("net_arch", kwargs)
        with self.assertRaises(ValueError):
            parse_config({**CONFIG, "ppo": {"lr": 0.01}})

    def test_self_play_sides(self):
        config = parse_config({**CONFIG, "self_play": {"rounds": 2, "steps_per_round": 10}})
        a = self_play_side(config, "a", "b.zip")
        self.assertEqual((a.learner.task, a.enemy.task, a.enemy.model), ("some.module:TaskA", "some.module:TaskB", "b.zip"))
        self.assertEqual(a.enemy.mode, "frozen")
        self.assertEqual(a.scenarios, config.scenarios)

        b = self_play_side(config, "b", "a.zip")
        self.assertEqual((b.learner.race, b.learner.task), ("Zerg", "some.module:TaskB"))
        self.assertEqual((b.enemy.race, b.enemy.task, b.enemy.model), ("Protoss", "some.module:TaskA", "a.zip"))
        self.assertEqual(b.scenarios[0].learner, {UnitTypeId.ROACH: 2})
        self.assertEqual(b.ppo, config.ppo)


class RewardTest(unittest.TestCase):
    def test_damage_shares(self):
        task = MicroTask()
        before = FightSnapshot(own_life=100, enemy_life=200, own_start=100, enemy_start=200)
        after = FightSnapshot(own_life=90, enemy_life=150, own_start=100, enemy_start=200)
        self.assertAlmostEqual(task.reward(before, after), 0.25 - 0.10)
        self.assertEqual(task.reward(after, after), 0.0)

    def test_terminal(self):
        task = MicroTask()
        outcomes = ("win", "loss", "tie", "timeout")
        self.assertEqual([task.terminal_reward(o, survivors=[]) for o in outcomes], [1.0, -1.0, 0.0, 0.0])

    def test_neutral_hooks(self):
        task = MicroTask()
        task.start_episode()
        self.assertEqual(task.unit_reward(unit=None, allies=[], enemies=[]), 0.0)

    def test_kite_task_punishes_timeouts(self):
        from tools.rl.examples.kite_task import KiteTask

        self.assertEqual(KiteTask().terminal_reward("timeout", survivors=[]), -1.0)
        self.assertEqual(KiteTask().terminal_reward("tie", survivors=[]), 0.0)


class StandInUnit:
    """Just enough of a python-sc2 Unit for CooldownKiteTask, recording the commands it gets."""

    def __init__(self, tag, x, y, cooldown=0.0, health=80, health_max=80, shield=80, shield_max=80, ground_range=6):
        from sc2.position import Point2

        self.tag, self.position = tag, Point2((x, y))
        self.weapon_cooldown, self.ground_range, self.radius = cooldown, ground_range, 0.5
        self.health, self.health_max, self.shield, self.shield_max = health, health_max, shield, shield_max
        self.commands = []

    def attack(self, target):
        self.commands.append(("attack", target))

    def move(self, point):
        self.commands.append(("move", point))

    def stop(self):
        self.commands.append(("stop",))


class CooldownKiteTaskTest(unittest.TestCase):
    def setUp(self):
        from tools.rl.examples.cooldown_kite_task import CooldownKiteTask

        self.task = CooldownKiteTask()
        self.stalker = StandInUnit(1, 10, 10, cooldown=12)
        self.roach = StandInUnit(2, 16, 10, ground_range=4)

    def test_observation_and_distance_change(self):
        first = self.task.observe(self.stalker, [], [self.roach])
        self.assertEqual(first.tolist(), [12.0, 6.0, 0.0])
        self.roach.position = self.roach.position.offset((2, 0))
        self.assertEqual(self.task.observe(self.stalker, [], [self.roach]).tolist(), [12.0, 8.0, 2.0])
        self.task.start_episode()
        self.assertEqual(self.task.observe(self.stalker, [], [self.roach])[2], 0.0)

    def test_no_enemies(self):
        self.assertEqual(self.task.observe(self.stalker, [], []).tolist(), [12.0, 0.0, 0.0])
        self.task.apply(self.stalker, 0, [], [])
        self.assertEqual(self.stalker.commands, [("stop",)])

    def test_actions(self):
        far_roach = StandInUnit(3, 30, 10)
        self.task.apply(self.stalker, 0, [], [far_roach, self.roach])
        self.assertEqual(self.stalker.commands[-1], ("attack", self.roach))

        self.task.apply(self.stalker, 1, [], [self.roach])
        kind, point = self.stalker.commands[-1]
        # Away from the roach, at its range (4) + both radii (1) + margin (4) = 9 from it.
        self.assertEqual(kind, "move")
        self.assertAlmostEqual(point.x, 16 - 9)
        self.assertAlmostEqual(point.y, 10)

    def test_rewards(self):
        self.assertEqual(self.task.unit_reward(self.stalker, [], [self.roach]), 1.0)
        self.assertEqual(self.task.unit_reward(StandInUnit(4, 0, 0, cooldown=0), [], []), 0.0)
        self.assertEqual(self.task.reward(None, None), 0.0)

        hurt = StandInUnit(5, 0, 0, health=40, shield=0)
        self.assertAlmostEqual(self.task.terminal_reward("win", [hurt]), 0.5)
        self.assertAlmostEqual(self.task.terminal_reward("win", [self.stalker]), 1.5)
        for outcome in ("loss", "tie", "timeout"):
            self.assertEqual(self.task.terminal_reward(outcome, [self.stalker]), 0.0)

    def test_survival_variant_scales_only_the_win_reward(self):
        from tools.rl.examples.cooldown_kite_task import SurvivalCooldownKiteTask

        survival = SurvivalCooldownKiteTask()
        hurt = StandInUnit(5, 0, 0, health=40, shield=0)
        self.assertAlmostEqual(survival.terminal_reward("win", [hurt]), 50.0)
        self.assertEqual(survival.terminal_reward("loss", [hurt]), 0.0)
        self.assertEqual(survival.unit_reward(self.stalker, [], [self.roach]), 1.0)


if __name__ == "__main__":
    unittest.main()
