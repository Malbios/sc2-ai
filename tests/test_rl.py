import math
import random
import unittest
from collections import Counter

import numpy as np

from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2

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

    def test_enemy_behavior(self):
        chase, leash = parse_scenarios([SCENARIOS[0], {**SCENARIOS[1], "enemy_behavior": "leash", "leash": [8, 14]}])
        self.assertEqual((chase.enemy_behavior, chase.leash), ("chase", None))
        self.assertEqual((leash.enemy_behavior, leash.leash), ("leash", (8.0, 14.0)))
        self.assertEqual(leash.swapped().leash, (8.0, 14.0))
        for bad in ({"enemy_behavior": "flee"}, {"enemy_behavior": "leash"}, {"enemy_behavior": "leash", "leash": [9, 3]}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse_scenarios([{**SCENARIOS[0], **bad}])


    def test_enemy_support(self):
        plain, supported = parse_scenarios([SCENARIOS[0], {**SCENARIOS[1], "enemy_support": {"overseer": 1}}])
        self.assertEqual(plain.enemy_support, {})
        self.assertEqual(supported.enemy_support, {UnitTypeId.OVERSEER: 1})
        self.assertEqual(supported.enemy, {UnitTypeId.ZERGLING: 4})  # support isn't part of the fight
        with self.assertRaises(ValueError):
            supported.swapped()
        for bad in ({"enemy_support": {"Overlordd": 1}}, {"enemy_support": {"Overseer": 0}}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse_scenarios([{**SCENARIOS[0], **bad}])


    def test_bile_dodge_reaction(self):
        plain, dodging = parse_scenarios([SCENARIOS[0], {**SCENARIOS[1], "bile_dodge_reaction": [0.3, 0.9]}])
        self.assertIsNone(plain.bile_dodge_reaction)
        self.assertEqual(dodging.bile_dodge_reaction, (0.3, 0.9))
        for bad in ([0.9, 0.3], [-1, 0.5], [0.5]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse_scenarios([{**SCENARIOS[0], "bile_dodge_reaction": bad}])

    def test_enemy_scans(self):
        plain, scanning = parse_scenarios([SCENARIOS[0], {**SCENARIOS[1], "enemy_scans": 2, "scan_reaction": [0.5, 1]}])
        self.assertEqual((plain.enemy_scans, plain.scan_reaction), (0, None))
        self.assertEqual((scanning.enemy_scans, scanning.scan_reaction), (2, (0.5, 1.0)))
        for bad in ({"enemy_scans": 1}, {"scan_reaction": [0.5, 1]}, {"enemy_scans": 5, "scan_reaction": [0.5, 1]},
                    {"enemy_scans": 1, "scan_reaction": [1, 0.5]}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse_scenarios([{**SCENARIOS[0], **bad}])


class ScanTargetTest(unittest.TestCase):
    """Loop 100; tag 1 burrowed since loop 70 (1.3 s), tag 2 since loop 95 (0.2 s)."""

    def setUp(self):
        self.burrowed = {1: (Point2((10, 10)), 70), 2: (Point2((40, 10)), 95)}

    def test_scans_the_unit_burrowed_long_enough(self):
        from tools.rl.enemies import scan_target

        self.assertEqual(scan_target(self.burrowed, 100, 1.0, 50, []), Point2((10, 10)))
        self.assertIsNone(scan_target(self.burrowed, 100, 1.5, 50, []))  # not long enough yet

    def test_needs_energy_and_skips_units_already_in_a_scan(self):
        from tools.rl.enemies import scan_target

        self.assertIsNone(scan_target(self.burrowed, 100, 1.0, 49, []))
        self.assertIsNone(scan_target(self.burrowed, 100, 1.0, 50, [Point2((15, 10))]))  # 5 away: inside
        self.assertEqual(scan_target(self.burrowed, 100, 0.1, 50, [Point2((15, 10))]), Point2((40, 10)))

    def test_burrowed_positions(self):
        from types import SimpleNamespace

        from tools.rl.enemies import burrowed_positions

        standing = SimpleNamespace(tag=1, position=Point2((1, 1)), type_id=UnitTypeId.ROACH, is_burrowed=False)
        listed_burrowed = SimpleNamespace(tag=2, position=Point2((2, 2)), type_id=UnitTypeId.ROACHBURROWED, is_burrowed=False)
        last_seen = {1: Point2((0, 0)), 3: Point2((3, 3))}  # tag 3 is no longer listed: it went down there
        self.assertEqual(burrowed_positions([standing, listed_burrowed], last_seen),
                         {2: Point2((2, 2)), 3: Point2((3, 3))})

    def test_detections_include_scans(self):
        from types import SimpleNamespace

        from sc2.ids.effect_id import EffectId

        from tools.rl.driver import detections
        from tools.rl.enemies import SCAN_RADIUS, Detection

        class StandInUnits(list):
            def __or__(self, other):
                return StandInUnits(self + other)

            def filter(self, keep):
                return StandInUnits(unit for unit in self if keep(unit))

        overseer = SimpleNamespace(position=Point2((5, 5)), detect_range=11.0, is_detector=True)
        marine = SimpleNamespace(position=Point2((6, 6)), detect_range=0.0, is_detector=False)
        scan = SimpleNamespace(id=EffectId.SCANNERSWEEP, positions=[(20.0, 20.0)])
        bile = SimpleNamespace(id=EffectId.RAVAGERCORROSIVEBILECP, positions=[(30.0, 30.0)])
        bot = SimpleNamespace(enemy_units=StandInUnits([overseer, marine]), enemy_structures=StandInUnits(),
                              state=SimpleNamespace(effects=[scan, bile]))
        self.assertEqual(detections(bot), [Detection(Point2((5, 5)), 11.0), Detection(Point2((20, 20)), SCAN_RADIUS)])


class DodgePointTest(unittest.TestCase):
    def test_steps_straight_out_of_a_bile_that_would_hit(self):
        from tools.rl.enemies import DODGE_MARGIN, dodge_point

        bile = (Point2((10, 10)), 0.5)
        clear = 1.5 + 2 * DODGE_MARGIN
        escape = dodge_point(Point2((10.5, 10)), 1.0, [bile])  # 0.5 east of the center, inside 0.5 + 1
        self.assertEqual(escape, Point2((10 + clear, 10)))
        edge = dodge_point(Point2((10, 11.5 + DODGE_MARGIN / 2)), 1.0, [bile])  # just outside: about to walk in
        self.assertEqual(edge, Point2((10, 10 + clear)))
        self.assertIsNone(dodge_point(Point2((13, 10)), 1.0, [bile]))  # well clear
        self.assertIsNone(dodge_point(Point2((10, 10)), 1.0, []))
        centered = dodge_point(Point2((10, 10)), 1.0, [bile])
        self.assertAlmostEqual(centered.distance_to(Point2((10, 10))), clear)


class ConfigTest(unittest.TestCase):
    def test_defaults(self):
        config = parse_config(CONFIG)
        self.assertEqual(config.map, "MicroTraining410")
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

    def test_upgrades(self):
        self.assertEqual(parse_config(CONFIG).learner.upgrades, ())
        learner = {**CONFIG["learner"], "upgrades": ["Burrow", "TUNNELINGCLAWS"]}
        self.assertEqual(parse_config({**CONFIG, "learner": learner}).learner.upgrades, ("BURROW", "TUNNELINGCLAWS"))
        with self.assertRaises(ValueError):
            parse_config({**CONFIG, "learner": {**learner, "upgrades": ["Blink2"]}})

    def test_builtin_build(self):
        from sc2.data import AIBuild

        from tools.rl.config import with_enemy
        from tools.rl.enemies import make_enemy_player

        self.assertEqual(parse_config(CONFIG).enemy.build, "RandomBuild")
        builtin = parse_config({**CONFIG, "enemy": {"mode": "builtin", "race": "Terran", "build": "Rush"}})
        self.assertEqual(make_enemy_player(builtin.enemy, 3).ai_build, AIBuild.Rush)
        air = with_enemy(builtin, build="Air")
        self.assertEqual((air.enemy.mode, air.enemy.build), ("builtin", "Air"))
        self.assertEqual(with_enemy(builtin, mode="scripted").enemy.build, "Rush")
        with self.assertRaises(ValueError):
            parse_config({**CONFIG, "enemy": {"mode": "builtin", "race": "Terran", "build": "Cheese"}})
        with self.assertRaises(ValueError):
            with_enemy(builtin, build="Cheese")

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


class BaselinePolicyTest(unittest.TestCase):
    def test_cooldown_kite_task_baselines(self):
        from tools.rl.examples.cooldown_kite_task import CooldownKiteTask

        policies = CooldownKiteTask().baseline_policies()
        ready, cooling = [0.0, 6.0, 0.0], [12.0, 6.0, 0.0]
        self.assertEqual([policies["attack"](ready), policies["attack"](cooling)], [0, 0])
        self.assertEqual([policies["retreat"](ready), policies["retreat"](cooling)], [1, 1])
        self.assertEqual([policies["kite"](ready), policies["kite"](cooling)], [0, 1])
        self.assertEqual(MicroTask().baseline_policies(), {})

    def test_situations(self):
        from tools.rl.examples.cooldown_kite_task import CooldownKiteTask

        situation = CooldownKiteTask().situation
        self.assertEqual(situation([0.0, 4.0, 0.0]), "weapon ready, < 5.5")
        self.assertEqual(situation([12.0, 6.0, 0.0]), "weapon cooling, 5.5-7.5")
        self.assertEqual(situation([3.0, 7.5, 0.0]), "weapon cooling, >= 7.5")
        self.assertEqual(MicroTask().situation([0.0]), "all")


class KiteMeterTest(unittest.TestCase):
    """Roach reach against the stand-in stalker: range 4 + radii 1 = 5, so gaps up to 7 are close."""

    def setUp(self):
        from tools.rl.kiting import KiteMeter

        self.meter = KiteMeter()
        self.stalker = StandInUnit(1, 10, 10, cooldown=12)
        self.roach = StandInUnit(2, 16, 10, ground_range=4)

    def step(self, stalker_x: float):
        self.stalker.position = Point2((stalker_x, 10))
        self.meter.update(self.stalker, [self.roach])

    def test_backing_off_during_cooldown_counts_as_kiting(self):
        self.step(10)  # first sighting: nothing to compare with yet
        self.step(9.5)
        self.step(9)
        self.assertEqual((self.meter.close_cooldown_steps, self.meter.backing_off_steps), (2, 2))
        self.assertEqual(self.meter.share, 1.0)

    def test_chasing_or_standing_does_not(self):
        self.step(10)
        self.step(10.5)
        self.step(10.5)
        self.assertEqual(self.meter.share, 0.0)
        self.assertEqual(self.meter.close_cooldown_steps, 2)

    def test_ready_weapon_or_distant_enemy_is_not_counted(self):
        self.stalker.weapon_cooldown = 0
        self.step(10)
        self.step(9)
        self.stalker.weapon_cooldown = 12
        self.roach.position = Point2((30, 10))
        self.step(8)
        self.step(7)
        self.assertEqual(self.meter.close_cooldown_steps, 0)
        self.assertEqual(self.meter.share, 0.0)
        self.meter.update(self.stalker, [])


class StandInGrid:
    """A pathing grid, pathable (1) everywhere except the cells listed as walls."""

    def __init__(self, width, height, walls=()):
        self.width, self.height, self.walls = width, height, set(walls)

    def __getitem__(self, cell):
        return 0 if cell in self.walls else 1


class FreeKiteTaskTest(unittest.TestCase):
    def setUp(self):
        from types import SimpleNamespace

        from tools.rl.examples.free_kite_task import FreeKiteTask

        self.task = FreeKiteTask()
        # A wall along x = 14 (4 east of the stalker); the grid ends at y = 20 (10 north).
        walls = [(14, y) for y in range(20)]
        self.task.start_game(SimpleNamespace(pathing_grid=StandInGrid(40, 20, walls)))
        self.stalker = StandInUnit(1, 10.5, 10.5, cooldown=15, shield=40)
        self.near = StandInUnit(2, 10.5, 5.5, health=100, health_max=145, shield=0, shield_max=0)
        self.far = StandInUnit(3, 4.5, 10.5, health=50, health_max=145, shield=0, shield_max=0)

    def test_observation(self):
        raw = self.task.observe(self.stalker, [], [self.far, self.near])
        self.assertEqual(raw.shape, self.task.observation_space.shape)
        observation = raw.astype(float).round(3).tolist()
        self.assertEqual(observation[:3], [0.5, 1.0, 0.5])
        self.assertEqual(observation[3:8], [1.0, 0.0, -0.5, 0.5, round(100 / 145, 3)])  # closest first
        self.assertEqual(observation[8:13], [1.0, -0.6, 0.0, 0.6, round(50 / 145, 3)])
        east, north, west, south = observation[13], observation[15], observation[17], observation[19]
        self.assertEqual([east, north, west, south], [0.3, 0.9, 1.0, 1.0])

    def test_missing_enemy_is_zeros(self):
        observation = self.task.observe(self.stalker, [], [self.near])
        self.assertEqual(observation[8:13].tolist(), [0.0] * 5)

    def test_actions(self):
        from tools.rl.examples.free_kite_task import FIRST_MOVE

        enemies = [self.far, self.near]
        self.task.apply(self.stalker, 0, [], enemies)
        self.task.apply(self.stalker, 1, [], enemies)
        self.task.apply(self.stalker, FIRST_MOVE + 2, [], enemies)  # north
        (_, closest), (_, weakest), (_, destination) = self.stalker.commands
        self.assertEqual((closest.tag, weakest.tag), (2, 3))
        self.assertEqual((round(destination.x, 3), round(destination.y, 3)), (10.5, 13.5))

    def test_baselines_and_situation(self):
        from tools.rl.examples.free_kite_task import FIRST_MOVE

        policies = self.task.baseline_policies()
        cooling = self.task.observe(self.stalker, [], [self.far, self.near])
        self.assertEqual(policies["kite"](cooling), FIRST_MOVE + 2)  # straight away from the roach south
        # Away from both roaches (south and west) points northeast, but the wall 3 east makes north the better run.
        self.assertEqual(policies["smart"](cooling), FIRST_MOVE + 2)
        self.assertEqual(self.task.situation(cooling), "weapon cooling, open ground")

        self.stalker.weapon_cooldown = 0
        self.stalker.position = Point2((12.5, 10.5))
        ready = self.task.observe(self.stalker, [], [self.far, self.near])
        self.assertEqual([policies["attack"](ready), policies["kite"](ready), policies["smart"](ready)], [0, 0, 1])
        self.assertEqual(self.task.situation(ready), "weapon ready, near a wall")


class TrackingKiteTaskTest(unittest.TestCase):
    def setUp(self):
        from types import SimpleNamespace

        from tools.rl.examples.free_kite_task import TrackingKiteTask

        self.game_info = SimpleNamespace(pathing_grid=StandInGrid(40, 40))
        self.task = TrackingKiteTask()
        self.task.start_game(self.game_info)
        self.stalker = StandInUnit(1, 20.5, 20.5, cooldown=15)
        self.a = StandInUnit(2, 20.5, 16.5)  # 4 south
        self.b = StandInUnit(3, 14.5, 20.5)  # 6 west

    def movement(self):
        return [round(value, 3) for value in self.task.observe(self.stalker, [], [self.a, self.b]).tolist()[-4:]]

    def test_movement_follows_each_enemy(self):
        self.assertEqual(self.movement(), [0.0, 0.0, 0.0, 0.0])  # nothing to compare with yet
        self.a.position = Point2((20.5, 16.08))  # a roach step away from the stalker
        self.assertEqual(self.movement(), [0.0, -1.0, 0.0, 0.0])
        self.a.position = Point2((20.5, 12.5))  # now farther than b: b is listed first
        self.b.position = Point2((14.92, 20.5))
        self.assertEqual(self.movement()[:2], [1.0, 0.0])
        self.task.start_episode()
        self.assertEqual(self.movement(), [0.0, 0.0, 0.0, 0.0])

    def test_keeps_free_kite_inputs_and_situation(self):
        from tools.rl.examples.free_kite_task import FreeKiteTask

        free = FreeKiteTask()
        free.start_game(self.game_info)
        observation = self.task.observe(self.stalker, [], [self.a, self.b])
        self.assertEqual(observation.shape, self.task.observation_space.shape)
        self.assertEqual(observation[:21].tolist(), free.observe(self.stalker, [], [self.a, self.b]).tolist())
        observation[-4:] = -5.0  # movement inputs must not look like a nearby wall
        self.assertEqual(self.task.situation(observation), "weapon cooling, open ground")


class StandInRoach(StandInUnit):
    def __init__(self, *args, burrowed=False, **kwargs):
        super().__init__(*args, shield=0, shield_max=0, **kwargs)
        self.is_burrowed = burrowed

    def __call__(self, ability):
        self.commands.append(("ability", ability))


class RoachKiteTaskTest(unittest.TestCase):
    def setUp(self):
        from types import SimpleNamespace

        from tools.rl.examples.roach_kite_task import RoachKiteTask

        self.task = RoachKiteTask()
        self.task.start_game(SimpleNamespace(pathing_grid=StandInGrid(40, 40)))
        self.roach = StandInRoach(1, 20.5, 20.5, cooldown=15, health=145, health_max=145)
        self.enemy = StandInRoach(2, 20.5, 16.5, health=145, health_max=145)

    def observe(self):
        return self.task.observe(self.roach, [], [self.enemy])

    def test_burrowed_input(self):
        from tools.rl.examples.roach_kite_task import BURROWED

        self.assertEqual(self.observe().shape, self.task.observation_space.shape)
        self.assertEqual(self.observe()[BURROWED], 0.0)
        self.roach.is_burrowed = True
        self.assertEqual(self.observe()[BURROWED], 1.0)
        self.assertEqual(self.observe()[2], 0.0)  # no shield

    def test_burrow_actions(self):
        from sc2.ids.ability_id import AbilityId

        from tools.rl.examples.roach_kite_task import BURROW, UNBURROW

        self.task.apply(self.roach, BURROW, [], [self.enemy])
        self.task.apply(self.roach, UNBURROW, [], [self.enemy])
        self.task.apply(self.roach, 0, [], [self.enemy])
        self.assertEqual(self.roach.commands, [("ability", AbilityId.BURROWDOWN_ROACH),
                                               ("ability", AbilityId.BURROWUP_ROACH), ("attack", self.enemy)])

    def test_burrow_rule(self):
        from tools.rl.examples.roach_kite_task import BURROW, UNBURROW

        policies = self.task.baseline_policies()
        self.assertEqual(set(policies), {"attack", "kite", "smart", "burrow", "careful_burrow"})

        def burrow_rule(health, burrowed):
            self.roach.health, self.roach.is_burrowed = health, burrowed
            observation = self.observe()
            return policies["burrow"](observation), policies["smart"](observation)

        burrow, smart = burrow_rule(145 * 0.5, False)
        self.assertEqual(burrow, smart)  # healthy enough: plays like the smart rule
        self.assertEqual(burrow_rule(145 * 0.35, False)[0], BURROW)
        self.assertEqual(burrow_rule(145 * 0.6, True)[0], BURROW)  # stays burrowed while healing
        self.assertEqual(burrow_rule(145 * 0.75, True)[0], UNBURROW)

    def test_detector_inputs(self):
        from types import SimpleNamespace

        from tools.rl.examples.roach_kite_task import DETECTION_MARGIN, DETECTOR_PRESENT

        self.assertEqual(self.observe()[DETECTOR_PRESENT:DETECTION_MARGIN + 1].tolist(), [0.0, 0.0])
        near = SimpleNamespace(position=Point2((20.5, 25.5)), detect_range=11.0)  # 5 away: 6 inside
        far = SimpleNamespace(position=Point2((40.5, 20.5)), detect_range=11.0)  # 20 away: 9 outside
        self.task.see_detectors([far, near])
        self.assertEqual([round(v, 3) for v in self.observe()[DETECTOR_PRESENT:DETECTION_MARGIN + 1].tolist()], [1.0, -0.6])
        self.task.see_detectors([far])
        self.assertAlmostEqual(float(self.observe()[DETECTION_MARGIN]), 0.9, places=5)

    def test_careful_burrow_rule(self):
        from types import SimpleNamespace

        from tools.rl.examples.roach_kite_task import BURROW, UNBURROW

        policies = self.task.baseline_policies()
        low = 145 * 0.3
        self.roach.health = low
        self.task.see_detectors([SimpleNamespace(position=Point2((20.5, 25.5)), detect_range=11.0)])
        inside = self.observe()
        self.assertEqual(policies["burrow"](inside), BURROW)
        self.assertEqual(policies["careful_burrow"](inside), policies["smart"](inside))  # detected: fight or run instead
        self.roach.is_burrowed = True
        self.assertEqual(policies["careful_burrow"](self.observe()), UNBURROW)

        self.roach.is_burrowed = False
        self.task.see_detectors([SimpleNamespace(position=Point2((40.5, 20.5)), detect_range=11.0)])
        self.assertEqual(policies["careful_burrow"](self.observe()), BURROW)  # out of detection: safe to burrow


class StandInCaster(StandInUnit):
    def __call__(self, ability, target=None):
        self.commands.append(("ability", ability, target))


class RavagerTaskTest(unittest.TestCase):
    def setUp(self):
        from types import SimpleNamespace

        from tools.rl.examples.ravager_task import RavagerTask

        self.task = RavagerTask()
        self.task.start_game(SimpleNamespace(pathing_grid=StandInGrid(40, 40)))
        self.ravager = StandInCaster(1, 20.5, 20.5, cooldown=15, health=120, health_max=120, shield=0, shield_max=0)
        self.near = StandInUnit(2, 26.5, 20.5, health=145, health_max=145, shield=0, shield_max=0)  # 6 east
        self.far = StandInUnit(3, 20.5, 12.5, health=145, health_max=145, shield=0, shield_max=0)  # 8 south

    def observe(self):
        return self.task.observe(self.ravager, [], [self.near, self.far])

    def test_bile_ready_input(self):
        from sc2.ids.ability_id import AbilityId

        from tools.rl.examples.ravager_task import BILE_READY

        self.assertEqual(self.observe().shape, self.task.observation_space.shape)
        self.assertEqual(self.observe()[BILE_READY], 0.0)
        self.task.see_abilities({1: {AbilityId.EFFECT_CORROSIVEBILE}})
        self.assertEqual(self.observe()[BILE_READY], 1.0)

    def test_bile_is_led_by_the_targets_movement(self):
        from sc2.ids.ability_id import AbilityId

        from tools.rl.examples.ravager_task import BILE_FLIGHT_SECONDS, DECISION_SECONDS, bile_action

        self.observe()
        self.near.position = Point2((26.1, 20.5))  # moved 0.4 toward the ravager since the last decision
        self.observe()
        enemies = [self.near, self.far]
        self.task.apply(self.ravager, bile_action(0, 0), [], enemies)  # where it is now
        self.task.apply(self.ravager, bile_action(0, 2), [], enemies)  # straight-line lead
        self.task.apply(self.ravager, bile_action(1, 0), [], enemies)  # the second closest
        (_, ability, now), (_, _, led), (_, _, second) = self.ravager.commands
        self.assertEqual(ability, AbilityId.EFFECT_CORROSIVEBILE)
        self.assertEqual(now, Point2((26.1, 20.5)))
        self.assertAlmostEqual(led.x, 26.1 - 0.4 * BILE_FLIGHT_SECONDS / DECISION_SECONDS, places=4)
        self.assertEqual(second, Point2((20.5, 12.5)))

    def test_bile_baselines(self):
        from sc2.ids.ability_id import AbilityId

        from tools.rl.examples.ravager_task import NOW, STRAIGHT_LINE, bile_action

        policies = self.task.baseline_policies()
        not_ready = self.observe()
        for name in ("bile_now", "bile_lead"):
            self.assertEqual(policies[name](not_ready), policies["smart"](not_ready))
        self.task.see_abilities({1: {AbilityId.EFFECT_CORROSIVEBILE}})
        ready = self.observe()
        self.assertEqual(policies["bile_now"](ready), bile_action(0, NOW))
        self.assertEqual(policies["bile_lead"](ready), bile_action(0, STRAIGHT_LINE))
        self.near.position = Point2((30.5, 20.5))
        self.far.position = Point2((20.5, 10.5))  # both out of bile range
        out_of_range = self.observe()
        self.assertEqual(policies["bile_lead"](out_of_range), policies["smart"](out_of_range))


class RavagerBileTaskTest(unittest.TestCase):
    def setUp(self):
        from types import SimpleNamespace

        from tools.rl.examples.ravager_task import RavagerBileTask

        self.task = RavagerBileTask()
        self.task.start_game(SimpleNamespace(pathing_grid=StandInGrid(40, 40)))
        self.ravager = StandInCaster(1, 20.5, 20.5, cooldown=15, health=120, health_max=120, shield=0, shield_max=0)
        self.roach = StandInUnit(2, 26.5, 20.5, health=145, health_max=145, shield=0, shield_max=0)

    def test_bile_actions_are_masked_until_bile_is_ready(self):
        from sc2.ids.ability_id import AbilityId

        from tools.rl.examples.ravager_task import FIRST_BILE

        self.assertTrue(self.task.uses_action_masks)
        waiting = self.task.action_mask(self.ravager, [], [self.roach])
        self.assertEqual(waiting.shape, (self.task.action_space.n,))
        self.assertTrue(waiting[:FIRST_BILE].all())
        self.assertFalse(waiting[FIRST_BILE:].any())
        self.task.see_abilities({1: {AbilityId.EFFECT_CORROSIVEBILE}})
        self.assertTrue(self.task.action_mask(self.ravager, [], [self.roach]).all())

    def test_paid_for_damage_dealt_not_for_reloading(self):
        before = FightSnapshot(own_life=120, enemy_life=290, own_start=120, enemy_start=290)
        bile_hit = FightSnapshot(own_life=100, enemy_life=230, own_start=120, enemy_start=290)
        self.assertAlmostEqual(self.task.reward(before, bile_hit), 100 * 60 / 290)  # damage taken costs nothing
        self.assertEqual(self.task.reward(before, before), 0.0)
        self.assertEqual(self.task.unit_reward(self.ravager, [], [self.roach]), 0.0)
        self.assertEqual(self.task.terminal_reward("win", [self.ravager]), 100.0)


class RavagerHybridTaskTest(unittest.TestCase):
    """A ravager at (20.5, 20.5), roaches 6 east and 8 south."""

    def setUp(self):
        from types import SimpleNamespace

        from tools.rl.examples.ravager_task import RavagerHybridTask

        self.task = RavagerHybridTask()
        self.task.start_game(SimpleNamespace(pathing_grid=StandInGrid(40, 40)))
        self.ravager = StandInCaster(1, 20.5, 20.5, cooldown=15, health=120, health_max=120, shield=0, shield_max=0)
        self.near = StandInUnit(2, 26.5, 20.5, health=145, health_max=145, shield=0, shield_max=0)
        self.far = StandInUnit(3, 20.5, 12.5, health=145, health_max=145, shield=0, shield_max=0)

    def observe(self):
        return self.task.observe(self.ravager, [], [self.near, self.far])

    def test_the_two_bile_actions_aim_with_the_straight_line_lead(self):
        from sc2.ids.ability_id import AbilityId

        from tools.rl.examples.ravager_task import BILE_FLIGHT_SECONDS, DECISION_SECONDS, FIRST_BILE

        self.assertEqual(self.task.action_space.n, 12)
        self.observe()
        self.near.position = Point2((26.1, 20.5))  # moved 0.4 toward the ravager since the last decision
        self.observe()
        enemies = [self.near, self.far]
        self.task.apply(self.ravager, FIRST_BILE, [], enemies)
        self.task.apply(self.ravager, FIRST_BILE + 1, [], enemies)
        (_, ability, led), (_, _, second) = self.ravager.commands
        self.assertEqual(ability, AbilityId.EFFECT_CORROSIVEBILE)
        self.assertAlmostEqual(led.x, 26.1 - 0.4 * BILE_FLIGHT_SECONDS / DECISION_SECONDS, places=4)
        self.assertEqual(second, Point2((20.5, 12.5)))

    def test_other_actions_and_masking(self):
        from sc2.ids.ability_id import AbilityId

        from tools.rl.examples.ravager_task import FIRST_BILE

        self.task.apply(self.ravager, 0, [], [self.near, self.far])
        self.assertEqual(self.ravager.commands[0][0], "attack")
        mask = self.task.action_mask(self.ravager, [], [self.near])
        self.assertEqual((mask.shape, bool(mask[:FIRST_BILE].all()), bool(mask[FIRST_BILE:].any())), ((12,), True, False))
        self.task.see_abilities({1: {AbilityId.EFFECT_CORROSIVEBILE}})
        self.assertTrue(self.task.action_mask(self.ravager, [], [self.near]).all())

    def test_lead_rule_baseline(self):
        from sc2.ids.ability_id import AbilityId

        from tools.rl.examples.ravager_task import FIRST_BILE

        policies = self.task.baseline_policies()
        self.assertNotIn("bile_now", policies)
        not_ready = self.observe()
        self.assertEqual(policies["bile_lead"](not_ready), policies["smart"](not_ready))
        self.task.see_abilities({1: {AbilityId.EFFECT_CORROSIVEBILE}})
        self.assertEqual(policies["bile_lead"](self.observe()), FIRST_BILE)


class LeashOrderTest(unittest.TestCase):
    """A roach spawned at (20, 20) with a leash of 8, against a stalker it may chase."""

    def setUp(self):
        from tools.rl.enemies import EnemyBriefing

        self.briefing = EnemyBriefing("leash", Point2((20, 20)), 8.0)
        self.stalker = StandInUnit(1, 30, 20)
        self.roach = StandInUnit(2, 25, 20)

    def order(self, roach_x, returning):
        from tools.rl.enemies import leash_order

        self.roach.position = Point2((roach_x, 20))
        return leash_order(self.roach, [self.stalker], self.briefing, returning)

    def test_attacks_inside_the_leash(self):
        target, returning = self.order(25, False)
        self.assertEqual((target.tag, returning), (1, False))

    def test_walks_home_past_the_leash_until_it_gets_there(self):
        self.assertEqual(self.order(28.5, False), (Point2((20, 20)), True))
        self.assertEqual(self.order(25, True), (Point2((20, 20)), True))  # back inside, still walking home
        target, returning = self.order(21, True)  # home: fights again
        self.assertEqual((target.tag, returning), (1, False))


class StandInGroup(list):
    """Just enough of python-sc2's Units for the driver's group decisions."""

    def find_by_tag(self, tag):
        return next((unit for unit in self if unit.tag == tag), None)

    def tags_not_in(self, tags):
        return StandInGroup(unit for unit in self if unit.tag not in tags)


class GroupDecisionTest(unittest.TestCase):
    def setUp(self):
        from types import SimpleNamespace

        from gymnasium import spaces

        from tools.rl.driver import Episode, GameDriver

        class GroupTask(MicroTask):
            observation_space = spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32)
            action_space = spaces.Discrete(4)
            group_slots = 3

            def observe(self, unit, allies, enemies):
                return np.array([unit.tag, len(allies)], dtype=np.float32)

            def apply(self, unit, action, allies, enemies):
                unit.commands.append(action)

        self.driver = GameDriver(SimpleNamespace(decision_interval=3), GroupTask(), random.Random())
        self.addCleanup(self.driver.loop.close)
        self.driver.episode = Episode(scenario=None, slot_tags=[5, 7])
        self.asked = []

        async def ask(event):
            self.asked.append(event)
            return np.array([1, 2, 3])

        self.driver._ask = ask

    def decide(self, *units):
        self.driver.loop.run_until_complete(self.driver._decide_as_group(StandInGroup(units), enemies=[]))

    def test_one_request_with_a_row_per_slot(self):
        first, second = StandInUnit(5, 0, 0), StandInUnit(7, 1, 0)
        self.decide(second, first)
        (kind, observations), = self.asked
        self.assertEqual(kind, "decide")
        np.testing.assert_array_equal(observations, [[5, 1], [7, 1], [0, 0]])
        self.assertEqual((first.commands, second.commands), ([1], [2]))

    def test_a_dead_unit_keeps_its_slot_empty(self):
        second = StandInUnit(7, 1, 0)
        self.decide(second)
        np.testing.assert_array_equal(self.asked[0][1], [[0, 0], [7, 0], [0, 0]])
        self.assertEqual(second.commands, [2])

    def test_rewards_land_in_their_units_slots(self):
        import asyncio

        units = StandInGroup([StandInUnit(5, 0, 0), StandInUnit(7, 1, 0)])
        self.driver._add_slot_rewards({5: 1.0, 7: 2.0}, units, enemies=[])
        self.driver._add_to_slots({7: 10.0, 99: 5.0})  # 99 has no slot
        np.testing.assert_array_equal(self.driver.pending_slot_rewards, [1.0, 12.0, 0.0])

        self.driver.events = asyncio.Queue()
        self.driver.game_task = self.driver.loop.create_future()
        self.addCleanup(self.driver.game_task.cancel)
        self.driver.events.put_nowait(("decide", np.zeros((3, 2), dtype=np.float32)))
        self.driver._waiter, self.driver._waiting_for = self.driver.loop.create_future(), "decide"
        _, reward, _, _, info = self.driver.step(np.array([0, 0, 0]))
        np.testing.assert_array_equal(info["slot_rewards"], [1.0, 12.0, 0.0])
        self.assertEqual(reward, 13.0)
        np.testing.assert_array_equal(self.driver.pending_slot_rewards, [0.0, 0.0, 0.0])

    def test_every_unit_gets_the_team_reward_by_default(self):
        self.assertEqual(self.driver.task.share_team_reward(2.0, [StandInUnit(5, 0, 0), StandInUnit(7, 1, 0)]),
                         {5: 2.0, 7: 2.0})


class ZerglingSurroundTaskTest(unittest.TestCase):
    """A zergling at (20.5, 20.5) with marines to its east and north."""

    def setUp(self):
        from types import SimpleNamespace

        from tools.rl.examples.zergling_surround_task import ZerglingSurroundTask

        self.task = ZerglingSurroundTask()
        self.task.start_game(SimpleNamespace(pathing_grid=StandInGrid(40, 40)))
        self.ling = self.zergling(1, 20.5, 20.5)
        self.east = StandInUnit(10, 25.5, 20.5, health=45, health_max=45, shield=0, shield_max=0)
        self.north = StandInUnit(11, 20.5, 26.5, health=45, health_max=45, shield=0, shield_max=0)

    def zergling(self, tag, x, y, health=35):
        return StandInUnit(tag, x, y, health=health, health_max=35, shield=0, shield_max=0)

    def flank(self, allies, enemies):
        return self.task.baseline_policies()["flank"](self.task.observe(self.ling, allies, enemies))

    def test_observation(self):
        from tools.rl.examples.zergling_surround_task import FIRST_ALLY, FIRST_ENEMY, FIRST_RAY, INPUTS

        allies = [self.zergling(3, 20.5, 17.5, health=14), self.zergling(2, 22.5, 20.5)]
        observation = self.task.observe(self.ling, allies, [self.north, self.east]).astype(float).round(3).tolist()
        self.assertEqual(len(observation), INPUTS)
        self.assertEqual(observation[:2], [0.0, 1.0])
        self.assertEqual(observation[FIRST_ENEMY:FIRST_ENEMY + 14],
                         [1.0, 0.5, 0.0, 0.5, 1.0, 0.0, 0.0, 1.0, 0.0, 0.6, 0.6, 1.0, 0.0, 0.0])  # closest first
        self.assertEqual(observation[FIRST_ENEMY + 14:FIRST_ALLY], [0.0] * 14)
        self.assertEqual(observation[FIRST_ALLY:FIRST_ALLY + 8], [1.0, 0.2, 0.0, 1.0, 1.0, 0.0, -0.3, 0.4])
        self.assertEqual(observation[FIRST_ALLY + 8:FIRST_RAY], [0.0] * 8)
        self.assertEqual(observation[FIRST_RAY], 1.0)

    def test_enemy_movement_since_the_last_decision(self):
        from tools.rl.examples.zergling_surround_task import FIRST_ENEMY

        self.task.observe(self.ling, [], [self.east])
        self.east.position = Point2((25.92, 20.5))
        self.assertAlmostEqual(float(self.task.observe(self.ling, [], [self.east])[FIRST_ENEMY + 5]), 1.0, places=3)

    def test_flank_runs_around_on_its_own_side(self):
        from tools.rl.examples.free_kite_task import FIRST_MOVE

        northeast, southeast = FIRST_MOVE + 1, FIRST_MOVE + 7
        self.assertEqual(self.flank([self.zergling(2, 20.5, 18.5)], [self.east]), northeast)  # allies south
        self.assertEqual(self.flank([self.zergling(2, 20.5, 22.5)], [self.east]), southeast)  # allies north

    def test_flank_attacks_when_close_or_past_the_enemies(self):
        from tools.rl.examples.free_kite_task import ATTACK_CLOSEST

        touching = StandInUnit(12, 21.5, 20.5, health=45, health_max=45, shield=0, shield_max=0)
        self.assertEqual(self.flank([self.zergling(2, 20.5, 18.5)], [touching]), ATTACK_CLOSEST)
        behind = StandInUnit(13, 17.5, 20.5, health=45, health_max=45, shield=0, shield_max=0)
        self.assertEqual(self.flank([self.zergling(2, 10.5, 20.5)], [behind]), ATTACK_CLOSEST)
        self.assertEqual(self.task.baseline_policies()["attack"](None), ATTACK_CLOSEST)

    def test_rewards(self):
        self.assertEqual(self.task.reward(FightSnapshot(100, 100, 100, 200), FightSnapshot(90, 70, 100, 200)), 15.0)
        self.task.observe(self.ling, [self.zergling(2, 22.5, 20.5), self.zergling(3, 20.5, 17.5)], [self.east])
        half = self.zergling(1, 20.5, 20.5, health=17.5)
        self.assertAlmostEqual(self.task.terminal_reward("win", [half]), 100 * 0.5 / 3)
        self.assertEqual(self.task.terminal_reward("loss", []), 0.0)
        self.task.start_episode()
        self.assertEqual(self.task.terminal_reward("win", [half]), 0.0)

    def test_damage_goes_to_the_zerglings_that_attacked(self):
        first, second = self.zergling(1, 20.5, 20.5), self.zergling(2, 21.5, 20.5)
        first.weapon_cooldown = second.weapon_cooldown = 0.0
        self.assertEqual(self.task.share_team_reward(6.0, [first, second]), {1: 3.0, 2: 3.0})  # nobody seen attacking
        second.weapon_cooldown = 11.0
        self.assertEqual(self.task.share_team_reward(6.0, [first, second]), {2: 6.0})
        second.weapon_cooldown = 8.0  # cooling down, not a new attack
        self.assertEqual(self.task.share_team_reward(4.0, [first, second]), {1: 2.0, 2: 2.0})
        self.task.start_episode()
        self.assertEqual(self.task.share_team_reward(4.0, [second]), {2: 4.0})

    def test_group_slots_cover_the_largest_calibration_group(self):
        from pathlib import Path

        from tools.rl.config import load_config

        config = load_config(Path(__file__).parents[1] / "tools/rl/configs/zergling_surround_calibration.yaml")
        largest = max(sum(scenario.learner.values()) for scenario in config.scenarios)
        self.assertLessEqual(largest, self.task.group_slots)


class StandInFighter(StandInUnit):
    """A StandInUnit with a fixed damage per second against anything, and python-sc2's reach check."""

    def __init__(self, tag, x, y, health, health_max, dps=0.0, ground_range=4):
        super().__init__(tag, x, y, health=health, health_max=health_max, shield=0, shield_max=0,
                         ground_range=ground_range)
        self.dps = dps
        self.order_target = None

    def calculate_dps_vs_target(self, target):
        return self.dps

    def target_in_range(self, target):
        return self.position.distance_to(target.position) <= self.radius + target.radius + self.ground_range


class RoachGroupTaskTest(unittest.TestCase):
    """A roach at (20.5, 20.5): a marine 5 to the east and a marauder 4.5 to the north are in
    range (4, plus both radii), a marine 10 to the east is not."""

    def setUp(self):
        from types import SimpleNamespace

        from tools.rl.examples.roach_group_task import RoachGroupTask

        self.task = RoachGroupTask()
        self.task.start_game(SimpleNamespace(pathing_grid=StandInGrid(40, 40)))
        self.roach = StandInFighter(1, 20.5, 20.5, health=145, health_max=145)
        self.marine = StandInFighter(10, 25.5, 20.5, health=45, health_max=45, dps=10.0, ground_range=5)
        self.marauder = StandInFighter(11, 20.5, 25.0, health=125, health_max=125, dps=20.0, ground_range=6)
        self.far_marine = StandInFighter(12, 30.5, 20.5, health=5, health_max=45, dps=10.0, ground_range=5)

    def target(self, action, enemies):
        self.roach.commands.clear()
        self.task.apply(self.roach, action, [], enemies)
        return self.roach.commands[0][1]

    def smart(self, enemies):
        return self.task.baseline_policies()["smart"](self.task.observe(self.roach, [], enemies))

    def test_observation(self):
        from tools.rl.examples.group_fight_task import FIRST_ALLY, FIRST_ENEMY, INPUTS

        observation = self.task.observe(self.roach, [], [self.marauder, self.marine]).astype(float).round(3).tolist()
        self.assertEqual(len(observation), INPUTS)
        self.assertEqual(observation[:2], [0.0, 1.0])
        self.assertEqual(observation[FIRST_ENEMY:FIRST_ENEMY + 16],
                         [1.0, 0.0, 0.45, 0.45, 1.0, 0.0, 0.0, 1.0, 1.0, 0.5, 0.0, 0.5, 1.0, 0.0, 0.0, 0.5])  # closest first
        self.assertEqual(observation[FIRST_ENEMY + 16:FIRST_ALLY], [0.0] * 16)

    def test_attacks_pick_their_target_among_the_enemies_in_range(self):
        from tools.rl.examples.group_fight_task import ATTACK_CLOSEST, ATTACK_DANGEROUS, ATTACK_WEAKEST

        enemies = [self.far_marine, self.marauder, self.marine]
        self.assertIs(self.target(ATTACK_CLOSEST, enemies), self.marauder)
        self.assertIs(self.target(ATTACK_WEAKEST, enemies), self.marine)  # the far marine is weaker but out of range
        self.assertIs(self.target(ATTACK_DANGEROUS, enemies), self.marine)  # 10 / 45 beats 20 / 125
        self.marauder.health = 50
        self.assertIs(self.target(ATTACK_DANGEROUS, enemies), self.marauder)  # 20 / 50

    def test_ties_keep_the_current_target_then_go_to_the_closest(self):
        from tools.rl.examples.group_fight_task import ATTACK_DANGEROUS, ATTACK_WEAKEST

        farther = StandInFighter(13, 16.0, 20.5, health=45, health_max=45, dps=10.0, ground_range=5)  # 4.5 west
        closer = StandInFighter(14, 20.5, 16.5, health=45, health_max=45, dps=10.0, ground_range=5)  # 4 south
        for action in (ATTACK_WEAKEST, ATTACK_DANGEROUS):
            self.roach.order_target = None
            self.assertIs(self.target(action, [farther, closer]), closer)
            self.roach.order_target = farther.tag
            self.assertIs(self.target(action, [farther, closer]), farther)

    def test_attacks_fall_back_to_the_closest_with_nobody_in_range(self):
        from tools.rl.examples.group_fight_task import ATTACK_DANGEROUS, ATTACK_WEAKEST

        self.assertIs(self.target(ATTACK_WEAKEST, [self.far_marine]), self.far_marine)
        self.assertIs(self.target(ATTACK_DANGEROUS, [self.far_marine]), self.far_marine)

    def test_smart_pulls_back_only_when_hurt_and_close(self):
        from tools.rl.examples.group_fight_task import ATTACK_DANGEROUS, FIRST_MOVE

        west = FIRST_MOVE + 4
        self.assertEqual(self.smart([self.marine]), ATTACK_DANGEROUS)
        self.roach.health = 40
        self.assertEqual(self.smart([self.marine]), west)
        self.assertEqual(self.smart([self.far_marine]), ATTACK_DANGEROUS)
        self.assertEqual({name: rule(None) for name, rule in self.task.baseline_policies().items()
                          if name in ("attack", "focus", "threat")},
                         {"attack": 0, "focus": 1, "threat": 2})

    def test_life_kite_backs_off_only_with_an_enemy_within_its_distance(self):
        from tools.rl.examples.group_fight_task import ATTACK_CLOSEST, FIRST_MOVE

        west = FIRST_MOVE + 4
        self.roach.health = 72  # half life
        self.roach.weapon_cooldown = 24.0  # 75% of the cooldown left
        near_marine = StandInFighter(10, 26.5, 20.5, health=45, health_max=45, dps=10.0, ground_range=5)  # 6 away
        self.assertEqual(self.task.baseline_policies()["life_kite_7"](self.task.observe(self.roach, [], [near_marine])), west)
        self.assertEqual(self.task.baseline_policies()["life_kite_5"](self.task.observe(self.roach, [], [near_marine])),
                         ATTACK_CLOSEST)

    def test_es_rotate_roach_starting_policy_is_life_kite_5(self):
        from tools.rl.es import FAMILIES, LIFE_KITE_THETA0

        policy, rule = FAMILIES["rotate_roach"].policy(LIFE_KITE_THETA0), self.task.baseline_policies()["life_kite_5"]
        self.roach.health = 72  # half life
        for distance in (4.5, 5.5):
            for cooldown in (8.0, 24.0):  # 25% and 75% of the cooldown left
                marine = StandInFighter(10, 20.5 + distance, 20.5, health=45, health_max=45, dps=10.0, ground_range=5)
                self.roach.weapon_cooldown = cooldown
                observation = self.task.observe(self.roach, [], [marine])
                with self.subTest(distance=distance, cooldown=cooldown):
                    self.assertEqual(policy(observation), rule(observation))

    def test_group_slots_cover_the_largest_group_in_the_new_roach_configs(self):
        from pathlib import Path

        from tools.rl.config import load_config

        for name in ("roach_group_sizes", "roach_group_builtin"):
            with self.subTest(name):
                config = load_config(Path(__file__).parents[1] / f"tools/rl/configs/{name}.yaml")
                largest = max(sum(scenario.learner.values()) for scenario in config.scenarios)
                self.assertLessEqual(largest, self.task.group_slots)

    def test_es_kite_roach_starting_policy_is_the_kite_rule_at_5(self):
        from tools.rl.es import FAMILIES, KITE_THETA0
        from tools.rl.examples.group_fight_task import kite_rule

        policy = FAMILIES["kite_roach"].policy(KITE_THETA0)
        self.roach.weapon_cooldown = 10.0
        for distance in (4.5, 5.5):
            marine = StandInFighter(10, 20.5 + distance, 20.5, health=45, health_max=45, dps=10.0, ground_range=5)
            observation = self.task.observe(self.roach, [], [marine])
            with self.subTest(distance=distance):
                self.assertEqual(policy(observation), kite_rule(observation, 5.0))

    def test_group_slots_cover_the_largest_calibration_group(self):
        from pathlib import Path

        from tools.rl.config import load_config

        config = load_config(Path(__file__).parents[1] / "tools/rl/configs/roach_group_calibration.yaml")
        largest = max(sum(scenario.learner.values()) for scenario in config.scenarios)
        self.assertLessEqual(largest, self.task.group_slots)


class MutaliskGroupTaskTest(unittest.TestCase):
    """A mutalisk at (5.5, 20.5) in a playable area from 2 to 38 on both axes; the pathing grid
    has a wall right next to it, which a flying unit ignores."""

    def setUp(self):
        from types import SimpleNamespace

        from tools.rl.examples.mutalisk_group_task import MutaliskGroupTask

        self.task = MutaliskGroupTask()
        walls = [(6, 20), (4, 20)]
        self.task.start_game(SimpleNamespace(pathing_grid=StandInGrid(40, 40, walls),
                                             playable_area=SimpleNamespace(x=2, y=2, width=36, height=36)))
        self.muta = StandInFighter(1, 5.5, 20.5, health=120, health_max=120, ground_range=3)

    def marine(self, tag, x, y):
        return StandInFighter(tag, x, y, health=45, health_max=45, dps=10.0, ground_range=5)

    def rule(self, name, allies, enemies):
        return self.task.baseline_policies()[name](self.task.observe(self.muta, allies, enemies))

    def test_rays_stop_at_the_playable_area_edge_only(self):
        from tools.rl.examples.free_kite_task import DIRECTIONS
        from tools.rl.examples.group_fight_task import FIRST_RAY

        rays = self.task.observe(self.muta, [], [])[FIRST_RAY:FIRST_RAY + len(DIRECTIONS)]
        self.assertAlmostEqual(float(rays[0]), 1.0, places=5)  # east: 10, the ray's length, over the wall
        self.assertAlmostEqual(float(rays[4]), 0.35, places=5)  # west: 3.5 to the edge

    def test_kite_steps_away_only_while_cooling_with_a_marine_close(self):
        from tools.rl.examples.group_fight_task import ATTACK_CLOSEST, FIRST_MOVE

        west = FIRST_MOVE + 4
        close, far = [self.marine(10, 10.5, 20.5)], [self.marine(11, 13.5, 20.5)]
        self.assertEqual(self.rule("kite_6", [], close), ATTACK_CLOSEST)  # weapon ready
        self.muta.weapon_cooldown = 10.0
        self.assertEqual(self.rule("kite_6", [], close), west)
        self.assertEqual(self.rule("kite_6", [], far), ATTACK_CLOSEST)  # 8 away
        self.assertEqual(self.rule("kite_8", [], far), west)

    def test_es_kite_starting_policy_is_kite_4_5(self):
        from tools.rl.es import FAMILIES, KITE_THETA0

        rule, policy = self.task.baseline_policies()["kite_4.5"], FAMILIES["kite"].policy(KITE_THETA0)
        cases = {"no marines": [], "ready, near": [self.marine(10, 9.5, 20.5)]}
        observations = {name: self.task.observe(self.muta, [], enemies) for name, enemies in cases.items()}
        self.muta.weapon_cooldown = 10.0
        observations["cooling, near"] = self.task.observe(self.muta, [], [self.marine(10, 9.5, 20.5)])  # 4 away
        observations["cooling, far"] = self.task.observe(self.muta, [], [self.marine(10, 13.5, 20.5)])  # 8 away
        for name, observation in observations.items():
            with self.subTest(name):
                self.assertEqual(policy(observation), rule(observation))

    def test_es_rotate_mutalisk_starting_policy_is_life_kite(self):
        from tools.rl.es import FAMILIES, LIFE_KITE_THETA0

        policy, rule = FAMILIES["rotate_mutalisk"].policy(LIFE_KITE_THETA0), self.task.baseline_policies()["life_kite"]
        self.muta.health = 60  # half life
        for distance in (4.0, 5.0):
            for cooldown in (6.0, 18.0):  # 25% and 75% of the cooldown left
                self.muta.weapon_cooldown = cooldown
                observation = self.task.observe(self.muta, [], [self.marine(10, 5.5 + distance, 20.5)])
                with self.subTest(distance=distance, cooldown=cooldown):
                    self.assertEqual(policy(observation), rule(observation))

    def test_es_kite_attacks_the_weakest_when_its_score_is_positive(self):
        from tools.rl.es import FAMILIES, KITE_FEATURES, KITE_THETA0
        from tools.rl.examples.group_fight_task import ATTACK_WEAKEST

        theta = KITE_THETA0.copy()
        theta[len(KITE_FEATURES)] = 1.0  # the attack decision's bias
        observation = self.task.observe(self.muta, [], [self.marine(10, 9.5, 20.5)])
        self.assertEqual(FAMILIES["kite"].policy(theta)(observation), ATTACK_WEAKEST)

    def test_clump_regroups_only_out_of_marine_reach(self):
        from tools.rl.examples.group_fight_task import ATTACK_CLOSEST, FIRST_MOVE

        north = FIRST_MOVE + 2
        ally = [StandInFighter(2, 5.5, 23.5, health=120, health_max=120)]  # 3 north
        self.assertEqual(self.rule("clump_kite_6", ally, [self.marine(10, 15.5, 20.5)]), north)
        self.assertEqual(self.rule("clump_kite_6_r3", ally, [self.marine(10, 15.5, 20.5)]), ATTACK_CLOSEST)
        self.assertEqual(self.rule("clump_kite_6", ally, [self.marine(10, 10.5, 20.5)]), ATTACK_CLOSEST)

    def test_hurt_kite_steps_away_only_when_hurt(self):
        from tools.rl.examples.group_fight_task import ATTACK_CLOSEST, FIRST_MOVE

        west = FIRST_MOVE + 4
        near = [self.marine(10, 9.5, 20.5)]  # 4 away
        self.muta.weapon_cooldown = 10.0
        self.assertEqual(self.rule("hurt_kite_0.5", [], near), ATTACK_CLOSEST)  # full life
        self.muta.health = 48  # 40% life
        self.assertEqual(self.rule("hurt_kite_0.5", [], near), west)
        self.muta.weapon_cooldown = 0.0
        self.assertEqual(self.rule("hurt_kite_0.5", [], near), ATTACK_CLOSEST)  # weapon ready

    def test_life_kite_backs_off_while_more_cooldown_than_life_is_left(self):
        from tools.rl.examples.group_fight_task import ATTACK_CLOSEST, FIRST_MOVE

        west = FIRST_MOVE + 4
        near, far = [self.marine(10, 9.5, 20.5)], [self.marine(10, 13.5, 20.5)]
        self.muta.health = 60  # half life
        self.muta.weapon_cooldown = 18.0  # 75% of the cooldown left
        self.assertEqual(self.rule("life_kite", [], near), west)
        self.assertEqual(self.rule("life_kite", [], far), ATTACK_CLOSEST)  # 8 away
        self.muta.weapon_cooldown = 6.0  # 25% left
        self.assertEqual(self.rule("life_kite", [], near), ATTACK_CLOSEST)

    def test_spread_flies_away_from_a_close_ally_only_while_cooling(self):
        from tools.rl.examples.group_fight_task import ATTACK_CLOSEST, FIRST_MOVE, closest_ally_offset

        south = FIRST_MOVE + 6
        marine = [self.marine(10, 9.5, 20.5)]
        close, far = [StandInFighter(2, 5.5, 21.5, health=120, health_max=120)], [StandInFighter(2, 5.5, 23.5, health=120, health_max=120)]
        observation = self.task.observe(self.muta, close, marine)
        offset = closest_ally_offset(observation)
        self.assertAlmostEqual(offset.x, 0.0, places=5)
        self.assertAlmostEqual(offset.y, 1.0, places=5)
        self.assertIsNone(closest_ally_offset(self.task.observe(self.muta, [], marine)))
        self.assertEqual(self.rule("spread_1.5", close, marine), ATTACK_CLOSEST)  # weapon ready
        self.muta.weapon_cooldown = 10.0
        self.assertEqual(self.rule("spread_1.5", close, marine), south)
        self.assertEqual(self.rule("spread_1.5", far, marine), ATTACK_CLOSEST)  # 3 away
        self.assertEqual(self.rule("spread_2.5", far, marine), ATTACK_CLOSEST)
        self.assertEqual(self.rule("spread_1.5_life_kite", far, marine), self.rule("life_kite", far, marine))

    def test_spread_always_keeps_the_spacing_with_the_weapon_ready(self):
        from tools.rl.examples.group_fight_task import ATTACK_CLOSEST, FIRST_MOVE

        south = FIRST_MOVE + 6
        marine = [self.marine(10, 9.5, 20.5)]
        close = [StandInFighter(2, 5.5, 21.3, health=120, health_max=120)]  # 0.8 north
        self.assertEqual(self.rule("spread_always_1", close, marine), south)
        self.assertEqual(self.rule("spread_1", close, marine), ATTACK_CLOSEST)
        self.assertEqual(self.rule("spread_always_0.75", close, marine), ATTACK_CLOSEST)

    def test_es_spread_mutalisk_starting_policy_is_spread_1(self):
        from tools.rl.es import FAMILIES, SPREAD_THETA0

        policy, rule = FAMILIES["spread_mutalisk"].policy(SPREAD_THETA0), self.task.baseline_policies()["spread_1"]
        marine = [self.marine(10, 9.5, 20.5)]
        allies = {"0.8 north": [StandInFighter(2, 5.5, 21.3, health=120, health_max=120)],
                  "3 north": [StandInFighter(2, 5.5, 23.5, health=120, health_max=120)], "none": []}
        for cooldown in (0.0, 10.0):
            self.muta.weapon_cooldown = cooldown
            for name, ally in allies.items():
                observation = self.task.observe(self.muta, ally, marine)
                with self.subTest(cooldown=cooldown, ally=name):
                    self.assertEqual(policy(observation), rule(observation))

    def test_group_slots_cover_the_largest_group_in_every_mutalisk_config(self):
        from pathlib import Path

        from tools.rl.config import load_config

        for name in ("mutalisk_group_calibration", "mutalisk_group", "mutalisk_group_sizes", "mutalisk_group_builtin",
                     "mutalisk_thor_calibration", "mutalisk_thor"):
            with self.subTest(name):
                config = load_config(Path(__file__).parents[1] / f"tools/rl/configs/{name}.yaml")
                largest = max(sum(scenario.learner.values()) for scenario in config.scenarios)
                self.assertLessEqual(largest, self.task.group_slots)


class HydraliskGroupTaskTest(unittest.TestCase):
    """A hydralisk at (20.5, 20.5) with a zealot 2.5 to its east."""

    def setUp(self):
        from types import SimpleNamespace

        from tools.rl.examples.hydralisk_group_task import HydraliskGroupTask

        self.task = HydraliskGroupTask()
        self.task.start_game(SimpleNamespace(pathing_grid=StandInGrid(40, 40)))
        self.hydra = StandInFighter(1, 20.5, 20.5, health=90, health_max=90, ground_range=6)
        self.zealot = StandInFighter(10, 23.0, 20.5, health=150, health_max=150, dps=18.6, ground_range=0.1)

    def rule(self, name):
        return self.task.baseline_policies()[name](self.task.observe(self.hydra, [], [self.zealot]))

    def test_kite_rules(self):
        from tools.rl.examples.group_fight_task import ATTACK_CLOSEST, ATTACK_DANGEROUS, FIRST_MOVE

        west = FIRST_MOVE + 4
        self.assertEqual(sorted(self.task.baseline_policies()),
                         ["attack", "kite_2", "kite_3", "kite_5", "kite_threat_2", "kite_threat_3", "kite_threat_5",
                          "life_kite_3", "life_kite_5"])
        self.assertEqual(self.rule("kite_threat_3"), ATTACK_DANGEROUS)  # weapon ready
        self.hydra.weapon_cooldown = 5.0
        self.assertEqual(self.rule("kite_3"), west)
        self.assertEqual(self.rule("kite_threat_3"), west)
        self.assertEqual(self.rule("kite_2"), ATTACK_CLOSEST)  # the zealot is 2.5 away

    def test_group_slots_cover_the_largest_calibration_group(self):
        from pathlib import Path

        from tools.rl.config import load_config

        config = load_config(Path(__file__).parents[1] / "tools/rl/configs/hydralisk_group_calibration.yaml")
        largest = max(sum(scenario.learner.values()) for scenario in config.scenarios)
        self.assertLessEqual(largest, self.task.group_slots)


class StandInGroupCaster(StandInFighter):
    """A StandInFighter of a given type that can cast abilities."""

    def __init__(self, tag, x, y, type_id, **kwargs):
        super().__init__(tag, x, y, **kwargs)
        self.type_id = type_id

    def __call__(self, ability, target=None):
        self.commands.append(("ability", ability, target))


class RoachRavagerGroupTaskTest(unittest.TestCase):
    """A ravager and a roach at (20.5, 20.5); a marine 5 east, a marauder 8 north."""

    def setUp(self):
        from types import SimpleNamespace

        from tools.rl.examples.roach_ravager_group_task import RoachRavagerGroupTask

        self.task = RoachRavagerGroupTask()
        self.task.start_game(SimpleNamespace(pathing_grid=StandInGrid(40, 40)))
        self.ravager = StandInGroupCaster(1, 20.5, 20.5, UnitTypeId.RAVAGER, health=120, health_max=120, ground_range=6)
        self.roach = StandInGroupCaster(2, 20.5, 20.5, UnitTypeId.ROACH, health=145, health_max=145)
        self.marine = StandInFighter(10, 25.5, 20.5, health=45, health_max=45, dps=10.0, ground_range=5)
        self.marauder = StandInFighter(11, 20.5, 28.5, health=125, health_max=125, dps=20.0, ground_range=6)
        self.enemies = [self.marauder, self.marine]

    def ready(self):
        from sc2.ids.ability_id import AbilityId

        self.task.see_abilities({1: {AbilityId.EFFECT_CORROSIVEBILE}})

    def test_ravager_and_bile_ready_inputs(self):
        from tools.rl.examples.roach_ravager_group_task import BILE_READY, IS_RAVAGER

        self.ready()
        ravager = self.task.observe(self.ravager, [], self.enemies)
        roach = self.task.observe(self.roach, [], self.enemies)
        self.assertEqual(ravager.shape, self.task.observation_space.shape)
        self.assertEqual((ravager[IS_RAVAGER], ravager[BILE_READY]), (1.0, 1.0))
        self.assertEqual((roach[IS_RAVAGER], roach[BILE_READY]), (0.0, 0.0))

    def test_bile_actions(self):
        from sc2.ids.ability_id import AbilityId

        from tools.rl.examples.ravager_task import BILE_FLIGHT_SECONDS, DECISION_SECONDS
        from tools.rl.examples.group_fight_task import SEEN_ENEMIES
        from tools.rl.examples.roach_ravager_group_task import BILE_LEADS, BILE_TARGETS, bile_action

        self.assertEqual(self.task.action_space.n, bile_action(BILE_TARGETS[-1], BILE_LEADS[-1]) + 1 + SEEN_ENEMIES)
        self.task.observe(self.ravager, [], self.enemies)
        self.marine.position = Point2((25.1, 20.5))  # moved 0.4 toward the ravager
        self.task.observe(self.ravager, [], self.enemies)
        self.marauder.health = 50  # 20 / 50 now beats the marine's 10 / 45
        for action in (bile_action("closest", 1.0), bile_action("closest", 0.25), bile_action("dangerous", 0.5)):
            self.task.apply(self.ravager, action, [], self.enemies)
        (_, ability, led), (_, _, quarter), (_, _, dangerous) = self.ravager.commands
        self.assertEqual(ability, AbilityId.EFFECT_CORROSIVEBILE)
        self.assertAlmostEqual(led.x, 25.1 - 0.4 * BILE_FLIGHT_SECONDS / DECISION_SECONDS, places=4)
        self.assertAlmostEqual(quarter.x, 25.1 - 0.25 * 0.4 * BILE_FLIGHT_SECONDS / DECISION_SECONDS, places=4)
        self.assertEqual(dangerous, Point2((20.5, 28.5)))  # it didn't move
        self.task.apply(self.roach, bile_action("closest", 1.0), [], self.enemies)
        self.assertEqual(self.roach.commands, [("attack", self.marine)])

    def test_rotating_bile_rule_biles_like_its_bile_rule_and_otherwise_rotates(self):
        from tools.rl.examples.group_fight_task import ATTACK_DANGEROUS, step_away
        from tools.rl.examples.roach_ravager_group_task import bile_action

        rule = self.task.baseline_policies()["bile_dangerous_lead_0_life_kite_5"]
        self.marine.position = Point2((24.5, 20.5))  # 4 away
        self.ready()
        self.assertEqual(rule(self.task.observe(self.ravager, [], self.enemies)), bile_action("dangerous", 0.0))
        self.assertEqual(rule(self.task.observe(self.roach, [], self.enemies)), ATTACK_DANGEROUS)  # healthy, ready
        self.roach.health, self.roach.weapon_cooldown = 72, 24.0  # half life, 75% of the cooldown left
        observation = self.task.observe(self.roach, [], self.enemies)
        self.assertEqual(rule(observation), step_away(observation))

    def test_bile_rules(self):
        from tools.rl.examples.group_fight_task import ATTACK_DANGEROUS
        from tools.rl.examples.roach_ravager_group_task import bile_action

        rules = self.task.baseline_policies()
        self.assertIn("bile_dangerous_lead_1.5_clump_2", rules)
        lead_half = rules["bile_closest_lead_0.5"]
        self.assertEqual(lead_half(self.task.observe(self.ravager, [], self.enemies)), ATTACK_DANGEROUS)  # not ready
        self.ready()
        observation = self.task.observe(self.ravager, [], self.enemies)
        self.assertEqual(lead_half(observation), bile_action("closest", 0.5))
        self.assertEqual(rules["bile_dangerous_lead_0"](observation), bile_action("dangerous", 0.0))
        self.assertEqual(lead_half(self.task.observe(self.roach, [], self.enemies)), ATTACK_DANGEROUS)
        self.marine.position = Point2((30.5, 20.5))
        self.marauder.position = Point2((20.5, 30.5))  # both 10 away, out of bile range
        self.assertEqual(lead_half(self.task.observe(self.ravager, [], self.enemies)), ATTACK_DANGEROUS)

    def test_training_config(self):
        from pathlib import Path

        from tools.rl.config import load_config

        for name in ("roach_ravager_train.yaml", "roach_ravager_hard.yaml"):
            config = load_config(Path(__file__).parents[1] / "tools/rl/configs" / name)
            self.assertEqual(config.learner.decide_every, 1)  # bile's lead assumes it
            self.assertLessEqual(max(sum(s.learner.values()) for s in config.scenarios), self.task.group_slots)

    def test_ravagers_share_damage_while_their_bile_may_land(self):
        from tools.rl.examples.roach_ravager_group_task import BILE_CREDIT_STEPS, bile_action

        self.roach.weapon_cooldown = self.ravager.weapon_cooldown = 0.0
        units = [self.ravager, self.roach]
        self.task.share_team_reward(0.0, units)
        self.task.observe(self.ravager, [], self.enemies)
        self.task.apply(self.ravager, bile_action("closest", 1.0), [], self.enemies)
        self.roach.weapon_cooldown = 20.0  # the roach fires in the next step
        self.assertEqual(self.task.share_team_reward(4.0, units), {1: 2.0, 2: 2.0})
        for _ in range(BILE_CREDIT_STEPS - 1):
            self.assertEqual(self.task.share_team_reward(4.0, units), {1: 4.0})  # nobody fired: bile's
        self.assertEqual(self.task.share_team_reward(4.0, units), {1: 2.0, 2: 2.0})  # no one: all share

    def test_bile_the_ith_closest_where_it_is(self):
        from tools.rl.examples.roach_ravager_group_task import bile_seen_action

        self.task.observe(self.ravager, [], self.enemies)
        self.marine.position = Point2((25.1, 20.5))  # moving: lead 0 ignores it
        self.task.observe(self.ravager, [], self.enemies)
        for index in (0, 1, 3):  # 3: past the last enemy, the farthest
            self.task.apply(self.ravager, bile_seen_action(index), [], self.enemies)
        self.assertEqual([target for _, _, target in self.ravager.commands],
                         [Point2((25.1, 20.5)), Point2((20.5, 28.5)), Point2((20.5, 28.5))])

    def test_es_starting_policy_is_the_clump_rule(self):
        from tools.rl.es import THETA0, bile_policy
        from tools.rl.examples.group_fight_task import ATTACK_DANGEROUS
        from tools.rl.examples.roach_ravager_group_task import bile_action, bile_seen_action

        rule, policy = self.task.baseline_policies()["bile_closest_lead_0_clump_1"], bile_policy(THETA0)
        same = {ATTACK_DANGEROUS: ATTACK_DANGEROUS, bile_action("closest", 0.0): bile_seen_action(0)}
        beside = StandInFighter(12, 25.5, 22.0, health=45, health_max=45, dps=10.0, ground_range=5)
        cases = {"not ready": self.enemies + [beside]}
        observations = {"not ready": self.task.observe(self.ravager, [], cases["not ready"])}
        self.ready()
        observations["alone"] = self.task.observe(self.ravager, [], self.enemies)
        observations["clumped"] = self.task.observe(self.ravager, [], self.enemies + [beside])
        self.marine.position, self.marauder.position = Point2((30.5, 20.5)), Point2((20.5, 30.5))
        beside.position = Point2((30.5, 22.0))
        observations["out of range"] = self.task.observe(self.ravager, [], self.enemies + [beside])
        for name, observation in observations.items():
            with self.subTest(name):
                self.assertEqual(policy(observation), same[rule(observation)])
        self.assertEqual(policy(observations["clumped"]), bile_seen_action(0))

    def test_es_update_climbs_a_known_fitness(self):
        from tools.rl.es import candidates_for, es_update

        rng = np.random.default_rng(0)
        target, theta = np.array([1.0, -2.0, 0.5]), np.zeros(3)
        for _ in range(300):
            directions = rng.standard_normal((4, 3))
            scores = np.array([-np.sum((c - target) ** 2) for c in candidates_for(theta, directions)])
            theta = es_update(theta, directions, scores)
        self.assertLess(np.linalg.norm(theta - target), 0.3)

    def test_clump_sizes(self):
        from tools.rl.examples.roach_ravager_group_task import clump_sizes

        beside = StandInFighter(12, 25.5, 22.0, health=45, health_max=45, dps=10.0, ground_range=5)
        observation = self.task.observe(self.ravager, [], self.enemies + [beside])
        self.assertEqual(clump_sizes(observation), [1, 1, 0])  # marine and beside, then the marauder

    def test_clump_condition(self):
        from tools.rl.examples.group_fight_task import ATTACK_DANGEROUS
        from tools.rl.examples.roach_ravager_group_task import bile_action

        self.ready()
        rules = self.task.baseline_policies()
        observation = self.task.observe(self.ravager, [], self.enemies)  # the marauder is 8.6 from the marine
        self.assertEqual(rules["bile_closest_lead_0.5_clump_1"](observation), ATTACK_DANGEROUS)
        beside = StandInFighter(12, 25.5, 22.0, health=45, health_max=45, dps=10.0, ground_range=5)  # 1.5 from it
        observation = self.task.observe(self.ravager, [], self.enemies + [beside])
        self.assertEqual(rules["bile_closest_lead_0.5_clump_1"](observation), bile_action("closest", 0.5))
        self.assertEqual(rules["bile_closest_lead_0.5_clump_2"](observation), ATTACK_DANGEROUS)

    def test_group_slots_cover_the_largest_calibration_group(self):
        from pathlib import Path

        from tools.rl.config import load_config

        for name in ("roach_ravager_group_calibration", "roach_ravager_builtin"):
            with self.subTest(name):
                config = load_config(Path(__file__).parents[1] / f"tools/rl/configs/{name}.yaml")
                largest = max(sum(scenario.learner.values()) for scenario in config.scenarios)
                self.assertLessEqual(largest, self.task.group_slots)


class RoachBurrowGroupTaskTest(unittest.TestCase):
    """A roach at (20.5, 20.5) with a marine 5 to the east."""

    def setUp(self):
        from types import SimpleNamespace

        from tools.rl.examples.roach_burrow_group_task import RoachBurrowGroupTask

        self.task = RoachBurrowGroupTask()
        self.task.start_game(SimpleNamespace(pathing_grid=StandInGrid(40, 40)))
        self.roach = StandInGroupCaster(1, 20.5, 20.5, UnitTypeId.ROACH, health=145, health_max=145)
        self.roach.is_burrowed = False
        self.marine = StandInFighter(10, 25.5, 20.5, health=45, health_max=45, dps=10.0, ground_range=5)

    def rule(self, name):
        return self.task.baseline_policies()[name](self.task.observe(self.roach, [], [self.marine]))

    def test_detection_inputs(self):
        from tools.rl.enemies import Detection
        from tools.rl.examples.roach_burrow_group_task import BURROWED, DETECTION_MARGIN, DETECTOR_PRESENT

        self.roach.is_burrowed = True
        self.task.see_detectors([Detection(Point2((20.5, 25.5)), 13.0)])  # 5 away: 8 inside
        observation = self.task.observe(self.roach, [], [self.marine])
        self.assertEqual(observation.shape, self.task.observation_space.shape)
        self.assertEqual(observation[BURROWED], 1.0)
        self.assertEqual(observation[DETECTOR_PRESENT], 1.0)
        self.assertAlmostEqual(float(observation[DETECTION_MARGIN]), -0.8, places=5)

    def test_burrow_rules(self):
        from tools.rl.enemies import Detection
        from tools.rl.examples.group_fight_task import ATTACK_DANGEROUS
        from tools.rl.examples.roach_burrow_group_task import BURROW, UNBURROW

        self.assertEqual(self.rule("burrow"), ATTACK_DANGEROUS)
        self.roach.health = 50  # 34%
        self.assertEqual(self.rule("burrow"), BURROW)
        self.assertEqual(self.rule("burrow_0.3_0.6"), ATTACK_DANGEROUS)
        self.roach.is_burrowed, self.roach.health = True, 90  # 62%
        self.assertEqual(self.rule("burrow"), BURROW)  # stays down until above 70%
        self.assertEqual(self.rule("burrow_0.3_0.6"), UNBURROW)
        self.task.see_detectors([Detection(Point2((20.5, 25.5)), 13.0)])
        self.assertEqual(self.rule("burrow"), BURROW)
        self.assertEqual(self.rule("careful_burrow"), UNBURROW)  # detected: come back up
        self.roach.is_burrowed, self.roach.health = False, 50
        self.assertEqual(self.rule("careful_burrow"), ATTACK_DANGEROUS)  # no burrowing where detected

    def test_burrow_actions(self):
        from sc2.ids.ability_id import AbilityId

        from tools.rl.examples.roach_burrow_group_task import BURROW, UNBURROW

        self.task.apply(self.roach, BURROW, [], [self.marine])
        self.task.apply(self.roach, UNBURROW, [], [self.marine])
        self.assertEqual(self.roach.commands, [("ability", AbilityId.BURROWDOWN_ROACH, None),
                                               ("ability", AbilityId.BURROWUP_ROACH, None)])

    def test_group_slots_cover_the_largest_group(self):
        from pathlib import Path

        from tools.rl.config import load_config

        config = load_config(Path(__file__).parents[1] / "tools/rl/configs/roach_burrow_group.yaml")
        largest = max(sum(scenario.learner.values()) for scenario in config.scenarios)
        self.assertLessEqual(largest, self.task.group_slots)


class DecideEveryTest(unittest.TestCase):
    def test_decides_on_the_first_and_every_kth_step(self):
        from types import SimpleNamespace

        from tools.rl.driver import Episode, GameDriver
        from tools.rl.examples.cooldown_kite_task import CooldownKiteTask

        config = SimpleNamespace(decision_interval=3, learner=SimpleNamespace(decide_every=3))
        driver = GameDriver(config, CooldownKiteTask(), random.Random())
        self.addCleanup(driver.loop.close)
        episode = Episode(scenario=None)
        self.assertEqual([driver._decides_now(episode) for _ in range(7)],
                         [True, False, False, True, False, False, True])

    def test_config(self):
        base = {"learner": {"task": "m:C"}, "scenarios": [{"name": "s", "learner": {"Zergling": 1}, "enemy": {"Marine": 1}}]}
        self.assertEqual(parse_config(base).learner.decide_every, 1)
        self.assertEqual(parse_config({**base, "learner": {"task": "m:C", "decide_every": 3}}).learner.decide_every, 3)
        with self.assertRaisesRegex(ValueError, "decide_every"):
            parse_config({**base, "learner": {"task": "m:C", "decide_every": 0}})


class PPOSettingsTest(unittest.TestCase):
    BASE = {"learner": {"task": "m:C"}, "scenarios": [{"name": "s", "learner": {"Zergling": 1}, "enemy": {"Marine": 1}}]}

    def test_defaults_keep_the_old_behavior(self):
        settings = parse_config(self.BASE).ppo
        self.assertEqual((settings.learning_rate_end, settings.clip_range, settings.target_kl, settings.normalize_reward),
                         (None, 0.2, None, False))
        kwargs = settings.as_kwargs()
        self.assertEqual(kwargs["learning_rate"], 3e-4)
        self.assertNotIn("normalize_reward", kwargs)
        self.assertNotIn("learning_rate_end", kwargs)

    def test_stability_settings(self):
        settings = parse_config({**self.BASE, "ppo": {"learning_rate": 1e-4, "learning_rate_end": 1e-5,
                                                      "clip_range": 0.1, "target_kl": 0.02, "normalize_reward": True}}).ppo
        kwargs = settings.as_kwargs()
        schedule = kwargs["learning_rate"]
        self.assertAlmostEqual(schedule(1.0), 1e-4)
        self.assertAlmostEqual(schedule(0.5), 5.5e-5)
        self.assertAlmostEqual(schedule(0.0), 1e-5)
        self.assertEqual((kwargs["clip_range"], kwargs["target_kl"], settings.normalize_reward), (0.1, 0.02, True))
        with self.assertRaisesRegex(ValueError, "unknown 'ppo' settings"):
            parse_config({**self.BASE, "ppo": {"learning_rate_finish": 1e-5}})


class StabilityTest(unittest.TestCase):
    def test_windows_and_holds(self):
        from tools.rl.stability import holds, windows

        rising = [(steps, 0.1 if steps <= 100_000 else 0.4) for steps in range(20_000, 2_500_001, 20_000)]
        start, peak, end = windows(rising)
        self.assertEqual((round(start, 2), round(peak, 2), round(end, 2)), (0.1, 0.4, 0.4))
        self.assertTrue(holds(start, peak, end))
        self.assertFalse(holds(0.1, 0.47, 0.0))  # the collapse of the earlier run
        self.assertFalse(holds(0.1, 0.3, 0.15))  # holding, but barely above the start
        self.assertFalse(holds(0.1, 0.47, 0.40))  # fell more than 5 points from the peak window

    def test_reads_progress_csv(self):
        import tempfile
        from pathlib import Path

        from tools.rl.stability import read_win_rates

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "progress.csv"
            path.write_text("time/total_timesteps,fights/all/win_rate\n16384,\n32768,0.25\n", encoding="utf-8")
            self.assertEqual(read_win_rates(path), [(32768, 0.25)])


class KiteOrderTest(unittest.TestCase):
    """A marine at (20, 20) against zerglings to its east."""

    def order(self, cooldown, *ling_positions):
        from tools.rl.enemies import kite_order

        marine = StandInUnit(1, 20, 20, cooldown=cooldown)
        lings = [StandInUnit(2 + index, x, y) for index, (x, y) in enumerate(ling_positions)]
        return kite_order(marine, lings)

    def test_attacks_the_closest_when_the_weapon_is_ready(self):
        self.assertEqual(self.order(0, (22, 20), (21, 20)).tag, 3)

    def test_steps_back_from_the_targets_while_cooling_down(self):
        self.assertEqual(self.order(5, (22, 21), (22, 19)), Point2((18, 20)))

    def test_attacks_while_cooling_down_when_nothing_is_close(self):
        self.assertEqual(self.order(5, (25, 20)).tag, 2)

    def test_kite_is_a_known_behavior(self):
        scenario = parse_scenarios([{"name": "k", "learner": {"Zergling": 8}, "enemy": {"Marine": 3},
                                     "enemy_behavior": "kite"}])[0]
        self.assertEqual(scenario.enemy_behavior, "kite")


class ResearchTest(unittest.TestCase):
    def test_buildings_for_each_upgrade(self):
        from sc2.ids.upgrade_id import UpgradeId

        from tools.rl.research import research_buildings

        self.assertEqual(research_buildings({UpgradeId.BURROW}), {UnitTypeId.HATCHERY})
        # The Lair that Tunneling Claws needs also researches Burrow, so no Hatchery.
        self.assertEqual(research_buildings({UpgradeId.BURROW, UpgradeId.TUNNELINGCLAWS}),
                         {UnitTypeId.LAIR, UnitTypeId.ROACHWARREN})
        self.assertEqual(research_buildings(set()), set())

    def test_spots_are_buildable_far_from_the_center_and_apart(self):
        from tools.rl.research import building_spots

        grid = StandInGrid(30, 30, walls=[(x, y) for x in range(30) for y in range(30) if x < 5 or y < 5])
        center = Point2((15, 15))
        spots = building_spots(grid, center, 3, spacing=8)
        self.assertEqual(len(spots), 3)
        self.assertTrue(all(spot.x >= 5 and spot.y >= 5 for spot in spots))
        self.assertTrue(all(a.distance_to(b) >= 8 for a in spots for b in spots if a != b))
        self.assertEqual(spots[0], Point2((29, 29)))  # the farthest buildable cell comes first
        with self.assertRaises(ValueError):
            building_spots(grid, center, 20, spacing=8)

    def test_nothing_to_research_is_done_at_once(self):
        import asyncio

        from tools.rl.research import UpgradeResearch

        self.assertTrue(asyncio.run(UpgradeResearch(()).done(bot=None)))


class LearnerBotTest(unittest.TestCase):
    def test_task_gets_the_map_when_the_game_starts(self):
        import asyncio
        from types import SimpleNamespace

        from tools.rl.driver import LearnerBot

        class RecordingTask(MicroTask):
            game_info = None

            def start_game(self, game_info):
                self.game_info = game_info

        class StandInClient:
            game_step = 1

            async def debug_show_map(self):
                pass

        task = RecordingTask()
        config = SimpleNamespace(decision_interval=3, learner=SimpleNamespace(upgrades=()))
        bot = LearnerBot(SimpleNamespace(config=config, task=task))
        bot.client, bot.game_info = StandInClient(), "the map"
        asyncio.run(bot.on_start())
        self.assertEqual(task.game_info, "the map")


class DriverCrashTest(unittest.TestCase):
    def setUp(self):
        import asyncio
        import contextlib
        import io
        from types import SimpleNamespace

        from sc2.protocol import ProtocolError

        from tools.rl.driver import GameDriver
        from tools.rl.examples.cooldown_kite_task import CooldownKiteTask

        self.asyncio, self.ProtocolError = asyncio, ProtocolError
        self.driver = GameDriver(SimpleNamespace(decision_interval=3), CooldownKiteTask(), random.Random())
        self.addCleanup(self.driver.loop.close)
        self.stderr = io.StringIO()
        redirect = contextlib.redirect_stderr(self.stderr)
        redirect.__enter__()
        self.addCleanup(redirect.__exit__, None, None, None)

    def start_crashing_game(self):
        async def crash():
            raise self.ProtocolError("Unable to complete the step request, not in a game.")

        self.games_started = getattr(self, "games_started", 0) + 1
        self.driver.events = self.asyncio.Queue()
        self.driver._waiter, self.driver._waiting_for = None, None
        self.driver.game_task = self.driver.loop.create_task(crash())

    def test_crash_during_a_fight_ends_it_as_game_over(self):
        self.start_crashing_game()
        seen = np.full(self.driver.last_observation.shape, 0.5, dtype=np.float32)
        self.driver.last_observation = seen
        self.driver._waiter, self.driver._waiting_for = self.driver.loop.create_future(), "decide"
        observation, _, terminated, truncated, info = self.driver.step(0)
        self.assertEqual((terminated, truncated, info), (False, True, {"outcome": "game_over"}))
        self.assertIs(observation, seen)
        self.assertIn("SC2 game crashed", self.stderr.getvalue())

    def test_timeout_ends_with_the_last_real_observation(self):
        seen = np.full(self.driver.last_observation.shape, 0.5, dtype=np.float32)
        self.driver.events = self.asyncio.Queue()
        self.driver.game_task = self.driver.loop.create_future()
        self.addCleanup(self.driver.game_task.cancel)
        self.driver.events.put_nowait(("decide", seen))
        self.driver.events.put_nowait(("end", {"outcome": "timeout"}))

        self.driver._waiter, self.driver._waiting_for = self.driver.loop.create_future(), "decide"
        self.assertIs(self.driver.step(0)[0], seen)
        self.driver._waiter = self.driver.loop.create_future()
        observation, _, terminated, truncated, _ = self.driver.step(0)
        self.assertEqual((terminated, truncated), (False, True))
        self.assertIs(observation, seen)

    def test_reset_restarts_crashed_games_then_gives_up(self):
        from tools.rl.driver import MAX_GAME_RESTARTS

        self.driver._start_game = self.start_crashing_game
        with self.assertRaises(RuntimeError) as raised:
            self.driver.reset(scenario=None)
        self.assertEqual(self.games_started, MAX_GAME_RESTARTS + 1)
        self.assertIsInstance(raised.exception.__cause__, self.ProtocolError)

    def test_every_action_is_allowed_by_default(self):
        from types import SimpleNamespace

        from tools.rl.env import SC2MicroEnv

        mask = self.driver.action_mask
        self.assertEqual((mask.shape, bool(mask.all())), ((self.driver.task.action_space.n,), True))
        env = SimpleNamespace(driver=self.driver)
        self.assertIs(SC2MicroEnv.action_masks(env), mask)
        self.assertFalse(self.driver.task.uses_action_masks)

    def test_leaves_the_game_between_fights_before_the_loop_limit(self):
        from types import SimpleNamespace

        from tools.rl.driver import GAME_LOOP_LIMIT

        class StandInClient:
            left = False

            async def leave(self):
                self.left = True

        asked = []

        class Asked(Exception):
            pass

        async def ask(event):
            asked.append(event)
            raise Asked  # stop here: the rest of the step needs a real game

        self.driver._ask = ask
        for game_loop, leaves in ((1000, False), (GAME_LOOP_LIMIT - 100, True)):
            with self.subTest(game_loop=game_loop):
                asked.clear()
                self.driver.episode = None
                bot = SimpleNamespace(client=StandInClient(), state=SimpleNamespace(game_loop=game_loop))
                try:
                    self.driver.loop.run_until_complete(self.driver.learner_step(bot))
                except Asked:
                    pass
                self.assertEqual((bot.client.left, self.driver.restarting), (leaves, leaves))
                self.assertEqual(asked, [] if leaves else [("ready", None)])

    def test_abilities_are_shared_only_with_tasks_that_want_them(self):
        from types import SimpleNamespace

        from sc2.ids.ability_id import AbilityId

        class RecordingTask(MicroTask):
            seen = None

            def see_abilities(self, available):
                self.seen = available

        class StandInBot:
            calls = 0

            async def get_available_abilities(self, units):
                self.calls += 1
                return [[AbilityId.EFFECT_CORROSIVEBILE, AbilityId.MOVE], [AbilityId.MOVE]][:len(units)]

        own = [SimpleNamespace(tag=7), SimpleNamespace(tag=8)]
        for wants in (False, True):
            with self.subTest(wants=wants):
                task, bot = RecordingTask(), StandInBot()
                task.wants_abilities = wants
                self.driver.task = task
                self.driver.loop.run_until_complete(self.driver._share_abilities(bot, own))
                self.assertEqual(bot.calls, int(wants))
                expected = {7: {AbilityId.EFFECT_CORROSIVEBILE, AbilityId.MOVE}, 8: {AbilityId.MOVE}}
                self.assertEqual(task.seen, expected if wants else None)

    def test_a_planned_restart_is_not_reported_as_a_crash(self):
        self.start_crashing_game()
        self.driver.restarting = True
        self.assertEqual(self.driver._next_event()[0], "game_over")
        self.assertIn("fresh SC2 game", self.stderr.getvalue())
        self.assertNotIn("crashed", self.stderr.getvalue())

    def test_a_failing_client_cancels_the_other(self):
        from tools.rl.driver import _run_together

        cancelled = []

        async def fails():
            raise self.ProtocolError("not in a game")

        async def runs_forever():
            try:
                await self.asyncio.sleep(3600)
            except self.asyncio.CancelledError:
                cancelled.append(True)
                raise

        with self.assertRaises(ExceptionGroup):
            self.driver.loop.run_until_complete(_run_together(runs_forever(), fails()))
        self.assertEqual(cancelled, [True])

    def test_close_cancels_each_client_only_once(self):
        from tools.rl.driver import _run_together

        cancelled_twice = []

        async def client():
            """Like a python-sc2 request: on cancel, it still waits for the answer before
            re-raising, and a second cancel during that wait is fatal."""
            try:
                await self.asyncio.sleep(3600)
            except self.asyncio.CancelledError:
                try:
                    await self.asyncio.sleep(0.01)
                except self.asyncio.CancelledError:
                    cancelled_twice.append(True)
                raise

        self.driver.loop.close = lambda: None  # close() closes the loop; setUp's cleanup does it too
        self.driver.game_task = self.driver.loop.create_task(_run_together(client(), client()))
        self.driver.loop.run_until_complete(self.asyncio.sleep(0))
        self.driver.close()
        self.assertEqual(cancelled_twice, [])


if __name__ == "__main__":
    unittest.main()
