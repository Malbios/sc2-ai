import math
import random
import unittest
from collections import Counter

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
        observation = self.task.observe(self.stalker, [], [self.far, self.near]).astype(float).round(3).tolist()
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
        bot = LearnerBot(SimpleNamespace(config=SimpleNamespace(decision_interval=3), task=task))
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
        self.driver._waiter, self.driver._waiting_for = self.driver.loop.create_future(), "decide"
        _, _, terminated, truncated, info = self.driver.step(0)
        self.assertEqual((terminated, truncated, info), (False, True, {"outcome": "game_over"}))
        self.assertIn("SC2 game crashed", self.stderr.getvalue())

    def test_reset_restarts_crashed_games_then_gives_up(self):
        from tools.rl.driver import MAX_GAME_RESTARTS

        self.driver._start_game = self.start_crashing_game
        with self.assertRaises(RuntimeError) as raised:
            self.driver.reset(scenario=None)
        self.assertEqual(self.games_started, MAX_GAME_RESTARTS + 1)
        self.assertIsInstance(raised.exception.__cause__, self.ProtocolError)

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
