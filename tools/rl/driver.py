"""Runs a python-sc2 game inside an event loop the caller owns, and exposes the learner's side of
it one decision at a time.

python-sc2 drives a game by calling the bot's on_step, while Gymnasium expects the trainer to call
env.step(action). The driver joins the two: the learner bot's on_step hands each decision request
to the driver and waits for the answer, and env.step() runs the event loop only until the next
request. SC2 waits in between, because the game only advances after on_step returns. Everything
stays on one thread, which python-sc2 requires (it installs a SIGINT handler when it starts SC2).

One game serves many episodes. A reset kills every non-structure unit and spawns the next
scenario, so the townhalls (and the learner's research buildings) survive and the game never ends
on its own.

Several learner units ("unit cycling"): each decision request is for one unit, and the game only
advances once every living learner unit has decided. The team reward for that game step is paid
out on the step that let the game advance.
"""

import asyncio
import math
import random
import sys
import time
import traceback
from dataclasses import dataclass, field

import numpy as np
from sc2 import maps
from sc2.bot_ai import BotAI
from sc2.data import Race
from sc2.main import _host_game, _join_game
from sc2.player import Bot
from sc2.portconfig import Portconfig
from sc2.position import Point2
from sc2.sc2process import KillSwitch
from sc2.units import Units

from tools.rl.config import TrainingConfig
from tools.rl.enemies import EnemyBriefing, make_enemy_player
from tools.rl.scenarios import Scenario, spawn_centers
from tools.rl.kiting import KiteMeter
from tools.rl.research import UpgradeResearch
from tools.rl.task import FightSnapshot, MicroTask, life

# Steps to wait for spawned units to show up before clearing and spawning again.
SPAWN_TIMEOUT_STEPS = 25
# How often a game may be restarted in a row before the driver gives up.
MAX_GAME_RESTARTS = 3
# SC2 ends every game at this game loop (6.5 game hours), so the driver starts a fresh game
# between fights shortly before that instead of losing a fight to it.
GAME_LOOP_LIMIT = 2**19
RESTART_MARGIN_LOOPS = int(22.4 * 60 * 10)
ABORT = object()


@dataclass
class Episode:
    scenario: Scenario
    phase: str = "clear"  # clear -> spawn -> wait -> fight
    learner_center: Point2 | None = None
    enemy_center: Point2 | None = None
    learner_tags: set[int] = field(default_factory=set)
    enemy_tags: set[int] = field(default_factory=set)
    dead_enemies: set[int] = field(default_factory=set)
    enemy_life: dict[int, float] = field(default_factory=dict)  # last known, per tag
    unseen_steps: int = 0  # steps where a living enemy of the fight was out of sight
    kite_meter: KiteMeter = field(default_factory=KiteMeter)
    start: FightSnapshot | None = None
    last: FightSnapshot | None = None
    start_loop: int = 0
    wait_steps: int = 0


class LearnerBot(BotAI):
    def __init__(self, driver: "GameDriver"):
        super().__init__()
        self.driver = driver
        self.research = UpgradeResearch(driver.config.learner.upgrades)
        driver.learner_bot = self

    async def on_start(self):
        self.client.game_step = self.driver.config.decision_interval
        # Full vision, so both sides of a fight are always known. Fights are short-range, so this
        # barely changes what a unit would see anyway. Only the learner may send this: it's one
        # toggle for the whole game, so a second client sending it turns vision off again.
        await self.client.debug_show_map()
        self.driver.task.start_game(self.game_info)

    async def on_step(self, iteration: int):
        if await self.research.done(self):
            await self.driver.learner_step(self)


async def _run_together(*coroutines):
    """Runs both clients of a game; if one fails, the other is cancelled (which shuts its SC2
    process down) instead of running on without a partner."""
    async with asyncio.TaskGroup() as group:
        tasks = [group.create_task(coroutine) for coroutine in coroutines]
    return [task.result() for task in tasks]


def _group_near(units: Units, center: Point2, expected: dict) -> Units | None:
    """The spawned group around `center`, once every expected unit has appeared."""
    radius = 4 + 1.5 * math.sqrt(sum(expected.values()))
    group = units.filter(lambda unit: unit.type_id in expected and unit.distance_to(center) < radius)
    for type_id, count in expected.items():
        if sum(1 for unit in group if unit.type_id == type_id) < count:
            return None
    return group


class GameDriver:
    def __init__(self, config: TrainingConfig, task: MicroTask, rng: random.Random, launch_delay: float = 0.0):
        self.config = config
        self.task = task
        self.rng = rng
        self.launch_delay = launch_delay
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.game_task: asyncio.Task | None = None
        self.learner_bot: LearnerBot | None = None
        self.events: asyncio.Queue | None = None
        self.episode: Episode | None = None
        self.restarting = False  # the game was left on purpose (see GAME_LOOP_LIMIT)
        self.enemy_briefing: EnemyBriefing | None = None  # read by the scripted enemy every step
        self.pending_reward = 0.0
        self._waiter: asyncio.Future | None = None
        self._waiting_for: str | None = None  # "ready" or "decide" once the env has seen the request
        self._zero_obs = np.zeros(task.observation_space.shape, dtype=np.float32)

    # ----- learner side (runs inside the game, on the event loop) -----

    async def _ask(self, event: tuple):
        self._waiter = self.loop.create_future()
        self.events.put_nowait(event)
        return await self._waiter

    async def learner_step(self, bot: BotAI):
        if self.episode is None:
            if bot.state.game_loop > GAME_LOOP_LIMIT - RESTART_MARGIN_LOOPS:
                # The enemy client then fails like after a crash, and the next reset starts a
                # fresh game; `restarting` keeps that from being reported as a crash.
                self.restarting = True
                await bot.client.leave()
                return
            self.episode = Episode(await self._ask(("ready", None)))
        episode = self.episode

        if episode.phase == "clear":
            doomed = bot.all_units.filter(lambda unit: unit.owner_id in (1, 2) and not unit.is_structure)
            if doomed:
                await bot.client.debug_kill_unit(doomed)
            episode.phase = "spawn"
            return

        if episode.phase == "spawn":
            center = bot.game_info.map_center
            learner_center, enemy_center = spawn_centers((center.x, center.y), episode.scenario, self.rng)
            episode.learner_center, episode.enemy_center = Point2(learner_center), Point2(enemy_center)
            scenario = episode.scenario
            leash = self.rng.uniform(*scenario.leash) if scenario.leash else 0.0
            self.enemy_briefing = EnemyBriefing(scenario.enemy_behavior, episode.enemy_center, leash)
            me, them = bot.player_id, 3 - bot.player_id
            await bot.client.debug_create_unit(
                [(type_id, count, episode.learner_center, me) for type_id, count in episode.scenario.learner.items()]
                + [(type_id, count, episode.enemy_center, them) for type_id, count in episode.scenario.enemy.items()]
                + [(type_id, count, episode.enemy_center, them) for type_id, count in episode.scenario.enemy_support.items()]
            )
            episode.phase, episode.wait_steps = "wait", 0
            return

        if episode.phase == "wait":
            own = _group_near(bot.units, episode.learner_center, episode.scenario.learner)
            enemies = _group_near(bot.enemy_units, episode.enemy_center, episode.scenario.enemy)
            if own is None or enemies is None:
                episode.wait_steps += 1
                if episode.wait_steps > SPAWN_TIMEOUT_STEPS:
                    episode.phase = "clear"
                return
            episode.learner_tags, episode.enemy_tags = own.tags, enemies.tags
            episode.start = episode.last = FightSnapshot(life(own), life(enemies), life(own), life(enemies))
            episode.start_loop = bot.state.game_loop
            episode.phase = "fight"
            self.task.start_episode()

        # An enemy only counts as dead when SC2 reports its death. One that is merely out of sight
        # keeps its last known life, so running away can never look like winning.
        episode.dead_enemies |= bot.state.dead_units & episode.enemy_tags
        living_enemy_tags = episode.enemy_tags - episode.dead_enemies
        enemies = bot.enemy_units.tags_in(living_enemy_tags)
        for enemy in enemies:
            episode.enemy_life[enemy.tag] = enemy.health + enemy.shield
        if len(enemies) < len(living_enemy_tags):
            episode.unseen_steps += 1
        enemy_life = sum(episode.enemy_life[tag] for tag in living_enemy_tags)

        own = bot.units.tags_in(episode.learner_tags)
        now = FightSnapshot(life(own), enemy_life, episode.start.own_start, episode.start.enemy_start)
        self.pending_reward += self.task.reward(episode.last, now)
        self.pending_reward += sum(self.task.unit_reward(unit, own.tags_not_in({unit.tag}), enemies) for unit in own)
        for unit in own:
            episode.kite_meter.update(unit, enemies)
        episode.last = now

        outcome = None
        if not own and not living_enemy_tags:
            outcome = "tie"
        elif not own:
            outcome = "loss"
        elif not living_enemy_tags:
            outcome = "win"
        elif (bot.state.game_loop - episode.start_loop) / 22.4 > episode.scenario.time_limit:
            outcome = "timeout"
        if outcome:
            self._finish(outcome, bot.state.game_loop, own)
            return

        self.task.see_detectors((bot.enemy_units | bot.enemy_structures).filter(lambda unit: unit.is_detector))
        await self._share_abilities(bot, own)
        for unit in sorted(own, key=lambda u: u.tag):
            allies = own.tags_not_in({unit.tag})
            action = await self._ask(("decide", self.task.observe(unit, allies, enemies)))
            if action is ABORT:
                self.episode = None
                return
            self.task.apply(unit, int(action), allies, enemies)

    async def _share_abilities(self, bot: BotAI, own: Units):
        if self.task.wants_abilities:
            abilities = await bot.get_available_abilities(own)
            self.task.see_abilities({unit.tag: set(usable) for unit, usable in zip(own, abilities)})

    def _finish(self, outcome: str, game_loop: int, survivors: Units):
        episode = self.episode
        self.pending_reward += self.task.terminal_reward(outcome, survivors)
        self.events.put_nowait(("end", {
            "scenario": episode.scenario.name,
            "outcome": outcome,
            "damage_dealt": 1 - episode.last.enemy_life / max(episode.start.enemy_start, 1.0),
            "damage_taken": 1 - episode.last.own_life / max(episode.start.own_start, 1.0),
            "game_seconds": (game_loop - episode.start_loop) / 22.4,
            "unseen_steps": episode.unseen_steps,  # should stay 0 with full vision
            "kite_share": episode.kite_meter.share,
        }))
        self.episode = None
        self.enemy_briefing = None

    # ----- env side (called by the Gymnasium env, runs the loop until the next event) -----

    def _start_game(self):
        if self.launch_delay:
            time.sleep(self.launch_delay)  # SC2 processes started at the same instant fail on Linux
            self.launch_delay = 0.0
        self.events = asyncio.Queue()
        self.episode = None
        self.restarting = False
        self._waiter, self._waiting_for = None, None

        learner = Bot(Race[self.config.learner.race], LearnerBot(self), name="Learner")
        self.enemy_briefing = None
        enemy = make_enemy_player(self.config.enemy, self.config.decision_interval, briefing=lambda: self.enemy_briefing)
        game_map = maps.get(self.config.map)

        if isinstance(enemy, Bot):
            portconfig = Portconfig()
            players = [learner, enemy]
            self.game_task = self.loop.create_task(_run_together(
                _host_game(game_map, players, portconfig=portconfig),
                _join_game(players, realtime=False, portconfig=portconfig),
            ))
        else:
            self.game_task = self.loop.create_task(_host_game(game_map, [learner, enemy]))

    def _next_event(self) -> tuple[str, object]:
        async def wait():
            getter = asyncio.ensure_future(self.events.get())
            done, _ = await asyncio.wait({getter, self.game_task}, return_when=asyncio.FIRST_COMPLETED)
            if getter in done:
                return getter.result()
            getter.cancel()
            return ("game_over", None)

        kind, payload = self.loop.run_until_complete(wait())
        if kind == "game_over" and not self.game_task.cancelled():
            error = self.game_task.exception()  # read even when expected, or asyncio reports it later
            if self.restarting:
                print("Starting a fresh SC2 game before SC2's game length limit", file=sys.stderr)
            elif error:
                payload = error
                print("SC2 game crashed, restarting on the next reset:", file=sys.stderr)
                traceback.print_exception(payload, file=sys.stderr)
        if kind in ("ready", "decide"):
            self._waiting_for = kind
        return kind, payload

    def _answer(self, value):
        self._waiter.set_result(value)
        self._waiting_for = None

    def reset(self, scenario: Scenario) -> np.ndarray:
        """Start a fight in `scenario` and return the first unit's observation."""
        restarts = 0
        if self.game_task is None or self.game_task.done():
            self._start_game()

        sent = False
        if self._waiting_for == "decide":
            self._answer(ABORT)
        elif self._waiting_for == "ready":
            self._answer(scenario)
            sent = True

        while True:
            kind, payload = self._next_event()
            if kind == "game_over":
                restarts += 1
                if restarts > MAX_GAME_RESTARTS:
                    raise RuntimeError(f"the SC2 game ended {restarts} times in a row during reset") from payload
                self._start_game()
                sent = False
            elif kind == "ready":
                self._answer(scenario)
                sent = True
            elif kind == "decide":
                if sent:
                    self.pending_reward = 0.0
                    return payload
                self._answer(ABORT)
            # "end" events of an episode that was already over are skipped

    def step(self, action: int):
        """Gymnasium step: (observation, reward, terminated, truncated, info)."""
        if self._waiting_for != "decide":
            raise RuntimeError("step() called without a pending decision; call reset() first")
        self._answer(action)
        kind, payload = self._next_event()
        reward, self.pending_reward = self.pending_reward, 0.0

        if kind == "decide":
            return payload, reward, False, False, {}
        if kind == "end":
            timed_out = payload["outcome"] == "timeout"
            return self._zero_obs, reward, not timed_out, timed_out, payload
        return self._zero_obs, reward, False, True, {"outcome": "game_over"}

    def save_replay(self, path: str):
        """Save a replay of the game so far: every fight since the game started. The request
        must not overlap one of the learner's, so first let the game run until the learner is
        waiting on the driver (right after a fight ends, it is still stepping the game)."""
        if self.game_task is None or self.game_task.done():
            raise RuntimeError("no running game to save a replay of")
        while self._waiting_for is None:
            kind, _ = self._next_event()
            if kind == "game_over":
                raise RuntimeError("the game ended before the replay could be saved")
        self.loop.run_until_complete(self.learner_bot.client.save_replay(path))

    def close(self):
        # Cancelling the game makes python-sc2 shut its SC2 processes down; cancelling everything
        # else (e.g. aiohttp's connection cleanup) keeps asyncio from warning about pending tasks.
        # The game goes first and alone: it cancels its clients itself, and python-sc2 exits the
        # process when a client's request is cancelled a second time.
        if self.game_task is not None and not self.game_task.done():
            self.game_task.cancel()
            self.loop.run_until_complete(asyncio.gather(self.game_task, return_exceptions=True))
        pending = asyncio.all_tasks(self.loop)
        for task in pending:
            task.cancel()
        self.loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        self.loop.run_until_complete(self.loop.shutdown_asyncgens())
        KillSwitch.kill_all()
        self.loop.close()
