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
out on the step that let the game advance. Group tasks (MicroTask.group_slots) instead get one
request per game step with every unit's observation, and every unit's action in the answer.
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
from sc2.ids.unit_typeid import UnitTypeId
from sc2.main import _host_game, _join_game
from sc2.player import Bot
from sc2.portconfig import Portconfig
from sc2.position import Point2
from sc2.sc2process import KillSwitch
from sc2.units import Units

from tools.rl.config import TrainingConfig
from tools.rl.enemies import SCAN_ENERGY, SCAN_RADIUS, Detection, EnemyBriefing, active_scans, make_enemy_player
from tools.rl.scenarios import Scenario, spawn_centers
from tools.rl.kiting import KiteMeter
from tools.rl.research import UpgradeResearch, building_spots
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
ENERGY = 1  # debug_set_unit_value's code for a unit's energy
SCANNER_SPOT = 6  # the enemy's Orbital Command takes the 7th spot from research.building_spots


@dataclass
class Episode:
    scenario: Scenario
    phase: str = "clear"  # clear -> spawn -> wait -> fight
    learner_center: Point2 | None = None
    enemy_center: Point2 | None = None
    learner_tags: set[int] = field(default_factory=set)
    slot_tags: list[int] = field(default_factory=list)  # group tasks: the unit deciding in each slot
    enemy_tags: set[int] = field(default_factory=set)
    dead_enemies: set[int] = field(default_factory=set)
    enemy_life: dict[int, float] = field(default_factory=dict)  # last known, per tag
    unseen_steps: int = 0  # steps where a living enemy of the fight was out of sight
    kite_meter: KiteMeter = field(default_factory=KiteMeter)
    start: FightSnapshot | None = None
    last: FightSnapshot | None = None
    start_loop: int = 0
    wait_steps: int = 0
    fight_steps: int = 0


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


def _scanner_spot(bot: BotAI) -> Point2:
    """Where the enemy's Orbital Command goes: far from the fights, past the learner's research
    buildings (which take the farthest spots)."""
    return building_spots(bot.game_info.placement_grid, bot.game_info.map_center, SCANNER_SPOT + 1)[SCANNER_SPOT]


def detections(bot: BotAI) -> list[Detection]:
    """Where the enemy can see burrowed units: its detector units and structures, and its scans."""
    detectors = (bot.enemy_units | bot.enemy_structures).filter(lambda unit: unit.is_detector)
    return ([Detection(unit.position, unit.detect_range) for unit in detectors]
            + [Detection(position, SCAN_RADIUS) for position in active_scans(bot.state.effects)])


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
        # Returned when a fight ends. PPO estimates the rest of a cut-off fight from it, so it must
        # be a real situation, not a blank one.
        slots = (task.group_slots,) if task.group_slots else ()
        self.last_observation = np.zeros(slots + task.observation_space.shape, dtype=np.float32)
        self.action_mask = np.ones(task.action_space.n, dtype=bool)  # for the pending decision
        self.pending_slot_rewards = np.zeros(task.group_slots, dtype=np.float32)  # group tasks: per slot

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
            reaction = self.rng.uniform(*scenario.bile_dodge_reaction) if scenario.bile_dodge_reaction else None
            scan_reaction = self.rng.uniform(*scenario.scan_reaction) if scenario.enemy_scans else None
            self.enemy_briefing = EnemyBriefing(scenario.enemy_behavior, episode.enemy_center, leash, reaction, scan_reaction)
            me, them = bot.player_id, 3 - bot.player_id
            scanner = []
            if scenario.enemy_scans and not bot.enemy_structures(UnitTypeId.ORBITALCOMMAND):
                scanner = [(UnitTypeId.ORBITALCOMMAND, 1, _scanner_spot(bot), them)]
            await bot.client.debug_create_unit(
                [(type_id, count, episode.learner_center, me) for type_id, count in episode.scenario.learner.items()]
                + [(type_id, count, episode.enemy_center, them) for type_id, count in episode.scenario.enemy.items()]
                + [(type_id, count, episode.enemy_center, them) for type_id, count in episode.scenario.enemy_support.items()]
                + scanner
            )
            episode.phase, episode.wait_steps = "wait", 0
            return

        if episode.phase == "wait":
            own = _group_near(bot.units, episode.learner_center, episode.scenario.learner)
            enemies = _group_near(bot.enemy_units, episode.enemy_center, episode.scenario.enemy)
            scanners = bot.enemy_structures(UnitTypeId.ORBITALCOMMAND)
            if own is None or enemies is None or (episode.scenario.enemy_scans and not scanners):
                episode.wait_steps += 1
                if episode.wait_steps > SPAWN_TIMEOUT_STEPS:
                    episode.phase = "clear"
                return
            if episode.scenario.enemy_scans:
                await bot.client.debug_set_unit_value(scanners, ENERGY, SCAN_ENERGY * episode.scenario.enemy_scans)
            episode.learner_tags, episode.enemy_tags = own.tags, enemies.tags
            episode.slot_tags = sorted(own.tags)[:self.task.group_slots]
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
        if self.task.group_slots:
            self._add_slot_rewards(self.task.share_team_reward(self.task.reward(episode.last, now), own), own, enemies)
        else:
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

        if not self._decides_now(episode):
            return
        self.task.see_detectors(detections(bot))
        await self._share_abilities(bot, own)
        if self.task.group_slots:
            await self._decide_as_group(own, enemies)
        else:
            await self._decide_one_by_one(own, enemies)

    def _add_slot_rewards(self, shares: dict[int, float], own: Units, enemies: Units):
        """Group tasks: each living unit's share of the team reward plus its own unit_reward, into
        its slot."""
        self._add_to_slots({unit.tag: shares.get(unit.tag, 0.0) + self.task.unit_reward(unit, own.tags_not_in({unit.tag}), enemies)
                            for unit in own})

    def _add_to_slots(self, rewards: dict[int, float]):
        for slot, tag in enumerate(self.episode.slot_tags):
            self.pending_slot_rewards[slot] += rewards.get(tag, 0.0)

    def _decides_now(self, episode: Episode) -> bool:
        """Counts the fight's steps; true on its first step and every learner.decide_every-th after."""
        episode.fight_steps += 1
        return (episode.fight_steps - 1) % self.config.learner.decide_every == 0

    async def _decide_one_by_one(self, own: Units, enemies: Units):
        for unit in sorted(own, key=lambda u: u.tag):
            allies = own.tags_not_in({unit.tag})
            observation = self.task.observe(unit, allies, enemies)
            if self.task.uses_action_masks:
                self.action_mask = self.task.action_mask(unit, allies, enemies)
            action = await self._ask(("decide", observation))
            if action is ABORT:
                self.episode = None
                return
            self.task.apply(unit, int(action), allies, enemies)

    async def _decide_as_group(self, own: Units, enemies: Units):
        """One request for all units: row i of the observation is the unit in slot i, zeros for a
        slot whose unit is dead, and action i is that unit's."""
        observations = np.zeros_like(self.last_observation)
        slot_units = {}
        for slot, tag in enumerate(self.episode.slot_tags):
            unit = own.find_by_tag(tag)
            if unit is not None:
                slot_units[slot] = unit
                observations[slot] = self.task.observe(unit, own.tags_not_in({tag}), enemies)
        actions = await self._ask(("decide", observations))
        if actions is ABORT:
            self.episode = None
            return
        for slot, unit in slot_units.items():
            self.task.apply(unit, int(actions[slot]), own.tags_not_in({unit.tag}), enemies)

    async def _share_abilities(self, bot: BotAI, own: Units):
        if self.task.wants_abilities:
            abilities = await bot.get_available_abilities(own)
            self.task.see_abilities({unit.tag: set(usable) for unit, usable in zip(own, abilities)})

    def _finish(self, outcome: str, game_loop: int, survivors: Units):
        episode = self.episode
        if self.task.group_slots:
            win_reward = self.task.terminal_reward(outcome, survivors)
            self._add_to_slots({unit.tag: win_reward for unit in survivors})
        else:
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
                    self.pending_slot_rewards[:] = 0.0
                    self.last_observation = payload
                    return payload
                self._answer(ABORT)
            # "end" events of an episode that was already over are skipped

    def step(self, action: int | np.ndarray):
        """Gymnasium step: (observation, reward, terminated, truncated, info). Group tasks pass one
        action per slot."""
        if self._waiting_for != "decide":
            raise RuntimeError("step() called without a pending decision; call reset() first")
        self._answer(action)
        kind, payload = self._next_event()
        reward, self.pending_reward = self.pending_reward, 0.0
        extra = {}
        if self.task.group_slots:
            slot_rewards, self.pending_slot_rewards = self.pending_slot_rewards, np.zeros_like(self.pending_slot_rewards)
            reward, extra = float(slot_rewards.sum()), {"slot_rewards": slot_rewards}

        if kind == "decide":
            self.last_observation = payload
            return payload, reward, False, False, extra
        if kind == "end":
            timed_out = payload["outcome"] == "timeout"
            return self.last_observation, reward, not timed_out, timed_out, {**payload, **extra}
        return self.last_observation, reward, False, True, {"outcome": "game_over", **extra}

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
