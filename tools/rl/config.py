"""Training configuration (YAML): map, decision interval, learner, enemy control, scenarios,
PPO settings and optional self-play."""

import importlib
from dataclasses import dataclass, fields, replace
from pathlib import Path

import yaml
from sc2.data import AIBuild, Difficulty, Race
from sc2.ids.upgrade_id import UpgradeId

from tools.rl.scenarios import Scenario, parse_scenarios

ENEMY_MODES = ("builtin", "scripted", "frozen")


@dataclass(frozen=True)
class LearnerConfig:
    race: str
    task: str  # "module:Class" of a MicroTask
    upgrades: tuple[str, ...] = ()  # UpgradeId names, researched once per game before the first fight
    # The units decide on every k-th game step of a fight and keep their orders in between; the
    # enemy still reacts every step.
    decide_every: int = 1


@dataclass(frozen=True)
class EnemyConfig:
    mode: str  # builtin | scripted | frozen
    race: str
    difficulty: str = "VeryEasy"  # builtin only
    # builtin only: its strategy (AIBuild). RandomBuild draws one per SC2 game, which then holds
    # for every fight in that game.
    build: str = "RandomBuild"
    task: str | None = None  # frozen and self-play: the enemy's MicroTask
    model: str | None = None  # frozen: the enemy's trained model (.zip)


@dataclass(frozen=True)
class SelfPlayConfig:
    rounds: int  # each round trains one side, then the sides swap
    steps_per_round: int


@dataclass(frozen=True)
class PPOConfig:
    learning_rate: float = 3e-4
    n_steps: int = 1024  # decisions collected per environment before each update
    batch_size: int = 256
    n_epochs: int = 10
    gamma: float = 0.99
    ent_coef: float = 0.01
    net_arch: tuple[int, ...] = (64, 64)  # hidden layer sizes
    learning_rate_end: float | None = None  # the learning rate falls linearly to this over the run
    clip_range: float = 0.2  # how far one update may move the model's choices
    target_kl: float | None = None  # an update stops its passes once the model changed this much
    normalize_reward: bool = False  # rescale rewards to a steady size while training (VecNormalize)

    def as_kwargs(self) -> dict:
        """PPO's keyword arguments. net_arch goes into policy_kwargs; normalize_reward is for the
        training envs, not PPO."""
        not_ppo = {"net_arch", "learning_rate_end", "normalize_reward"}
        settings = {field.name: getattr(self, field.name) for field in fields(self) if field.name not in not_ppo}
        if self.learning_rate_end is not None:
            settings["learning_rate"] = linear_schedule(self.learning_rate, self.learning_rate_end)
        return {**settings, "policy_kwargs": {"net_arch": list(self.net_arch)}}


def linear_schedule(start: float, end: float):
    """An SB3 schedule: SB3 passes the share of training still to go, from 1 down to 0."""
    return lambda progress_remaining: end + (start - end) * progress_remaining


@dataclass(frozen=True)
class TrainingConfig:
    map: str
    decision_interval: int  # game frames between decisions (python-sc2's game_step)
    learner: LearnerConfig
    enemy: EnemyConfig
    scenarios: tuple[Scenario, ...]
    self_play: SelfPlayConfig | None = None
    ppo: PPOConfig = PPOConfig()


def _parse_ppo(settings: dict) -> PPOConfig:
    known = {field.name for field in fields(PPOConfig)}
    unknown = set(settings) - known
    if unknown:
        raise ValueError(f"unknown 'ppo' settings: {', '.join(sorted(unknown))}. Known: {', '.join(sorted(known))}")
    if "net_arch" in settings:
        settings = {**settings, "net_arch": tuple(int(size) for size in settings["net_arch"])}
    return PPOConfig(**settings)


def _check_enum(value: str, enum, what: str) -> str:
    if value not in enum.__members__:
        raise ValueError(f"Unknown {what} '{value}'. Valid: {', '.join(enum.__members__)}")
    return value


def load_class(path: str):
    if ":" not in path:
        raise ValueError(f"expected 'module:Class', got '{path}'")
    module_name, class_name = path.split(":")
    return getattr(importlib.import_module(module_name), class_name)


def parse_config(data: dict) -> TrainingConfig:
    learner = data.get("learner") or {}
    enemy = data.get("enemy") or {}
    if "task" not in learner:
        raise ValueError("'learner.task' is required")

    enemy_config = EnemyConfig(
        mode=enemy.get("mode", "scripted"),
        race=_check_enum(enemy.get("race", "Terran"), Race, "race"),
        difficulty=_check_enum(enemy.get("difficulty", "VeryEasy"), Difficulty, "difficulty"),
        build=_check_enum(enemy.get("build", "RandomBuild"), AIBuild, "build"),
        task=enemy.get("task"),
        model=enemy.get("model"),
    )
    if enemy_config.mode not in ENEMY_MODES:
        raise ValueError(f"'enemy.mode' must be one of {', '.join(ENEMY_MODES)}")

    self_play = None
    if data.get("self_play"):
        self_play = SelfPlayConfig(
            rounds=int(data["self_play"]["rounds"]),
            steps_per_round=int(data["self_play"]["steps_per_round"]),
        )
        if not enemy_config.task:
            raise ValueError("self-play needs 'enemy.task', the model the other side trains")
    elif enemy_config.mode == "frozen" and not (enemy_config.task and enemy_config.model):
        raise ValueError("enemy mode 'frozen' needs 'enemy.task' and 'enemy.model'")

    config = TrainingConfig(
        map=data.get("map", "MicroTraining410"),
        decision_interval=int(data.get("decision_interval", 4)),
        learner=LearnerConfig(
            race=_check_enum(learner.get("race", "Terran"), Race, "race"),
            task=learner["task"],
            upgrades=tuple(_check_enum(str(name).upper(), UpgradeId, "upgrade") for name in learner.get("upgrades") or ()),
            decide_every=int(learner.get("decide_every", 1)),
        ),
        enemy=enemy_config,
        scenarios=tuple(parse_scenarios(data.get("scenarios"))),
        self_play=self_play,
        ppo=_parse_ppo(data.get("ppo") or {}),
    )
    if config.decision_interval < 1:
        raise ValueError("'decision_interval' must be at least 1")
    if config.learner.decide_every < 1:
        raise ValueError("'learner.decide_every' must be at least 1")
    return config


def self_play_side(config: TrainingConfig, side: str, other_model: str) -> TrainingConfig:
    """The config for training one side of a self-play pair against a frozen copy of the other.
    Side "a" is the configured learner; side "b" is the configured enemy, with every scenario
    seen from its side."""
    if side == "a":
        learner = config.learner
        enemy = EnemyConfig(mode="frozen", race=config.enemy.race, task=config.enemy.task, model=other_model)
        scenarios = config.scenarios
    elif side == "b":
        learner = LearnerConfig(race=config.enemy.race, task=config.enemy.task)
        enemy = EnemyConfig(mode="frozen", race=config.learner.race, task=config.learner.task, model=other_model)
        scenarios = tuple(scenario.swapped() for scenario in config.scenarios)
    else:
        raise ValueError(f"side must be 'a' or 'b', got '{side}'")
    return replace(config, learner=learner, enemy=enemy, scenarios=scenarios)


def with_enemy(config: TrainingConfig, mode: str | None = None, build: str | None = None) -> TrainingConfig:
    """The config with the enemy's mode and the built-in AI's build replaced where given."""
    enemy = config.enemy
    if mode:
        enemy = replace(enemy, mode=mode)
    if build:
        enemy = replace(enemy, build=_check_enum(build, AIBuild, "build"))
    return replace(config, enemy=enemy)


def load_config(path: str | Path) -> TrainingConfig:
    with open(path, encoding="utf-8") as f:
        return parse_config(yaml.safe_load(f) or {})
