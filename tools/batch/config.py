"""Batch run configuration: which bot plays which built-in AI opponents on which maps."""

from dataclasses import dataclass
from pathlib import Path

import yaml
from sc2.data import AIBuild, Difficulty, Race


@dataclass(frozen=True)
class Opponent:
    race: str
    difficulty: str = "Easy"
    build: str = "RandomBuild"

    def label(self) -> str:
        return f"{self.race}-{self.difficulty}-{self.build}"


@dataclass(frozen=True)
class BatchConfig:
    bot: str  # "module:Class", e.g. "bot:CompetitiveBot"
    race: str
    maps: tuple[str, ...]
    opponents: tuple[Opponent, ...]
    games_per_matchup: int = 1
    parallel: int = 4
    game_time_limit: int | None = 1800  # game seconds; a game that runs longer is a Tie
    wall_timeout: int = 1800  # real seconds before a game's process is killed


@dataclass(frozen=True)
class GameSpec:
    index: int
    map: str
    opponent: Opponent

    def name(self) -> str:
        return f"{self.index:04d}-{self.map}-{self.opponent.label()}"


def _check_enum(value: str, enum, what: str) -> str:
    if value not in enum.__members__:
        raise ValueError(f"Unknown {what} '{value}'. Valid: {', '.join(enum.__members__)}")
    return value


def parse_config(data: dict) -> BatchConfig:
    for key in ("maps", "opponents"):
        if not data.get(key):
            raise ValueError(f"'{key}' must be a non-empty list")

    opponents = tuple(
        Opponent(
            race=_check_enum(o["race"], Race, "race"),
            difficulty=_check_enum(o.get("difficulty", "Easy"), Difficulty, "difficulty"),
            build=_check_enum(o.get("build", "RandomBuild"), AIBuild, "AI build"),
        )
        for o in data["opponents"]
    )

    bot = data.get("bot", "bot:CompetitiveBot")
    if ":" not in bot:
        raise ValueError(f"'bot' must look like 'module:Class', got '{bot}'")

    config = BatchConfig(
        bot=bot,
        race=_check_enum(data.get("race", "Zerg"), Race, "race"),
        maps=tuple(data["maps"]),
        opponents=opponents,
        games_per_matchup=int(data.get("games_per_matchup", 1)),
        parallel=int(data.get("parallel", 4)),
        game_time_limit=data.get("game_time_limit", 1800),
        wall_timeout=int(data.get("wall_timeout", 1800)),
    )
    if config.games_per_matchup < 1 or config.parallel < 1:
        raise ValueError("'games_per_matchup' and 'parallel' must be at least 1")
    return config


def load_config(path: str | Path) -> BatchConfig:
    with open(path, encoding="utf-8") as f:
        return parse_config(yaml.safe_load(f) or {})


def expand_games(config: BatchConfig) -> list[GameSpec]:
    """Every map x opponent combination, games_per_matchup times, interleaved so that a run
    cut short still covers every matchup roughly evenly."""
    games = []
    for _ in range(config.games_per_matchup):
        for game_map in config.maps:
            for opponent in config.opponents:
                games.append(GameSpec(len(games), game_map, opponent))
    return games
