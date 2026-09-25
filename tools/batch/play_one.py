"""Plays ONE game of the bot against the built-in AI and writes the outcome as JSON.

Started by tools.batch.run as its own process: python-sc2 kills every SC2 process in the
Python process when a game ends, so games can't share one.
"""

import argparse
import importlib
import json
import time
import traceback

from sc2 import maps
from sc2.data import AIBuild, Difficulty, Race
from sc2.main import run_game
from sc2.player import Bot, Computer


def load_class(path: str):
    module_name, class_name = path.split(":")
    return getattr(importlib.import_module(module_name), class_name)


def make_recording_bot(bot_class, record: dict):
    """The configured bot, plus remembering the game loop it ended on."""

    class RecordingBot(bot_class):
        async def on_end(self, result):
            state = getattr(self, "state", None)
            record["game_loop"] = state.game_loop if state else 0
            await super().on_end(result)

    RecordingBot.__name__ = bot_class.__name__
    return RecordingBot()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bot", required=True)
    parser.add_argument("--race", required=True)
    parser.add_argument("--map", required=True)
    parser.add_argument("--opponent-race", required=True)
    parser.add_argument("--difficulty", required=True)
    parser.add_argument("--build", required=True)
    parser.add_argument("--game-time-limit", type=int, default=None)
    parser.add_argument("--replay", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args()

    record = {"game_loop": 0}
    started = time.monotonic()
    try:
        bot_class = load_class(args.bot)
        result = run_game(
            maps.get(args.map),
            [
                Bot(Race[args.race], make_recording_bot(bot_class, record), name=bot_class.__name__),
                Computer(Race[args.opponent_race], Difficulty[args.difficulty], AIBuild[args.build]),
            ],
            realtime=False,
            save_replay_as=args.replay,
            game_time_limit=args.game_time_limit,
        )
        record["result"] = result.name
    except Exception:
        record["result"] = "Crash"
        record["error"] = traceback.format_exc(limit=8)
        traceback.print_exc()

    record["game_seconds"] = round(record.pop("game_loop") / 22.4, 1)
    record["wall_seconds"] = round(time.monotonic() - started, 1)
    with open(args.result, "w", encoding="utf-8") as f:
        json.dump(record, f)


if __name__ == "__main__":
    main()
