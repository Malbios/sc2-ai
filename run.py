import argparse
import asyncio
import random
from pathlib import Path

import aiohttp

from sc2.controller import Controller
from sc2.data import Difficulty, Race
from sc2.main import play_from_websocket
from sc2.maps import Map
from sc2.player import Bot, Computer

from bot import CompetitiveBot
import config


async def play_remote_game(host: str, port: int, map_name: str, bot_race: Race,
                            opponent_race: Race, difficulty: Difficulty, realtime: bool):
    url = f"ws://{host}:{port}/sc2api"
    print(f"Connecting to SC2 at {url} ...")

    async with aiohttp.ClientSession() as session:
        try:
            ws = await session.ws_connect(url, timeout=30)
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            raise RuntimeError(
                f"Could not reach SC2 at {url}: {e}\n"
                "Check that SC2 is running on that machine with "
                f"'-listen 0.0.0.0 -port {port}', and that no firewall is blocking it."
            ) from e

        controller = Controller(ws, process=None)
        bot = Bot(bot_race, CompetitiveBot(), name=config.BOT_NAME)
        computer = Computer(opponent_race, difficulty)

        game_map = Map(Path(map_name + ".SC2Map"))
        await controller.create_game(game_map, [bot, computer], realtime=realtime)

        return await play_from_websocket(ws, bot, realtime=realtime)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", type=str, default=None,
                         help=f"Map to play on. Default: random from {', '.join(config.MAP_POOL)}")
    parser.add_argument("--realtime", action="store_true", default=config.REALTIME,
                         help=f"Play in realtime. Default: {config.REALTIME}")
    return parser.parse_args()


def main():
    args = parse_arguments()

    map_name = args.map or random.choice(config.MAP_POOL)
    bot_race = Race[config.BOT_RACE.capitalize()]
    opponent_race = Race[config.OPPONENT_RACE.capitalize()]
    difficulty = Difficulty[config.OPPONENT_DIFFICULTY]

    print(f"===== {config.BOT_NAME} ({config.BOT_RACE}) =====")
    print(f"Map: {map_name}")
    print(f"Opponent: {config.OPPONENT_RACE} {config.OPPONENT_DIFFICULTY}")
    print(f"Realtime: {args.realtime}")

    result = asyncio.run(
        play_remote_game(
            config.REMOTE_HOST, config.REMOTE_PORT, map_name,
            bot_race, opponent_race, difficulty, args.realtime,
        )
    )

    print(f"Result: {result}")


if __name__ == "__main__":
    main()
