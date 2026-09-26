"""The entry point AI Arena runs. tools/create_ladder_zip.py packs this file into the zip as run.py."""

import argparse
import asyncio

import aiohttp
from sc2.data import Race
from sc2.main import play_from_websocket
from sc2.player import Bot
from sc2.portconfig import Portconfig

import config
from bot import CompetitiveBot


def parse_ladder_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--GamePort", type=int, required=True)
    parser.add_argument("--StartPort", type=int, required=True)
    parser.add_argument("--LadderServer", default="127.0.0.1")
    parser.add_argument("--OpponentId")
    parser.add_argument("--RealTime", action="store_true")
    arguments, _ = parser.parse_known_args()
    return arguments


def ladder_portconfig(start_port: int) -> Portconfig:
    """The port layout every ladder bot derives from StartPort, so both players agree on it."""
    return Portconfig(
        server_ports=[start_port + 2, start_port + 3],
        player_ports=[[start_port + 4, start_port + 5]],
    )


async def play_ladder_game(arguments: argparse.Namespace):
    bot = Bot(Race[config.BOT_RACE.capitalize()], CompetitiveBot(), name=config.BOT_NAME)
    # Connecting here instead of passing play_from_websocket a URL: in python-sc2 7.3.0 a failed
    # connection from a URL raises an AttributeError from its cleanup, hiding the real error.
    async with aiohttp.ClientSession() as session:
        websocket = await session.ws_connect(f"ws://{arguments.LadderServer}:{arguments.GamePort}/sc2api", timeout=120)
        return await play_from_websocket(
            websocket, bot, realtime=arguments.RealTime, portconfig=ladder_portconfig(arguments.StartPort)
        )


def main():
    arguments = parse_ladder_arguments()
    result = asyncio.run(play_ladder_game(arguments))
    print(f"Result: {result} (opponent {arguments.OpponentId})")


if __name__ == "__main__":
    main()
