# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project state

This repository is at an early/exploratory stage. There is no build system, package manifest (no
`requirements.txt`/`pyproject.toml`), README, or test suite yet. The only content is
`helper-scripts/`, a set of standalone Python scripts used to manually probe/drive a StarCraft II
instance over its API before any bot framework is chosen.

## What the helper scripts do

Both scripts talk to SC2's raw API directly over a websocket (`ws://<host>:5001/sc2api`), sending
serialized `sc2api_pb2.Request` protobuf messages and parsing `sc2api_pb2.Response` back. They do
**not** use a higher-level bot framework (e.g. `python-sc2`/`burnysc2`) — requests are built and
sent manually. Runtime dependencies: `websocket-client` and `s2clientprotocol`.

- `helper-scripts/ping_sc2.py` — minimal connectivity check. Connects to a host's SC2 API port and
  sends a `RequestPing`, printing status/version info. Usage: `python ping_sc2.py <host>`.
- `helper-scripts/bot_join.py` — joins a running SC2 game as a client, sends a chat message to
  prove it can issue actions, then polls `RequestObservation` in a loop (realtime mode, no
  `RequestStep`) printing loop/minerals/gas/supply/unit-count roughly once per second until the
  game ends. Usage: `python bot_join.py <host> [--race terran|zerg|protoss|random]`.

The port constants (`SERVER_GAME`/`SERVER_BASE` = 5100/5101, `CLIENT_GAME`/`CLIENT_BASE` =
5102/5103) and the `bot_join.py` comment about "Both SC2 processes are on Windows" describe a
two-host setup: an SC2 process the script connects out to (via `<host>:5001/sc2api`), and a second
local SC2 process this script's `RequestJoinGame` configures ports for. Keep this cross-host
port layout in mind when editing connection logic — the game/base port pairs are not
interchangeable.
