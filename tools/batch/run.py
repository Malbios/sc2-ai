"""Plays a batch of games of the bot against the built-in AI, several at a time.

    python -m tools.batch.run --config tools/batch/example.yaml [--parallel 4] [--games-per-matchup 3]

Each game runs in its own process (tools.batch.play_one). Results, replays and per-game logs go
to runs/<time>-<commit>/, and a summary is printed at the end.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

from tools.batch.config import BatchConfig, GameSpec, expand_games, load_config
from tools.batch.summary import summarize

REPO_ROOT = Path(__file__).resolve().parents[2]
# SC2 processes launched at the same instant fail on Linux (noted in python-sc2's main.py).
LAUNCH_GAP_SECONDS = 5


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()


def start_game(config: BatchConfig, game: GameSpec, run_dir: Path) -> tuple[subprocess.Popen, float]:
    command = [
        sys.executable, "-m", "tools.batch.play_one",
        "--bot", config.bot,
        "--race", config.race,
        "--map", game.map,
        "--opponent-race", game.opponent.race,
        "--difficulty", game.opponent.difficulty,
        "--build", game.opponent.build,
        "--replay", str(run_dir / "replays" / f"{game.name()}.SC2Replay"),
        "--result", str(run_dir / "results" / f"{game.name()}.json"),
    ]
    if config.game_time_limit:
        command += ["--game-time-limit", str(config.game_time_limit)]

    log = open(run_dir / "logs" / f"{game.name()}.log", "w", encoding="utf-8")
    # A new session puts the game and its SC2 process in one process group, so a timeout can
    # kill both.
    process = subprocess.Popen(
        command, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
    )
    log.close()
    return process, time.monotonic()


def kill_game(process: subprocess.Popen):
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    else:
        process.kill()
    process.wait()


def finish_game(game: GameSpec, run_dir: Path, outcome: dict) -> dict:
    record = {"game": game.name(), "map": game.map, "opponent": game.opponent.label(), **outcome}
    with open(run_dir / "results.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return record


def read_outcome(game: GameSpec, run_dir: Path, exit_code: int) -> dict:
    path = run_dir / "results" / f"{game.name()}.json"
    if not path.exists():
        return {"result": "Crash", "error": f"no result file (exit code {exit_code})"}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def run_batch(config: BatchConfig, out_dir: Path) -> Path:
    commit = git("rev-parse", "--short", "HEAD") or "nogit"
    dirty = bool(git("status", "--porcelain", "--untracked-files=no"))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir = out_dir / f"{stamp}-{commit}{'-dirty' if dirty else ''}"
    for sub in ("replays", "results", "logs"):
        (run_dir / sub).mkdir(parents=True)
    (run_dir / "meta.json").write_text(
        json.dumps({"commit": git("rev-parse", "HEAD"), "dirty": dirty, "config": asdict(config)}, indent=2),
        encoding="utf-8",
    )

    pending = expand_games(config)
    total = len(pending)
    running: dict[GameSpec, tuple[subprocess.Popen, float]] = {}
    done = 0
    last_launch = 0.0
    print(f"{total} games, {config.parallel} at a time -> {run_dir}", flush=True)

    try:
        while pending or running:
            if pending and len(running) < config.parallel and time.monotonic() - last_launch >= LAUNCH_GAP_SECONDS:
                game = pending.pop(0)
                running[game] = start_game(config, game, run_dir)
                last_launch = time.monotonic()

            for game, (process, started) in list(running.items()):
                if process.poll() is not None:
                    outcome = read_outcome(game, run_dir, process.returncode)
                elif time.monotonic() - started > config.wall_timeout:
                    kill_game(process)
                    outcome = {"result": "Timeout", "wall_seconds": config.wall_timeout}
                else:
                    continue
                del running[game]
                done += 1
                record = finish_game(game, run_dir, outcome)
                print(f"[{done}/{total}] {record['game']}: {record['result']}"
                      f" ({record.get('game_seconds', '?')} game s, {record.get('wall_seconds', '?')} wall s)",
                      flush=True)
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Interrupted, stopping running games...", flush=True)
        for process, _ in running.values():
            kill_game(process)
        raise

    return run_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--parallel", type=int, help="overrides the config")
    parser.add_argument("--games-per-matchup", type=int, help="overrides the config")
    parser.add_argument("--out", default="runs")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.parallel:
        config = replace(config, parallel=args.parallel)
    if args.games_per_matchup:
        config = replace(config, games_per_matchup=args.games_per_matchup)

    run_dir = run_batch(config, REPO_ROOT / args.out)
    print()
    print(summarize(run_dir))


if __name__ == "__main__":
    main()
