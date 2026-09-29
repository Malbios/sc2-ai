"""Evaluates several models of one config at the same time, each in its own process.

    python -m tools.rl.evaluate_many --config tools/rl/configs/ravager_bile.yaml --episodes 200 \\
        --parallel 4 models/ravager-bile/checkpoints/rl_model_1000000_steps.zip models/ravager-bile/final.zip

Each model's full output goes to <model>.eval.log next to it. When a model is done, its summary
table (see tools.rl.evaluate) is printed under its name, in the order the models finish.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

from tools.batch.run import LAUNCH_GAP_SECONDS, REPO_ROOT, kill_game
from tools.rl.config import ENEMY_MODES

SUMMARY_HEADER = "scenario"
CRASH_LINE = "SC2 game crashed"


def evaluate_command(config: str, model: str, episodes: int, enemy_mode: str | None,
                     stochastic: bool = False) -> list[str]:
    command = [sys.executable, "-m", "tools.rl.evaluate", "--config", config, "--model", model,
               "--episodes", str(episodes)]
    if enemy_mode:
        command += ["--enemy-mode", enemy_mode]
    if stochastic:
        command.append("--stochastic")
    return command


def summary_table(log: str) -> str:
    """The last summary table in an evaluate log: its header line up to the next empty line."""
    lines = log.splitlines()
    starts = [i for i, line in enumerate(lines) if line.startswith(SUMMARY_HEADER) and "fights" in line]
    if not starts:
        return ""
    table = []
    for line in lines[starts[-1]:]:
        if not line.strip():
            break
        table.append(line)
    return "\n".join(table)


def crash_count(log: str) -> int:
    return sum(CRASH_LINE in line for line in log.splitlines())


def log_path(model: str) -> Path:
    return Path(model).with_suffix(".eval.log")


def display_name(model: str) -> str:
    """The model's path inside the repo (runs often share file names like final.zip)."""
    path = Path(model)
    return path.relative_to(REPO_ROOT).as_posix() if path.is_relative_to(REPO_ROOT) else path.as_posix()


def report(model: str, exit_code: int) -> str:
    log_file = log_path(model)
    log = log_file.read_text(encoding="utf-8", errors="replace")
    name = display_name(model)
    if exit_code != 0:
        return f"== FAILED {name} (exit code {exit_code}), see {log_file}"
    lines = [f"== {name}"]
    crashes = crash_count(log)
    if crashes:
        lines.append(f"SC2 crashed {crashes} times; fights in progress then are not counted")
    lines.append(summary_table(log))
    return "\n".join(lines)


def start(command: list[str], model: str) -> subprocess.Popen:
    with open(log_path(model), "w", encoding="utf-8") as log:
        # A new session puts the evaluation and its SC2 processes in one process group, so
        # stopping it early can kill them all.
        return subprocess.Popen(command, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)


def evaluate_many(models: list[str], config: str, episodes: int, enemy_mode: str | None, parallel: int,
                  stochastic: bool = False):
    pending = list(models)
    running: dict[str, subprocess.Popen] = {}
    last_launch = 0.0
    print(f"{len(models)} models, {parallel} at a time", flush=True)
    try:
        while pending or running:
            if pending and len(running) < parallel and time.monotonic() - last_launch >= LAUNCH_GAP_SECONDS:
                model = pending.pop(0)
                running[model] = start(evaluate_command(config, model, episodes, enemy_mode, stochastic), model)
                last_launch = time.monotonic()

            for model, process in list(running.items()):
                if process.poll() is not None:
                    del running[model]
                    print(report(model, process.returncode), flush=True)
                    print(flush=True)
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Interrupted, stopping running evaluations...", flush=True)
        for process in running.values():
            kill_game(process)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--episodes", type=int, default=20, help="fights per scenario")
    parser.add_argument("--enemy-mode", choices=ENEMY_MODES, help="overrides the config's enemy mode")
    parser.add_argument("--stochastic", action="store_true", help="sample actions as in training (see tools.rl.evaluate)")
    parser.add_argument("--parallel", type=int, default=4, help="evaluations running at the same time")
    parser.add_argument("models", nargs="+", help="trained models (.zip)")
    args = parser.parse_args()
    if args.parallel < 1:
        parser.error("--parallel must be at least 1")

    models = [str(Path(model).resolve()) for model in args.models]
    evaluate_many(models, str(Path(args.config).resolve()), args.episodes, args.enemy_mode, args.parallel, args.stochastic)


if __name__ == "__main__":
    main()
