"""Scores a trained model on every scenario of a config, separately.

    python -m tools.rl.evaluate --config tools/rl/configs/stalker_vs_roaches.yaml --model models/stalker/final.zip
    python -m tools.rl.evaluate --config ... --random          # baseline: random actions
    python -m tools.rl.evaluate --config ... --model ... --enemy-mode builtin
    python -m tools.rl.evaluate --config ... --model ... --replay models/stalker/eval.SC2Replay

The model acts deterministically (always its most likely action). --replay saves every evaluated
fight, back to back, as one SC2 4.10 replay (open it in sc2-observer).
"""

import argparse
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

from tools.rl.config import ENEMY_MODES, load_config
from tools.rl.env import SC2MicroEnv


def summarize(fights: list[dict]) -> str:
    """One row per scenario: fights, win/loss/tie shares, average damage dealt and taken (as a
    share of each side's starting life) and average fight length."""
    by_scenario = defaultdict(list)
    for fight in fights:
        by_scenario[fight["scenario"]].append(fight)

    width = max([8] + [len(name) for name in by_scenario])
    lines = [f"{'scenario':<{width}}  fights   win  loss   tie  dealt  taken  seconds"]
    for name, rows in sorted(by_scenario.items()):
        count = len(rows)
        share = lambda outcomes: sum(r["outcome"] in outcomes for r in rows) / count
        mean = lambda key: sum(r[key] for r in rows) / count
        lines.append(
            f"{name:<{width}}  {count:>6}  {share({'win'}):>4.0%}  {share({'loss'}):>4.0%}"
            f"  {share({'tie', 'timeout'}):>4.0%}  {mean('damage_dealt'):>5.0%}  {mean('damage_taken'):>5.0%}"
            f"  {mean('game_seconds'):>7.1f}"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model", help="trained model (.zip)")
    source.add_argument("--random", action="store_true", help="random actions, as a baseline")
    parser.add_argument("--episodes", type=int, default=20, help="fights per scenario")
    parser.add_argument("--enemy-mode", choices=ENEMY_MODES, help="overrides the config's enemy mode")
    parser.add_argument("--replay", help="save all evaluated fights as one .SC2Replay at this path")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.enemy_mode:
        config = replace(config, enemy=replace(config.enemy, mode=args.enemy_mode))

    model = None
    if args.model:
        from stable_baselines3 import PPO

        model = PPO.load(args.model, device="cpu")

    env = SC2MicroEnv(config)
    fights = []
    try:
        for scenario in config.scenarios:
            for episode in range(args.episodes):
                obs, _ = env.reset(options={"scenario": scenario.name})
                while True:
                    action = model.predict(obs, deterministic=True)[0] if model else env.action_space.sample()
                    obs, _, terminated, truncated, info = env.step(action)
                    if terminated or truncated:
                        break
                if "scenario" in info:
                    fights.append(info)
                print(f"{scenario.name} {episode + 1}/{args.episodes}: {info.get('outcome')}", flush=True)
        if args.replay:
            replay = Path(args.replay).resolve()
            replay.parent.mkdir(parents=True, exist_ok=True)
            env.driver.save_replay(str(replay))
            print(f"Replay saved to {replay}", flush=True)
    finally:
        env.close()

    print()
    print(summarize(fights))


if __name__ == "__main__":
    main()
