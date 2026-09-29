"""Scores a trained model on every scenario of a config, separately.

    python -m tools.rl.evaluate --config tools/rl/configs/stalker_vs_roaches.yaml --model models/stalker/final.zip
    python -m tools.rl.evaluate --config ... --random          # baseline: random actions
    python -m tools.rl.evaluate --config ... --baseline kite   # a task's hand-written policy
    python -m tools.rl.evaluate --config ... --model ... --enemy-mode builtin
    python -m tools.rl.evaluate --config ... --model ... --replay models/stalker/eval.SC2Replay
    python -m tools.rl.evaluate --config ... --model ... --compare-with kite
    python -m tools.rl.evaluate --config ... --model ... --stochastic

The model acts deterministically (always its most likely action) unless --stochastic makes it
sample its actions as in training; in group tasks, every unit
applies the model or baseline to its own observation. --replay saves every evaluated
fight, back to back, as one SC2 4.10 replay (open it in sc2-observer).
"""

import argparse
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np

from tools.rl.config import ENEMY_MODES, load_config
from tools.rl.env import SC2MicroEnv, make_env


def summarize(fights: list[dict]) -> str:
    """One row per scenario: fights, win/loss/tie shares, average damage dealt and taken (as a
    share of each side's starting life), average fight length, and the kiting share (how often
    units backed off while their weapon cooled down with an enemy close)."""
    by_scenario = defaultdict(list)
    for fight in fights:
        by_scenario[fight["scenario"]].append(fight)

    width = max([8] + [len(name) for name in by_scenario])
    lines = [f"{'scenario':<{width}}  fights   win  loss   tie  dealt  taken  seconds  kite"]
    for name, rows in sorted(by_scenario.items()):
        count = len(rows)
        share = lambda outcomes: sum(r["outcome"] in outcomes for r in rows) / count
        mean = lambda key: sum(r.get(key, 0.0) for r in rows) / count
        lines.append(
            f"{name:<{width}}  {count:>6}  {share({'win'}):>4.0%}  {share({'loss'}):>4.0%}"
            f"  {share({'tie', 'timeout'}):>4.0%}  {mean('damage_dealt'):>5.0%}  {mean('damage_taken'):>5.0%}"
            f"  {mean('game_seconds'):>7.1f}  {mean('kite_share'):>4.0%}"
        )
    return "\n".join(lines)


def compare(decisions: list[dict]) -> str:
    """One table per scenario, a row per situation: how many decisions fell into it, their share
    of the scenario's decisions, and how often the model picked the baseline's action, overall
    and within won and lost fights ('-' when there were none)."""
    by_scenario = defaultdict(lambda: defaultdict(list))
    for decision in decisions:
        by_scenario[decision["scenario"]][decision["situation"]].append(decision)

    agreement = lambda rows: f"{sum(r['agree'] for r in rows) / len(rows):.0%}" if rows else "-"
    tables = []
    for scenario, situations in sorted(by_scenario.items()):
        total = sum(len(rows) for rows in situations.values())
        width = max([9] + [len(name) for name in situations])
        lines = [
            scenario,
            f"  {'situation':<{width}}  decisions  share  agree  agree in wins  agree in losses",
        ]
        for name, rows in sorted(situations.items()):
            wins = [r for r in rows if r["outcome"] == "win"]
            losses = [r for r in rows if r["outcome"] == "loss"]
            lines.append(
                f"  {name:<{width}}  {len(rows):>9}  {len(rows) / total:>5.0%}  {agreement(rows):>5}"
                f"  {agreement(wins):>13}  {agreement(losses):>15}"
            )
        tables.append("\n".join(lines))
    return "\n\n".join(tables)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model", help="trained model (.zip)")
    source.add_argument("--random", action="store_true", help="random actions, as a baseline")
    source.add_argument("--baseline", help="one of the task's hand-written policies, e.g. kite")
    parser.add_argument("--episodes", type=int, default=20, help="fights per scenario")
    parser.add_argument("--enemy-mode", choices=ENEMY_MODES, help="overrides the config's enemy mode")
    parser.add_argument("--replay", help="save all evaluated fights as one .SC2Replay at this path")
    parser.add_argument("--stochastic", action="store_true",
                        help="with --model: sample actions as in training instead of always the most likely one")
    parser.add_argument("--compare-with", help="with --model: also report where the model's actions differ from this baseline's")
    args = parser.parse_args()
    if args.compare_with and not args.model:
        parser.error("--compare-with needs --model")
    if args.stochastic and not args.model:
        parser.error("--stochastic needs --model")

    config = load_config(args.config)
    if args.enemy_mode:
        config = replace(config, enemy=replace(config.enemy, mode=args.enemy_mode))

    env = make_env(config)
    group = bool(env.task.group_slots)
    if group and args.compare_with:
        parser.error("--compare-with doesn't support group tasks yet")
    if args.model:
        from tools.rl.train import model_class

        model = model_class(config).load(args.model, device="cpu")
        if env.task.uses_action_masks:
            choose_action = lambda observation: model.predict(
                observation, deterministic=not args.stochastic, action_masks=env.action_masks())[0]
        else:
            choose_action = lambda observation: model.predict(observation, deterministic=not args.stochastic)[0]
    elif args.baseline:
        policy = _baseline(env, args.baseline, parser)
        choose_action = (lambda observations: np.array([policy(row) for row in observations])) if group else policy
    else:
        choose_action = lambda observation: env.action_space.sample()
    rule = _baseline(env, args.compare_with, parser) if args.compare_with else None

    fights, decisions = [], []
    try:
        for scenario in config.scenarios:
            for episode in range(args.episodes):
                obs, _ = env.reset(options={"scenario": scenario.name})
                fight_decisions = []
                while True:
                    action = choose_action(obs)
                    if rule:
                        fight_decisions.append({
                            "scenario": scenario.name,
                            "situation": env.task.situation(obs),
                            "agree": int(action) == int(rule(obs)),
                        })
                    obs, _, terminated, truncated, info = env.step(action)
                    if terminated or truncated:
                        break
                if "scenario" in info:
                    fights.append(info)
                    decisions += [{**d, "outcome": info["outcome"]} for d in fight_decisions]
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
    if rule:
        print()
        print(f"Model vs baseline '{args.compare_with}':")
        print(compare(decisions))


def _baseline(env: SC2MicroEnv, name: str, parser: argparse.ArgumentParser):
    baselines = env.task.baseline_policies()
    if name not in baselines:
        parser.error(f"the task has no baseline '{name}'. Available: {', '.join(baselines) or 'none'}")
    return baselines[name]


if __name__ == "__main__":
    main()
