"""Evolution strategies: tune a hand rule's decisions by searching over weights.

No per-step rewards: each candidate plays whole fights, and the search moves toward the
candidates that did better. Each policy family is a hand rule with learned weights, starting
exactly at the rule:
- bile (roach and ravager fight): every seen enemy within bile range gets a score (weights times
  features), and a ravager with bile ready biles the best one when its score is above 0.
  Everything else (shooting the most dangerous in range, bile where the target is now) stays
  the clump rule's. THETA0 is the clump rule.
- kite (mutalisk fight), kite_roach, kite_hydra: one score decides whether to back off, another
  whether to attack the weakest in range instead of the closest. KITE_THETA0 is the kite rule at
  the family's distance (kite_4.5 for mutalisks).
- rotate_roach: the same decisions, starting at life_kite_5 instead (LIFE_KITE_THETA0).

    python -m tools.rl.es search --config tools/rl/configs/roach_ravager_hard.yaml --out models/es-bile
    python -m tools.rl.es evaluate --config tools/rl/configs/roach_ravager_hard.yaml --theta models/es-bile/theta.json --fights 200
    python -m tools.rl.es search --family kite --config tools/rl/configs/mutalisk_group.yaml --out models/es-kite
"""

import argparse
import json
import multiprocessing
import multiprocessing.util
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np

from tools.rl.config import load_config
from tools.rl.evaluate import summarize
from tools.rl.examples.free_kite_task import DISTANCE_SCALE, MOVEMENT_SCALE
from tools.rl.examples.group_fight_task import (
    ATTACK_CLOSEST,
    ATTACK_DANGEROUS,
    ATTACK_WEAKEST,
    COOLDOWN,
    ENEMY_INPUTS,
    FIRST_ALLY,
    FIRST_ENEMY,
    LIFE,
    THREAT,
    allies_center,
    closest_enemy_distance,
    step_away,
)
from tools.rl.examples.ravager_task import BILE_RANGE
from tools.rl.examples.roach_ravager_group_task import BILE_READY, bile_seen_action, clump_sizes

FEATURES = ("bias", "closest", "clumped", "clump size", "distance", "life", "threat", "speed")
THETA0 = np.array([-1.5, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # the clump rule

SIGMA = 0.3
LEARNING_RATE = 0.1
DIRECTIONS = 4  # each scored mirrored: 8 candidates per generation
FIGHTS_PER_TASK = 5
WORKERS = 6


def enemy_features(observation: np.ndarray) -> np.ndarray:
    """One row per seen enemy (closest first) with the FEATURES."""
    clumps = clump_sizes(observation)
    rows = []
    for index, clump in enumerate(clumps):
        start = FIRST_ENEMY + index * ENEMY_INPUTS
        speed = float(np.hypot(*observation[start + 5:start + 7])) * MOVEMENT_SCALE
        rows.append([1.0, 1.0 if index == 0 else 0.0, 1.0 if clump else 0.0, clump / 3,
                     observation[start + 3], observation[start + 4], observation[start + THREAT], speed])
    return np.array(rows, dtype=np.float32).reshape(len(rows), len(FEATURES))


def bile_policy(theta: np.ndarray) -> Callable[[np.ndarray], int]:
    """The policy for one unit's observation: bile the best-scoring seen enemy within bile range
    when bile is ready and that score is above 0, else shoot the most dangerous in range."""
    def policy(observation: np.ndarray) -> int:
        if not observation[BILE_READY]:
            return ATTACK_DANGEROUS
        features = enemy_features(observation)
        distances = features[:, FEATURES.index("distance")] * DISTANCE_SCALE
        scores = np.where(distances <= BILE_RANGE, features @ theta, -np.inf)
        if not len(scores) or scores.max() <= 0:
            return ATTACK_DANGEROUS
        return bile_seen_action(int(scores.argmax()))
    return policy


KITE_FEATURES = ("bias", "cooling", "enemy within D", "cooldown left", "closest distance", "life",
                 "enemies within D + 1.5", "allies center distance")
MUTALISK_KITE_DISTANCE = 4.5
ROACH_KITE_DISTANCE = 5.0  # center to center: a roach's range 4 plus both radii
HYDRALISK_KITE_DISTANCE = 3.0
CROWD_MARGIN = 1.5
# Back off: bias -1.5, cooling +1, within D +1 (the kite rule at D); attack the weakest: never.
KITE_THETA0 = np.array([-1.5, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
# Back off: bias -1.5, within D +1.5, cooldown left +1, life -1 (life_kite_rule at D: with an
# enemy within D, cooldown left above life; without one, never); attack the weakest: never.
LIFE_KITE_THETA0 = np.array([-1.5, 0.0, 1.5, 1.0, 0.0, -1.0, 0.0, 0.0,
                             -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


def kite_features(observation: np.ndarray, distance: float) -> np.ndarray:
    """The KITE_FEATURES of one unit's observation (with at least one enemy in sight), D being
    `distance`."""
    closest = closest_enemy_distance(observation)
    distances = [observation[start + 3] * DISTANCE_SCALE
                 for start in range(FIRST_ENEMY, FIRST_ALLY, ENEMY_INPUTS) if observation[start]]
    center = allies_center(observation)
    return np.array([1.0, 1.0 if observation[COOLDOWN] > 0 else 0.0, 1.0 if closest <= distance else 0.0,
                     observation[COOLDOWN], closest / DISTANCE_SCALE, observation[LIFE],
                     sum(d <= distance + CROWD_MARGIN for d in distances) / 4,
                     center.length / DISTANCE_SCALE if center is not None else 0.0], dtype=np.float32)


def kite_policy(theta: np.ndarray, distance: float) -> Callable[[np.ndarray], int]:
    """Back off (step away from the enemies) when the first score is above 0; otherwise attack,
    the weakest in range when the second score is above 0, else the closest."""
    back_off, weakest = theta[:len(KITE_FEATURES)], theta[len(KITE_FEATURES):]

    def policy(observation: np.ndarray) -> int:
        if not observation[FIRST_ENEMY]:
            return ATTACK_CLOSEST
        features = kite_features(observation, distance)
        if features @ back_off > 0:
            return step_away(observation)
        return ATTACK_WEAKEST if features @ weakest > 0 else ATTACK_CLOSEST
    return policy


@dataclass(frozen=True)
class Family:
    theta0: np.ndarray
    policy: Callable[[np.ndarray], Callable[[np.ndarray], int]]
    fights_per_scenario: int  # per candidate and generation


FAMILIES = {
    "bile": Family(THETA0, bile_policy, 15),
    # more weights than bile: more fights against the noise, 40 per candidate in all
    "kite": Family(KITE_THETA0, partial(kite_policy, distance=MUTALISK_KITE_DISTANCE), 20),
    "kite_roach": Family(KITE_THETA0, partial(kite_policy, distance=ROACH_KITE_DISTANCE), 20),
    "kite_hydra": Family(KITE_THETA0, partial(kite_policy, distance=HYDRALISK_KITE_DISTANCE), 40),
    "rotate_roach": Family(LIFE_KITE_THETA0, partial(kite_policy, distance=ROACH_KITE_DISTANCE), 20),
}


def centered_ranks(values: np.ndarray) -> np.ndarray:
    """The values' ranks, scaled to -0.5 (worst) to 0.5 (best)."""
    ranks = np.empty(len(values))
    ranks[np.argsort(values)] = np.arange(len(values))
    return ranks / (len(values) - 1) - 0.5


def candidates_for(theta: np.ndarray, directions: np.ndarray, sigma: float = SIGMA) -> list[np.ndarray]:
    """Mirrored samples: theta + sigma * d for each direction d, then theta - sigma * d."""
    return [theta + sigma * d for d in directions] + [theta - sigma * d for d in directions]


def es_update(theta: np.ndarray, directions: np.ndarray, scores: np.ndarray, sigma: float = SIGMA,
              learning_rate: float = LEARNING_RATE) -> np.ndarray:
    """Moves theta toward the better candidates (scored in candidates_for's order), by rank."""
    ranks = centered_ranks(scores)
    count = len(directions)
    gradient = ((ranks[:count] - ranks[count:]) @ directions) / (2 * count * sigma)
    return theta + learning_rate * gradient


def fight_fitness(fight: dict) -> float:
    """1 for a win, plus the share of enemy life dealt, minus the share of own life lost."""
    return (fight["outcome"] == "win") + fight["damage_dealt"] - fight["damage_taken"]


# ----- workers: each runs its own SC2 game -----

_env = None


def _start_worker(config_path: str, counter):
    from tools.rl.env import make_env

    global _env
    with counter.get_lock():
        rank = counter.value
        counter.value += 1
    _env = make_env(load_config(config_path), rank)
    # Runs when the worker exits after pool.close(), so its SC2 processes don't outlive it.
    multiprocessing.util.Finalize(None, _env.close, exitpriority=10)


def _play(job: tuple[int, str, list, str, int]) -> tuple[int, list[dict]]:
    """Plays `count` fights of `scenario` with the family's policy for `theta`; returns them with
    the job's candidate index."""
    candidate, family, theta, scenario, count = job
    policy = FAMILIES[family].policy(np.array(theta))
    fights = []
    for _ in range(count):
        observations, _ = _env.reset(options={"scenario": scenario})
        while True:
            actions = np.array([policy(row) for row in observations])
            observations, _, terminated, truncated, info = _env.step(actions)
            if terminated or truncated:
                break
        if "scenario" in info:
            fights.append(info)
    return candidate, fights


def _pool(config_path: str, workers: int):
    context = multiprocessing.get_context("spawn")
    return context.Pool(workers, initializer=_start_worker, initargs=(config_path, context.Value("i", 0)))


def _close(pool) -> None:
    """Lets the workers exit on their own (closing their games); terminate() would orphan SC2."""
    pool.close()
    pool.join()


def play_candidates(pool, family: str, thetas: list[np.ndarray], scenarios: list[str],
                    fights_per_scenario: int) -> list[list[dict]]:
    """Every candidate's fights, spread over the pool in jobs of FIGHTS_PER_TASK."""
    jobs = [(index, family, theta.tolist(), scenario, FIGHTS_PER_TASK)
            for index, theta in enumerate(thetas) for scenario in scenarios
            for _ in range(fights_per_scenario // FIGHTS_PER_TASK)]
    results = [[] for _ in thetas]
    for candidate, fights in pool.imap_unordered(_play, jobs):
        results[candidate] += fights
    return results


def win_rates(fights: list[dict]) -> dict[str, float]:
    by_scenario = {}
    for fight in fights:
        by_scenario.setdefault(fight["scenario"], []).append(fight["outcome"] == "win")
    return {name: round(sum(wins) / len(wins), 3) for name, wins in sorted(by_scenario.items())}


def search(config_path: str, family: str, out: Path, generations: int, workers: int, seed: int):
    """Resumes from out/theta.json if it exists. Each generation's log line has the theta it
    scored; out/best.json gets the one with the best mean fitness so far."""
    out.mkdir(parents=True, exist_ok=True)
    state_file, best_file = out / "theta.json", out / "best.json"
    state = json.loads(state_file.read_text()) if state_file.exists() else {"generation": 0, "theta": FAMILIES[family].theta0.tolist()}
    best = json.loads(best_file.read_text()) if best_file.exists() else {"mean_fitness": -np.inf}
    theta = np.array(state["theta"])
    rng = np.random.default_rng(seed + state["generation"])
    scenarios = [scenario.name for scenario in load_config(config_path).scenarios]
    pool = _pool(config_path, workers)
    try:
        with open(out / "log.jsonl", "a") as log:
            for generation in range(state["generation"], generations):
                directions = rng.standard_normal((DIRECTIONS, len(theta)))
                played = play_candidates(pool, family, candidates_for(theta, directions), scenarios,
                                         FAMILIES[family].fights_per_scenario)
                scores = np.array([np.mean([fight_fitness(f) for f in fights]) if fights else -1.0 for fights in played])
                entry = {"generation": generation + 1, "theta": theta.round(4).tolist(),
                         "mean_fitness": round(float(scores.mean()), 3), "best_fitness": round(float(scores.max()), 3),
                         "win_rates": win_rates([fight for fights in played for fight in fights])}
                if entry["mean_fitness"] > best["mean_fitness"]:
                    best = entry
                    best_file.write_text(json.dumps(best))
                theta = es_update(theta, directions, scores)
                log.write(json.dumps(entry) + "\n")
                log.flush()
                state_file.write_text(json.dumps({"generation": generation + 1, "theta": theta.tolist()}))
                print(json.dumps(entry), flush=True)
    finally:
        _close(pool)


def evaluate(config_path: str, family: str, theta: np.ndarray, fights: int, workers: int):
    scenarios = [scenario.name for scenario in load_config(config_path).scenarios]
    pool = _pool(config_path, workers)
    try:
        played = play_candidates(pool, family, [theta], scenarios, fights)[0]
    finally:
        _close(pool)
    print(summarize(played))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("search", "evaluate"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--family", choices=sorted(FAMILIES), default="bile", help="which rule's decisions to tune")
    parser.add_argument("--out", help="search: output folder (theta.json, log.jsonl); resumes if theta.json exists")
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--theta", help="evaluate: theta.json or best.json of a search, or 'rule' for the family's starting weights")
    parser.add_argument("--fights", type=int, default=200, help="evaluate: fights per scenario (a multiple of 5)")
    parser.add_argument("--workers", type=int, default=WORKERS)
    args = parser.parse_args()
    if args.command == "search":
        if not args.out:
            parser.error("search needs --out")
        search(args.config, args.family, Path(args.out), args.generations, args.workers, args.seed)
    else:
        if not args.theta:
            parser.error("evaluate needs --theta")
        theta = (FAMILIES[args.family].theta0 if args.theta == "rule"
                 else np.array(json.loads(Path(args.theta).read_text())["theta"]))
        evaluate(args.config, args.family, theta, args.fights, args.workers)


if __name__ == "__main__":
    main()
