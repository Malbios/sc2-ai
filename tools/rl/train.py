"""Trains a micro model with PPO (Stable-Baselines3) on several SC2 environments in parallel.

    python -m tools.rl.train --config tools/rl/configs/stalker_vs_roaches.yaml --timesteps 200000 --out models/stalker
    python -m tools.rl.train --config ... --timesteps 100000 --out models/stalker2 --resume models/stalker/final.zip
    python -m tools.rl.train --config ... --timesteps 100000 --out models/wider --warm-start models/stalker/final.zip

With `self_play` in the config, both sides learn in turns instead (see train_self_play), and
--timesteps is ignored in favor of rounds x steps_per_round.

Output folder: final.zip (the model), checkpoints/, logs/ (progress.csv) and a copy of the config.
"""

import argparse
import shutil
from collections import deque
from functools import partial
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.logger import configure
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, VecMonitor

from tools.rl.config import TrainingConfig, load_class, load_config, self_play_side
from tools.rl.env import SC2MicroEnv, make_env

REPO_ROOT = Path(__file__).resolve().parents[2]


def model_class(config: TrainingConfig) -> type[PPO]:
    """PPO, or sb3-contrib's MaskablePPO for tasks that mask actions (see MicroTask.action_mask)."""
    if load_class(config.learner.task).uses_action_masks:
        from sb3_contrib import MaskablePPO

        return MaskablePPO
    return PPO


def new_model(config: TrainingConfig, env) -> PPO:
    return model_class(config)("MlpPolicy", env, device="cpu", verbose=1, **config.ppo.as_kwargs())


def load_model(path: str | Path, config: TrainingConfig, env) -> PPO:
    """A saved model that continues with the config's PPO settings. The network shape is the one
    exception: it is part of the saved model and can't change."""
    settings = config.ppo.as_kwargs()
    settings.pop("policy_kwargs")
    return model_class(config).load(path, env=env, device="cpu", **settings)


INPUT_LAYERS = ("mlp_extractor.policy_net.0.weight", "mlp_extractor.value_net.0.weight")


def warm_start_model(path: str | Path, config: TrainingConfig, env) -> PPO:
    """A new model that starts with a saved model's weights, for a task whose inputs are the saved
    model's inputs plus more at the end. The extra inputs start with zero weight, so at first the
    new model plays exactly like the saved one."""
    model = new_model(config, env)
    saved = PPO.load(path, device="cpu")
    model.policy.load_state_dict(widen_inputs(saved.policy.state_dict(), model.policy.state_dict()))
    return model


def widen_inputs(old: dict, new: dict) -> dict:
    """`old`'s weights in `new`'s shapes: the input layers get zero columns for the extra inputs,
    everything else must already match."""
    widened = {}
    for name, target in new.items():
        source = old[name]
        if source.shape == target.shape:
            widened[name] = source.clone()
        elif name in INPUT_LAYERS and source.shape[0] == target.shape[0] and source.shape[1] < target.shape[1]:
            widened[name] = target.new_zeros(target.shape)
            widened[name][:, :source.shape[1]] = source
        else:
            raise ValueError(f"{name}: can't turn shape {tuple(source.shape)} into {tuple(target.shape)}")
    return widened


class FightStatsCallback(BaseCallback):
    """Adds win rate and damage traded over the last 100 fights, overall and per scenario, to the
    training log next to PPO's own numbers."""

    def __init__(self, window: int = 100):
        super().__init__()
        self.window = window
        self.fights: dict[str, deque] = {}

    def _on_step(self) -> bool:
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            if done and "outcome" in info and "scenario" in info:
                fight = (info["outcome"] == "win", info["damage_dealt"] - info["damage_taken"])
                for key in ("all", info["scenario"]):
                    self.fights.setdefault(key, deque(maxlen=self.window)).append(fight)
        for key, fights in self.fights.items():
            self.logger.record(f"fights/{key}/win_rate", float(np.mean([won for won, _ in fights])))
            self.logger.record(f"fights/{key}/damage_traded", float(np.mean([traded for _, traded in fights])))
        return True


class SlotVecEnv(VecEnv):
    """Presents each slot of each group env (SC2GroupEnv) to SB3 as an env of its own, so every
    unit's decisions form their own stream with the team reward (independent PPO with one shared
    network). A fight ends for all its slots at once. The fight's info (outcome, damage, ...)
    goes to slot 0 only, so every fight is counted once."""

    def __init__(self, games: VecEnv, slots: int):
        self.games, self.slots = games, slots
        per_slot = spaces.Box(-np.inf, np.inf, shape=games.observation_space.shape[1:], dtype=np.float32)
        super().__init__(games.num_envs * slots, per_slot, spaces.Discrete(int(games.action_space.nvec[0])))

    def reset(self) -> np.ndarray:
        return self._flat(self.games.reset())

    def step_async(self, actions: np.ndarray) -> None:
        self.games.step_async(np.asarray(actions).reshape(self.games.num_envs, self.slots))

    def step_wait(self):
        observations, _, dones, game_infos = self.games.step_wait()
        rewards = np.stack([info["slot_rewards"] for info in game_infos]).astype(np.float32)
        infos = []
        for info, done in zip(game_infos, dones):
            for slot in range(self.slots):
                slot_info = {}
                if done:
                    slot_info["terminal_observation"] = info["terminal_observation"][slot]
                    slot_info["TimeLimit.truncated"] = info.get("TimeLimit.truncated", False)
                    if slot == 0:
                        slot_info.update({key: value for key, value in info.items()
                                          if key not in ("slot_rewards", "terminal_observation")})
                infos.append(slot_info)
        return self._flat(observations), rewards.reshape(-1), np.repeat(dones, self.slots), infos

    def close(self) -> None:
        self.games.close()

    def get_attr(self, attr_name, indices=None):
        return [self.games.get_attr(attr_name, [index // self.slots])[0] for index in self._indices(indices)]

    def set_attr(self, attr_name, value, indices=None) -> None:
        self.games.set_attr(attr_name, value, sorted({index // self.slots for index in self._indices(indices)}))

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        return [self.games.env_method(method_name, *method_args, indices=[index // self.slots], **method_kwargs)[0]
                for index in self._indices(indices)]

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [self.games.env_is_wrapped(wrapper_class, [index // self.slots])[0] for index in self._indices(indices)]

    def _flat(self, observations: np.ndarray) -> np.ndarray:
        return observations.reshape((-1,) + observations.shape[2:])

    def _indices(self, indices) -> list[int]:
        if indices is None:
            return list(range(self.num_envs))
        return [indices] if isinstance(indices, int) else list(indices)


def make_vec_env(config: TrainingConfig, n_envs: int):
    factories = [partial(make_env, config, rank) for rank in range(n_envs)]
    # "spawn" gives every env a fresh process: python-sc2 kills all SC2 processes of a Python
    # process when one game ends, so environments must not share one.
    venv = SubprocVecEnv(factories, start_method="spawn") if n_envs > 1 else DummyVecEnv(factories)
    slots = load_class(config.learner.task).group_slots
    return VecMonitor(SlotVecEnv(venv, slots) if slots else venv)


def train(config: TrainingConfig, n_envs: int, timesteps: int, out: Path, resume: str | None,
          warm_start: str | None = None) -> Path:
    venv = make_vec_env(config, n_envs)
    try:
        if resume:
            model = load_model(resume, config, venv)
        elif warm_start:
            model = warm_start_model(warm_start, config, venv)
        else:
            model = new_model(config, venv)
        model.set_logger(configure(str(out / "logs"), ["stdout", "csv"]))
        checkpoints = CheckpointCallback(save_freq=max(50_000 // venv.num_envs, 1), save_path=str(out / "checkpoints"))
        model.learn(timesteps, callback=CallbackList([checkpoints, FightStatsCallback()]),
                    reset_num_timesteps=not resume)
        model.save(out / "final.zip")
    finally:
        venv.close()
    return out / "final.zip"


def _initial_model(config: TrainingConfig, path: Path) -> Path:
    """An untrained model with the right spaces, so the other side has an opponent to start with.
    Building the env doesn't start SC2."""
    env = DummyVecEnv([partial(SC2MicroEnv, config)])
    try:
        new_model(config, env).save(path)
    finally:
        env.close()
    return path


def train_self_play(config: TrainingConfig, n_envs: int, out: Path) -> dict[str, Path]:
    """Both sides learn in turns: round 0 trains side a against a frozen side b, round 1 trains
    b against the frozen result of round 0, and so on. Each side keeps improving its own model."""
    rounds, steps = config.self_play.rounds, config.self_play.steps_per_round
    latest: dict[str, Path] = {}
    for side in ("a", "b"):
        (out / side).mkdir(parents=True, exist_ok=True)
        side_config = self_play_side(config, side, other_model="unused")
        latest[side] = _initial_model(side_config, out / side / "initial.zip")

    for round_number in range(rounds):
        side, other = ("a", "b") if round_number % 2 == 0 else ("b", "a")
        print(f"=== self-play round {round_number + 1}/{rounds}: training side {side} against frozen {latest[other].name}")
        side_config = self_play_side(config, side, other_model=str(latest[other]))
        venv = make_vec_env(side_config, n_envs)
        try:
            model = PPO.load(latest[side], env=venv, device="cpu")
            model.set_logger(configure(str(out / side / "logs" / f"round{round_number}"), ["stdout", "csv"]))
            model.learn(steps, callback=FightStatsCallback(), reset_num_timesteps=False)
            latest[side] = out / side / f"round{round_number}.zip"
            model.save(latest[side])
        finally:
            venv.close()

    for side in ("a", "b"):
        shutil.copy(latest[side], out / side / "final.zip")
    return {side: out / side / "final.zip" for side in ("a", "b")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True, help="output folder, e.g. models/stalker")
    parser.add_argument("--timesteps", type=int, default=200_000, help="decisions to train on (not self-play)")
    parser.add_argument("--n-envs", type=int, default=4, help="parallel environments (4 fill 8 cores)")
    start = parser.add_mutually_exclusive_group()
    start.add_argument("--resume", help="continue training this model (.zip) instead of starting fresh")
    start.add_argument("--warm-start", help="start from this model's weights (.zip), trained on a task "
                                            "with the same inputs minus some at the end")
    args = parser.parse_args()

    config = load_config(args.config)
    n_envs = args.n_envs
    out = REPO_ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, out / "config.yaml")

    if config.self_play:
        results = train_self_play(config, n_envs, out)
        print(f"Done: side a -> {results['a']}, side b -> {results['b']}")
    else:
        print(f"Done: {train(config, n_envs, args.timesteps, out, args.resume, args.warm_start)}")


if __name__ == "__main__":
    main()
