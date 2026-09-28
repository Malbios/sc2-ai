"""SC2MicroEnv: a Gymnasium environment for one learner unit type fighting through scenarios.
SC2GroupEnv: the same for group tasks, where all units decide together."""

import random

import gymnasium as gym
import numpy as np

from tools.rl.config import TrainingConfig, load_class
from tools.rl.driver import GameDriver
from tools.rl.scenarios import ScenarioSampler

# Seconds between SC2 launches of parallel environments (simultaneous launches fail on Linux).
LAUNCH_GAP_SECONDS = 5.0


class SC2MicroEnv(gym.Env):
    """Each step is one learner unit's decision. SC2 starts on the first reset, not on creation,
    so building the env (e.g. to read its spaces) is cheap."""

    metadata = {"render_modes": []}

    def __init__(self, config: TrainingConfig, rank: int = 0):
        super().__init__()
        self.task = load_class(config.learner.task)()
        self.observation_space = self.task.observation_space
        self.action_space = self.task.action_space
        self.sampler = ScenarioSampler(list(config.scenarios), random.Random())
        self.driver = GameDriver(config, self.task, random.Random(), launch_delay=rank * LAUNCH_GAP_SECONDS)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """options={"scenario": name} picks a scenario instead of sampling one."""
        super().reset(seed=seed)
        if seed is not None:
            self.sampler.rng.seed(seed)
            self.driver.rng.seed(seed + 1)
        name = (options or {}).get("scenario")
        scenario = self.sampler.by_name(name) if name else self.sampler.sample()
        return self.driver.reset(scenario), {"scenario": scenario.name}

    def step(self, action):
        return self.driver.step(int(action))

    def action_masks(self) -> np.ndarray:
        """The actions the pending decision may pick; MaskablePPO asks for this before each one."""
        return self.driver.action_mask

    def close(self):
        self.driver.close()


class SC2GroupEnv(SC2MicroEnv):
    """Each step is one decision for every unit of the fight: observations are stacked by slot
    (see MicroTask.group_slots), and the action has one entry per slot. info["slot_rewards"] has
    each slot's own reward (see MicroTask.share_team_reward), so each slot can be trained as a
    stream of its own (see train.SlotVecEnv); the reward is their sum."""

    def __init__(self, config: TrainingConfig, rank: int = 0):
        super().__init__(config, rank)
        slots = self.task.group_slots
        if self.task.uses_action_masks:
            raise ValueError("group tasks can't mask actions yet")
        for scenario in config.scenarios:
            if sum(scenario.learner.values()) > slots:
                raise ValueError(f"scenario '{scenario.name}' has more learner units than the task's {slots} slots")
        shape = (slots,) + self.task.observation_space.shape
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=shape, dtype=np.float32)
        self.action_space = gym.spaces.MultiDiscrete([self.task.action_space.n] * slots)

    def step(self, action):
        return self.driver.step(np.asarray(action))


def make_env(config: TrainingConfig, rank: int = 0) -> SC2MicroEnv:
    """The env the config's task needs: SC2GroupEnv for group tasks, else SC2MicroEnv."""
    group = load_class(config.learner.task).group_slots
    return (SC2GroupEnv if group else SC2MicroEnv)(config, rank)
