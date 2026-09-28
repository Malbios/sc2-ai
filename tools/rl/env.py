"""SC2MicroEnv: a Gymnasium environment for one learner unit type fighting through scenarios."""

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
