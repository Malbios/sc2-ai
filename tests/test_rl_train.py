import importlib.util
import unittest

import numpy as np

HAS_SB3 = importlib.util.find_spec("stable_baselines3") is not None


def _model(inputs: int, hidden: int = 16):
    import gymnasium as gym
    from stable_baselines3 import PPO

    class StandInEnv(gym.Env):
        observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(inputs,), dtype=np.float32)
        action_space = gym.spaces.Discrete(3)

    return PPO("MlpPolicy", StandInEnv(), device="cpu", policy_kwargs={"net_arch": [hidden, hidden]})


@unittest.skipUnless(HAS_SB3, "needs Stable-Baselines3")
class WidenInputsTest(unittest.TestCase):
    def test_extra_inputs_change_nothing_at_first(self):
        import torch
        from tools.rl.train import widen_inputs

        old, new = _model(3), _model(5)
        new.policy.load_state_dict(widen_inputs(old.policy.state_dict(), new.policy.state_dict()))

        rng = np.random.default_rng(0)
        shared = rng.normal(size=(50, 3)).astype(np.float32)
        extra = rng.normal(size=(50, 2)).astype(np.float32) * 10
        widened = np.concatenate([shared, extra], axis=1)

        old_actions, _ = old.predict(shared, deterministic=True)
        new_actions, _ = new.predict(widened, deterministic=True)
        np.testing.assert_array_equal(old_actions, new_actions)
        with torch.no_grad():
            old_values = old.policy.predict_values(torch.as_tensor(shared))
            new_values = new.policy.predict_values(torch.as_tensor(widened))
        np.testing.assert_allclose(old_values.numpy(), new_values.numpy(), rtol=1e-5, atol=1e-6)

    def test_other_network_size_is_refused(self):
        from tools.rl.train import widen_inputs

        with self.assertRaisesRegex(ValueError, "policy_net"):
            widen_inputs(_model(3, hidden=16).policy.state_dict(), _model(5, hidden=32).policy.state_dict())

    def test_fewer_inputs_are_refused(self):
        from tools.rl.train import widen_inputs

        with self.assertRaisesRegex(ValueError, "policy_net.0.weight"):
            widen_inputs(_model(5).policy.state_dict(), _model(3).policy.state_dict())


@unittest.skipUnless(HAS_SB3 and importlib.util.find_spec("sb3_contrib"), "needs Stable-Baselines3 and sb3-contrib")
class ModelClassTest(unittest.TestCase):
    def test_masked_tasks_train_with_maskable_ppo(self):
        from types import SimpleNamespace

        from sb3_contrib import MaskablePPO
        from stable_baselines3 import PPO

        from tools.rl.train import model_class

        def config(task):
            return SimpleNamespace(learner=SimpleNamespace(task=task))

        self.assertIs(model_class(config("tools.rl.examples.ravager_task:RavagerBileTask")), MaskablePPO)
        self.assertIs(model_class(config("tools.rl.examples.ravager_task:RavagerTask")), PPO)


class StandInGames:
    """Two group envs with 3 slots of 2 inputs, as a VecEnv would return them."""

    num_envs = 2

    def __init__(self, step_result):
        import gymnasium as gym

        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(3, 2), dtype=np.float32)
        self.action_space = gym.spaces.MultiDiscrete([4, 4, 4])
        self.step_result = step_result
        self.actions = None

    def reset(self):
        return np.arange(12, dtype=np.float32).reshape(2, 3, 2)

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        return self.step_result


@unittest.skipUnless(HAS_SB3, "needs Stable-Baselines3")
class SlotVecEnvTest(unittest.TestCase):
    def setUp(self):
        from tools.rl.train import SlotVecEnv

        observations = np.arange(12, dtype=np.float32).reshape(2, 3, 2)
        last = np.full((3, 2), 9.0, dtype=np.float32)
        infos = [
            {"slot_rewards": np.array([0.5, 0.5, 0.0])},
            {"slot_rewards": np.array([1.0, 0.0, 1.0]), "terminal_observation": last,
             "TimeLimit.truncated": True, "outcome": "timeout", "scenario": "s"},
        ]
        self.games = StandInGames((observations, np.array([0.5, 1.0]), np.array([False, True]), infos))
        self.venv = SlotVecEnv(self.games, slots=3)

    def test_every_slot_is_an_env(self):
        self.assertEqual((self.venv.num_envs, self.venv.observation_space.shape, self.venv.action_space.n), (6, (2,), 4))
        self.assertEqual(self.venv.reset().shape, (6, 2))
        self.venv.step_async(np.arange(6))
        np.testing.assert_array_equal(self.games.actions, [[0, 1, 2], [3, 4, 5]])

    def test_rewards_dones_and_infos_per_slot(self):
        observations, rewards, dones, infos = self.venv.step_wait()
        self.assertEqual(observations.shape, (6, 2))
        np.testing.assert_array_equal(rewards, [0.5, 0.5, 0.0, 1.0, 0.0, 1.0])
        np.testing.assert_array_equal(dones, [False, False, False, True, True, True])
        self.assertEqual(infos[0], {})
        for slot in (3, 4, 5):
            np.testing.assert_array_equal(infos[slot]["terminal_observation"], [9.0, 9.0])
            self.assertTrue(infos[slot]["TimeLimit.truncated"])
        self.assertEqual((infos[3]["outcome"], infos[3]["scenario"]), ("timeout", "s"))
        self.assertNotIn("outcome", infos[4])


if __name__ == "__main__":
    unittest.main()
