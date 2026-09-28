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


if __name__ == "__main__":
    unittest.main()
