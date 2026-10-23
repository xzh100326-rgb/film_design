"""Expose native GAE lambda without changing PPO, rewards or default behavior."""
from copy import deepcopy
from pathlib import Path
import unittest

from fixtures import base_config
from film_rl.config import validate
from film_rl.hppo import build_configs


class GAEConfigurationTests(unittest.TestCase):
    def test_historical_default_and_explicit_native_parameter(self):
        cfg = base_config(Path(__file__).parent)
        self.assertEqual(build_configs(cfg)[0].policy.collect.gae_lambda, .95)
        for value in (0., .99, 1.):
            trial = deepcopy(cfg)
            trial["training"]["gae_lambda"] = value
            validate(trial)
            main, _ = build_configs(trial)
            self.assertEqual(main.policy.collect.gae_lambda, value)
            self.assertEqual(main.policy.collect.discount_factor, 1.)

    def test_invalid_gae_lambda_is_rejected(self):
        for value in (True, -.01, 1.01, float("nan"), "1"):
            cfg = base_config(Path(__file__).parent)
            cfg["training"]["gae_lambda"] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "gae_lambda"):
                validate(cfg)


if __name__ == "__main__":
    unittest.main()
