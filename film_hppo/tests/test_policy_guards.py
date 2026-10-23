"""Safety checks for hybrid sample shapes and policy divergence; no updates."""

import unittest

import torch

from film_rl.policy import policy_drift, scalar_action_sample


class PolicyGuardTests(unittest.TestCase):
    def setUp(self):
        self.logits = {"action_type": torch.tensor([[0., 1., 2.], [2., 1., 0.]]),
                       "action_args": {"mu": torch.zeros(2, 2), "sigma": torch.ones(2, 2)}}
        self.action = {"action_type": torch.tensor([2, 0]), "action_args": torch.zeros(2, 2)}

    def test_scalar_action_prevents_categorical_cross_sample_broadcast(self):
        sample = {"action": {"action_type": torch.tensor([2]), "action_args": torch.ones(2)}}
        normalized = scalar_action_sample(sample)
        self.assertEqual(normalized["action"]["action_type"].shape, torch.Size([]))
        self.assertEqual(sample["action"]["action_type"].shape, torch.Size([1]))
        malformed = {**self.action, "action_type": torch.tensor([[2], [0]])}
        with self.assertRaisesRegex(ValueError, "never"):
            policy_drift(self.logits, self.logits, malformed)

    def test_exact_gaussian_drift_matches_analytic_result(self):
        self.assertEqual(policy_drift(self.logits, self.logits, self.action), (0., 0.))
        changed = {**self.logits, "action_args": {"mu": torch.ones(2, 2), "sigma": torch.ones(2, 2)}}
        # First row is EOS (zero); second row uses one N(0,1)->N(1,1).
        kl, ratio = policy_drift(changed, self.logits, self.action)
        self.assertAlmostEqual(kl, .25)
        self.assertAlmostEqual(ratio, .5)

    def test_unused_arguments_and_eos_do_not_trigger_drift(self):
        changed = {**self.logits, "action_args": {
            "mu": torch.tensor([[100., -100.], [0., 100.]]),
            "sigma": torch.tensor([[.1, 5.], [1., .1]])}}
        self.assertEqual(policy_drift(changed, self.logits, self.action), (0., 0.))

    def test_nonfinite_logits_and_invalid_scales_are_rejected(self):
        for value in (0., -1., float("nan")):
            changed = {**self.logits, "action_args": {"mu": torch.zeros(2, 2), "sigma": torch.full((2, 2), value)}}
            with self.subTest(value=value), self.assertRaises(FloatingPointError):
                policy_drift(changed, self.logits, self.action)


if __name__ == "__main__":
    unittest.main()
