"""Selected Gaussian likelihood, EOS masking and real native-optimizer checks."""
import math
import unittest

import torch
from ding.config import compile_config
from ding.rl_utils import ppo_policy_data, ppo_policy_error_continuous

from film_rl.hppo import build_configs, build_model
from film_rl.policy import FilmPPOPolicy, selected_continuous_policy_error
from fixtures import base_config


class SelectedContinuousLossTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.mu = torch.randn(4, 3, requires_grad=True)
        self.log_sigma = torch.randn(4, 3, requires_grad=True)
        self.new = {"mu": self.mu, "sigma": self.log_sigma.exp()}
        self.old = {"mu": self.mu.detach().clone(), "sigma": self.log_sigma.detach().exp()}
        self.action = {"action_type": torch.tensor([0, 2, 3, 1]), "action_args": torch.randn(4, 3)}
        self.adv = torch.tensor([1., -.5, .7, -.9])

    def test_unselected_and_eos_policy_gradients_are_zero(self):
        loss, _ = selected_continuous_policy_error(self.new, self.old, self.action, self.adv)
        gm, gs = torch.autograd.grad(loss.policy_loss, (self.mu, self.log_sigma))
        mask = torch.tensor([[True, False, False], [False, False, True],
                             [False, False, False], [False, True, False]])
        self.assertEqual(torch.count_nonzero(gm[~mask]), 0)
        self.assertEqual(torch.count_nonzero(gs[~mask]), 0)
        self.assertTrue((gm[mask].abs() > 0).all())

    def test_changing_unexecuted_arguments_preserves_loss_and_gradient(self):
        loss, _ = selected_continuous_policy_error(self.new, self.old, self.action, self.adv)
        original = torch.autograd.grad(loss.policy_loss, self.mu, retain_graph=True)[0]
        modified = self.action["action_args"].clone()
        mask = torch.tensor([[True, False, False], [False, False, True],
                             [False, False, False], [False, True, False]])
        modified[~mask] = 1000.
        other, _ = selected_continuous_policy_error(
            self.new, self.old, {**self.action, "action_args": modified}, self.adv)
        changed = torch.autograd.grad(other.policy_loss, self.mu)[0]
        torch.testing.assert_close(loss.policy_loss, other.policy_loss, atol=0, rtol=0)
        torch.testing.assert_close(original, changed, atol=0, rtol=0)

    def test_all_eos_has_zero_continuous_loss_and_finite_zero_gradients(self):
        action = {**self.action, "action_type": torch.full((4,), 3)}
        loss, info = selected_continuous_policy_error(self.new, self.old, action, self.adv)
        gm, gs = torch.autograd.grad(loss.policy_loss, (self.mu, self.log_sigma))
        self.assertEqual(loss.policy_loss.item(), 0.)
        self.assertEqual(info.approx_kl, 0.)
        self.assertEqual(info.clipfrac, 0.)
        self.assertEqual(torch.count_nonzero(gm), 0)
        self.assertEqual(torch.count_nonzero(gs), 0)

    def test_scalar_native_loss_parity_without_eos(self):
        new = {"mu": self.mu[:, :1], "sigma": self.log_sigma[:, :1].exp()}
        old = {"mu": torch.zeros(4, 1), "sigma": torch.ones(4, 1)}
        action = {"action_type": torch.zeros(4, dtype=torch.long), "action_args": self.action["action_args"][:, :1]}
        expected, info = ppo_policy_error_continuous(
            ppo_policy_data(new, old, action["action_args"], self.adv, None), .2)
        actual, actual_info = selected_continuous_policy_error(new, old, action, self.adv)
        torch.testing.assert_close(actual.policy_loss, expected.policy_loss)
        self.assertEqual(actual_info, info)

    def test_clipping_weights_and_full_batch_denominator(self):
        # At action=0, sigma_old=1, sigma_new=1/ratio => density ratio exactly ratio.
        ratios = torch.tensor([1.5, .5, 2., 1.1])
        new = {"mu": torch.zeros(4, 3), "sigma": (1/ratios[:, None]).expand(4, 3)}
        old = {"mu": torch.zeros(4, 3), "sigma": torch.ones(4, 3)}
        action = {**self.action, "action_args": torch.zeros(4, 3)}
        weights = torch.tensor([1., .5, 2., 1.])
        adv = torch.tensor([1., -1., 10., -1.])
        loss, _ = selected_continuous_policy_error(new, old, action, adv, weights)
        # EOS term is zero; divide by all four rows, not the three material rows.
        expected = -(1.2 + (-.8)*.5 + 0. + (-1.1))/4
        self.assertAlmostEqual(loss.policy_loss.item(), expected, places=6)


class SelectedPolicyUpdateTests(unittest.TestCase):
    def make_policy(self):
        cfg = base_config(".")
        cfg["training"].update(batch_size=4, epoch_per_collect=1, learning_rate=1e-4)
        main, create = build_configs(cfg)
        create.policy.type += "_command"
        compiled = compile_config(main, seed=0, auto=True, create_cfg=create, save_cfg=False)
        policy = FilmPPOPolicy(compiled.policy, model=build_model(cfg, main), enable_field=["learn", "collect"])
        return cfg, policy

    def samples(self, cfg, policy, count=8, eos_only=False):
        observations = torch.zeros(count, cfg["max_layers"]*5+1)
        observations[:, 0] = 1
        observations[:, 4] = .5
        observations[:, -1] = 1/cfg["max_layers"]  # EOS is legal after one layer.
        with torch.no_grad():
            output = policy._model(observations, mode="compute_actor_critic")
        samples = []
        for i in range(count):
            kind = 4 if eos_only or i % 4 == 0 else (i % 3 + 1)
            samples.append({
                "obs": observations[i].clone(),
                "logit": {"action_type": output["logit"]["action_type"][i].clone(),
                          "action_args": {k:v[i].clone() for k,v in output["logit"]["action_args"].items()}},
                "action": {"action_type": torch.tensor([kind]),
                           "action_args": output["logit"]["action_args"]["mu"][i].clone()+.1},
                "value": output["value"][i].clone(), "adv": torch.tensor(.1*(i+1)),
                "reward": torch.tensor([.8]), "done": True,
            })
        return samples

    def test_native_optimizer_updates_with_selected_loss(self):
        cfg, policy = self.make_policy()
        data = self.samples(cfg, policy)
        before = {k:v.clone() for k,v in policy._model.state_dict().items()}
        logs = policy.learn_mode.forward(data)
        self.assertEqual(len(logs), 2)
        self.assertEqual(policy.last_update["optimizer_steps"], 2)
        self.assertTrue(all(math.isfinite(v) for row in logs for v in row.values()))
        self.assertTrue(any(not torch.equal(before[k],v) for k,v in policy._model.state_dict().items()))
        self.assertEqual(policy._cfg.learn.batch_size, 4)
        self.assertEqual(policy._cfg.learn.epoch_per_collect, 1)
        self.assertEqual(data[0]["action"]["action_type"].shape, (1,))

    def test_all_eos_still_trains_critic_and_discrete_policy(self):
        cfg, policy = self.make_policy()
        data = self.samples(cfg, policy, count=4, eos_only=True)
        with torch.no_grad():
            out = policy._model(torch.stack([s["obs"] for s in data]), mode="compute_actor_critic")
            logits = out["logit"]
            expected_entropy = (
                torch.distributions.Categorical(logits=logits["action_type"]).entropy()
                + torch.distributions.Normal(logits["action_args"]["mu"], logits["action_args"]["sigma"]).entropy().sum(-1)
            ).mean().item()
        logs = policy.learn_mode.forward(data)
        self.assertAlmostEqual(logs[0]["entropy_loss"], expected_entropy, places=6)
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in policy._model.critic_head.parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in policy._model.actor_head[0].parameters()))
        # Original Gaussian entropy can still update sigma; it cannot update mu.
        mu_parameters = [(n,p) for n,p in policy._model.actor_head[1].named_parameters() if "log_sigma" not in n]
        self.assertTrue(mu_parameters)
        self.assertTrue(all(p.grad is None or torch.count_nonzero(p.grad)==0 for n,p in mu_parameters))

    def test_singleton_tail_keeps_all_samples_and_valid_updates(self):
        cfg, policy = self.make_policy()
        logs = policy.learn_mode.forward(self.samples(cfg, policy, count=9))
        self.assertEqual(len(logs), 3)
        self.assertTrue(all(math.isfinite(v) for row in logs for v in row.values()))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
