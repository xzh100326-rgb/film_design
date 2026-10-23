"""Bounded update regressions on disposable models, never a full training run."""

from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np
import torch
from ding.config import compile_config
from ding.envs import BaseEnvTimestep
from ding.policy import create_policy
from ding.utils import set_pkg_seed
from ding.rl_utils import ppo_policy_data, ppo_policy_error

from film_rl.config import load_config
from film_rl.env import ThinFilmEnv
from film_rl.hppo import build_configs, build_model
from film_rl.policy import policy_drift


class StabilityTests(unittest.TestCase):
    def make_policy(self, epochs=3):
        set_pkg_seed(0, use_cuda=False)
        cfg = load_config()
        main, create = build_configs(cfg)
        main.policy.learn.epoch_per_collect = epochs
        main.policy.learn.batch_size = 32
        create.policy.type += "_command"
        compiled = compile_config(main, seed=0, auto=True, create_cfg=create, save_cfg=False)
        model = build_model(cfg, main)
        policy = create_policy(compiled.policy, model=model, enable_field=["learn", "collect", "eval", "command"])
        return cfg, model, policy

    @staticmethod
    def samples(cfg, policy, count=128):
        env = ThinFilmEnv(cfg)
        obs = torch.as_tensor(env.reset())
        transitions, samples = [], []
        try:
            for index in range(count):
                output = policy.collect_mode.forward({0: obs})[0]
                action = {k: v.detach().cpu().numpy() for k, v in output["action"].items()}
                step = env.step(action)
                step = BaseEnvTimestep(torch.as_tensor(step.obs), torch.as_tensor(step.reward), step.done, step.info)
                transitions.append(policy.collect_mode.process_transition(obs, output, step))
                if step.done or index == count - 1:
                    samples.extend(policy.collect_mode.get_train_sample(transitions))
                    transitions = []
                obs = torch.as_tensor(env.reset()) if step.done else step.obs
            return samples
        finally:
            env.close()

    def test_real_tmm_rollouts_and_native_backward_remain_finite(self):
        cfg, model, policy = self.make_policy()
        original = {k: v.clone() for k, v in model.state_dict().items()}
        for _ in range(2):
            samples = self.samples(cfg, policy)
            frozen_samples = deepcopy(samples)
            logs = policy.learn_mode.forward(samples)
            self.assertGreater(len(logs), 0)
            self.assertLessEqual(len(logs), 12)
            self.assertEqual(len(logs), policy.last_update["optimizer_steps"])
            self.assertTrue(all(np.isfinite(float(value)) for log in logs for value in log.values()))
            self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
            for before, after in zip(frozen_samples, samples):
                torch.testing.assert_close(before["adv"], after["adv"])
                torch.testing.assert_close(before["value"], after["value"])
                torch.testing.assert_close(before["logit"]["action_type"], after["logit"]["action_type"])
        self.assertTrue(any(not torch.equal(v, original[k]) for k, v in model.state_dict().items()))

    def test_stale_extreme_log_ratio_is_rejected_before_update(self):
        cfg, model, policy = self.make_policy()
        samples = self.samples(cfg, policy, 32)
        for sample in samples:
            # Old policy assigns vanishing density to the sampled action, causing
            # exp(new_log_prob - old_log_prob) to overflow in the unguarded loss.
            sample["logit"]["action_args"]["mu"] += 1000
        before = {k: v.clone() for k, v in model.state_dict().items()}
        with self.assertRaisesRegex(FloatingPointError, "before any update"):
            policy.learn_mode.forward(samples)
        self.assertTrue(all(torch.equal(v, before[k]) for k, v in model.state_dict().items()))
        self.assertEqual(policy._cfg.learn.epoch_per_collect, 3)

    def test_kl_limit_stops_remaining_updates(self):
        cfg, model, policy = self.make_policy()
        samples = self.samples(cfg, policy, 64)
        with patch("film_rl.policy.policy_drift", side_effect=[(0., 0.), (1., 1.)]):
            with self.assertWarnsRegex(RuntimeWarning, "reuse stopped"):
                logs = policy.learn_mode.forward(samples)
        self.assertEqual(len(logs), 1)
        self.assertTrue(policy.last_update["early_stop"])
        self.assertEqual(policy.last_update["optimizer_steps"], 1)
        self.assertEqual(policy._cfg.learn.epoch_per_collect, 3)

    def test_nonfinite_gradient_blocked_before_optimizer_corrupts_weights(self):
        _, model, policy = self.make_policy()
        parameter = next(model.parameters())
        parameter.grad = torch.full_like(parameter, float("nan"))
        before = parameter.detach().clone()
        with self.assertRaisesRegex(FloatingPointError, "gradient"):
            policy._optimizer.step()
        torch.testing.assert_close(parameter, before)

    def test_singleton_tail_and_sigma_bounds(self):
        cfg, model, policy = self.make_policy(epochs=1)
        samples = self.samples(cfg, policy, 33)
        logs = policy.learn_mode.forward(samples)
        self.assertEqual(len(logs), 2)
        for log_scale in [-1000., 1000.]:
            with torch.no_grad():
                model.actor_head[1].log_sigma_param.fill_(log_scale)
            model.zero_grad(set_to_none=True)
            env = ThinFilmEnv(cfg)
            obs = torch.as_tensor(env.reset()).repeat(2, 1)
            output = model(obs, "compute_actor")
            sigma = output["logit"]["action_args"]["sigma"]
            self.assertTrue(torch.isfinite(sigma).all())
            self.assertTrue((sigma >= np.exp(-5) - 1e-8).all())
            self.assertTrue((sigma <= np.exp(2) + 1e-6).all())
            sigma.sum().backward()
            self.assertTrue(torch.isfinite(model.actor_head[1].log_sigma_param.grad).all())

    def test_native_loss_gets_scalar_categorical_actions(self):
        cfg, model, policy = self.make_policy(epochs=1)
        samples = self.samples(cfg, policy, 32)
        # Reproduce the shape delivered by the original DI-engine collector.
        for sample in samples:
            sample["action"]["action_type"] = sample["action"]["action_type"].reshape(1)
        calls = []

        def checked_loss(data, *args, **kwargs):
            self.assertEqual(data.action.shape, (32,))
            self.assertEqual(torch.distributions.Categorical(logits=data.logit_new).log_prob(data.action).shape, (32,))
            calls.append(True)
            return ppo_policy_error(data, *args, **kwargs)

        with patch("ding.policy.ppo.ppo_policy_error", side_effect=checked_loss):
            logs = policy.learn_mode.forward(samples)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(logs), 1)
        self.assertEqual(samples[0]["action"]["action_type"].shape, (1,))

    def test_broadcast_bug_overflows_despite_small_valid_action_kl(self):
        old = torch.full((2, 6), -100.)
        old[0, 0], old[1, 1] = 0., 0.
        new = old.clone()
        new[0, 1], new[1, 0] = -5., -5.
        action = torch.tensor([0, 1])
        args = {"mu": torch.zeros(2, 5), "sigma": torch.ones(2, 5)}
        kl, ratio = policy_drift({"action_type": new, "action_args": args},
                                 {"action_type": old, "action_args": args},
                                 {"action_type": action, "action_args": torch.zeros(2, 5)})
        self.assertLess(kl, .02)
        self.assertLess(ratio, .01)
        adv = torch.tensor([-1., 1.])
        wrong, _ = ppo_policy_error(ppo_policy_data(new, old, action[:, None], adv, None), .2)
        self.assertFalse(torch.isfinite(wrong.policy_loss))
        correct, _ = ppo_policy_error(ppo_policy_data(new, old, action, adv, None), .2)
        self.assertTrue(torch.isfinite(correct.policy_loss))


if __name__ == "__main__":
    unittest.main()
