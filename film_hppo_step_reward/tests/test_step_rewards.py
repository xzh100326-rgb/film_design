"""Signed mixed-score increments and native evaluation; never trains a model."""

from copy import deepcopy
import json
import math
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

from film_rl.config import PROJECT_DIR, load_config, material_count
from film_rl.env import ThinFilmEnv
from film_rl.optics import FilmOptics
import test_material_modes as fixtures


def mixed(score, cfg):
    alpha = cfg.get("reward_alpha", 0.)
    epsilon = cfg.get("reward_epsilon", 1e-6)
    return (1-alpha)*score + alpha*(-math.log10(max(1-score, epsilon))/-math.log10(epsilon))


class ScriptedOptics:
    """Known optical scores indexed by layer count, to test delayed improvements."""
    def __init__(self, cfg, scores):
        self.cfg = cfg
        self.scores = scores
        self.evaluation_count = 0
        self.stacks = []

    def evaluate(self, layers):
        self.evaluation_count += 1
        self.stacks.append(list(layers))
        score = self.scores[len(layers)]
        return {"score": score, "mean_transmission": score,
                "reward": mixed(score, self.cfg), "layer_count": len(layers),
                "layers": [{"material_id": i, "thickness_nm": d} for i, d in layers],
                "reward_mode": "linear_log_mixture", "reward_alpha": self.cfg["reward_alpha"]}


class StepRewardTests(unittest.TestCase):
    setUp = fixtures.MaterialModeTests.setUp

    def scripted(self, scores, cfg=None):
        cfg = deepcopy(self.cfg if cfg is None else cfg)
        optics = ScriptedOptics(cfg, scores)
        with patch("film_rl.env.FilmOptics", return_value=optics):
            env = ThinFilmEnv(cfg)
            env.reset()
        self.addCleanup(env.close)
        return env, optics

    @staticmethod
    def add(env, material):
        return env.step({"action_type": material, "action_args": np.zeros(env.material_count)})

    def test_signed_differences_and_no_duplicate_eos_reward(self):
        scores = [.95, .998, .970, .999]
        env, optics = self.scripted(scores)
        rewards = []
        for position, material in enumerate([0, 1, 2], start=1):
            step = self.add(env, material)
            reward = float(step.reward[0])
            self.assertAlmostEqual(reward, mixed(scores[position], self.cfg)-mixed(scores[position-1], self.cfg), places=7)
            self.assertEqual(len(optics.stacks[-1]), position)
            self.assertTrue(env.reward_space.contains(step.reward))
            rewards.append(reward)
        self.assertGreater(rewards[0], 0)
        self.assertLess(rewards[1], 0)
        self.assertGreater(rewards[2], 0)
        final = env.step({"action_type": env.EOS})
        self.assertEqual(float(final.reward[0]), 0.)
        self.assertEqual(optics.evaluation_count, 5)  # baseline, 3 layers, EOS
        self.assertEqual(final.info["reward_mode"], "mixed_score_delta")
        self.assertAlmostEqual(sum(rewards), mixed(.999, self.cfg)-mixed(.95, self.cfg), places=7)
        self.assertEqual(sum(rewards), final.info["episode_training_return"])
        self.assertEqual(final.info["eval_episode_return"], .999)
        self.assertEqual(final.info["reward"], mixed(.999, self.cfg))

    def test_log_amplification_is_applied_before_differencing(self):
        env, _ = self.scripted([.95, .998, .999])
        self.add(env, 0)
        step = self.add(env, 1)
        self.assertAlmostEqual(float(step.reward[0]), .025585832971998457, places=7)
        self.assertGreater(float(step.reward[0]), 20*.001)

    def test_alpha_zero_and_one_and_epsilon_saturation(self):
        for alpha in [0., .5, 1.]:
            cfg = deepcopy(self.cfg)
            cfg["reward_alpha"] = alpha
            env, _ = self.scripted([.95, .998, .999], cfg)
            first = self.add(env, 0)
            last = self.add(env, 1)
            expected = mixed(.999, cfg)-mixed(.95, cfg)
            self.assertAlmostEqual(float(first.reward[0])+float(last.reward[0]), expected, places=7)
        cfg["reward_alpha"] = 1.
        env, _ = self.scripted([.95, 1.-1e-7, 1.-1e-8], cfg)
        self.add(env, 0)
        self.assertEqual(float(self.add(env, 1).reward[0]), 0.)

    def test_last_layer_at_max_layers_can_still_have_negative_reward(self):
        cfg = deepcopy(self.cfg)
        cfg["max_layers"] = 2
        env, optics = self.scripted([.95, .999, .97], cfg)
        first = self.add(env, 0)
        last = self.add(env, 1)
        self.assertTrue(last.done)
        self.assertEqual(last.info["termination_reason"], "max_layers")
        self.assertLess(float(last.reward[0]), 0.)
        self.assertAlmostEqual(float(last.reward[0]), mixed(.97, cfg)-mixed(.999, cfg), places=7)
        self.assertAlmostEqual(float(first.reward[0])+float(last.reward[0]), mixed(.97, cfg)-mixed(.95, cfg), places=7)
        self.assertEqual(optics.evaluation_count, 3)

    def test_thickness_budget_termination_returns_only_increment(self):
        cfg = deepcopy(self.cfg)
        cfg["max_total_thickness_nm"] = cfg["thickness_min_nm"]
        env, _ = self.scripted([.99, .95], cfg)
        step = self.add(env, 0)
        self.assertTrue(step.done)
        self.assertEqual(step.info["termination_reason"], "thickness_budget")
        self.assertLess(float(step.reward[0]), 0.)
        self.assertAlmostEqual(float(step.reward[0]), mixed(.95, cfg)-mixed(.99, cfg), places=7)

    def test_bare_eos_has_zero_return_and_baseline_is_cached_on_reset(self):
        cfg = deepcopy(self.cfg)
        cfg["min_layers"] = 0
        env, optics = self.scripted([.95, .99], cfg)
        final = env.step({"action_type": env.EOS})
        self.assertEqual(float(final.reward[0]), 0.)
        self.assertEqual(final.info["episode_training_return"], 0.)
        self.assertEqual(final.info["eval_episode_return"], .95)
        self.assertEqual(optics.evaluation_count, 2)
        env.reset()
        self.assertEqual(optics.evaluation_count, 2)
        self.assertEqual(env._previous_mixed_score, mixed(.95, cfg))
        self.add(env, 0)
        second = env.step({"action_type": env.EOS})
        self.assertAlmostEqual(second.info["episode_training_return"], mixed(.99, cfg)-mixed(.95, cfg), places=7)

    def test_actual_tmm_return_telescopes_in_fixed_and_file_modes(self):
        for mode in ["fixed_193nm", "lm"]:
            cfg = deepcopy(self.cfg)
            cfg["material_source"] = mode
            env = ThinFilmEnv(cfg)
            self.addCleanup(env.close)
            env.reset()
            rewards = []
            for material in [0, 1, 0]:
                step = self.add(env, material)
                rewards.append(float(step.reward[0]))
            final = env.step({"action_type": env.EOS})
            absolute = FilmOptics(cfg).evaluate(env.layers)
            self.assertAlmostEqual(final.info["mixed_score"], absolute["reward"], places=12)
            self.assertAlmostEqual(sum(rewards), final.info["mixed_score"]-env._baseline_mixed_score, places=7)
            self.assertEqual(final.info["eval_episode_return"], absolute["score"])
            self.assertTrue(all(np.isfinite(rewards)))

    def test_saved_design_separates_absolute_score_from_last_step(self):
        cfg = deepcopy(self.cfg)
        cfg["result_dir"] = str(Path(self.folder.name) / "designs")
        env, _ = self.scripted([.95, .999], cfg)
        self.add(env, 0)
        final = env.step({"action_type": env.EOS})
        path = next(Path(cfg["result_dir"]).glob("best_*.json"))
        saved = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(saved["step_reward"], 0.)
        self.assertEqual(saved["score"], .999)
        self.assertEqual(saved["mixed_score"], mixed(.999, cfg))
        self.assertAlmostEqual(saved["episode_training_return"], mixed(.999, cfg)-mixed(.95, cfg), places=7)

    def test_native_evaluator_uses_transmission_and_not_training_return(self):
        from ding.envs import BaseEnvManager
        from ding.worker import InteractionSerialEvaluator
        from easydict import EasyDict
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        cfg = deepcopy(self.cfg)
        count = material_count(cfg)

        class OneLayerPolicy:
            def get_attribute(self, name):
                return EasyDict(type="film_hppo")
            def reset(self, *args):
                pass
            def forward(self, observations, **kwargs):
                return {i: {"action": {"action_type": torch.tensor(0 if obs[-1] == 0 else count),
                                        "action_args": torch.zeros(count)}} for i, obs in observations.items()}

        manager = BaseEnvManager(env_fn=[lambda: ThinFilmEnv(cfg)], cfg=BaseEnvManager.default_config())
        relative = Path(self.folder.name).relative_to(PROJECT_DIR) / "native_eval"
        options = InteractionSerialEvaluator.default_config()
        options.update(n_episode=2, stop_value=.995)
        evaluator = None
        previous = Path.cwd()
        try:
            os.chdir(PROJECT_DIR)
            with patch("film_rl.env.FilmOptics", side_effect=lambda c: ScriptedOptics(c, [.95, .999])):
                evaluator = InteractionSerialEvaluator(options, env=manager, policy=OneLayerPolicy(), exp_name=relative.as_posix())
                stop, info = evaluator.eval(train_iter=1, envstep=0)
                self.assertTrue(stop)
                np.testing.assert_allclose(info["eval_episode_return"], [.999, .999], rtol=0, atol=1e-6)
                np.testing.assert_allclose(info["training_return"], [mixed(.999,cfg)-mixed(.95,cfg)]*2, atol=1e-6)
                evaluator.close()
            events = EventAccumulator(str(PROJECT_DIR / relative / "log/evaluator/evaluator_tb_logger"))
            events.Reload()
            self.assertAlmostEqual(events.Scalars("evaluator_iter/reward_mean")[-1].value, .999, places=6)
        finally:
            if evaluator is not None:
                evaluator.close()
                # Native close flushes TensorBoard but leaves text handlers open.
                # Release only this test's own files before TemporaryDirectory cleanup.
                for handler in list(evaluator._logger.handlers):
                    filename = getattr(handler, "baseFilename", None)
                    if filename and Path(filename).resolve().is_relative_to(Path(self.folder.name).resolve()):
                        evaluator._logger.removeHandler(handler)
                        handler.close()
            manager.close()
            os.chdir(previous)


if __name__ == "__main__":
    unittest.main()
