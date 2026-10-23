"""Direct per-step score, EOS and real evaluator semantics; no training."""

from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from ding.config import compile_config
from ding.envs import BaseEnvManager
from ding.worker import InteractionSerialEvaluator

from film_rl.config import PROJECT_DIR, load_config
from film_rl.env import ThinFilmEnv
from film_rl.hppo import build_configs, build_model
from film_rl.optics import FilmOptics
from film_rl.policy import FilmPPOPolicy


class StepScoreTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config()

    @staticmethod
    def action(material):
        return {"action_type": material, "action_args": np.zeros(5, dtype=np.float32)}

    @staticmethod
    def score(value):
        return {"reward": value, "mae": 1 - value, "mean_transmission": value}

    def test_real_tmm_reward_is_absolute_after_every_action_including_eos(self):
        env = ThinFilmEnv(self.cfg)
        env.reset()
        self.assertEqual(env.optics.evaluation_count, 0)
        first = env.step(self.action(3))
        second = env.step(self.action(4))
        end = env.step({"action_type": 5})
        self.assertEqual(env.optics.evaluation_count, 3)
        for step in (first, second, end):
            self.assertAlmostEqual(float(step.reward[0]), 1 - step.info["mae"], places=6)
            self.assertEqual(np.asarray(step.info["T"]).shape, (10, 1))
            self.assertEqual(step.info["reward_mode"], "absolute_step_score")
        self.assertNotAlmostEqual(float(second.reward[0]), second.info["reward"] - first.info["reward"], places=5)
        self.assertAlmostEqual(float(second.reward[0]), float(end.reward[0]), places=6)
        expected = sum(step.info["reward"] for step in (first, second, end))
        self.assertAlmostEqual(end.info["training_return"], expected)
        self.assertEqual(end.info["eval_episode_return"], end.info["reward"])
        env.reset()
        self.assertEqual(env._training_return, 0.)

    def test_lower_score_is_still_a_positive_absolute_reward(self):
        env = ThinFilmEnv(self.cfg)
        env.reset()
        with patch.object(env.optics, "evaluate", side_effect=[self.score(.9), self.score(.7), self.score(.7)]):
            steps = [env.step(self.action(3)), env.step(self.action(4)), env.step({"action_type": 5})]
        np.testing.assert_allclose([float(step.reward[0]) for step in steps], [.9, .7, .7])
        self.assertAlmostEqual(steps[-1].info["training_return"], 2.3)
        self.assertAlmostEqual(steps[-1].info["eval_episode_return"], .7)

    def test_nonunit_target_still_uses_one_minus_mae(self):
        cfg = deepcopy(self.cfg)
        cfg["target_transmission"] = .6
        env = ThinFilmEnv(cfg)
        env.reset()
        step = env.step(self.action(3))
        expected = 1 - np.abs(np.asarray(step.info["T"]) - .6).mean()
        self.assertAlmostEqual(float(step.reward[0]), expected, places=6)
        self.assertNotAlmostEqual(float(step.reward[0]), step.info["mean_transmission"], places=4)

    def test_invalid_actions_do_not_evaluate_or_accumulate_reward(self):
        env = ThinFilmEnv(self.cfg)
        env.reset()
        with self.assertRaises(ValueError):
            env.step({"action_type": 5})
        self.assertEqual(env.optics.evaluation_count, 0)
        self.assertEqual(env._training_return, 0.)

    def test_best_design_uses_final_score_not_larger_training_return(self):
        scratch = PROJECT_DIR / "outputs"
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as folder:
            cfg = deepcopy(self.cfg)
            cfg.update(max_layers=3, max_total_thickness_nm=None, result_dir=folder)
            env = ThinFilmEnv(cfg)
            env.reset()
            with patch.object(env.optics, "evaluate", side_effect=[self.score(.7)] * 3 + [self.score(.9)] * 2):
                for material in (3, 4, 3):
                    end_a = env.step(self.action(material))
                env.reset()
                env.step(self.action(3))
                end_b = env.step({"action_type": 5})
            self.assertGreater(end_a.info["training_return"], end_b.info["training_return"])
            saved = json.loads(next(Path(folder).glob("best_*.json")).read_text(encoding="utf-8"))
            self.assertEqual(saved["reward"], .9)
            self.assertEqual(saved["layer_count"], 1)

    def test_native_evaluator_eight_episodes_do_not_stop_on_sum_above_one(self):
        cfg = deepcopy(self.cfg)
        cfg.update(max_layers=2, max_total_thickness_nm=None)
        main, create = build_configs(cfg)
        main.policy.cuda = False
        create.policy.type += "_command"
        compiled = compile_config(main, seed=cfg["training"]["seed"], auto=True,
                                  create_cfg=create, save_cfg=False)
        model = build_model(cfg, main)
        policy = FilmPPOPolicy(compiled.policy, model=model, enable_field=["learn", "collect", "eval"])
        before = {key: value.clone() for key, value in model.state_dict().items()}
        scratch = PROJECT_DIR / "outputs"
        scratch.mkdir(exist_ok=True)
        previous_dir = Path.cwd()
        with tempfile.TemporaryDirectory(dir=scratch) as folder:
            evaluator = None
            manager = BaseEnvManager(env_fn=[lambda: ThinFilmEnv(cfg)], cfg=BaseEnvManager.default_config())
            try:
                os.chdir(folder)
                evaluator = InteractionSerialEvaluator(compiled.policy.eval.evaluator, manager,
                                                       policy.eval_mode, exp_name="check")
                with patch.object(FilmOptics, "evaluate", return_value=self.score(.8)) as solver:
                    stop, report = evaluator.eval(train_iter=1)
                self.assertFalse(stop)
                self.assertEqual(solver.call_count, 16)
                np.testing.assert_allclose(report["eval_episode_return"], [.8] * 8)
                np.testing.assert_allclose(report["training_return"], [1.6] * 8)
                self.assertAlmostEqual(report["final_score_mean"], .8)
                self.assertTrue(all(torch.equal(value, before[key]) for key, value in model.state_dict().items()))
            finally:
                if evaluator is not None:
                    logger = evaluator._logger
                    evaluator.close()
                    # Native close() closes TensorBoard and env, but leaves
                    # FileHandlers open. Release only this test's handlers so
                    # Windows can remove its own temporary log directory.
                    for handler in list(logger.handlers):
                        handler.close()
                        logger.removeHandler(handler)
                else:
                    manager.close()
                os.chdir(previous_dir)


if __name__ == "__main__":
    unittest.main()
