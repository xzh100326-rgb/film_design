"""Budget, absolute per-step reward and export checks; no network updates."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from film_rl.config import PROJECT_DIR, load_config, validate
from film_rl.env import ThinFilmEnv
from film_rl.optics import FilmOptics


class TotalThicknessTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config()
        self.cfg.update(angles_deg=[0.], thickness_min_nm=30., thickness_max_nm=210.,
                        max_total_thickness_nm=400., min_layers=1, max_layers=50)

    @staticmethod
    def action(value=0., material=3):
        return {"action_type": material, "action_args": np.full(5, value, dtype=np.float32)}

    def test_budget_caps_next_layer_and_evaluates_once_per_step(self):
        env = ThinFilmEnv(self.cfg)
        env.reset()
        with patch.object(env.optics, "evaluate", wraps=env.optics.evaluate) as solver:
            first = env.step(self.action(100.))
            self.assertFalse(first.done)
            self.assertAlmostEqual(float(first.reward[0]), first.info["reward"], places=6)
            solver.assert_called_once()
            end = env.step(self.action(100., material=4))
            self.assertTrue(end.done)
            self.assertEqual(env.layers, [(3, 210.), (4, 190.)])
            self.assertEqual(end.info["total_thickness_nm"], 400.)
            self.assertEqual(end.info["termination_reason"], "thickness_budget")
            self.assertAlmostEqual(float(end.reward[0]), end.info["reward"], places=6)
            self.assertEqual(solver.call_count, 2)
        with self.assertRaises(RuntimeError):
            env.step(self.action())
        env.reset()
        self.assertEqual(env.layers, [])
        self.assertEqual(env.step(self.action(100.)).info["total_thickness_nm"], 210.)

    def test_insufficient_remainder_ends_without_filling_budget(self):
        env = ThinFilmEnv(self.cfg)
        env.reset()
        for index, value in enumerate((100., 0., 0.)):
            end = env.step(self.action(value, material=3 + index % 2))
        self.assertEqual([d for _, d in env.layers], [210., 110., 55.])
        self.assertTrue(end.done)
        self.assertEqual(end.info["total_thickness_nm"], 375.)
        self.assertEqual(end.info["termination_reason"], "thickness_budget")

    def test_exact_minimum_remainder_can_be_used(self):
        cfg = deepcopy(self.cfg)
        cfg["max_total_thickness_nm"] = 240.
        env = ThinFilmEnv(cfg)
        env.reset()
        self.assertFalse(env.step(self.action(100.)).done)
        end = env.step(self.action(material=4))
        self.assertTrue(end.done)
        self.assertEqual([d for _, d in env.layers], [210., 30.])
        self.assertEqual(end.info["total_thickness_nm"], 240.)

    def test_budget_reserves_thickness_for_required_minimum_layers(self):
        cfg = deepcopy(self.cfg)
        cfg["min_layers"] = 13
        env = ThinFilmEnv(cfg)
        env.reset()
        for index in range(13):
            end = env.step(self.action(100., material=3 + index % 2))
            self.assertEqual(end.done, index == 12)
        self.assertEqual([d for _, d in env.layers], [40.] + [30.] * 12)
        self.assertEqual(end.info["total_thickness_nm"], 400.)

    def test_eos_and_export_record_actual_total_thickness(self):
        scratch = PROJECT_DIR / "outputs"
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as folder:
            cfg = deepcopy(self.cfg)
            cfg["result_dir"] = folder
            env = ThinFilmEnv(cfg)
            env.reset()
            env.step(self.action())
            end = env.step({"action_type": 5})
            self.assertEqual(end.info["termination_reason"], "eos")
            self.assertEqual(end.info["total_thickness_nm"], 120.)
            saved = json.loads(next(Path(folder).glob("best_*.json")).read_text(encoding="utf-8"))
            self.assertEqual(saved["total_thickness_nm"], 120.)

    def test_optics_rejects_over_budget_design_before_tmm(self):
        optics = FilmOptics(self.cfg)
        with self.assertRaisesRegex(ValueError, "Total coating thickness"):
            optics.evaluate([(3, 210.), (4, 210.)])
        self.assertEqual(optics.evaluation_count, 0)

    def test_invalid_budgets_and_optional_disabled_constraint(self):
        for value in (True, "400", float("nan"), float("inf"), 0., 29.):
            cfg = deepcopy(self.cfg)
            cfg["max_total_thickness_nm"] = value
            with self.assertRaises(ValueError):
                validate(cfg)
        cfg = deepcopy(self.cfg)
        cfg["min_layers"] = 14
        with self.assertRaises(ValueError):
            validate(cfg)
        for missing in (False, True):
            cfg = deepcopy(self.cfg)
            cfg["max_total_thickness_nm"] = None
            if missing:
                cfg.pop("max_total_thickness_nm")
            env = ThinFilmEnv(cfg)
            env.reset()
            for index in range(3):
                step = env.step(self.action(100., material=3 + index % 2))
                self.assertFalse(step.done)
            self.assertEqual(step.info["total_thickness_nm"], 630.)

    def test_random_rollouts_respect_bounds_and_sum_step_scores(self):
        env = ThinFilmEnv(self.cfg)
        env.seed(7)
        for _ in range(20):
            env.reset()
            episode_return = 0.
            for _ in range(self.cfg["max_layers"] + 1):
                step = env.step(env.random_action())
                self.assertLessEqual(step.info["total_thickness_nm"], 400. + 1e-9)
                self.assertTrue(all(30. <= d <= 210. for _, d in env.layers))
                self.assertTrue(all(a[0] != b[0] for a, b in zip(env.layers, env.layers[1:])))
                self.assertTrue(env.observation_space.contains(step.obs))
                episode_return += float(step.reward[0])
                self.assertAlmostEqual(float(step.reward[0]), 1 - step.info["mae"], places=6)
                if step.done:
                    break
            else:
                self.fail("Episode did not terminate")
            self.assertAlmostEqual(episode_return, step.info["training_return"], places=6)
            self.assertEqual(step.info["eval_episode_return"], step.info["reward"])


if __name__ == "__main__":
    unittest.main()
