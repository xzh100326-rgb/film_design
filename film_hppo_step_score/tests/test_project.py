"""Behavioral and physical checks. Never calls learn, backward, or optimizer.step."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from film_rl.config import PROJECT_DIR, grid, load_config, validate
from film_rl.env import ThinFilmEnv
from film_rl.hppo import build_model
from film_rl.optics import FilmOptics, read_index


class ProjectTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config()
        # Fixed optical fixtures, independent of the user's experiment settings.
        self.cfg.update(angles_deg=[0.], thickness_min_nm=30., thickness_max_nm=210.,
                        max_total_thickness_nm=None, max_layers=50)

    @staticmethod
    def action(material=0, value=0):
        return {"action_type": material, "action_args": np.full(5, value, dtype=np.float32)}

    def test_real_material_interpolation_and_absorption(self):
        optics = FilmOptics(self.cfg)
        self.assertEqual(optics.material_indices.shape, (5, 1))
        self.assertAlmostEqual(optics.material_indices[0, 0].imag, 0.0377 + .6 * (0.0045 - .0377))
        self.assertGreater(optics.material_indices[2, 0].imag, 0)
        self.assertEqual(len(optics.names), 5)
        self.assertTrue(all("CaF2" not in name for name in optics.names))
        with self.assertRaises(ValueError):
            read_index(Path(self.cfg["material_dir"]) / self.cfg["material_files"][0], np.array([189.]))

    def test_each_step_evaluates_and_observation_preserves_stack(self):
        env = ThinFilmEnv(self.cfg)
        obs = env.reset()
        self.assertEqual(obs.shape, (301,))
        with patch.object(env.optics, "evaluate", wraps=env.optics.evaluate) as solver:
            step = env.step(self.action(3))
            self.assertFalse(step.done)
            self.assertAlmostEqual(float(step.reward[0]), step.info["reward"], places=6)
            self.assertAlmostEqual(float(step.reward[0]), 1. - step.info["mae"], places=6)
            solver.assert_called_once()
            self.assertEqual(env.layers, [(3, 120.)])
            self.assertEqual(step.obs[3], 1.)
            self.assertEqual(step.obs[5], .5)
            self.assertTrue(env.observation_space.contains(step.obs))
            # EOS ignores all continuous arguments, including an omitted argument.
            end = env.step({"action_type": 5})
            self.assertTrue(end.done)
            self.assertEqual(solver.call_count, 2)
            self.assertAlmostEqual(end.info["reward"], end.info["mean_transmission"])
            self.assertAlmostEqual(float(end.reward[0]), float(step.reward[0]), places=6)
            self.assertAlmostEqual(end.info["training_return"], 2 * end.info["reward"])
            self.assertEqual(end.info["eval_episode_return"], end.info["reward"])
            self.assertEqual(end.info["termination_reason"], "eos")
        with self.assertRaises(RuntimeError):
            env.step(self.action())
        env.reset()
        self.assertEqual(len(env.layers), 0)

    def test_bounds_and_selected_material_argument(self):
        env = ThinFilmEnv(self.cfg)
        env.reset()
        args = np.array([-100., 100., 0., 1., -1.], dtype=np.float32)
        for material in [0, 1, 2]:
            env.step({"action_type": material, "action_args": args})
        self.assertEqual([d for _, d in env.layers], [30., 210., 120.])

    def test_maximum_50_layers_auto_terminates(self):
        env = ThinFilmEnv(self.cfg)
        env.reset()
        for index in range(50):
            step = env.step(self.action(3 + index % 2))
            self.assertEqual(step.done, index == 49)
        self.assertEqual(step.info["termination_reason"], "max_layers")
        self.assertEqual(env.optics.evaluation_count, 50)
        self.assertTrue(np.isfinite(step.info["reward"]))

    def test_eos_mask_in_collect_and_learning_forwards(self):
        cfg = deepcopy(self.cfg)
        cfg["min_layers"] = 2
        model = build_model(cfg)
        obs = torch.zeros(3, 301)
        obs[:, -1] = torch.tensor([0., 1 / 50, 2 / 50])
        for mode in ("compute_actor", "compute_actor_critic"):
            with torch.no_grad():
                logits = model(obs, mode)["logit"]
            self.assertEqual(logits["action_args"]["mu"].shape, (3, 5))
            self.assertTrue((logits["action_type"][:2, 5] < -1e8).all())
            self.assertGreater(float(logits["action_type"][2, 5]), -1e8)

    def test_invalid_actions_do_not_mutate_stack(self):
        env = ThinFilmEnv(self.cfg)
        env.reset()
        for action in ({"action_type": 5}, self.action(6), self.action(1.5), self.action(0, np.nan),
                       {"action_type": 1, "action_args": [0.]}):
            with self.assertRaises(ValueError):
                env.step(action)
            self.assertEqual(len(env.layers), 0)

    def test_fresnel_bare_substrate_and_lossless_conservation(self):
        optics = FilmOptics(self.cfg)
        result = optics.evaluate([])
        n0, ns = self.cfg["incident_n"], optics.substrate_index[0]
        expected_r = abs((n0 - ns) / (n0 + ns)) ** 2
        expected_t = 4 * n0 * ns.real / abs(n0 + ns) ** 2
        self.assertAlmostEqual(result["R"][0][0], expected_r, places=10)
        self.assertAlmostEqual(result["T"][0][0], expected_t, places=10)
        # Analytic quarter-wave film at normal incidence independently checks units/order.
        optics.material_indices[0, :] = 1.4 + 0j
        optics.substrate_index[:] = 1.6 + 0j
        thickness = 193. / (4 * 1.4)
        result = optics.evaluate([(0, thickness)])
        expected_r = ((n0 * 1.6 - 1.4**2) / (n0 * 1.6 + 1.4**2))**2
        self.assertAlmostEqual(result["R"][0][0], expected_r, places=10)
        self.assertAlmostEqual(result["T"][0][0], 1 - expected_r, places=10)

    def test_angle_range_polarizations_and_target_reward(self):
        cfg = deepcopy(self.cfg)
        cfg["angles_deg"] = {"start": 0, "stop": 60, "num": 4}
        cfg["wavelengths_nm"] = [193., 200.]
        cfg["target_transmission"] = [.8, .9]
        results = {}
        for pol in ("s", "p", "unpolarized"):
            cfg["polarization"] = pol
            results[pol] = FilmOptics(cfg).evaluate([(3, 35.)])
        avg = np.array(results["unpolarized"]["T"])
        self.assertEqual(avg.shape, (4, 2))
        np.testing.assert_allclose(avg, (np.array(results["s"]["T"]) + results["p"]["T"]) / 2)
        self.assertAlmostEqual(results["unpolarized"]["reward"], 1 - np.abs(avg - [.8, .9]).mean())

    def test_seed_reproducibility(self):
        a, b = ThinFilmEnv(self.cfg), ThinFilmEnv(self.cfg)
        a.seed(12)
        b.seed(12)
        a.reset()
        b.reset()
        for _ in range(5):
            x, y = a.random_action(), b.random_action()
            self.assertEqual(x["action_type"], y["action_type"])
            np.testing.assert_array_equal(x["action_args"], y["action_args"])

    def test_optional_empty_stack_and_best_design_export(self):
        cfg = deepcopy(self.cfg)
        cfg["min_layers"] = 0
        scratch = (PROJECT_DIR / "outputs").resolve()
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as folder:
            cfg["result_dir"] = folder
            env = ThinFilmEnv(cfg)
            env.reset()
            step = env.step({"action_type": 5})
            self.assertTrue(step.done)
            self.assertEqual(step.info["layer_count"], 0)
            files = list(Path(folder).glob("best_*.json"))
            self.assertEqual(len(files), 1)
            self.assertEqual(json.loads(files[0].read_text(encoding="utf-8"))["T"], step.info["T"])

    def test_bad_grid_and_config(self):
        for value in ([20, 0], [0, 0], [], {"start": 0, "stop": 10, "num": 1}):
            with self.assertRaises(ValueError):
                grid(value, "test")
        for field, value in (("angles_deg", [90]), ("target_transmission", 100), ("min_layers", 51)):
            cfg = deepcopy(self.cfg)
            cfg[field] = value
            with self.assertRaises(ValueError):
                validate(cfg)


if __name__ == "__main__":
    unittest.main()
