"""Optical modes and four/five-material HPPO interfaces; no training updates."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from film_rl.config import DEFAULT_CONFIG, load_config, material_count, optical_data_files, validate
from film_rl.env import ThinFilmEnv
from film_rl.hppo import build_configs, build_model
from film_rl.optics import FilmOptics, read_substrate_index
from fixtures import base_config, write_substrate, yaml_config


class MaterialModeTests(unittest.TestCase):
    def setUp(self):
        scratch = DEFAULT_CONFIG.parents[1] / "outputs"
        scratch.mkdir(exist_ok=True)
        self.folder = tempfile.TemporaryDirectory(prefix="material_mode_test_", dir=scratch)
        assert Path(self.folder.name).resolve().is_relative_to(scratch.resolve())
        self.addCleanup(self.folder.cleanup)
        directory = Path(self.folder.name)
        self.cfg = base_config(directory)
        write_substrate(directory)

    def test_deprecated_lm_configs_fail_with_migration_message(self):
        cfg = deepcopy(self.cfg)
        cfg["material_source"] = "lm"
        with self.assertRaisesRegex(ValueError, "support was removed"):
            validate(cfg)
        del cfg["material_source"]
        cfg["material_files"] = ["legacy.lm"]
        with self.assertRaisesRegex(ValueError, "support was removed"):
            material_count(cfg)

    def test_fixed_constants_and_only_active_optical_files(self):
        optics = FilmOptics(self.cfg)
        np.testing.assert_allclose(optics.material_indices[:, 0], [1.42, 1.696, 1.394, 1.676])
        self.assertEqual(optical_data_files(self.cfg), [Path(self.cfg["material_dir"]) / "substrate.sub"])

    def test_substrate_interpolation_and_range(self):
        path = Path(self.cfg["material_dir"]) / "substrate.sub"
        path.write_text(json.dumps({"wavelength": [190., 200.], "n": [1.4, 1.6], "k": [.01, .03]}), encoding="utf-8")
        np.testing.assert_allclose(read_substrate_index(path, np.asarray([193., 197.])), [1.46+.016j, 1.54+.024j])
        with self.assertRaisesRegex(ValueError, "no extrapolation"):
            read_substrate_index(path, np.asarray([189.]))
        with self.assertRaisesRegex(ValueError, ".sub"):
            read_substrate_index(path.with_suffix(".lm"), np.asarray([193.]))

    def test_fixed_data_reject_other_wavelengths(self):
        for wavelengths in ([193., 200.], [193., 193.+1e-10], [193.4], [550.]):
            cfg = deepcopy(self.cfg)
            cfg["wavelengths_nm"] = wavelengths
            with self.subTest(wavelengths=wavelengths), self.assertRaisesRegex(ValueError, "only at 193 nm"):
                validate(cfg)

    def test_invalid_fixed_data_are_rejected(self):
        for key, value in (("n", 0), ("n", float("nan")), ("n", True), ("k", -1), ("k", None)):
            cfg = deepcopy(self.cfg)
            cfg["fixed_materials_193nm"][0][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate(cfg)
        cfg = deepcopy(self.cfg)
        cfg["fixed_materials_193nm"][1]["name"] = cfg["fixed_materials_193nm"][0]["name"]
        with self.assertRaisesRegex(ValueError, "distinct"):
            validate(cfg)
        cfg["material_source"] = "typo"
        with self.assertRaisesRegex(ValueError, "material_source"):
            validate(cfg)

    def test_four_material_env_terminal_reward_and_metadata(self):
        env = ThinFilmEnv(self.cfg)
        self.addCleanup(env.close)
        initial = env.reset()
        self.assertEqual(initial.shape, (101,))
        self.assertEqual(env.action_space["action_type"].n, 5)
        self.assertEqual(env.action_space["action_args"].shape, (4,))
        self.assertEqual(env.EOS, 4)
        with self.assertRaises(ValueError):
            env.step({"action_type": 4})
        for kind in [0, 1, 2, 3]:
            step = env.step({"action_type": kind, "action_args": np.zeros(4)})
            self.assertEqual(float(step.reward[0]), 0.)
            self.assertFalse(step.done)
            encoded = step.obs[:-1].reshape(env.max_layers, 5)
            self.assertEqual(encoded[kind, kind], 1.)
            self.assertEqual(encoded[kind, 4], .5)
        self.assertEqual(env.optics.evaluation_count, 0)
        step = env.step({"action_type": 4})
        self.assertTrue(step.done)
        self.assertEqual(env.optics.evaluation_count, 1)
        self.assertEqual(step.info["material_source"], "fixed_193nm")
        self.assertEqual([layer["material"] for layer in step.info["layers"]], ["MgF2", "LaF3", "AlF3", "GdF3"])
        self.assertAlmostEqual(float(step.reward[0]), step.info["reward"], places=6)
        self.assertEqual(step.info["eval_episode_return"], step.info["score"])
        json.dumps(step.info)

    def test_env_and_model_shapes_masks_for_four_and_five_materials(self):
        for cfg in (self.cfg, yaml_config(self.folder.name, count=4), yaml_config(self.folder.name, count=5)):
            count = material_count(cfg)
            with self.subTest(source=cfg["material_source"], count=count):
                env = ThinFilmEnv(cfg)
                self.addCleanup(env.close)
                main, _ = build_configs(cfg)
                self.assertEqual(main.policy.model.obs_shape, env.observation_space.shape[0])
                model = build_model(cfg, main)
                empty = env.reset()
                occupied = env.step({"action_type": count-1, "action_args": np.zeros(count)}).obs
                states = torch.from_numpy(np.stack([empty, occupied] * 64))
                with torch.no_grad():
                    result = model(states, "compute_actor_critic")
                logits = result["logit"]
                self.assertEqual(tuple(logits["action_type"].shape), (128, count+1))
                self.assertEqual(tuple(logits["action_args"]["mu"].shape), (128, count))
                self.assertTrue((logits["action_type"][::2, count] < -1e8).all())
                self.assertTrue((logits["action_type"][1::2, count-1] < -1e8).all())
                self.assertTrue((logits["action_type"][1::2, count] > -1e8).all())
                for _ in range(20):
                    action = env.random_action()
                    self.assertNotEqual(action["action_type"], count-1)
                    self.assertEqual(action["action_args"].shape, (count,))

    def test_invalid_actions_and_repeat_leave_state_unchanged(self):
        env = ThinFilmEnv(self.cfg)
        env.reset()
        for action in ({"action_type": 5}, {"action_type": 0, "action_args": np.zeros(5)}):
            with self.assertRaises(ValueError):
                env.step(action)
            self.assertEqual(env.layers, [])
        env.step({"action_type": 3, "action_args": np.zeros(4)})
        before = list(env.layers)
        with self.assertRaises(ValueError):
            env.step({"action_type": 3, "action_args": np.zeros(4)})
        self.assertEqual(env.layers, before)

    def test_fixed_tmm_matches_quarter_wave_analytic_result(self):
        cfg = deepcopy(self.cfg)
        cfg["angles_deg"] = [0.]
        n1 = cfg["fixed_materials_193nm"][0]["n"]
        result = FilmOptics(cfg).evaluate([(0, 193./(4*n1))])
        expected_r = ((1.5-n1*n1)/(1.5+n1*n1))**2
        self.assertAlmostEqual(result["R"][0][0], expected_r, places=12)
        self.assertAlmostEqual(result["T"][0][0], 1-expected_r, places=12)

    def test_explicit_absorption_and_budget_are_preserved(self):
        cfg = deepcopy(self.cfg)
        cfg["fixed_materials_193nm"][0]["k"] = .001
        result = FilmOptics(cfg).evaluate([(0, 50.)])
        self.assertGreater(min(row[0] for row in result["A"]), 0)
        cfg["max_total_thickness_nm"] = cfg["thickness_min_nm"]
        env = ThinFilmEnv(cfg)
        env.reset()
        step = env.step({"action_type": 0, "action_args": np.zeros(4)})
        self.assertTrue(step.done)
        self.assertEqual(step.info["termination_reason"], "thickness_budget")
        self.assertEqual(step.info["total_thickness_nm"], 5.)

    def test_seed_batch_snapshots_fixed_values_and_hashes_only_active_files(self):
        import run_seed_batch as batch_runner
        project = Path(self.folder.name) / "batch_project"
        (project / "film_rl").mkdir(parents=True)
        for name in ("plot_training.py", "run_seed_batch.py", "summarize_seed_batch.py"):
            (project / name).write_text("# test fixture\n", encoding="utf-8")
        cfg = deepcopy(self.cfg)
        path = project / "config.json"
        path.write_text(json.dumps(cfg), encoding="utf-8")
        with patch.object(batch_runner, "PROJECT", project):
            batch = batch_runner.prepare(config=path, group="fixed193_test", seeds=[0, 9])
        manifest = json.loads((batch / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(len(manifest["material_hashes"]), 1)
        self.assertTrue(next(iter(manifest["material_hashes"])).endswith("substrate.sub"))
        for entry in manifest["runs"]:
            snapshot = load_config(entry["config"])
            self.assertEqual(snapshot["fixed_materials_193nm"], cfg["fixed_materials_193nm"])
            self.assertEqual(material_count(snapshot), 4)
            self.assertFalse(Path(entry["run_dir"]).exists())


if __name__ == "__main__":
    unittest.main()
