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
from film_rl.optics import FilmOptics, read_index


class MaterialModeTests(unittest.TestCase):
    def setUp(self):
        scratch = DEFAULT_CONFIG.parents[1] / "outputs"
        scratch.mkdir(exist_ok=True)
        self.folder = tempfile.TemporaryDirectory(prefix="material_mode_test_", dir=scratch)
        assert Path(self.folder.name).resolve().is_relative_to(scratch.resolve())
        self.addCleanup(self.folder.cleanup)
        directory = Path(self.folder.name)
        self.cfg = load_config()
        # Stable synthetic fixture; user-selected default indices may change.
        self.cfg["fixed_materials_193nm"] = [
            {"name": name, "n": n, "k": 0.0}
            for name, n in zip(["MgF2", "LaF3", "AlF3", "GdF3"], [1.42, 1.696, 1.394, 1.676])
        ]
        self.cfg["material_dir"] = str(directory)
        self.cfg.pop("result_dir", None)
        self.cfg["substrate_file"] = "substrate.sub"
        self.cfg["material_files"] = [f"coating{i}.lm" for i in range(5)]
        for i, name in enumerate(self.cfg["material_files"]):
            (directory / name).write_text(json.dumps({
                "wavelength": [190., 195., 200.],
                "n": [1.4 + .05*i, 1.5 + .05*i, 1.6 + .05*i],
                "k": [.01, .02, .03],
            }), encoding="utf-8")
        (directory / "substrate.sub").write_text(json.dumps({
            "wavelength": [190., 200.], "n": [1.5, 1.5], "k": [0., 0.],
        }), encoding="utf-8")

    def lm_config(self, count=5):
        cfg = deepcopy(self.cfg)
        cfg["material_source"] = "lm"
        cfg["material_files"] = cfg["material_files"][:count]
        return cfg

    def test_default_values_and_unchanged_training_settings(self):
        fixed = load_config()
        legacy = load_config(DEFAULT_CONFIG.with_name("duv193_lm.json"))
        self.assertEqual([e["name"] for e in fixed["fixed_materials_193nm"]], ["MgF2", "LaF3", "AlF3", "GdF3"])
        manifest = json.loads((DEFAULT_CONFIG.parents[1] / "outputs/source_copy_manifest.json").read_text(encoding="utf-8"))
        snapshot = deepcopy(manifest["source_config"])
        snapshot["material_dir"] = str(Path(snapshot["material_dir"]).resolve())
        self.assertEqual(fixed, snapshot)
        for key in legacy:
            if key != "material_source":
                self.assertEqual(fixed[key], legacy[key], key)

    def test_fixed_values_do_not_read_inactive_lm_files(self):
        cfg = deepcopy(self.cfg)
        cfg["material_files"] = ["does_not_exist.lm"]
        optics = FilmOptics(cfg)
        np.testing.assert_allclose(optics.material_indices[:, 0], [1.42, 1.696, 1.394, 1.676])
        self.assertEqual(optical_data_files(cfg), ["substrate.sub"])

    def test_file_mode_interpolates_both_n_and_k(self):
        cfg = self.lm_config()
        cfg["wavelengths_nm"] = [193., 197.]
        optics = FilmOptics(cfg)
        np.testing.assert_allclose(optics.material_indices[0], [1.46+.016j, 1.54+.024j])
        self.assertEqual(material_count(cfg), 5)
        self.assertEqual(len(optical_data_files(cfg)), 6)
        with self.assertRaisesRegex(ValueError, "no extrapolation"):
            read_index(Path(cfg["material_dir"]) / "coating0.lm", np.array([189.]))

    def test_missing_source_preserves_legacy_interpolation(self):
        cfg = self.lm_config()
        del cfg["material_source"]
        self.assertEqual(material_count(cfg), 5)
        self.assertEqual(FilmOptics(cfg).material_source, "lm")
        self.assertEqual(ThinFilmEnv(cfg).EOS, 5)

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

    def test_four_material_env_step_delta_and_metadata(self):
        env = ThinFilmEnv(self.cfg)
        self.addCleanup(env.close)
        initial = env.reset()
        self.assertEqual(initial.shape, (101,))
        self.assertEqual(env.action_space["action_type"].n, 5)
        self.assertEqual(env.action_space["action_args"].shape, (4,))
        self.assertEqual(env.EOS, 4)
        with self.assertRaises(ValueError):
            env.step({"action_type": 4})
        previous = env._baseline_mixed_score
        rewards = []
        for kind in [0, 1, 2, 3]:
            step = env.step({"action_type": kind, "action_args": np.zeros(4)})
            self.assertAlmostEqual(float(step.reward[0]), step.info["mixed_score"]-previous, places=7)
            previous = step.info["mixed_score"]
            rewards.append(float(step.reward[0]))
            self.assertFalse(step.done)
            encoded = step.obs[:-1].reshape(env.max_layers, 5)
            self.assertEqual(encoded[kind, kind], 1.)
            self.assertEqual(encoded[kind, 4], .5)
        self.assertEqual(env.optics.evaluation_count, 5)
        step = env.step({"action_type": 4})
        self.assertTrue(step.done)
        self.assertEqual(env.optics.evaluation_count, 6)
        self.assertEqual(step.info["material_source"], "fixed_193nm")
        self.assertEqual([layer["material"] for layer in step.info["layers"]], ["MgF2", "LaF3", "AlF3", "GdF3"])
        self.assertEqual(float(step.reward[0]), 0.)
        self.assertAlmostEqual(sum(rewards), step.info["mixed_score"]-step.info["baseline_mixed_score"], places=7)
        self.assertEqual(step.info["eval_episode_return"], step.info["score"])
        json.dumps(step.info)

    def test_env_and_model_shapes_masks_for_four_and_five_materials(self):
        for cfg in (self.cfg, self.lm_config(4), self.lm_config(5)):
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

    def test_historical_audit_does_not_inherit_fixed_source(self):
        from diagnostics.audit_multiangle import restore_env
        old = self.lm_config()
        del old["material_source"]
        restored = restore_env({"env": old})
        self.assertEqual(restored["material_source"], "lm")
        self.assertEqual(material_count(restored), 5)
        np.testing.assert_allclose(FilmOptics(restored).material_indices[0], [1.46+.016j])

    def test_seed_batch_snapshots_fixed_values_and_hashes_only_active_files(self):
        import run_seed_batch as batch_runner
        project = Path(self.folder.name) / "batch_project"
        (project / "film_rl").mkdir(parents=True)
        for name in ("plot_training.py", "run_seed_batch.py", "summarize_seed_batch.py"):
            (project / name).write_text("# test fixture\n", encoding="utf-8")
        cfg = deepcopy(self.cfg)
        cfg["material_files"] = ["missing_inactive.lm"]
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
