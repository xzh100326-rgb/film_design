"""R/T/A target spectra: hand-calculated losses, passive physics and pipeline IO."""

import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from film_rl.config import DEFAULT_CONFIG, load_config, validate
from film_rl.env import ThinFilmEnv
from fixtures import yaml_config
from film_rl.optics import FilmOptics
from film_rl.targets import SpectralObjective, objective_for_config


class SpectralTargetTests(unittest.TestCase):
    def setUp(self):
        scratch = DEFAULT_CONFIG.parents[1] / "outputs"
        scratch.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="spectral_target_test_", dir=scratch)
        self.root = Path(self.temp.name).resolve()
        assert self.root.is_relative_to(scratch.resolve())
        self.addCleanup(self.temp.cleanup)
        self.cfg = yaml_config(self.root, optical_constants=[("clear", 1.4, 0.), ("lossy", 2., .3)])
        self.cfg.update(substrate_file="clear.sub", wavelengths_nm=[400., 500., 600.], angles_deg=[0., 45.])
        (self.root / "substrate.sub").rename(self.root / "clear.sub")
        self.targets = [
            {"quantity": "R", "wavelength_range_nm": [400., 400.], "target": 1., "weight": 1.},
            {"quantity": "A", "wavelength_range_nm": [500., 500.], "target": 1., "weight": 2.},
            {"quantity": "T", "wavelength_range_nm": [600., 600.], "target": 1., "weight": 1.}]

    def objective(self, targets):
        return SpectralObjective({"spectral_targets": targets}, [400., 500., 600.], [0., 45.])

    def test_weighted_rta_error_matches_manual_calculation(self):
        objective = self.objective(self.targets)
        r = np.asarray([[.8, .1, .1], [.8, .1, .1]])
        t = np.asarray([[.1, .2, .9], [.1, .2, .9]])
        result = objective.evaluate(r, t, 1-r-t)
        # R error .2, A error .3, T error .1; weights 1:2:1.
        self.assertAlmostEqual(result["mae"], .225)
        self.assertAlmostEqual(result["score"], .775)
        self.assertEqual(result["evaluation_metric"], "score")
        np.testing.assert_allclose([x["normalized_weight"] for x in result["target_errors"]], [.25, .5, .25])

    def test_generic_s_interface_uses_same_formula_for_r_t_a(self):
        for quantity in "RTA":
            with self.subTest(quantity=quantity):
                objective = self.objective([{"quantity": quantity, "target": 1}])
                result = objective.evaluate_spectra({quantity: np.full((2, 3), .7)})
                self.assertAlmostEqual(result["score"], .7)
                self.assertEqual(result["spectral_aggregation"], "weighted_band_mae")
                self.assertIn("S_b - S_target_b", result["score_formula"])

    def test_generic_mixed_interface_matches_legacy_signature(self):
        objective = self.objective(self.targets)
        r = np.full((2, 3), .1)
        t = np.full((2, 3), .2)
        self.assertEqual(objective.evaluate_spectra({"R": r, "T": t, "A": 1-r-t}),
                         objective.evaluate(r, t, 1-r-t))
        with self.assertRaisesRegex(ValueError, "Missing target response"):
            objective.evaluate_spectra({"R": r})
        with self.assertRaisesRegex(ValueError, "generic symbol"):
            objective.evaluate_spectra({"S": r})

    def test_pointwise_spectrum_error_does_not_collapse_to_band_means(self):
        objective = self.objective([{"quantity": "R", "target": [0., 1., 0.]}])
        actual = np.asarray([[1., 0., 0.], [1., 0., 0.]])
        result = objective.evaluate_spectra({"R": actual})
        # Same target/actual mean, but opposite spectral positions still fail.
        self.assertAlmostEqual(result["target_errors"][0]["mean_actual"], 1/3)
        self.assertAlmostEqual(result["mae"], 2/3)
        self.assertAlmostEqual(result["score"], 1/3)

    def test_sample_count_weights_recover_full_grid_s_mae(self):
        objective = self.objective([
            {"quantity": "R", "wavelength_range_nm": [400, 500], "target": 1., "weight": 2.},
            {"quantity": "T", "wavelength_range_nm": [600, 600], "target": 1., "weight": 1.}])
        result = objective.evaluate_spectra({"R": np.full((2,3), .8), "T": np.full((2,3), .4)})
        self.assertAlmostEqual(result["mae"], (.2+.2+.6)/3)

    def test_each_band_is_averaged_before_applying_its_weight(self):
        objective = self.objective([
            {"quantity": "R", "wavelength_range_nm": [400, 500], "target": 1},
            {"quantity": "T", "wavelength_range_nm": [600, 600], "target": 1}])
        r = np.asarray([[.8, .8, 0], [.8, .8, 0]])
        t = np.asarray([[0, 0, .4], [0, 0, .4]])
        result = objective.evaluate(r, t, 1-r-t)
        self.assertAlmostEqual(result["mae"], .4)  # (.2+.6)/2, not weighted by band width

    def test_unscored_gap_does_not_affect_reward(self):
        objective = self.objective([{"quantity": "R", "wavelength_range_nm": [400, 400], "target": 1}])
        r = np.asarray([[.8, 0., 0.], [.8, 1., 1.]])
        t = np.zeros_like(r)
        self.assertAlmostEqual(objective.evaluate(r, t, 1-r-t)["score"], .8)

    def test_arbitrary_curve_and_angle_dependent_target(self):
        target = [[.2, .4, .6], [.3, .5, .7]]
        objective = self.objective([{"quantity": "R", "target": target}])
        r = np.asarray(target)
        result = objective.evaluate(r, 1-r, np.zeros_like(r))
        self.assertAlmostEqual(result["score"], 1.)
        self.assertEqual(result["evaluation_metric"], "score")
        self.assertEqual(result["target_errors"][0]["target"], target)

    def test_pure_absorption_and_reflection_metric(self):
        r = np.full((2, 3), .1)
        t = np.full((2, 3), .2)
        for quantity, expected, label in [("A", .7, "absorption"), ("R", .1, "reflection"), ("T", .2, "transmission")]:
            with self.subTest(quantity=quantity):
                result = self.objective([{"quantity": quantity, "target": 1}]).evaluate(r, t, 1-r-t)
                self.assertAlmostEqual(result["score"], expected)
                self.assertEqual(result["evaluation_metric"], label)

    def test_legacy_transmission_is_numerically_identical(self):
        cfg = {"target_transmission": [.9, .8, .7]}
        objective = SpectralObjective(cfg, [400, 500, 600], [0, 45])
        r = np.full((2, 3), .1)
        t = np.asarray([[.8, .7, .6], [.9, .9, .5]])
        result = objective.evaluate(r, t, 1-r-t)
        self.assertEqual(result["mae"], float(np.mean(np.abs(t-np.asarray(cfg["target_transmission"])))))
        self.assertEqual(result["objective_mode"], "target_transmission")

    def test_new_targets_take_precedence_over_legacy_field(self):
        cfg = {"target_transmission": 1, "spectral_targets": [{"quantity": "A", "target": 1}]}
        objective = SpectralObjective(cfg, [400, 500, 600], [0, 45])
        self.assertEqual(objective.evaluation_metric, "absorption")
        self.assertIsNone(objective.legacy_target)

    def test_conflicting_overlapping_targets_are_rejected(self):
        cases = [
            [{"quantity": "R", "target": 1}, {"quantity": "A", "target": 1}],
            [{"quantity": "R", "target": .9}, {"quantity": "R", "target": .8}],
            [{"quantity": "R", "target": .1}, {"quantity": "T", "target": .2}, {"quantity": "A", "target": .3}],
            [{"quantity": "R", "wavelength_range_nm": [400, 500], "target": 1},
             {"quantity": "T", "wavelength_range_nm": [500, 600], "target": 1}]]
        for targets in cases:
            with self.subTest(targets=targets), self.assertRaises(ValueError):
                self.objective(targets)

    def test_consistent_rta_targets_can_overlap(self):
        objective = self.objective([{"quantity": "R", "target": .2}, {"quantity": "T", "target": .3},
                                    {"quantity": "A", "target": .5}])
        r = np.full((2, 3), .2)
        t = np.full((2, 3), .3)
        self.assertAlmostEqual(objective.evaluate(r, t, 1-r-t)["score"], 1.)

    def test_invalid_weights_ranges_targets_and_names(self):
        cases = [[], [{"quantity": "X", "target": 1}], [{"quantity": "R", "target": 1, "weight": 0}],
                 [{"quantity": "R", "target": 1, "weight": -1}],
                 [{"quantity": "R", "target": 1, "weight": float("inf")}],
                 [{"quantity": "R", "target": [1, 1]}], [{"quantity": "R", "target": 1.1}],
                 [{"quantity": "R", "target": 1, "wavelength_range_nm": [300, 500]}],
                 [{"quantity": "R", "target": 1, "wavelength_range_nm": [410, 490]}],
                 [{"quantity": "R", "target": 1, "weigth": 2}]]
        for targets in cases:
            with self.subTest(targets=targets), self.assertRaises(ValueError):
                self.objective(targets)

    def test_full_tmm_mixed_targets_preserve_log_shaping(self):
        self.cfg["spectral_targets"] = self.targets
        self.cfg.pop("target_transmission")
        validate(self.cfg)
        result = FilmOptics(self.cfg).evaluate([(1, 50), (0, 60)])
        manual = sum(x["normalized_weight"]*x["mae"] for x in result["target_errors"])
        self.assertAlmostEqual(result["mae"], manual)
        expected_log = -np.log10(max(manual, self.cfg["reward_epsilon"]))/-np.log10(self.cfg["reward_epsilon"])
        expected = (1-self.cfg["reward_alpha"])*(1-manual)+self.cfg["reward_alpha"]*expected_log
        self.assertAlmostEqual(result["reward"], expected)
        self.assertIsNone(result["target_transmission"])
        self.assertGreater(result["mean_absorption"], 0)
        np.testing.assert_allclose(np.asarray(result["R"])+result["T"]+result["A"], 1, atol=1e-10)
        json.dumps(result)

    def test_absorption_excludes_power_entering_absorbing_substrate(self):
        (self.root / "clear.sub").write_text(json.dumps({"wavelength": [400., 2000.],
                                                         "n": [1.5, 1.5], "k": [.1, .1]}), encoding="utf-8")
        self.cfg["angles_deg"] = [0.]
        self.cfg["spectral_targets"] = [{"quantity": "A", "target": 1.}]
        result = FilmOptics(self.cfg).evaluate([])
        self.assertLess(abs(result["mean_absorption"]), 1e-12)
        self.assertGreater(result["mean_transmission"], .9)
        self.assertLess(abs(result["score"]), 1e-12)

    def test_environment_and_best_design_use_composite_score(self):
        self.cfg["spectral_targets"] = self.targets
        self.cfg["reward_alpha"] = 0.
        self.cfg["result_dir"] = str(self.root / "designs")
        env = ThinFilmEnv(self.cfg)
        self.addCleanup(env.close)
        env.reset()
        first = env.step({"action_type": 1, "action_args": np.zeros(2)})
        self.assertEqual(float(first.reward[0]), 0)
        final = env.step({"action_type": 2})
        self.assertTrue(final.done)
        self.assertEqual(final.info["eval_episode_return"], final.info["score"])
        self.assertAlmostEqual(float(final.reward[0]), final.info["score"], places=7)
        self.assertEqual(final.info["reward_formula"], final.info["score_formula"])
        self.assertEqual(env.optics.evaluation_count, 1)
        self.assertNotAlmostEqual(final.info["score"], final.info["mean_transmission"])
        saved = json.loads(next((self.root / "designs").glob("best_*.json")).read_text(encoding="utf-8"))
        self.assertEqual(saved["evaluation_metric"], "score")
        self.assertEqual(len(saved["target_errors"]), 3)
        self.assertEqual(saved["design_format"], "compact_v1")
        self.assertEqual(saved["layers"], final.info["layers"])
        self.assertEqual(saved["score"], final.info["score"])
        self.assertEqual(saved["score_formula"], final.info["score_formula"])
        self.assertEqual(saved["reward_formula"], final.info["reward_formula"])
        self.assertNotIn("R", saved)
        self.assertNotIn("spectral_targets", saved)

    def test_batch_snapshot_uses_score_plot_label_for_mixed_targets(self):
        import run_seed_batch as batch_runner
        project = self.root / "batch_project"
        (project / "film_rl").mkdir(parents=True)
        for name in ("plot_training.py", "run_seed_batch.py", "summarize_seed_batch.py"):
            (project / name).write_text("# fixture\n", encoding="utf-8")
        self.cfg["spectral_targets"] = self.targets
        self.cfg.pop("target_transmission")
        path = project / "config.json"
        path.write_text(json.dumps(self.cfg), encoding="utf-8")
        with patch.object(batch_runner, "PROJECT", project), patch("sys.stdout", new_callable=io.StringIO):
            batch = batch_runner.prepare(config=path, seeds=[0])
        manifest = json.loads((batch / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["runs"][0]["evaluation_metric"], "score")
        self.assertFalse(manifest["runs"][0]["reward_as_transmission"])
        self.assertIn("targetsRTA", manifest["group"])

    def test_config_loads_generic_s_and_legacy_targets_independently(self):
        # Experiment JSONs may be renamed/archived; test the real config loader
        # against controlled inputs rather than depending on those mutable files.
        self.cfg["spectral_targets"] = self.targets
        generic = self.root / "generic_s.json"
        generic.write_text(json.dumps(self.cfg), encoding="utf-8")
        loaded = load_config(generic)
        self.assertEqual(objective_for_config(loaded).evaluation_metric, "score")
        self.assertEqual(len(objective_for_config(loaded).wave), 3)
        self.cfg.pop("spectral_targets")
        legacy = self.root / "legacy_t.json"
        legacy.write_text(json.dumps(self.cfg), encoding="utf-8")
        self.assertNotIn("spectral_targets", load_config(legacy))
        self.assertEqual(objective_for_config(load_config(legacy)).evaluation_metric, "transmission")

    def test_plot_auto_labels_and_rejects_mislabelled_transmission(self):
        from torch.utils.tensorboard import SummaryWriter
        run = self.root / "synthetic_plot_interface_check"
        (run / "designs").mkdir(parents=True)
        (run / "designs/best_fixture.json").write_text(json.dumps({"evaluation_metric": "score"}), encoding="utf-8")
        writer = SummaryWriter(str(run / "log/serial"))
        try:
            for step in [0, 100]:
                writer.add_scalar("evaluator_iter/reward_mean", .5+step/1000, step)
                writer.add_scalar("evaluator_iter/reward_max", .6+step/1000, step)
        finally:
            writer.close()
        command = [sys.executable, str(DEFAULT_CONFIG.parents[1] / "plot_training.py"), "--run", str(run),
                   "--output", str(self.root / "plot.png")]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        self.assertTrue((self.root / "plot.png").is_file())
        rejected = subprocess.run(command+["--reward-as-transmission"], capture_output=True, text=True)
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn("cannot label", rejected.stderr)


if __name__ == "__main__":
    unittest.main()
