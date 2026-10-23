"""Offline dataset selection, wavelength/absorption handling and real TMM; no training."""

from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import yaml

from film_rl.config import DEFAULT_CONFIG, load_config, material_count, optical_data_files, validate
from film_rl.optics import FilmOptics
from fixtures import base_config
from film_rl.refractiveindex import candidates, read_refractiveindex, resolve_materials


class RefractiveIndexTests(unittest.TestCase):
    def setUp(self):
        scratch = DEFAULT_CONFIG.parents[1] / "outputs"
        scratch.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="rii_test_", dir=scratch)
        assert Path(self.temp.name).resolve().is_relative_to(scratch.resolve())
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.database = self.root / "repo" / "database"
        self.database.mkdir(parents=True)
        self.cfg = base_config(self.root)
        self.cfg.update(material_source="refractiveindex", materials=[{"name": "MgF2"}, {"name": "LaF3"}],
                        refractiveindex_dir=str(self.database.parent), material_dir=str(self.root),
                        substrate_file="substrate.sub", wavelengths_nm=[190., 193., 200.], angles_deg=[0., 45.])
        (self.root / "substrate.sub").write_text(json.dumps({
            "wavelength": [180., 210.], "n": [1.5, 1.5], "k": [0., 0.]}), encoding="utf-8")
        self.a = self.make("MgF2", "a.yml")
        self.b = self.make("MgF2", "b.yaml", n=1.6)
        self.c = self.make("LaF3", "only.yml", n=1.7)

    def make(self, name, version, n=1.4, blocks=None):
        path = self.database / "data" / "main" / name / "nk" / version
        path.parent.mkdir(parents=True, exist_ok=True)
        document = {"COMMENTS": "test film; source conditions", "REFERENCES": "fixture DOI",
                    "DATA": blocks or [{"type": "tabulated nk",
                                        "data": f"0.190 {n} 0.001\n0.200 {n+.1} 0.003\n"}]}
        path.write_text(yaml.safe_dump(document), encoding="utf-8")
        return path

    def test_nm_conversion_and_complex_interpolation(self):
        index, meta = read_refractiveindex(self.a, [190, 193, 200])
        np.testing.assert_allclose(index, [1.4+.001j, 1.43+.0016j, 1.5+.003j])
        self.assertEqual(meta["file"], str(self.a.resolve()))
        self.assertEqual(len(meta["sha256"]), 64)
        self.assertFalse(meta["assumed_zero_k"])

    def test_separate_n_k_grids_and_no_extrapolation(self):
        path = self.make("Separate", "nk.yml", blocks=[
            {"type": "tabulated n", "data": "0.18 1.4\n0.22 1.8\n"},
            {"type": "tabulated k", "data": "0.19 0.01\n0.20 0.02\n"}])
        result, _ = read_refractiveindex(path, [195])
        np.testing.assert_allclose(result, [1.55+.015j])
        with self.assertRaisesRegex(ValueError, "k coverage.*no extrapolation"):
            read_refractiveindex(path, [205])
        with self.assertRaisesRegex(ValueError, "no extrapolation"):
            read_refractiveindex(self.a, [189])

    def test_formula_and_explicit_missing_k_assumption(self):
        path = self.make("Formula", "li.yml", blocks=[
            {"type": "formula 1", "wavelength_range": "0.18 0.3", "coefficients": "0.1 0.5 0.1"}])
        with self.assertRaisesRegex(ValueError, "k is not provided"):
            read_refractiveindex(path, [200])
        index, meta = read_refractiveindex(path, [200, 250], assume_zero_k=True)
        self.assertAlmostEqual(index[0].real, np.sqrt(1.1 + 2/3))
        self.assertTrue(meta["assumed_zero_k"])
        self.assertTrue(np.all(index.imag == 0))

    def test_all_database_dispersion_formulas_at_known_points(self):
        # Simple independently calculable cases; lambda=.2 um, n=1.5.
        cases = {1: "1.25", 2: "1.25", 3: "2.25", 4: "2.25 0 0 1 1 0 0 1 1",
                 5: "1.5", 6: "0.5", 7: "1.5 0 0 0 0 0", 8: "0.29411764705882354 0 1 0",
                 9: "2.25 0 1 0 1 1"}
        for number, coefficients in cases.items():
            with self.subTest(number=number):
                path = self.make(f"Formula{number}", "data.yml", blocks=[{
                    "type": f"formula {number}", "wavelength_range": "0.19 0.3", "coefficients": coefficients}])
                index, _ = read_refractiveindex(path, [200], assume_zero_k=True)
                self.assertAlmostEqual(index[0].real, 1.5)

    def test_formula_with_k_and_range_is_checked_separately(self):
        path = self.make("FormulaK", "data.yml", blocks=[
            {"type": "formula 5", "wavelength_range": "0.19 0.21", "coefficients": "1.5 0.01 -2"},
            {"type": "tabulated k", "data": "0.19 0.001\n0.20 0.003"}])
        index, _ = read_refractiveindex(path, [200])
        self.assertAlmostEqual(index[0].real, 1.75)
        self.assertAlmostEqual(index[0].imag, .003)
        with self.assertRaisesRegex(ValueError, "k coverage"):
            read_refractiveindex(path, [205])

    def test_single_wavelength_measurement_is_not_extrapolated(self):
        path = self.make("Single", "data.yml", blocks=[{"type": "tabulated nk", "data": "0.193 1.394 0.001"}])
        index, _ = read_refractiveindex(path, [193])
        np.testing.assert_allclose(index, [1.394+.001j])
        with self.assertRaisesRegex(ValueError, "no extrapolation"):
            read_refractiveindex(path, [193, 194])

    def test_herzberger_without_trailing_zero(self):
        path = self.make("Herzberger", "data.yml", blocks=[{
            "type": "formula 7", "wavelength_range": "0.19 0.3", "coefficients": "1.5 0 0 2 0"}])
        index, _ = read_refractiveindex(path, [200], assume_zero_k=True)
        self.assertAlmostEqual(index[0].real, 1.58)

    def test_multiple_versions_require_user_selection(self):
        with self.assertRaisesRegex(ValueError, "multiple YAML versions"):
            resolve_materials(self.cfg)
        with patch("sys.stdout", new_callable=io.StringIO):
            selected = resolve_materials(self.cfg, interactive=True, input_fn=lambda prompt: "2")
        self.assertTrue(selected["materials"][0]["file"].endswith("b.yaml"))
        self.assertTrue(selected["materials"][1]["file"].endswith("only.yml"))
        self.assertNotIn("file", self.cfg["materials"][0])

    def test_invalid_choice_retries_and_pinned_choice_never_prompts(self):
        answers = iter(["bad", "99", "1"])
        with patch("sys.stdout", new_callable=io.StringIO):
            cfg = resolve_materials(self.cfg, interactive=True, input_fn=lambda prompt: next(answers))
        with patch("builtins.input", side_effect=AssertionError("must not prompt")):
            self.assertEqual(resolve_materials(cfg), cfg)
        cfg["materials"][0]["file"] = "b.yaml"
        self.assertTrue(resolve_materials(cfg)["materials"][0]["file"].endswith("b.yaml"))

    def test_missing_alf3_fails_before_any_prompt(self):
        self.cfg["materials"].append({"name": "AlF3"})
        with self.assertRaisesRegex(FileNotFoundError, "AlF3"), patch("builtins.input", side_effect=AssertionError):
            resolve_materials(self.cfg, interactive=True)

    def test_external_yaml_can_supply_missing_material(self):
        self.cfg["materials"].append({"name": "AlF3", "file": str(self.c)})
        self.cfg["materials"][0]["file"] = str(self.a)
        cfg = resolve_materials(self.cfg)
        self.assertEqual(material_count(cfg), 3)
        self.assertEqual(len(optical_data_files(cfg)), 4)

    def test_persist_only_chosen_entries_and_preserve_relative_paths(self):
        raw = deepcopy(self.cfg)
        raw["refractiveindex_dir"] = "repo"
        raw["material_dir"] = "."
        path = self.root / "config.json"
        path.write_text(json.dumps(raw), encoding="utf-8")
        with patch("builtins.input", return_value="1"), patch("sys.stdout", new_callable=io.StringIO):
            cfg = load_config(path, resolve_materials=True, interactive=True, persist_selection=True)
        saved = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(saved["material_dir"], ".")
        self.assertEqual(saved["refractiveindex_dir"], "repo")
        self.assertEqual(saved["training"], raw["training"])
        self.assertEqual(saved["materials"], cfg["materials"])
        before = path.read_bytes()
        load_config(path, resolve_materials=True, persist_selection=True)
        self.assertEqual(before, path.read_bytes())

    def test_real_tmm_multiangle_multiwavelength_and_provenance(self):
        self.cfg["materials"][0]["file"] = "a.yml"
        cfg = resolve_materials(self.cfg)
        optics = FilmOptics(cfg)
        result = optics.evaluate([(0, 35), (1, 30)])
        self.assertEqual(np.asarray(result["T"]).shape, (2, 3))
        self.assertTrue(np.isfinite(result["T"]).all())
        self.assertGreater(np.min(result["A"]), 0)
        self.assertEqual(result["material_source"], "refractiveindex")
        self.assertEqual(result["material_datasets"][0]["sha256"], optics.material_datasets[0]["sha256"])
        self.assertEqual(result["layers"][0]["material"], "MgF2")
        self.assertAlmostEqual(result["score"], np.mean(result["T"]))
        json.dumps(result)

    def test_yaml_edits_invalidate_cached_data(self):
        first, meta = read_refractiveindex(self.a, [193])
        self.make("MgF2", "a.yml", n=1.8)
        second, changed = read_refractiveindex(self.a, [193])
        self.assertGreater(second[0].real, first[0].real)
        self.assertNotEqual(meta["sha256"], changed["sha256"])

    def test_pinned_paths_do_not_scan_database(self):
        self.cfg["materials"][0]["file"] = str(self.a)
        self.cfg["materials"][1]["file"] = str(self.c)
        with patch.object(Path, "rglob", side_effect=AssertionError("Pinned paths must not scan")):
            selected = resolve_materials(self.cfg)
        self.assertEqual(material_count(selected), 2)

    def test_new_materials_appear_without_restart(self):
        entry = {"name": "AddedLater"}
        self.assertEqual(candidates(self.cfg, entry), [])
        path = self.make("AddedLater", "new.yml")
        self.assertEqual(candidates(self.cfg, entry), [path.resolve()])

    def test_all_unavailable_versions_fail_before_prompt(self):
        self.cfg["wavelengths_nm"] = [300.]
        with patch("sys.stdout", new_callable=io.StringIO), \
                self.assertRaisesRegex(ValueError, "no usable YAML"), \
                patch("builtins.input", side_effect=AssertionError("Nothing can be selected")):
            resolve_materials(self.cfg, interactive=True)

    def test_malformed_yaml_blocks_raise_readable_errors(self):
        for block in ({"type": None}, {"type": "tabulated nk"}, {"type": "formula 1"}):
            with self.subTest(block=block):
                path = self.make("Bad", "malformed.yml", blocks=[block])
                with self.assertRaises(ValueError):
                    read_refractiveindex(path, [193])

    def test_materials_command_does_not_require_substrate_or_solver(self):
        from film_rl.__main__ import main
        self.cfg["materials"][0]["file"] = str(self.a)
        self.cfg["substrate_file"] = "missing_but_not_used_here.sub"
        path = self.root / "materials_only.json"
        path.write_text(json.dumps(self.cfg), encoding="utf-8")
        with patch("sys.argv", ["film_rl", "materials", "--config", str(path), "--non-interactive"]), \
                patch("film_rl.optics.coh_tmm", side_effect=AssertionError("No solver needed")), \
                patch("sys.stdout", new_callable=io.StringIO) as output:
            main()
        report = json.loads(output.getvalue())
        self.assertFalse(report["training_executed"])
        self.assertEqual(len(report["datasets"]), 2)

    def test_bad_tables_unknown_formulas_and_unsafe_yaml_rejected(self):
        blocks = [
            {"type": "tabulated nk", "data": "0.20 1.4 0\n0.19 1.5 0"},
            {"type": "tabulated nk", "data": "0.19 1.4 -1\n0.20 1.5 0"},
            {"type": "formula 10", "coefficients": "1", "wavelength_range": "0.19 0.2"}]
        for block in blocks:
            path = self.make("Bad", "bad.yml", blocks=[block])
            with self.assertRaises(ValueError):
                read_refractiveindex(path, [193])
        self.a.write_text("!!python/object/apply:os.system ['echo forbidden']", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "invalid YAML"):
            read_refractiveindex(self.a, [193])

    def test_config_validation_and_material_order(self):
        validate(self.cfg)
        self.cfg["materials"][1]["name"] = "mgf2"
        with self.assertRaisesRegex(ValueError, "distinct"):
            validate(self.cfg)

    def test_batch_selects_once_and_hashes_yaml_in_seed_snapshots(self):
        import run_seed_batch as batch_runner
        project = self.root / "batch_project"
        (project / "film_rl").mkdir(parents=True)
        for name in ("plot_training.py", "run_seed_batch.py", "summarize_seed_batch.py"):
            (project / name).write_text("# fixture\n", encoding="utf-8")
        path = project / "config.json"
        path.write_text(json.dumps(self.cfg), encoding="utf-8")
        with patch.object(batch_runner, "PROJECT", project), patch("sys.stdin.isatty", return_value=True), \
                patch("builtins.input", return_value="1") as prompt, patch("sys.stdout", new_callable=io.StringIO):
            batch = batch_runner.prepare(config=path, group="yaml_test", seeds=[0, 1])
        self.assertEqual(prompt.call_count, 1)
        manifest = json.loads((batch / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(len(manifest["material_hashes"]), 3)
        self.assertIn(str(self.a.resolve()), manifest["material_hashes"])
        for entry in manifest["runs"]:
            with patch("builtins.input", side_effect=AssertionError("workers must not ask")):
                cfg = load_config(entry["config"], resolve_materials=True)
                optics = FilmOptics(cfg)
            self.assertEqual(material_count(cfg), 2)
            np.testing.assert_allclose(optics.material_indices[0], [1.4+.001j, 1.43+.0016j, 1.5+.003j])
            self.assertFalse(Path(entry["run_dir"]).exists())


if __name__ == "__main__":
    unittest.main()
