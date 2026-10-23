"""Saved-config discovery and end-to-end enhancement evaluation."""
import json
from pathlib import Path
import tempfile
import unittest

import torch

from calculate_enhancement import calculate, design_config
from fixtures import yaml_config, write_substrate


class EnhancementCliTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.folder = tempfile.TemporaryDirectory(
            prefix="enhancement_cli_", dir=Path(__file__).resolve().parents[1] / "outputs")
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.cfg = yaml_config(self.root, optical_constants=[("Air1", 1., 0.), ("Air2", 1., 0.)])
        write_substrate(self.root, n=1.)
        self.cfg.update(wavelengths_nm=[400., 500., 600., 700., 800., 1000.])
        self.run = self.root / "seed0"
        (self.run / "designs").mkdir(parents=True)
        self.design = self.run / "designs" / "best.json"
        self.design.write_text(json.dumps({"layers": [
            {"material_id": 0, "material": "Air1", "thickness_nm": 50.}]}), encoding="utf-8")

    def snapshot(self, env=None):
        path = self.run / "total_config.py"
        # A realistic DI-engine nonliteral manager must not prevent optical reads.
        path.write_text("exp_config = {'env': " + repr(env or self.cfg)[:-1]
                        + ", 'manager': {'episode_num': float('inf')}}}\n", encoding="utf-8")
        return path

    def test_training_snapshot_and_transparent_filter_end_to_end(self):
        path = self.snapshot()
        result = calculate(self.design, wavelength_step_nm=10., angle_order=8, chunk_size=20)
        self.assertEqual(result["config"]["path"], str(path.resolve()))
        self.assertEqual(result["layer_count"], 1)
        for row in result["enhancement"]:
            self.assertAlmostEqual(row["chi"], 1., places=10)
            self.assertAlmostEqual(row["temperature_k"], row["bare_temperature_k"], places=8)

    def test_explicit_json_overrides_ancestor_snapshot(self):
        self.snapshot()
        path = self.root / "explicit.json"
        cfg = {**self.cfg, "min_layers": 0}
        path.write_text(json.dumps(cfg), encoding="utf-8")
        actual, source = design_config(self.design, path)
        self.assertEqual(actual["min_layers"], 0)
        self.assertEqual(source["path"], str(path.resolve()))

    def test_snapshot_is_parsed_without_executing_code(self):
        path = self.snapshot()
        marker = self.root / "MUST_NOT_EXIST"
        with path.open("a", encoding="utf-8") as stream:
            stream.write(f"open({str(marker)!r}, 'w').write('executed')\n")
        design_config(self.design)
        self.assertFalse(marker.exists())
        path.write_text("exp_config = {'env': dict()}\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "literal env"):
            design_config(self.design)

    def test_missing_snapshot_and_changed_material_data_fail_clearly(self):
        with self.assertRaisesRegex(ValueError, "--config"):
            design_config(self.design)
        self.snapshot()
        data = json.loads(self.design.read_text(encoding="utf-8"))
        data["material_datasets"] = [{"material": "Air1", "sha256": "wrong"}]
        self.design.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Material data changed"):
            calculate(self.design, wavelength_step_nm=10., angle_order=8)


if __name__ == "__main__":
    unittest.main()
