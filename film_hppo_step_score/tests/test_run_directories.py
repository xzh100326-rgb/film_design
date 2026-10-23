"""Check real DI-engine config compilation and design export without training."""

from copy import deepcopy
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from ding.config import compile_config

from film_rl.config import PROJECT_DIR, load_config
from film_rl.env import ThinFilmEnv
from film_rl.hppo import build_configs


class RunDirectoryTests(unittest.TestCase):
    def test_new_name_keeps_compiled_logs_and_designs_in_one_directory(self):
        cfg = load_config()
        cfg["training"]["exp_name"] = "runs/repeated_experiment"
        original = deepcopy(cfg)
        scratch = PROJECT_DIR / "outputs"
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as folder:
            root = Path(folder)
            base = root / cfg["training"]["exp_name"]
            base.mkdir(parents=True)
            (base / "existing.txt").write_text("original run", encoding="utf-8")
            occupied = base.with_name(base.name + "_261014_170803")
            occupied.mkdir()
            with patch("film_rl.hppo.PROJECT_DIR", root), patch("film_rl.hppo.datetime") as clock:
                clock.now.return_value.strftime.return_value = "_261014_170803"
                main, create = build_configs(cfg, new_run=True)
            expected = base.with_name(base.name + "_261014_170803_1")
            self.assertEqual(root / main.exp_name, expected)
            self.assertEqual(Path(main.env.result_dir), expected / "designs")
            self.assertFalse(expected.exists())
            create.policy.type += "_command"
            previous_dir = Path.cwd()
            try:
                os.chdir(root)
                # Same native compiler/options as serial_pipeline_onpolicy.
                compiled = compile_config(main, seed=cfg["training"]["seed"], auto=True,
                                          create_cfg=create, save_cfg=True, renew_dir=True)
            finally:
                os.chdir(previous_dir)
            self.assertEqual(compiled.exp_name, main.exp_name)
            self.assertTrue((expected / "total_config.py").is_file())
            self.assertEqual(Path(compiled.env.result_dir), expected / "designs")
            env = ThinFilmEnv(compiled.env)
            try:
                env.reset()
                env.step({"action_type": 3, "action_args": np.zeros(5, dtype=np.float32)})
                env.step({"action_type": 5})
                self.assertEqual(len(list((expected / "designs").glob("best_*.json"))), 1)
                self.assertFalse((base / "designs").exists())
                self.assertEqual((base / "existing.txt").read_text(encoding="utf-8"), "original run")
            finally:
                env.close()
        self.assertEqual(cfg, original)

    def test_unused_name_and_read_only_config_keep_requested_directory(self):
        cfg = load_config()
        cfg["training"]["exp_name"] = "runs/new_experiment"
        scratch = PROJECT_DIR / "outputs"
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as folder:
            root = Path(folder)
            with patch("film_rl.hppo.PROJECT_DIR", root):
                main, _ = build_configs(cfg, new_run=True)
                self.assertEqual(root / main.exp_name, root / cfg["training"]["exp_name"])
                self.assertFalse((root / main.exp_name).exists())
                (root / main.exp_name).mkdir(parents=True)
                checked, _ = build_configs(cfg)
                self.assertEqual(Path(checked.env.result_dir), root / main.exp_name / "designs")
                self.assertEqual(Path(checked.exp_name), Path(cfg["training"]["exp_name"]))


if __name__ == "__main__":
    unittest.main()
