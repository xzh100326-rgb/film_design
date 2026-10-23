"""Batch postprocessing with compact single designs and Pareto archives."""

import csv
import json
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

from run_seed_batch import run_one, select_saved_design, write_json
from summarize_seed_batch import summarize


def candidate(score, attained=False):
    return {
        "score": score, "reward": score, "layers": [{"material_id": 0, "thickness_nm": 40.}],
        "layer_count": 1, "total_thickness_nm": 40.,
        "mean_transmission": .4, "mean_reflection": .6, "mean_absorption": 0.,
        "target_errors": [], "design_format": "compact_v1",
        "application_metrics": {
            "application_targets_met": attained, "balanced_harmonic": .7 if attained else .8,
            "visible_T": .76 if attained else .4, "visible_R": .2 if attained else .6,
            "visible_A": .04 if attained else 0., "UV_R": .8, "IR_R": .9,
            "IR_subband_R": [.8, .85, .9, .92, .95], "IR_min_subband_R": .8,
        },
    }


class SeedBatchTests(unittest.TestCase):
    def test_pareto_list_uses_application_rule_and_score_mode_remains_compatible(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            write_json(run / "designs/best_a.json", candidate(.99))
            write_json(run / "designs/pareto_a.json", [candidate(.98), candidate(.65, True)])
            path, index, design, count = select_saved_design(run, "lamp_pareto")
            self.assertEqual((path.name, index, count), ("pareto_a.json", 1, 3))
            self.assertTrue(design["application_metrics"]["application_targets_met"])
            self.assertEqual(select_saved_design(run, "score")[2]["score"], .99)

    def test_legacy_reward_only_design_and_missing_lamp_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            design = candidate(.8)
            del design["score"]
            del design["application_metrics"]
            write_json(run / "designs/best_a.json", design)
            self.assertEqual(select_saved_design(run)[2]["reward"], .8)
            with self.assertRaisesRegex(ValueError, "application_metrics"):
                select_saved_design(run, "lamp_pareto")

    def test_empty_run_reports_no_saved_designs(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError, "no saved designs"):
                select_saved_design(Path(temporary))

    def test_worker_export_and_csv_with_mocked_training_and_plot_processes(self):
        # Exercise real postprocessing; no training or optical calculations run.
        with tempfile.TemporaryDirectory() as temporary:
            batch = Path(temporary) / "outputs"
            run = Path(temporary) / "runs/seed0"
            config = batch / "configs/seed0.json"
            write_json(config, {"design_selection": "lamp_pareto"})
            entry = {"seed": 0, "name": "seed0", "run_dir": str(run), "config": str(config),
                     "record": str(batch / "status/seed0.json"), "plot": str(batch / "seed0.png"),
                     "evaluation_metric": "score", "status": "pending"}
            write_json(batch / "manifest.json", {"runs": [entry]})

            class FakeTrain:
                pid = 123

                def __init__(self, *args, **kwargs):
                    write_json(run / "designs/best_a.json", candidate(.99))
                    write_json(run / "designs/pareto_a.json", [candidate(.65, True)])
                    with (run / "result.pkl").open("wb") as stream:
                        pickle.dump({"train_iter": 2, "env_step": 8, "eval_value": .8}, stream)

                def wait(self):
                    return 0

            def fake_plot(command, **kwargs):
                path = Path(command[command.index("--output") + 1])
                path.write_bytes(b"mock plot")
                path.with_suffix(".csv").write_text(
                    "tag,value\nevaluator_iter/reward_mean,0.8\n", encoding="utf-8")
                return type("Completed", (), {"returncode": 0})()

            with patch("run_seed_batch.subprocess.Popen", FakeTrain), patch("run_seed_batch.subprocess.run", fake_plot):
                result = run_one(entry, batch, {})
            self.assertEqual(result["status"], "complete", result.get("error"))
            self.assertEqual(result["best_saved_score"], .65)
            self.assertEqual(result["best_design_source_index"], 0)
            exported = json.loads(Path(result["best_design"]).read_text(encoding="utf-8"))
            self.assertIsInstance(exported, dict)
            self.assertEqual(exported["layers"], candidate(.65, True)["layers"])
            rows = summarize(batch)
            self.assertEqual(rows[0]["visible_T"], .76)
            self.assertEqual(rows[0]["IR_R_710_900_nm"], .8)
            self.assertTrue(rows[0]["application_targets_met"])
            self.assertEqual(rows[0]["best_evaluation_mean"], .8)
            with (batch / "results.csv").open(encoding="utf-8-sig", newline="") as stream:
                self.assertEqual(next(csv.DictReader(stream))["visible_T"], "0.76")


if __name__ == "__main__":
    unittest.main()
