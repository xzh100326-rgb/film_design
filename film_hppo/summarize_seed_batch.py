"""Inspect a seed batch, or export its completed results without training."""

import argparse
import csv
import json
from pathlib import Path
import re
import tempfile
import time


def latest_match(path, pattern):
    if not path.is_file():
        return None
    matches = re.findall(pattern, path.read_text(encoding="utf-8", errors="replace"))
    return matches[-1] if matches else None


def summarize(batch):
    manifest = json.loads((batch / "manifest.json").read_text(encoding="utf-8"))
    rows = []
    for entry in manifest["runs"]:
        record = Path(entry.get("record", batch / f"{entry['name']}.json"))
        if record.is_file():
            entry = json.loads(record.read_text(encoding="utf-8"))
        run = Path(entry["run_dir"])
        result = entry.get("result", {})
        application = entry.get("best_application_metrics") or {}
        row = {
            "group": entry.get("group", manifest.get("group")),
            "batch_id": entry.get("batch_id", manifest.get("batch_id")),
            "seed": entry["seed"], "run": entry["name"], "status": entry["status"],
            "optimizer_updates": result.get("train_iter") or latest_match(
                run / "log/learner/learner_logger.txt", r"Training Iteration (\d+) Result"),
            "env_steps": result.get("env_step") or latest_match(
                run / "log/collector/collector_logger.txt", r"total_envstep_count: (\d+)"),
            "training_seconds": entry.get("training_seconds"),
            "best_saved_design_transmission": entry.get("best_saved_transmission", entry.get("best_saved_reward")),
            "best_saved_design_reflection": entry.get("best_saved_reflection"),
            "best_saved_design_absorption": entry.get("best_saved_absorption"),
            "evaluation_metric": entry.get("evaluation_metric"),
            "design_selection": entry.get("design_selection", "score"),
            "has_constraints": entry.get('has_constraints', False),
            "best_saved_reward": entry.get('best_saved_reward'),
            "constraint_penalty": entry.get('best_constraint_penalty'),
            "constraints_satisfied": entry.get('best_constraints_satisfied'),
            "visible_T": application.get("visible_T"),
            "visible_R": application.get("visible_R"),
            "visible_A": application.get("visible_A"),
            "UV_R": application.get("UV_R"),
            "IR_R": application.get("IR_R"),
            "IR_min_subband_R": application.get("IR_min_subband_R"),
            "application_targets_met": application.get("application_targets_met"),
            "best_target_errors": json.dumps(entry["best_target_errors"], ensure_ascii=False)
                                  if entry.get("best_target_errors") is not None else None,
            "best_saved_design_score": entry.get("best_saved_score", entry.get("best_saved_reward")),
            "best_saved_training_reward": entry.get("best_saved_reward"),
            "best_layer_count": entry.get("best_layer_count"),
            "best_total_thickness_nm": entry.get("best_total_thickness_nm"),
            "last_evaluation_mean": result.get("eval_value"),
            "best_evaluation_mean": None,
            "best_design": entry.get("best_design"), "plot": entry["plot"], "run_dir": str(run),
        }
        for index, label in enumerate(("710_900", "910_1200", "1210_1700", "1710_2300", "2310_3000")):
            values = application.get("IR_subband_R") or []
            row[f"IR_R_{label}_nm"] = values[index] if index < len(values) else None
        scalar_path = Path(entry["plot"]).with_suffix(".csv")
        if entry["status"] == "complete" and scalar_path.is_file():
            with scalar_path.open(encoding="utf-8-sig", newline="") as stream:
                values = [float(item["value"]) for item in csv.DictReader(stream)
                          if item["tag"] == "evaluator_iter/reward_mean"]
            row["best_evaluation_mean"] = max(values) if values else None
        rows.append(row)
    # Atomic replacement prevents readers seeing a partially refreshed table.
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8-sig", newline="", dir=batch,
                                     prefix="results_", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    try:
        temporary.replace(batch / "results.csv")
    except PermissionError:
        # An open Excel file can lock the destination; training should continue.
        print(f"CSV is locked; close it to allow refresh: {batch / 'results.csv'}", flush=True)
    finally:
        temporary.unlink(missing_ok=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch_dir", type=Path)
    parser.add_argument("--watch", action="store_true", help="Refresh CSV every 30 seconds until all runs end")
    args = parser.parse_args()
    batch = args.batch_dir.resolve()
    while True:
        rows = summarize(batch)
        if not args.watch:
            print(json.dumps(rows, indent=2, ensure_ascii=False))
            return
        ended = sum(row["status"] in ("complete", "failed") for row in rows)
        print(f"Ended {ended}/{len(rows)}; results: {batch / 'results.csv'}", flush=True)
        if ended == len(rows):
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
