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
            "best_saved_design_score": entry.get("best_saved_score", entry.get("best_saved_reward")),
            "best_saved_mixed_score": entry.get("best_saved_mixed_score", entry.get("best_saved_reward")),
            "best_saved_training_return": entry.get("best_saved_training_return"),
            "best_saved_last_step_reward": entry.get("best_saved_step_reward"),
            "best_layer_count": entry.get("best_layer_count"),
            "best_total_thickness_nm": entry.get("best_total_thickness_nm"),
            "last_evaluation_mean": result.get("eval_value"),
            "best_evaluation_mean": None,
            "best_design": entry.get("best_design"), "plot": entry["plot"], "run_dir": str(run),
        }
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
