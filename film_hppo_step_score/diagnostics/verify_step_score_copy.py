"""Verify source preservation and unchanged runtime/config files; no training."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def main():
    print("Verifying copied source snapshot...", flush=True)
    record = json.loads((ROOT / "outputs/source_copy_manifest.json").read_text(encoding="utf-8-sig"))
    source = Path(record["source"])
    changed_source = [entry["path"] for entry in record["files"]
                      if not (source / entry["path"]).is_file()
                      or sha256(source / entry["path"]) != entry["sha256"]]
    unchanged = ["configs/duv193.json", "film_rl/hppo.py", "film_rl/policy.py", "film_rl/optics.py",
                 "film_rl/config.py", "film_rl/__init__.py", "plot_training.py", "pyproject.toml"]
    mismatches = [name for name in unchanged if sha256(source / name) != sha256(ROOT / name)]
    report = {"source": str(source), "copy": str(ROOT), "copied_files": record["copied_files"],
              "source_files_changed_since_copy": changed_source,
              "unchanged_runtime_and_config_files": unchanged,
              "runtime_or_config_mismatches": mismatches,
              "training_executed": False,
              "reward_mode": "Every valid step including EOS returns current 1-MAE, no subtraction"}
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    assert not changed_source, "Original snapshot changed since copying"
    assert not mismatches, "A runtime/config file outside reward implementation changed"
    (ROOT / "outputs/step_score_copy_verification.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Verification report saved.", flush=True)


if __name__ == "__main__":
    main()
