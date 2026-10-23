"""Batch training: runs/<group>/<timestamp>/seedN, with matching outputs."""

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import re
import subprocess
import sys
import time

import numpy as np

from film_rl.config import grid, load_config, optical_data_files
from summarize_seed_batch import summarize


PROJECT = Path(__file__).resolve().parent
SHANGHAI = timezone(timedelta(hours=8))


def now():
    return datetime.now(SHANGHAI)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    # Windows readers may briefly hold the destination without delete sharing.
    for attempt in range(6):
        try:
            temporary.replace(path)
            break
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(.05 * (attempt + 1))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def project_path(path):
    path = Path(path)
    path = (path if path.is_absolute() else PROJECT / path).resolve()
    if not path.is_relative_to(PROJECT.resolve()):
        raise ValueError(f"Path outside project: {path}")
    return path


def group_name(value):
    if (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,119}", value)
            or value.endswith(".")
            or value.split(".")[0].upper() in {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"}
            or re.fullmatch(r"(?:COM|LPT)[1-9](?:\..*)?", value, re.IGNORECASE)):
        raise ValueError("Group must be one directory name: letters, numbers, _, - or .; no paths/reserved names")
    return value


def number(value):
    return format(float(value), ".8g").replace(".", "p")


def automatic_group(cfg):
    wavelengths = grid(cfg["wavelengths_nm"], "wavelengths_nm")
    angles = grid(cfg["angles_deg"], "angles_deg")
    wave_tag = ("duv" if wavelengths.max() < 400 else "wl") + number(wavelengths[0])
    if len(wavelengths) > 1:
        wave_tag += "-" + number(wavelengths[-1])
    angle_tag = number(angles[0])
    if len(angles) > 1:
        angle_tag += "-" + number(angles[-1])
    alpha = cfg.get("reward_alpha", 0.0)
    reward_tag = "linear" if alpha == 0 else "log" if alpha == 1 else "mix_a" + number(alpha)
    return group_name(f"{wave_tag}_angle{angle_tag}_step_delta_{reward_tag}")


def prepare(config=None, group=None, seeds=None):
    source = Path(config) if config is not None else PROJECT / "configs/duv193.json"
    source = (source if source.is_absolute() else PROJECT / source).resolve()
    cfg = load_config(source)  # Resolve relative material paths BEFORE copying the config.
    seeds = list(range(10)) if seeds is None else list(seeds)
    if not seeds or any(type(seed) is not int or seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be distinct nonnegative integers")
    group = group_name(group) if group is not None else automatic_group(cfg)
    paths = sorted((PROJECT / "film_rl").glob("*.py"))
    paths += [PROJECT / name for name in ("plot_training.py", "run_seed_batch.py", "summarize_seed_batch.py")]
    source_hashes = {str(path.relative_to(PROJECT)): digest(path) for path in paths}
    directory = Path(cfg["material_dir"])
    material_hashes = {str((directory / name).resolve()): digest(directory / name)
                       for name in optical_data_files(cfg)}
    stamp = now().strftime("%Y%m%d_%H%M%S")
    index = 0
    while True:
        batch_id = stamp if index == 0 else f"{stamp}_{index:02d}"
        run_root = project_path(PROJECT / "runs" / group / batch_id)
        batch = project_path(PROJECT / "outputs" / group / batch_id)
        if not run_root.exists() and not batch.exists():
            try:
                # Reserve batch parents only. Existing seed run directories make
                # DI-engine rename its logs and can separate them from designs.
                batch.mkdir(parents=True, exist_ok=False)
                run_root.mkdir(parents=True, exist_ok=False)
                break
            except FileExistsError:
                pass
        index += 1
    write_json(batch / "base_config.json", cfg)
    manifest = {
        "schema_version": 2, "created_at": now().isoformat(),
        "group": group, "batch_id": batch_id, "batch_dir": str(batch), "run_root": str(run_root),
        "python": sys.executable, "source_config": str(source),
        "base_config_sha256": digest(source),
        "runtime_environment": {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        "source_hashes": source_hashes, "material_hashes": material_hashes, "runs": [],
    }
    for seed in seeds:
        name = f"seed{seed}"
        run = project_path(run_root / name)
        run_cfg = deepcopy(cfg)
        run_cfg["training"].update(seed=seed, exp_name=run.relative_to(PROJECT).as_posix())
        config_path = batch / "configs" / f"{name}.json"
        write_json(config_path, run_cfg)
        manifest["runs"].append({
            "seed": seed, "name": name, "group": group, "batch_id": batch_id,
            "config": str(config_path), "config_sha256": digest(config_path), "run_dir": str(run),
            "plot": str(batch / f"{name}.png"), "record": str(batch / "status" / f"{name}.json"),
            "train_log": str(batch / "logs" / f"{name}.train.log"),
            "plot_log": str(batch / "logs" / f"{name}.plot.log"),
            "reward_as_transmission": bool(np.all(np.asarray(cfg["target_transmission"]) == 1)),
            "status": "pending",
        })
    write_json(batch / "manifest.json", manifest)
    summarize(batch)
    print(f"Experiment group: {group}\nTraining results: {run_root}\nBatch outputs: {batch}", flush=True)
    return batch


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    return value.tolist() if hasattr(value, "tolist") else value


def run_one(entry, batch, runtime_environment):
    entry = deepcopy(entry)
    record = Path(entry.get("record", batch / f"{entry['name']}.json"))
    entry.update(started_at=now().isoformat(), status="training")
    environment = os.environ.copy()
    environment.update(runtime_environment)
    environment.update(PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", MPLBACKEND="Agg")
    started = time.perf_counter()
    try:
        if Path(entry["run_dir"]).exists():
            raise RuntimeError(f"Run already exists; refusing to overwrite or rename it: {entry['run_dir']}")
        write_json(record, entry)
        train_log = Path(entry.get("train_log", batch / f"{entry['name']}.train.log"))
        train_log.parent.mkdir(parents=True, exist_ok=True)
        entry["train_log"] = str(train_log)
        command = [sys.executable, "-m", "film_rl", "train", "--config", entry["config"]]
        entry["training_command"] = command
        print(f"START {entry['name']}: {entry['run_dir']}", flush=True)
        with train_log.open("w", encoding="utf-8") as stream:
            process = subprocess.Popen(command, cwd=PROJECT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
            entry["training_pid"] = process.pid
            write_json(record, entry)
            return_code = process.wait()
        entry.update(training_seconds=time.perf_counter() - started, training_exit_code=return_code)
        if return_code != 0:
            raise RuntimeError(f"Training exited {return_code}; see {train_log}")
        run = Path(entry["run_dir"])
        if not (run / "result.pkl").is_file():
            raise RuntimeError(f"Expected completed run missing: {run}")
        # Only read the result generated by this project's own subprocess.
        with (run / "result.pkl").open("rb") as stream:
            entry["result"] = json_safe(pickle.load(stream))
        designs = []
        for path in sorted((run / "designs").glob("*.json")):
            design = json.loads(path.read_text(encoding="utf-8-sig"))
            designs.append((float(design.get("score", design["reward"])), path, design))
        if not designs:
            raise RuntimeError("Training produced no saved designs")
        score, best_path, best = max(designs, key=lambda item: item[0])
        entry.update(best_saved_reward=float(best["reward"]), best_saved_score=score,
                     best_saved_mixed_score=float(best["mixed_score"]),
                     best_saved_training_return=float(best["episode_training_return"]),
                     best_saved_step_reward=float(best["step_reward"]),
                     best_saved_transmission=float(best["mean_transmission"]), best_design=str(best_path),
                     best_layer_count=best["layer_count"], best_total_thickness_nm=best["total_thickness_nm"],
                     design_count=len(designs), status="plotting")
        write_json(record, entry)
        command = [sys.executable, "plot_training.py", "--run", entry["run_dir"], "--output", entry["plot"]]
        if entry.get("reward_as_transmission", True):
            command.append("--reward-as-transmission")
        entry["plot_command"] = command
        plot_log = Path(entry.get("plot_log", batch / f"{entry['name']}.plot.log"))
        plot_log.parent.mkdir(parents=True, exist_ok=True)
        with plot_log.open("w", encoding="utf-8") as stream:
            plotted = subprocess.run(command, cwd=PROJECT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
        if plotted.returncode or not Path(entry["plot"]).is_file() or not Path(entry["plot"]).with_suffix(".csv").is_file():
            raise RuntimeError(f"Plotting failed: exit={plotted.returncode}; see {plot_log}")
        entry["status"] = "complete"
    except Exception as error:
        entry.update(status="failed", error=str(error))
    entry.update(finished_at=now().isoformat(), total_seconds=time.perf_counter() - started)
    write_json(record, entry)
    print(f"{entry['status'].upper()} {entry['name']}: {entry['total_seconds']:.1f}s, "
          f"best_score={entry.get('best_saved_score')}", flush=True)
    return entry


def refresh_summary(batch, manifest, workers, started):
    rows = summarize(batch)
    entries = []
    for source in manifest["runs"]:
        record = Path(source.get("record", batch / f"{source['name']}.json"))
        entry = json.loads(record.read_text(encoding="utf-8")) if record.is_file() else deepcopy(source)
        row = next(row for row in rows if row["seed"] == entry["seed"])
        entry["best_evaluation_mean"] = row["best_evaluation_mean"]
        entries.append(entry)
    summary = {
        "group": manifest.get("group"), "batch_id": manifest.get("batch_id"),
        "batch_dir": str(batch), "run_root": manifest.get("run_root"), "workers": workers,
        "elapsed_seconds": time.perf_counter() - started, "updated_at": now().isoformat(), "runs": entries,
        "all_complete": all(entry["status"] == "complete" for entry in entries),
    }
    source_config = Path(manifest.get("source_config", PROJECT / "configs/duv193.json"))
    summary["source_config_unchanged"] = source_config.is_file() and digest(source_config) == manifest["base_config_sha256"]
    summary["source_files_unchanged"] = all((PROJECT / path).is_file() and digest(PROJECT / path) == expected
                                           for path, expected in manifest["source_hashes"].items())
    write_json(batch / "summary.json", summary)
    finished = sum(entry["status"] == "complete" for entry in entries)
    failed = sum(entry["status"] == "failed" for entry in entries)
    print(f"Progress: {finished}/{len(entries)} complete, {failed} failed; {batch / 'results.csv'}", flush=True)
    return summary


def execute(batch, workers):
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
    batch = project_path(batch)
    manifest = json.loads((batch / "manifest.json").read_text(encoding="utf-8"))
    for relative, expected in manifest["source_hashes"].items():
        if digest(PROJECT / relative) != expected:
            raise RuntimeError(f"Source changed since preparation: {relative}")
    for path, expected in manifest["material_hashes"].items():
        if digest(Path(path)) != expected:
            raise RuntimeError(f"Material changed since preparation: {path}")
    for entry in manifest["runs"]:
        project_path(entry["run_dir"])
        if Path(entry["run_dir"]).exists():
            raise RuntimeError(f"Output already exists: {entry['run_dir']}")
        if entry.get("config_sha256") and digest(Path(entry["config"])) != entry["config_sha256"]:
            raise RuntimeError(f"Prepared config changed: {entry['config']}; prepare a new batch instead")
    started = time.perf_counter()
    refresh_summary(batch, manifest, workers, started)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(run_one, entry, batch, manifest["runtime_environment"]) for entry in manifest["runs"]}
        while pending:
            done, pending = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
            for future in done:
                future.result()
            summary = refresh_summary(batch, manifest, workers, started)
    print(f"Batch finished; all_complete={summary['all_complete']}; summary={batch / 'summary.json'}", flush=True)
    return 0 if summary["all_complete"] else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Base config (default: configs/duv193.json)")
    parser.add_argument("--group", help="Experiment group; default derived from wavelength, angles and reward alpha")
    parser.add_argument("--seeds", nargs="+", type=int, help="Distinct seeds (default: 0 through 9)")
    parser.add_argument("--workers", type=int, default=3, help="Concurrent training processes (default: 3)")
    parser.add_argument("--prepare-only", action="store_true", help="Create configs and directories without training")
    parser.add_argument("--batch-dir", type=Path, help="Execute an existing, not-yet-started prepared batch")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.batch_dir and any(value is not None for value in (args.config, args.group, args.seeds)):
        parser.error("--batch-dir uses saved settings; do not combine with --config, --group or --seeds")
    try:
        batch = project_path(args.batch_dir) if args.batch_dir else prepare(args.config, args.group, args.seeds)
        if args.batch_dir and not (batch / "manifest.json").is_file():
            raise ValueError(f"Batch manifest missing: {batch}")
        return 0 if args.prepare_only else execute(batch, args.workers)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
