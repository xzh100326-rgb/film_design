"""One-time, lossless organization of the 2026-10-14--20 experiment results.

Original trees are moved intact into archive; selected working copies receive
consistent names. Historical bytes are verified before any metadata rewriting.
Use --apply explicitly. No model training or deletion is performed.
"""
import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil

PROJECT = Path(__file__).resolve().parents[1]
PERIOD = "week_20261014_20261020"
ARCHIVE = PROJECT/"archive"/"results_before_organization_20261020"
WEEK = PROJECT/"outputs"/PERIOD
SELECTIONS = [
    ("runs/duv193_angle0-45_terminal_mix_a0p5/20261018_160311", "runs/duv193__hppo__nk_v1__a00-45/20261018_160311"),
    ("runs/duv193_angle0-45_terminal_mix_a0p5/20261019_143507", "runs/duv193__hppo__nk_v2__a00-45/20261019_143507"),
    ("runs/spectral_rta_example_261020_004725", "runs/abs400-2000__hppo__baseline/20261020_004725/seed1"),
    ("outputs/duv193_angle0-45_terminal_mix_a0p5/20261018_160311", "outputs/duv193__hppo__nk_v1__a00-45/20261018_160311"),
    ("outputs/duv193_angle0-45_terminal_mix_a0p5/20261019_143507", "outputs/duv193__hppo__nk_v2__a00-45/20261019_143507"),
    ("outputs/absorption_400_2000/20261020_010902", "outputs/abs400-2000__hppo__multiseed/20261020_010902"),
    ("outputs/absorption_improvement_20261020", "outputs/abs400-2000__hybrid__pilot/20261020_140821"),
    ("outputs/improvement_20261020_155601", "outputs/abs400-2000__hybrid__search/20261020_155601"),
    ("outputs/improvement_20261020_164544", "outputs/duv193__hybrid__nk_v2__a00-45/20261020_164544"),
    ("outputs/paper_curve_reproduction", "outputs/references/yang2016_digitized_nk"),
    ("outputs/duv_improve_diagnosis_20261020.json", "outputs/duv193__diagnostics/20261020_164829/nk_comparison.json"),
]
HISTORICAL_ALIASES = [
    # These stale names already appeared inside manifests before organization.
    ("runs/fluoride4_fixed193/20261018_160311", "runs/duv193__hppo__nk_v1__a00-45/20261018_160311"),
    ("outputs/fluoride4_fixed193/20261018_160311", "outputs/duv193__hppo__nk_v1__a00-45/20261018_160311"),
]


def checked(path):
    path = Path(path).resolve()
    if not path.is_relative_to(PROJECT) or path == PROJECT:
        raise ValueError(f"Refusing operation outside project: {path}")
    return path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = checked(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


def all_files(path):
    return [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())


def path_mappings():
    pairs = [(PROJECT/a, PROJECT/b) for a, b in SELECTIONS+HISTORICAL_ALIASES]
    pairs += [(PROJECT/"runs", ARCHIVE/"raw/runs"), (PROJECT/"outputs", ARCHIVE/"raw/outputs")]
    pairs.sort(key=lambda pair: len(str(pair[0])), reverse=True)
    variants = []
    for old, new in pairs:
        for a, b in [(str(old), str(new)), (old.as_posix(), new.as_posix()),
                     (str(old.relative_to(PROJECT)), str(new.relative_to(PROJECT))),
                     (old.relative_to(PROJECT).as_posix(), new.relative_to(PROJECT).as_posix())]:
            variants.append((a, b))
    return variants


def replace_path(value):
    if isinstance(value, str):
        # Already-curated paths must not hit the generic archive fallback.
        for _, target in SELECTIONS:
            p = PROJECT/target
            for prefix in (str(p), p.as_posix(), str(Path(target)), Path(target).as_posix()):
                if value == prefix or value.startswith(prefix+"/") or value.startswith(prefix+"\\"):
                    return value
        for old, new in path_mappings():
            if value == old or value.startswith(old+"/") or value.startswith(old+"\\"):
                return new+value[len(old):]
    elif isinstance(value, list):
        return [replace_path(v) for v in value]
    elif isinstance(value, dict):
        return {k: replace_path(v) for k, v in value.items()}
    return value


def group_fields(value, group):
    if isinstance(value, dict):
        return {k: group if k == "group" else group_fields(v, group) for k, v in value.items()}
    if isinstance(value, list):
        return [group_fields(v, group) for v in value]
    return value


def rewrite_batch(batch):
    """Update navigation paths on copies; immutable original manifests remain archived."""
    group = batch.parent.name
    jsons = list(batch.rglob("*.json"))
    for p in jsons:
        d = json.loads(p.read_text(encoding="utf-8-sig"))
        write_json(p, group_fields(replace_path(d), group))
    # Configs changed only in directory references, so refresh their lookup hashes.
    for p in jsons:
        d = json.loads(p.read_text(encoding="utf-8-sig"))
        def refresh(value):
            if isinstance(value, dict):
                if "config_sha256" in value and value.get("config") and Path(value["config"]).is_file():
                    value["config_sha256"] = digest(value["config"])
                for v in value.values(): refresh(v)
            elif isinstance(value, list):
                for v in value: refresh(v)
        refresh(d)
        write_json(p, d)
    for p in batch.glob("*.csv"):
        with p.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            fieldnames = reader.fieldnames
            rows = [{k: group if k == "group" else replace_path(v) for k, v in row.items()} for row in reader]
        with p.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader(); writer.writerows(rows)


def apply():
    if ARCHIVE.exists() or WEEK.exists():
        raise FileExistsError("This organization already has an archive/report. Inspect it instead of applying again.")
    original_roots = [p for base in (PROJECT/"runs", PROJECT/"outputs") for p in sorted(base.iterdir())]
    inventory = []
    for p in original_roots:
        checked(p)
        for q in all_files(p):
            if q.is_symlink(): raise ValueError(f"Unexpected link: {q}")
            stat = q.stat()
            # Leave actively written training logs alone.
            if (q.suffix == ".txt" or q.suffix == ".log" or "tfevents" in q.name) and datetime.now().timestamp()-stat.st_mtime < 90:
                raise RuntimeError(f"Possibly active experiment, postpone organization: {q}")
            inventory.append({"original": q.relative_to(PROJECT).as_posix(), "size": stat.st_size,
                              "mtime_ns": stat.st_mtime_ns, "sha256": digest(q)})
    for old, new in SELECTIONS:
        if not (PROJECT/old).exists(): raise FileNotFoundError(old)
        if (PROJECT/new).exists(): raise FileExistsError(new)
        checked(PROJECT/old); checked(PROJECT/new)
    copied_bytes = sum(sum(q.stat().st_size for q in all_files(PROJECT/a)) for a, _ in SELECTIONS)
    if shutil.disk_usage(PROJECT).free < copied_bytes+100*1024**2:
        raise RuntimeError("Insufficient free disk space for curated copies")
    ARCHIVE.mkdir(parents=True)
    write_json(ARCHIVE/"inventory_before.json", inventory)
    write_json(ARCHIVE/"plan.json", {"selected": SELECTIONS, "archive": str(ARCHIVE),
                                   "period": PERIOD, "original_roots": [str(p) for p in original_roots]})
    config_backup = ARCHIVE/"config_before"
    config_backup.mkdir()
    for p in (PROJECT/"configs").glob("*.json"):
        shutil.copy2(p, config_backup/p.name)
    shutil.copy2(PROJECT/"README.md", ARCHIVE/"README_before.md")
    for p in original_roots:
        destination = checked(ARCHIVE/"raw"/p.relative_to(PROJECT))
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Both resolved absolute source and final target were checked above.
        shutil.move(str(checked(p)), str(destination))
    for row in inventory:
        q = checked(ARCHIVE/"raw"/row["original"])
        if not q.is_file() or digest(q) != row["sha256"]:
            raise RuntimeError(f"Archive verification failed: {q}")
    copies = []
    for old, new in SELECTIONS:
        source, destination = checked(ARCHIVE/"raw"/old), checked(PROJECT/new)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir(): shutil.copytree(source, destination)
        else: shutil.copy2(source, destination)
        for q in all_files(source):
            relative = q.relative_to(source) if source.is_dir() else Path(destination.name)
            target = destination/relative if source.is_dir() else destination
            if digest(q) != digest(target): raise RuntimeError(f"Curated-copy verification failed: {target}")
        copies.append({"original": old, "current": new, "archive": str(source)})
    for path in [PROJECT/b for a, b in SELECTIONS if (PROJECT/b/"manifest.json").is_file() and "/__unused__/" not in b]:
        # Only seed-batch manifests use operational absolute paths; search source
        # snapshots and proof manifests remain unmodified as scientific provenance.
        doc = json.loads((path/"manifest.json").read_text(encoding="utf-8-sig"))
        if "runs" in doc: rewrite_batch(path)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_updates = []
    names = {"400-2000absob.json": "abs400-2000__hppo__warm",
             "400-2000absob_cold_start.json": "abs400-2000__hppo__cold",
             "duv193_fixed193.json": "duv193__hppo__nk_v2__a00-45",
             "duv193.json": "duv193__hppo__yaml__a00-45"}
    for name, group in names.items():
        p = PROJECT/"configs"/name
        if not p.exists(): continue
        old = json.loads(p.read_text(encoding="utf-8-sig")); new = json.loads(json.dumps(old))
        new["training"]["exp_name"] = f"runs/{group}/{stamp}/seed{new['training']['seed']}"
        if old["training"].get("initial_model"):
            new["training"]["initial_model"] = replace_path(old["training"]["initial_model"])
            if not (PROJECT/new["training"]["initial_model"]).is_file():
                raise RuntimeError("Warm-start model path was not repaired")
        write_json(p, new)
        config_updates.append({"config": str(p), "exp_name_before": old["training"]["exp_name"],
                               "exp_name_after": new["training"]["exp_name"],
                               "initial_model_after": new["training"].get("initial_model")})
    # Repair current documentation references; historical copies stay unchanged.
    readme = (PROJECT/"README.md").read_text(encoding="utf-8-sig")
    for old, new in SELECTIONS:
        readme = readme.replace(old, new)
    (PROJECT/"README.md").write_text(readme, encoding="utf-8")
    WEEK.mkdir(parents=True)
    migration = {"completed_at": datetime.now().astimezone().isoformat(), "period": PERIOD,
                 "original_files_verified": len(inventory), "original_bytes": sum(x["size"] for x in inventory),
                 "archive": str(ARCHIVE), "selected": copies, "config_path_changes_only": config_updates,
                 "deleted_files": 0,
                 "note": "Original contents/hashes preserved in archive. Active copies may have navigation paths rewritten; historical snapshots remain unchanged."}
    write_json(WEEK/"organization_log.json", migration)
    write_json(ARCHIVE/"completed.json", migration)
    # A compact path map covers archived, selected, and absent-before-organization paths.
    write_json(WEEK/"path_mapping.json", {"selected": copies, "fallback_archive_root": str(ARCHIVE/"raw")})
    print(json.dumps({k: migration[k] for k in ("original_files_verified", "original_bytes", "deleted_files", "archive")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if args.apply: apply()
    else:
        print(json.dumps({"selected": SELECTIONS, "archive": str(ARCHIVE), "period": PERIOD}, indent=2))
