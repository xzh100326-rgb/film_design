"""Offline refractiveindex.info YAML data and explicit, reproducible selection.

Wavelengths in the database are micrometres; the project API uses nanometres.
No network access or input prompts take place inside an environment or TMM call.
"""

from copy import deepcopy
from functools import lru_cache
import hashlib
import io
from pathlib import Path

import numpy as np
import yaml

from .config import grid, material_source

def database_root(directory):
    directory = Path(directory).expanduser().resolve()
    for candidate in (directory / "database", directory):
        if (candidate / "data").is_dir():
            return candidate
    raise FileNotFoundError(f"No refractiveindex database/data directory under {directory}")


def dataset_path(cfg, entry):
    file = Path(entry["file"])
    return (file if file.is_absolute() else database_root(cfg["refractiveindex_dir"]) / file).resolve()


def candidates(cfg, entry):
    root = database_root(cfg["refractiveindex_dir"])
    supplied = Path(entry["file"]) if entry.get("file") else None
    if supplied is not None and (supplied.is_absolute() or len(supplied.parts) > 1):
        return [(supplied if supplied.is_absolute() else root / supplied).resolve()]
    # Do not cache directory listings: users may add measured materials at runtime.
    folders = [folder for folder in (root / "data").rglob(entry["name"])
               if folder.is_dir() and folder.name.casefold() == entry["name"].casefold()]
    if not folders:
        folders = [folder for folder in (root / "data").rglob("*")
                   if folder.is_dir() and folder.name.casefold() == entry["name"].casefold()]
    paths = []
    for folder in folders:
        source = folder / "nk" if (folder / "nk").is_dir() else folder
        paths.extend(p for p in source.rglob("*") if p.is_file()
                     and p.suffix.lower() in (".yml", ".yaml") and p.stem.lower() != "about")
    if supplied is not None:
        # Accept a basename inside the material's nk directory, a database-relative
        # path, or an explicitly supplied absolute path for external measurements.
        found = [p for p in paths if p.name.casefold() == supplied.name.casefold()] or [root / supplied]
        return sorted(set(p.resolve() for p in found))
    return sorted(set(p.resolve() for p in paths))


@lru_cache(maxsize=256)
def _document(path, mtime_ns, size):
    del mtime_ns, size  # included in the cache key so edits invalidate the cache
    raw = Path(path).read_bytes()
    try:
        document = yaml.safe_load(raw.decode("utf-8-sig"))
    except (UnicodeError, yaml.YAMLError) as error:
        raise ValueError(f"{path}: invalid YAML: {error}") from error
    if not isinstance(document, dict) or not isinstance(document.get("DATA"), list):
        raise ValueError(f"{path}: require a YAML mapping with a DATA list")
    return document, hashlib.sha256(raw).hexdigest()


def _load(path):
    path = Path(path).resolve()
    stat = path.stat()
    return _document(str(path), stat.st_mtime_ns, stat.st_size)


def _numbers(value):
    if isinstance(value, str):
        return np.asarray([float(v) for v in value.split()], dtype=np.float64)
    return np.asarray(value, dtype=np.float64)


def _formula(number, coefficients, wave):
    """Database dispersion formulas 1--9 (database/doc/Dispersion formulas.pdf)."""
    c = _numbers(coefficients)
    if c.ndim != 1 or c.size == 0 or not np.isfinite(c).all():
        raise ValueError("Invalid dispersion coefficients")
    if number in (1, 2, 3, 5, 6) and c.size % 2 != 1:
        raise ValueError("Dispersion formula requires a constant followed by coefficient pairs")
    w2 = wave ** 2
    with np.errstate(divide="raise", invalid="raise", over="raise"):
        if number in (1, 2):
            value = np.full_like(wave, 1 + c[0])
            for a, b in zip(c[1::2], c[2::2]):
                value += a * w2 / (w2 - (b*b if number == 1 else b))
            return np.sqrt(value)
        if number in (3, 5):
            value = np.full_like(wave, c[0])
            for a, b in zip(c[1::2], c[2::2]):
                value += a * wave ** b
            return np.sqrt(value) if number == 3 else value
        if number == 4:
            if c.size < 9 or (c.size - 9) % 2:
                raise ValueError("Formula 4 requires nine coefficients and optional pairs")
            value = np.full_like(wave, c[0])
            for i in (1, 5):
                if c[i] != 0:
                    value += c[i] * wave ** c[i+1] / (w2 - c[i+2] ** c[i+3])
            for a, b in zip(c[9::2], c[10::2]):
                value += a * wave ** b
            return np.sqrt(value)
        if number == 6:
            value = np.full_like(wave, 1 + c[0])
            for a, b in zip(c[1::2], c[2::2]):
                value += a / (b - wave ** -2)
            return value
        if number == 7 and c.size in (5, 6):
            # Edwards (Si) omits the zero lambda^6 coefficient.
            c = np.pad(c, (0, 6-c.size))
            return c[0] + c[1] / (w2 - .028) + c[2] / (w2 - .028)**2 + c[3]*w2 + c[4]*w2**2 + c[5]*w2**3
        if number == 8 and c.size == 4:
            q = c[0] + c[1]*w2/(w2-c[2]) + c[3]*w2
            return np.sqrt((1 + 2*q) / (1-q))
        if number == 9 and c.size == 6:
            return np.sqrt(c[0] + c[1]/(w2-c[2]) + c[3]*(wave-c[4])/((wave-c[4])**2+c[5]))
    raise ValueError(f"Unsupported formula or coefficient count: formula {number}, {c.size} coefficients")


def read_refractiveindex(path, wavelengths_nm, assume_zero_k=False):
    """Return complex n+i*k and source metadata, without extrapolation.

    Supports tabulated n/k/nk, separate n and k grids, and formulas 1--9.
    Missing k requires an explicit per-material assume_zero_k=true.
    """
    path = Path(path).resolve()
    doc, sha = _load(path)
    wave = np.atleast_1d(np.asarray(wavelengths_nm, dtype=np.float64)) / 1000.0
    if wave.ndim != 1 or not wave.size or not np.isfinite(wave).all() or np.any(wave <= 0):
        raise ValueError("Wavelengths must be a positive finite 1-D array in nm")
    channels = {"n": [], "k": []}
    definitions = []
    for block in doc["DATA"]:
        if not isinstance(block, dict):
            raise ValueError(f"{path}: invalid DATA block")
        kind = block.get("type", "")
        if not isinstance(kind, str):
            raise ValueError(f"{path}: DATA type must be a string")
        if kind.startswith("tabulated "):
            suffix = kind.split()[1]
            if suffix not in ("n", "k", "nk"):
                raise ValueError(f"{path}: unsupported data type {kind}")
            if not isinstance(block.get("data"), str) or not block["data"].strip():
                raise ValueError(f"{path}: {kind} requires a nonempty data table")
            table = np.loadtxt(io.StringIO(block["data"]), ndmin=2)
            width = 3 if suffix == "nk" else 2
            if table.shape[1] != width or len(table) < 1 or not np.isfinite(table).all():
                raise ValueError(f"{path}: invalid {kind} table")
            if np.any(table[:, 0] <= 0) or not np.all(np.diff(table[:, 0]) > 0):
                raise ValueError(f"{path}: wavelengths must be positive and strictly increasing")
            if (suffix in ("n", "nk") and np.any(table[:, 1] <= 0)
                    or suffix == "k" and np.any(table[:, 1] < 0)
                    or suffix == "nk" and np.any(table[:, 2] < 0)):
                raise ValueError(f"{path}: require n > 0 and k >= 0")
            for column, channel in enumerate(suffix, start=1):
                channels[channel].append((table[0, 0], table[-1, 0], ("table", table[:, 0], table[:, column])))
        elif kind.startswith("formula "):
            number = int(kind.split()[1])
            if number not in range(1, 10):
                raise ValueError(f"{path}: unsupported {kind}")
            if "wavelength_range" not in block or "coefficients" not in block:
                raise ValueError(f"{path}: {kind} requires wavelength_range and coefficients")
            bounds = _numbers(block["wavelength_range"])
            if bounds.shape != (2,) or not np.isfinite(bounds).all() or not 0 < bounds[0] < bounds[1]:
                raise ValueError(f"{path}: invalid wavelength_range")
            channels["n"].append((*bounds, ("formula", number, block["coefficients"])))
        else:
            raise ValueError(f"{path}: unsupported data type {kind!r}")
        definitions.append(kind)
    if not channels["n"]:
        raise ValueError(f"{path}: no refractive index n provided")
    missing_k = not channels["k"]
    if missing_k and not assume_zero_k:
        raise ValueError(f"{path.name}: k is not provided; select n,k data or explicitly set assume_zero_k=true")

    values = {}
    for channel, sections in channels.items():
        if not sections:
            values[channel] = np.zeros_like(wave)
            continue
        value = np.full_like(wave, np.nan)
        for low, high, definition in sections:
            mask = (wave >= low) & (wave <= high) & np.isnan(value)
            if not mask.any():
                continue
            try:
                if definition[0] == "table":
                    value[mask] = np.interp(wave[mask], definition[1], definition[2])
                else:
                    value[mask] = _formula(definition[1], definition[2], wave[mask])
            except (FloatingPointError, OverflowError) as error:
                raise ValueError(f"{path.name}: dispersion formula is undefined at requested wavelength") from error
        if np.isnan(value).any():
            ranges = ", ".join(f"[{lo*1000:g}, {hi*1000:g}]" for lo, hi, _ in sections)
            raise ValueError(f"{path.name}: requested wavelength outside {channel} coverage {ranges} nm; no extrapolation")
        if not np.isfinite(value).all() or np.any(value <= 0 if channel == "n" else value < 0):
            raise ValueError(f"{path.name}: invalid calculated {channel}")
        values[channel] = value
    return values["n"] + 1j*values["k"], {
        "file": str(path), "sha256": sha, "reference": doc.get("REFERENCES", ""),
        "comments": doc.get("COMMENTS", ""), "conditions": doc.get("CONDITIONS", {}),
        "data_types": definitions, "assumed_zero_k": bool(missing_k),
    }


def load_coatings(cfg, wavelengths_nm):
    """Read selected coatings without loading a substrate or initializing TMM."""
    if material_source(cfg) == "fixed_193nm":
        entries = cfg["fixed_materials_193nm"]
        if len(wavelengths_nm) != 1 or not np.isclose(wavelengths_nm[0], 193., rtol=0, atol=1e-9):
            raise ValueError("fixed_193nm data are valid only at 193 nm")
        return ([e["name"] for e in entries],
                np.asarray([[complex(e["n"], e["k"])] for e in entries]), [])
    names, indices, datasets = [], [], []
    for entry in cfg["materials"]:
        index, metadata = read_refractiveindex(dataset_path(cfg, entry), wavelengths_nm,
                                               entry.get("assume_zero_k", False))
        names.append(entry["name"])
        indices.append(index)
        datasets.append({"material": entry["name"], **metadata})
    return names, np.stack(indices), datasets


def resolve_materials(cfg, interactive=False, input_fn=None):
    """Resolve every material before prompting; return a copy with pinned files."""
    cfg = deepcopy(cfg)
    if material_source(cfg) != "refractiveindex":
        return cfg
    cfg["material_source"] = "refractiveindex"
    input_fn = input if input_fn is None else input_fn
    root = database_root(cfg["refractiveindex_dir"])
    found = []
    missing = []
    for entry in cfg["materials"]:
        choices = candidates(cfg, entry)
        if not choices or any(not path.is_file() for path in choices):
            missing.append(entry["name"] + (f" ({entry['file']})" if entry.get("file") else ""))
        found.append(choices)
    if missing:
        raise FileNotFoundError(
            "Missing refractiveindex YAML material data: " + ", ".join(missing)
            + f". Database: {root}. Add measured YAML under data/main/<material>/nk/ "
            "or set materials[].file to an absolute YAML path. No fixed-index fallback is used."
        )
    wave = grid(cfg["wavelengths_nm"], "wavelengths_nm")
    for entry, choices in zip(cfg["materials"], found):
        if len(choices) > 1:
            if not interactive:
                listing = ", ".join(str(p.relative_to(root)) if p.is_relative_to(root) else str(p) for p in choices)
                raise ValueError(f"{entry['name']}: multiple YAML versions: {listing}. "
                                 "Run python -m film_rl materials --config <config.json> in a terminal "
                                 "to select, or set materials[].file explicitly.")
            print(f"\n{entry['name']}: choose a YAML dataset (target wavelengths: {wave.min():g}--{wave.max():g} nm)", flush=True)
            usable = set()
            for i, path in enumerate(choices, start=1):
                try:
                    _, meta = read_refractiveindex(path, wave, entry.get("assume_zero_k", False))
                    usable.add(i)
                    status = "OK; " + " ".join(str(meta["comments"]).split())[:300]
                except (ValueError, KeyError, TypeError, FloatingPointError) as error:
                    status = "UNAVAILABLE: " + str(error)
                print(f"  [{i}] {path.relative_to(root)} | {status}", flush=True)
            if not usable:
                raise ValueError(f"{entry['name']}: no usable YAML versions for the requested wavelengths and n/k requirements")
            while True:
                try:
                    reply = input_fn("Dataset number (q to cancel): ").strip()
                except EOFError as exc:
                    raise ValueError("No interactive input; specify materials[].file before training") from exc
                if reply.lower() == "q":
                    raise ValueError("Material selection cancelled")
                if reply.isdigit() and 1 <= int(reply) <= len(choices):
                    if int(reply) not in usable:
                        print("Cannot use this dataset; select a version marked OK.", flush=True)
                        continue
                    selected = choices[int(reply)-1]
                    break
                print("Enter one of the listed dataset numbers.", flush=True)
        else:
            selected = choices[0]
        read_refractiveindex(selected, wave, entry.get("assume_zero_k", False))
        entry["file"] = selected.relative_to(root).as_posix() if selected.is_relative_to(root) else str(selected)
    return cfg
