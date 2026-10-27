"""Configuration and strict grid validation; all optical lengths are in nm."""

import json
from pathlib import Path
import uuid

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_DIR / "configs" / "duv193.json"
THICKNESS_TOL_NM = 1e-9


def grid(value, name):
    if isinstance(value, dict):
        if set(value) != {"start", "stop", "num"}:
            raise ValueError(f"{name}: use start/stop/num")
        num = value["num"]
        if isinstance(num, bool) or not isinstance(num, int) or num < 1:
            raise ValueError(f"{name}.num must be a positive integer")
        if num == 1 and value["start"] != value["stop"]:
            raise ValueError(f"{name}: num=1 requires start=stop")
        value = np.linspace(value["start"], value["stop"], num)
    array = np.atleast_1d(np.asarray(value, dtype=np.float64))
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a nonempty finite 1-D grid")
    if array.size > 1 and not np.all(np.diff(array) > 0):
        raise ValueError(f"{name} must be strictly increasing")
    return array


def material_source(cfg):
    source = cfg.get("material_source", "refractiveindex")
    if source == "lm" or ("material_source" not in cfg and "material_files" in cfg and "materials" not in cfg):
        raise ValueError(".lm coating support was removed; migrate to material_source='refractiveindex' and materials[].file")
    if source not in ("refractiveindex", "fixed_193nm"):
        raise ValueError("material_source must be 'refractiveindex' or 'fixed_193nm'")
    return source


def material_count(cfg):
    key = "materials" if material_source(cfg) == "refractiveindex" else "fixed_materials_193nm"
    return len(cfg[key])


def optical_data_files(cfg):
    """Absolute paths of active optical files, for batch change detection."""
    if material_source(cfg) == "refractiveindex":
        from .refractiveindex import dataset_path
        if any(not entry.get("file") for entry in cfg["materials"]):
            raise ValueError("Resolve materials[].file before hashing optical data")
        coatings = [dataset_path(cfg, entry) for entry in cfg["materials"]]
    else:
        coatings = []
    return coatings + [(Path(cfg["material_dir"]) / cfg["substrate_file"]).resolve()]


def validate(cfg):
    mode = cfg.get('reward_objective', 'spectral')
    if mode not in ('spectral', 'enhancement'):
        raise ValueError("reward_objective must be spectral or enhancement")
    if mode == 'enhancement':
        from .enhancement import reward_settings
        reward_settings(cfg)
        wave = grid(cfg['wavelengths_nm'], 'wavelengths_nm')
        if wave[0] > 400 or wave[-1] < 780:
            raise ValueError('Enhancement reward requires wavelengths covering 400-780 nm')
        if cfg['polarization'] != 'unpolarized' or cfg.get('substrate_thickness_nm') is not None:
            raise ValueError('Enhancement reward requires unpolarized light and no finite substrate')
        if complex(cfg['incident_n']) != 1:
            raise ValueError('Enhancement reward requires air incidence')
        if cfg.get('design_selection', 'score') != 'score':
            raise ValueError('Enhancement reward requires design_selection=score (select by enhancement reward)')
    if cfg.get('design_selection', 'score') not in ('score', 'lamp_pareto'):
        raise ValueError('design_selection must be score or lamp_pareto')
    sigma = cfg.get('training', {}).get('initial_sigma')
    if sigma is not None and (isinstance(sigma, bool) or not isinstance(sigma, (int, float))
            or not np.isfinite(sigma) or not np.exp(-5) <= sigma <= np.exp(2)):
        raise ValueError('training.initial_sigma must lie within exp(-5)..exp(2)')
    for key in ("entropy_weight_discrete", "entropy_weight_continuous"):
        if key in cfg.get("training", {}):
            value = cfg["training"][key]
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not np.isfinite(value) or value < 0):
                raise ValueError(f"training.{key} must be finite and nonnegative")
    gae_lambda = cfg.get("training", {}).get("gae_lambda", 0.95)
    if (isinstance(gae_lambda, bool) or not isinstance(gae_lambda, (int, float))
            or not np.isfinite(gae_lambda) or not 0 <= gae_lambda <= 1):
        raise ValueError("training.gae_lambda must be finite and within [0, 1]")
    # Missing fields preserve the reward used by historical experiment configs.
    for name, default in (("reward_alpha", 0.0), ("reward_epsilon", 1e-6)):
        value = cfg.get(name, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError(f"{name} must be a finite number")
    if not 0 <= cfg.get("reward_alpha", 0.0) <= 1:
        raise ValueError("reward_alpha must be within [0, 1]")
    if not 0 < cfg.get("reward_epsilon", 1e-6) < 1:
        raise ValueError("reward_epsilon must be within (0, 1)")
    if not isinstance(cfg.get("forbid_adjacent_same_material", True), bool):
        raise ValueError("forbid_adjacent_same_material must be boolean")
    source = material_source(cfg)
    if source == "refractiveindex":
        if not isinstance(cfg.get("refractiveindex_dir"), str) or not cfg["refractiveindex_dir"].strip():
            raise ValueError("refractiveindex_dir must point to the local database repository or database folder")
        materials = cfg.get("materials")
        if not isinstance(materials, list) or len(materials) < 2:
            raise ValueError("materials must contain at least two named materials")
        names = []
        for entry in materials:
            if (not isinstance(entry, dict) or not isinstance(entry.get("name"), str)
                    or not entry["name"].strip() or entry["name"] in (".", "..")
                    or any(c in entry["name"] for c in '/\\:')):
                raise ValueError("Each material requires a nonempty folder name, not a path")
            names.append(entry["name"].casefold())
            if "file" in entry and (not isinstance(entry["file"], str) or not entry["file"].strip()
                                    or Path(entry["file"]).suffix.lower() not in (".yaml", ".yml")):
                raise ValueError("materials[].file must be a YAML file path")
            if not isinstance(entry.get("assume_zero_k", False), bool):
                raise ValueError("materials[].assume_zero_k must be boolean")
        if len(names) != len(set(names)):
            raise ValueError("Material names must be distinct")
    elif source == "fixed_193nm":
        materials = cfg.get("fixed_materials_193nm")
        if not isinstance(materials, list) or len(materials) < 2:
            raise ValueError("fixed_materials_193nm must contain at least two materials")
        names = []
        for entry in materials:
            if not isinstance(entry, dict) or not isinstance(entry.get("name"), str) or not entry["name"].strip():
                raise ValueError("Each fixed material requires a nonempty name")
            names.append(entry["name"])
            for key in ("n", "k"):
                value = entry.get(key)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
                    raise ValueError(f"Each fixed material requires a finite numeric {key}")
            if entry["n"] <= 0 or entry["k"] < 0:
                raise ValueError("Fixed materials require n > 0 and k >= 0")
        if len(set(names)) != len(names):
            raise ValueError("Fixed material names must be distinct")
    if Path(cfg["substrate_file"]).suffix.lower() != ".sub":
        raise ValueError("substrate_file must be a .sub file")
    substrate_thickness = cfg.get("substrate_thickness_nm")
    if substrate_thickness is not None:
        if (isinstance(substrate_thickness, bool) or not isinstance(substrate_thickness, (int, float))
                or not np.isfinite(substrate_thickness) or substrate_thickness <= 0):
            raise ValueError("substrate_thickness_nm must be positive and finite or null (semi-infinite)")
    exit_n = cfg.get("exit_n", 1.0)
    if (isinstance(exit_n, bool) or not isinstance(exit_n, (int, float))
            or not np.isfinite(exit_n) or exit_n <= 0):
        raise ValueError("exit_n must be positive, finite and real")
    if substrate_thickness is None and "exit_n" in cfg:
        raise ValueError("exit_n requires a finite substrate_thickness_nm")
    for name in ("min_layers", "max_layers"):
        if isinstance(cfg[name], bool) or not isinstance(cfg[name], int):
            raise ValueError(f"{name} must be an integer")
    if not 0 <= cfg["min_layers"] <= cfg["max_layers"] or cfg["max_layers"] < 1:
        raise ValueError("Require 0 <= min_layers <= max_layers and max_layers >= 1")
    low, high = cfg["thickness_min_nm"], cfg["thickness_max_nm"]
    if not np.isfinite([low, high]).all() or not 0 < low < high:
        raise ValueError("Require 0 < thickness_min_nm < thickness_max_nm")
    budget = cfg.get("max_total_thickness_nm")
    if budget is not None:
        if isinstance(budget, bool) or not isinstance(budget, (int, float)) or not np.isfinite(budget):
            raise ValueError("max_total_thickness_nm must be a finite number or null")
        if budget < max(1, cfg["min_layers"]) * low:
            raise ValueError("Total thickness budget cannot accommodate the required minimum layers")
    if not np.isfinite(cfg["incident_n"]) or cfg["incident_n"] <= 0:
        raise ValueError("incident_n must be positive and real")
    wavelengths, angles = grid(cfg["wavelengths_nm"], "wavelengths_nm"), grid(cfg["angles_deg"], "angles_deg")
    if source == "fixed_193nm" and (wavelengths.size != 1
            or not np.isclose(wavelengths[0], 193.0, rtol=0, atol=1e-9)):
        raise ValueError("fixed_193nm data are valid only at 193 nm; use material_source='refractiveindex' for other wavelengths")
    if np.any(wavelengths <= 0) or np.any(angles < 0) or np.any(angles >= 90):
        raise ValueError("Require wavelength > 0 and 0 <= angle < 90 degrees")
    if cfg["polarization"] not in ("s", "p", "unpolarized"):
        raise ValueError("polarization must be s, p, or unpolarized")
    from .targets import SpectralObjective
    SpectralObjective(cfg, wavelengths, angles)


def load_config(path=DEFAULT_CONFIG, *, resolve_materials=False, interactive=False, persist_selection=False):
    path = Path(path).resolve()
    original = path.read_bytes()
    cfg = json.loads(original.decode("utf-8-sig"))
    cfg["material_source"] = material_source(cfg)
    material_dir = Path(cfg["material_dir"]).expanduser()
    if not material_dir.is_absolute():
        material_dir = path.parent / material_dir
    cfg["material_dir"] = str(material_dir.resolve())
    validate(cfg)
    if cfg.get("material_source") == "refractiveindex":
        directory = Path(cfg["refractiveindex_dir"]).expanduser()
        cfg["refractiveindex_dir"] = str((directory if directory.is_absolute() else path.parent / directory).resolve())
    if resolve_materials and cfg.get("material_source") == "refractiveindex":
        from .refractiveindex import resolve_materials as resolve
        cfg = resolve(cfg, interactive=interactive)
        if persist_selection:
            saved = json.loads(original.decode("utf-8-sig"))
            if saved.get("materials") != cfg["materials"]:
                saved["materials"] = cfg["materials"]
                if path.read_bytes() != original:
                    raise RuntimeError("Config changed during material selection; refusing to overwrite it")
                temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
                try:
                    temporary.write_text(json.dumps(saved, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                    temporary.replace(path)
                finally:
                    temporary.unlink(missing_ok=True)
    return cfg
