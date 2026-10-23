"""Configuration and strict grid validation; all optical lengths are in nm."""

import json
from pathlib import Path

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


def validate(cfg):
    if not isinstance(cfg.get("forbid_adjacent_same_material", True), bool):
        raise ValueError("forbid_adjacent_same_material must be boolean")
    files = cfg["material_files"]
    if len(files) != 5 or len(set(files)) != 5 or any(Path(f).suffix.lower() != ".lm" for f in files):
        raise ValueError("Exactly five distinct .lm coating materials are required")
    if Path(cfg["substrate_file"]).suffix.lower() != ".sub":
        raise ValueError("substrate_file must be a .sub file")
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
    if np.any(wavelengths <= 0) or np.any(angles < 0) or np.any(angles >= 90):
        raise ValueError("Require wavelength > 0 and 0 <= angle < 90 degrees")
    if cfg["polarization"] not in ("s", "p", "unpolarized"):
        raise ValueError("polarization must be s, p, or unpolarized")
    target = np.asarray(cfg["target_transmission"], dtype=np.float64)
    if not np.isfinite(target).all() or np.any(target < 0) or np.any(target > 1):
        raise ValueError("Target transmission must be finite and within [0, 1]")
    try:
        np.broadcast_to(target, (len(angles), len(wavelengths)))
    except ValueError as exc:
        raise ValueError("Target must be scalar, [wavelength], or [angle, wavelength]") from exc


def load_config(path=DEFAULT_CONFIG):
    path = Path(path).resolve()
    cfg = json.loads(path.read_text(encoding="utf-8-sig"))
    material_dir = Path(cfg["material_dir"]).expanduser()
    if not material_dir.is_absolute():
        material_dir = path.parent / material_dir
    cfg["material_dir"] = str(material_dir.resolve())
    validate(cfg)
    return cfg
