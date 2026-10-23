"""Calculate Appendix B enhancement from one design, without starting training.

python calculate_enhancement.py --design runs/experiment/designs/best_xxx.json
"""
import argparse
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from evaluate_baseline import calculate_spectra, read_layers
from film_rl.config import grid, load_config, validate
from film_rl.enhancement import enhancement, hemisphere_quadrature, positive
from film_rl.optics import FilmOptics


def source_record(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    return raw, {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}


def design_config(design_path, config_path=None):
    """Read the nearest saved configuration; never execute a training snapshot."""
    if config_path is None:
        for parent in Path(design_path).resolve().parents:
            config_path = next((parent / name for name in
                                ("config_snapshot.json", "total_config.py")
                                if (parent / name).is_file()), None)
            if config_path is not None:
                break
        else:
            raise ValueError("No saved config found beside design; supply --config ORIGINAL_CONFIG.json")
    config_path = Path(config_path).resolve()
    raw, source = source_record(config_path)
    if config_path.suffix.lower() == ".json":
        cfg = load_config(config_path, resolve_materials=True, interactive=False)
    elif config_path.name == "total_config.py":
        tree = ast.parse(raw.decode("utf-8-sig"), filename=str(config_path))
        exp = next((node.value for node in tree.body if isinstance(node, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "exp_config" for t in node.targets)), None)
        if not isinstance(exp, ast.Dict):
            raise ValueError("Training snapshot must define a literal exp_config dictionary")
        env = next((value for key, value in zip(exp.keys, exp.values)
                    if isinstance(key, ast.Constant) and key.value == "env"), None)
        if not isinstance(env, ast.Dict):
            raise ValueError("Training snapshot has no literal env dictionary")
        # DI-engine's manager includes float('inf'); it is unrelated to optics.
        # All retained fields must be literals, including paths and material IDs.
        cfg = {ast.literal_eval(key): ast.literal_eval(value)
               for key, value in zip(env.keys, env.values)
               if not (isinstance(key, ast.Constant) and key.value == "manager")}
        for key in ("material_dir", "refractiveindex_dir"):
            if key in cfg:
                path = Path(cfg[key]).expanduser()
                cfg[key] = str((path if path.is_absolute() else config_path.parent / path).resolve())
        validate(cfg)
    else:
        raise ValueError("Config must be a JSON config or saved total_config.py")
    return cfg, source


def calculate(design_path, config_path=None, *, view_factors=(1., .95),
              wavelength_step_nm=2., angle_order=64, chunk_size=128,
              power_w=100., area_mm2=20., radiating_sides=2., outside_reflectance=0.):
    raw, source = source_record(design_path)
    data = json.loads(raw.decode("utf-8-sig"))
    cfg, config_source = design_config(design_path, config_path)
    cfg = deepcopy(cfg)
    wave = grid(cfg["wavelengths_nm"], "wavelengths_nm")
    step = positive(wavelength_step_nm, "wavelength_step_nm")
    cfg["wavelengths_nm"] = np.unique(np.r_[np.arange(wave[0], wave[-1], step), wave[-1]]).tolist()
    cfg["polarization"] = "unpolarized"
    # Small offline evaluations run predictably without sharing the training GPU.
    cfg["tmm_device"] = "cpu"
    angles, weights = hemisphere_quadrature(angle_order)
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    factors = list(view_factors)
    if not factors or len(set(factors)) != len(factors):
        raise ValueError("view_factors must be nonempty and distinct")
    settings = dict(power_w=power_w, area_mm2=area_mm2, radiating_sides=radiating_sides,
                    outside_reflectance=outside_reflectance)
    # Validate all thermal settings before expensive TMM work.
    for factor in factors:
        enhancement(np.asarray(cfg["wavelengths_nm"]), np.zeros(len(cfg["wavelengths_nm"])),
                    view_factor=factor, **settings)
    optics = FilmOptics(cfg)
    if (not np.allclose(optics.incident_index, 1) or not np.allclose(optics.substrate_index, 1)
            or optics.substrate_thickness_nm is not None):
        raise ValueError("Appendix B model requires air on both sides and no finite substrate")
    layers = read_layers(data, optics)
    saved_datasets = data.get("material_datasets", [])
    current = {row["material"]: row for row in optics.material_datasets}
    for row in saved_datasets:
        now = current.get(row["material"])
        if now is None or row.get("sha256") != now.get("sha256"):
            raise ValueError(f"Material data changed since design export: {row['material']}")
        if row.get("assumed_zero_k", False) != now.get("assumed_zero_k", False):
            raise ValueError(f"Material absorption assumption changed: {row['material']}")
    spectra = calculate_spectra(optics, layers, angles, chunk_size)
    hemi = {q: weights @ spectra[q] for q in "RTA"}
    results = [enhancement(optics.wavelengths_nm, hemi["R"], view_factor=factor, **settings)
               for factor in factors]
    visible = (optics.wavelengths_nm >= 480) & (optics.wavelengths_nm <= 700)
    return {
        "model": "OML-PPO Appendix B reflection-only, equal-input-power blackbody model",
        "assumptions": [
            "Unpolarized hemispherical R: integral of 2*sin(theta)*cos(theta)*R(theta).",
            "Effective emission = 1 - view_factor^2 * R; film absorption/re-emission is not modeled.",
            "Eye-weighted visible integral: 400-780 nm; training score/weights/bounds are not inputs.",
            "Default two emitting faces is an inferred normalization, not a specified paper geometry.",
            "Outside the optical window, R is constant at outside_reflectance (default 0); no nk extrapolation.",
        ],
        "design": source, "config": config_source,
        "layer_count": len(layers), "total_thickness_nm": sum(d for _, d in layers),
        "layers": data["layers"], "material_datasets": optics.material_datasets,
        "material_hashes_checked": bool(saved_datasets),
        "numerics": {"wavelength_step_nm": step, "wavelength_count": len(optics.wavelengths_nm),
                     "angle_order": angle_order, "wavelength_chunk_size": chunk_size, "device": "cpu"},
        "visible_480_700_hemispherical_mean":
            {q: float(hemi[q][visible].mean()) for q in "RTA"} if visible.any() else None,
        "enhancement": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", required=True, type=Path, help="Design JSON path")
    parser.add_argument("--config", type=Path, help="Original config; default: nearest saved training/improve snapshot")
    parser.add_argument("--view-factors", type=float, nargs="+", default=[1., .95])
    parser.add_argument("--wavelength-step-nm", type=float, default=2.)
    parser.add_argument("--angle-order", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--power-w", type=float, default=100.)
    parser.add_argument("--area-mm2", type=float, default=20., help="Emitting area per face")
    parser.add_argument("--radiating-sides", type=float, default=2.)
    parser.add_argument("--outside-reflectance", type=float, default=0.)
    parser.add_argument("--output", type=Path, help="Optional new result JSON; default: console only")
    args = parser.parse_args()
    try:
        if args.output and args.output.exists():
            raise FileExistsError(f"Output already exists: {args.output}")
        torch.set_num_threads(1)
        print("Recalculating hemispherical spectrum (CPU); no training is started...", flush=True)
        result = calculate(args.design, args.config, view_factors=args.view_factors,
                           wavelength_step_nm=args.wavelength_step_nm, angle_order=args.angle_order,
                           chunk_size=args.chunk_size, power_w=args.power_w, area_mm2=args.area_mm2,
                           radiating_sides=args.radiating_sides, outside_reflectance=args.outside_reflectance)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    except (ValueError, KeyError, OSError, SyntaxError) as error:
        parser.error(str(error))
    print(f"Design: {result['design']['path']}")
    print(f"Config: {result['config']['path']}")
    print(f"Layers: {result['layer_count']}; total thickness: {result['total_thickness_nm']:.3f} nm")
    for row in result["enhancement"]:
        print(f"f={row['view_factor']:g}: chi={row['chi']:.8f}; T={row['temperature_k']:.2f} K; "
              f"bare T={row['bare_temperature_k']:.2f} K")
    print(f"Model: reflection-only; 400-780 nm eye response; {args.power_w:g} W; "
          f"{args.radiating_sides:g} faces x {args.area_mm2:g} mm^2; "
          f"outside-window R={args.outside_reflectance:g}.")
    if args.output:
        print(f"Result JSON: {args.output.resolve()}")


if __name__ == "__main__":
    main()
