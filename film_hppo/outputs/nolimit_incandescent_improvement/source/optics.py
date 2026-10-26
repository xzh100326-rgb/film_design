"""YAML or fixed 193 nm coatings, tabulated .sub substrates and TMM-fast."""

import json
import math
from pathlib import Path

import numpy as np
import torch
from tmm_fast import coh_tmm

from .config import THICKNESS_TOL_NM, grid, material_source, validate
from .refractiveindex import load_coatings, resolve_materials
from .targets import SpectralObjective


def read_substrate_index(path, wavelengths_nm):
    path = Path(path)
    if path.suffix.lower() != ".sub":
        raise ValueError("Substrate optical data must be a .sub file")
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if data.get("nType", 0) != 0 or data.get("kType", 0) != 0:
        raise ValueError(f"{path}: only tabulated n/k data are supported")
    arrays = [np.asarray(data[key], dtype=np.float64) for key in ("wavelength", "n", "k")]
    wave, n, k = arrays
    if any(a.ndim != 1 for a in arrays) or not (len(wave) == len(n) == len(k) >= 2):
        raise ValueError(f"{path}: wavelength/n/k must be equally sized 1-D arrays")
    if not all(np.isfinite(a).all() for a in arrays):
        raise ValueError(f"{path}: nonfinite optical data")
    if not np.all(np.diff(wave) > 0) or np.any(wave <= 0) or np.any(n <= 0) or np.any(k < 0):
        raise ValueError(f"{path}: require ordered wavelengths, n > 0 and k >= 0")
    if wavelengths_nm.min() < wave[0] or wavelengths_nm.max() > wave[-1]:
        raise ValueError(f"{path.name}: requested wavelength outside [{wave[0]}, {wave[-1]}] nm; no extrapolation")
    # Explicit linear interpolation, including measured absorption. N = n + i*k.
    return np.interp(wavelengths_nm, wave, n) + 1j * np.interp(wavelengths_nm, wave, k)


class FilmOptics:
    def __init__(self, cfg):
        validate(cfg)
        cfg = resolve_materials(cfg, interactive=False)
        self.cfg = cfg
        self.wavelengths_nm = grid(cfg["wavelengths_nm"], "wavelengths_nm")
        self.angles_deg = grid(cfg["angles_deg"], "angles_deg")
        directory = Path(cfg["material_dir"])
        self.material_source = material_source(cfg)
        self.names, self.material_indices, self.material_datasets = load_coatings(cfg, self.wavelengths_nm)
        self.substrate_index = read_substrate_index(directory / cfg["substrate_file"], self.wavelengths_nm)
        self.substrate_thickness_nm = cfg.get("substrate_thickness_nm")
        self.incident_index = np.full(self.wavelengths_nm.shape, complex(cfg["incident_n"]))
        self.objective = SpectralObjective(cfg, self.wavelengths_nm, self.angles_deg)
        self.target = self.objective.legacy_target
        self.evaluation_count = 0

    def evaluate(self, layers):
        """layers are (material_id, thickness_nm), ordered from incident side to substrate.

        With no substrate thickness, T enters the semi-infinite substrate and
        A excludes substrate absorption. With a finite substrate, coherent TMM
        includes its rear interface; T enters exit_n, and A includes absorption
        in both the coatings and the finite substrate.
        """
        layers = list(layers)
        if len(layers) > self.cfg["max_layers"]:
            raise ValueError("Too many coating layers")
        for position, (material_id, thickness) in enumerate(layers):
            if (isinstance(material_id, bool) or not isinstance(material_id, (int, np.integer))
                    or not 0 <= material_id < len(self.names)):
                raise ValueError(f"Material ID must be an integer in [0, {len(self.names) - 1}]")
            if not np.isfinite(thickness) or not self.cfg["thickness_min_nm"] <= thickness <= self.cfg["thickness_max_nm"]:
                raise ValueError("Film thickness outside configured bounds")
            if (self.cfg.get("forbid_adjacent_same_material", True) and position > 0
                    and material_id == layers[position - 1][0]):
                raise ValueError("Adjacent coating layers cannot use the same material")
        total_thickness = float(sum(d for _, d in layers))
        budget = self.cfg.get("max_total_thickness_nm")
        if budget is not None and total_thickness > budget + THICKNESS_TOL_NM:
            raise ValueError("Total coating thickness exceeds max_total_thickness_nm")
        indices = [self.incident_index] + [self.material_indices[i] for i, _ in layers] + [self.substrate_index]
        # Explicit batch dimension: N[S,L,W], d[S,L], theta[A], lambda[W].
        # Both wavelength and thickness must be converted from nm to metres.
        thicknesses_nm = [np.inf] + [d for _, d in layers]
        if self.substrate_thickness_nm is not None:
            indices.append(np.full_like(self.incident_index, complex(self.cfg.get("exit_n", 1.0))))
            thicknesses_nm.append(self.substrate_thickness_nm)
        n_stack = np.stack(indices)[None, :, :]
        thickness_m = np.asarray([thicknesses_nm + [np.inf]]) * 1e-9
        pols = ("s", "p") if self.cfg["polarization"] == "unpolarized" else (self.cfg["polarization"],)
        results = []
        with torch.no_grad():
            for pol in pols:
                result = coh_tmm(pol, n_stack.copy(), thickness_m.copy(), np.deg2rad(self.angles_deg),
                                 self.wavelengths_nm * 1e-9, device=self.cfg["tmm_device"])
                results.append({key: self._array(result[key])[0] for key in ("R", "T")})
        reflection = np.mean([r["R"] for r in results], axis=0)
        transmission = np.mean([r["T"] for r in results], axis=0)
        for name, spectrum in (("R", reflection), ("T", transmission)):
            if spectrum.shape != self.objective.shape or not np.isfinite(spectrum).all():
                raise RuntimeError(f"Invalid TMM {name} shape or nonfinite result")
            if np.any(spectrum < -1e-7) or np.any(spectrum > 1 + 1e-7):
                raise RuntimeError(f"Nonphysical TMM {name}; check material convention and solver")
        if np.any(reflection + transmission > 1 + 1e-7):
            raise RuntimeError("TMM violated passive-stack energy conservation")
        reflection, transmission = np.clip(reflection, 0, 1), np.clip(transmission, 0, 1)
        absorption = np.clip(1-reflection-transmission, 0, 1)
        # S is selected independently per target term: quantity=R, T or A.
        # All tasks share the same spectral-curve error and PPO reward path.
        objective = self.objective.evaluate_spectra({"R": reflection, "T": transmission, "A": absorption})
        error, score = objective["mae"], objective["score"]
        # Transform the weighted target error once, after aggregating all bands.
        alpha = float(self.cfg.get("reward_alpha", 0.0))
        epsilon = float(self.cfg.get("reward_epsilon", 1e-6))
        log_reward = -math.log10(max(error, epsilon)) / -math.log10(epsilon)
        reward = (1.0 - alpha) * score + alpha * log_reward
        reward_formula = (objective["score_formula"] if alpha == 0.0 else
                          "(1 - reward_alpha) * score + reward_alpha * log_reward")
        self.evaluation_count += 1
        return {
            "reward": reward,
            "reward_formula": reward_formula,
            "log_reward": log_reward,
            "reward_mode": "linear" if alpha == 0.0 else "log" if alpha == 1.0 else "linear_log_mixture",
            "reward_alpha": alpha,
            "reward_epsilon": epsilon,
            **objective,
            "mean_transmission": float(transmission.mean()),
            "min_transmission": float(transmission.min()),
            "mean_reflection": float(reflection.mean()),
            "min_reflection": float(reflection.min()),
            "mean_absorption": float(absorption.mean()),
            "min_absorption": float(absorption.min()),
            "wavelengths_nm": self.wavelengths_nm.tolist(),
            "angles_deg": self.angles_deg.tolist(),
            "polarization": self.cfg["polarization"],
            "target_transmission": self.target.tolist() if self.target is not None else None,
            "spectral_targets": self.cfg.get("spectral_targets"),
            "T": transmission.tolist(), "R": reflection.tolist(),
            "A": absorption.tolist(),
            "absorption_scope": ("finite coating layers and finite substrate"
                                 if self.substrate_thickness_nm is not None
                                 else "finite coating layers only; excludes the semi-infinite substrate"),
            "layer_count": len(layers),
            "total_thickness_nm": total_thickness,
            "forbid_adjacent_same_material": self.cfg.get("forbid_adjacent_same_material", True),
            "material_source": self.material_source,
            "material_reference": self.cfg.get("fixed_materials_reference") if self.material_source == "fixed_193nm" else None,
            "material_datasets": self.material_datasets,
            "coating_optical_constants": [
                {"material": name, "n": index.real.tolist(), "k": index.imag.tolist()}
                for name, index in zip(self.names, self.material_indices)
            ],
            "layers": [{"material_id": int(i), "material": self.names[i], "thickness_nm": float(d)} for i, d in layers],
            "substrate": self.cfg["substrate_file"],
            "substrate_thickness_nm": self.substrate_thickness_nm,
            "exit_n": self.cfg.get("exit_n", 1.0) if self.substrate_thickness_nm is not None else None,
            "substrate_optical_constants": {
                "n": self.substrate_index.real.tolist(), "k": self.substrate_index.imag.tolist()
            },
            "stack_order": ("incident medium -> layers in listed order -> finite substrate -> exit medium"
                            if self.substrate_thickness_nm is not None
                            else "incident medium -> layers in listed order -> semi-infinite substrate"),
        }

    @staticmethod
    def _array(value):
        return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
