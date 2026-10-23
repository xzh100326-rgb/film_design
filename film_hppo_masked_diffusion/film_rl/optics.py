"""Read tabulated .lm/.sub data and evaluate air/coatings/CaF2 with TMM-fast."""

import json
from pathlib import Path

import numpy as np
import torch
from tmm_fast import coh_tmm

from .config import THICKNESS_TOL_NM, grid, validate


def read_index(path, wavelengths_nm):
    path = Path(path)
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
        self.cfg = cfg
        self.wavelengths_nm = grid(cfg["wavelengths_nm"], "wavelengths_nm")
        self.angles_deg = grid(cfg["angles_deg"], "angles_deg")
        directory = Path(cfg["material_dir"])
        self.names = [Path(f).stem for f in cfg["material_files"]]
        self.material_indices = np.stack([read_index(directory / f, self.wavelengths_nm) for f in cfg["material_files"]])
        self.substrate_index = read_index(directory / cfg["substrate_file"], self.wavelengths_nm)
        self.incident_index = np.full(self.wavelengths_nm.shape, complex(cfg["incident_n"]))
        self.target = np.broadcast_to(np.asarray(cfg["target_transmission"], dtype=np.float64),
                                      (len(self.angles_deg), len(self.wavelengths_nm))).copy()
        self.evaluation_count = 0

    def evaluate_slots(self, slots):
        """None is MASK; the first {token: EOS} truncates the physical stack."""
        if not isinstance(slots, list) or len(slots) > self.cfg["max_layers"]:
            raise ValueError("slots must be a list within max_layers")
        layers, active_positions = [], []
        eos_position = None
        for position, slot in enumerate(slots):
            if slot is None:
                continue
            if not isinstance(slot, dict):
                raise ValueError("Each slot must be null, an EOS object, or a layer object")
            if slot.get("token") == "EOS":
                eos_position = position
                break
            material = slot["material_id"]
            if (isinstance(material, bool) or not isinstance(material, int)
                    or not 0 <= material < len(self.names)):
                raise ValueError("Invalid slot material ID")
            if "material" in slot and slot["material"] != self.names[material]:
                raise ValueError("Material ID/name mismatch: check the config ordering")
            layers.append((material, slot["thickness_nm"]))
            active_positions.append(position)
        report = self.evaluate(layers)
        report.update(active_positions=active_positions, eos_position=eos_position, slots=slots)
        return report

    def evaluate(self, layers):
        """layers are (material_id, thickness_nm), ordered from incident side to substrate.

        The substrate is semi-infinite: T is power entering CaF2 at the coated
        surface, not the transmission through both surfaces of a finite lens.
        """
        layers = list(layers)
        if len(layers) > self.cfg["max_layers"]:
            raise ValueError("Too many coating layers")
        for position, (material_id, thickness) in enumerate(layers):
            if isinstance(material_id, bool) or not isinstance(material_id, (int, np.integer)) or not 0 <= material_id < len(self.names):
                raise ValueError("Material ID outside configured material list")
            if not np.isfinite(thickness) or not self.cfg["thickness_min_nm"] <= thickness <= self.cfg["thickness_max_nm"]:
                raise ValueError("Film thickness outside configured bounds")
            if (self.cfg.get("forbid_adjacent_same_material", False) and position > 0
                    and material_id == layers[position - 1][0]):
                raise ValueError("Adjacent coating layers cannot use the same material")
        total_thickness = float(sum(d for _, d in layers))
        budget = self.cfg.get("max_total_thickness_nm")
        if budget is not None and total_thickness > budget + THICKNESS_TOL_NM:
            raise ValueError("Total coating thickness exceeds max_total_thickness_nm")
        indices = [self.incident_index] + [self.material_indices[i] for i, _ in layers] + [self.substrate_index]
        # Explicit batch dimension: N[S,L,W], d[S,L], theta[A], lambda[W].
        # Both wavelength and thickness must be converted from nm to metres.
        n_stack = np.stack(indices)[None, :, :]
        thickness_m = np.array([[np.inf] + [d * 1e-9 for _, d in layers] + [np.inf]])
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
            if spectrum.shape != self.target.shape or not np.isfinite(spectrum).all():
                raise RuntimeError(f"Invalid TMM {name} shape or nonfinite result")
            if np.any(spectrum < -1e-7) or np.any(spectrum > 1 + 1e-7):
                raise RuntimeError(f"Nonphysical TMM {name}; check material convention and solver")
        if np.any(reflection + transmission > 1 + 1e-7):
            raise RuntimeError("TMM violated passive-stack energy conservation")
        reflection, transmission = np.clip(reflection, 0, 1), np.clip(transmission, 0, 1)
        error = float(np.mean(np.abs(transmission - self.target)))
        mse = float(np.mean((transmission - self.target) ** 2))
        metric = self.cfg.get("reward_metric", "mae")
        self.evaluation_count += 1
        return {
            "reward": 1.0 - (error if metric == "mae" else mse),
            "mae": error,
            "mse": mse,
            "reward_metric": metric,
            "mean_transmission": float(transmission.mean()),
            "min_transmission": float(transmission.min()),
            "wavelengths_nm": self.wavelengths_nm.tolist(),
            "angles_deg": self.angles_deg.tolist(),
            "polarization": self.cfg["polarization"],
            "target_transmission": self.target.tolist(),
            "T": transmission.tolist(), "R": reflection.tolist(),
            "A": (1 - reflection - transmission).tolist(),
            "layer_count": len(layers),
            "total_thickness_nm": total_thickness,
            "forbid_adjacent_same_material": self.cfg.get("forbid_adjacent_same_material", False),
            "layers": [{"material_id": int(i), "material": self.names[i], "thickness_nm": float(d)} for i, d in layers],
            "substrate": self.cfg["substrate_file"],
            "stack_order": "incident medium -> layers in listed order -> semi-infinite substrate",
        }

    @staticmethod
    def _array(value):
        return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
