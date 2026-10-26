"""Generic S-spectrum matching; each target binds S to physical R, T or A."""

import numpy as np


SCORE_FORMULA = "1 - sum_b(normalized_weight_b * mean(abs(S_b - S_target_b)))"


class SpectralObjective:
    """Average error within each band, then average terms using their weights.

    Bounds are inclusive. Targets may be scalar, [band wavelength], or
    [angle, band wavelength]. Samples outside all target bands are unscored.
    Without spectral_targets, preserve the historical target_transmission loss.
    S is the response selected by each term's quantity (R/T/A), not a fourth
    optical channel. One score formula is shared by all supported tasks.
    """

    def __init__(self, cfg, wavelengths_nm, angles_deg):
        self.wave = np.asarray(wavelengths_nm, dtype=np.float64)
        self.shape = (len(angles_deg), len(self.wave))
        self.mode = "spectral_targets" if "spectral_targets" in cfg else "target_transmission"
        if self.mode == "spectral_targets":
            specifications = cfg["spectral_targets"]
            if not isinstance(specifications, list) or not specifications:
                raise ValueError("spectral_targets must be a nonempty list")
        else:
            if "target_transmission" not in cfg:
                raise ValueError("Provide spectral_targets or target_transmission")
            specifications = [{"quantity": "T", "target": cfg["target_transmission"]}]
        self.terms = []
        specified = {q: np.full(self.shape, np.nan) for q in "RTA"}
        for i, specification in enumerate(specifications):
            prefix = f"spectral_targets[{i}]"
            if not isinstance(specification, dict):
                raise ValueError(f"{prefix} must be an object")
            unknown = set(specification) - {"quantity", "wavelength_range_nm", "target", "weight"}
            if unknown:
                raise ValueError(f"{prefix}: unknown fields {sorted(unknown)}")
            quantity = specification.get("quantity")
            if quantity not in ("R", "T", "A"):
                raise ValueError(f"{prefix}.quantity must be R, T or A")
            weight = specification.get("weight", 1.0)
            if (isinstance(weight, bool) or not isinstance(weight, (int, float))
                    or not np.isfinite(weight) or weight <= 0):
                raise ValueError(f"{prefix}.weight must be positive and finite")
            if "wavelength_range_nm" in specification:
                bounds = np.asarray(specification["wavelength_range_nm"], dtype=np.float64)
                if (bounds.shape != (2,) or not np.isfinite(bounds).all() or bounds[0] > bounds[1]
                        or bounds[0] < self.wave[0] or bounds[1] > self.wave[-1]):
                    raise ValueError(f"{prefix}: wavelength_range_nm must lie inside the configured wavelength grid")
                mask = (self.wave >= bounds[0]) & (self.wave <= bounds[1])
            else:
                bounds = np.asarray([self.wave[0], self.wave[-1]])
                mask = np.ones(len(self.wave), dtype=bool)
            if not mask.any():
                raise ValueError(f"{prefix}: no sampled wavelengths in target band")
            if "target" not in specification:
                raise ValueError(f"{prefix}.target is required")
            try:
                target = np.asarray(specification["target"], dtype=np.float64)
                target = np.broadcast_to(target, (self.shape[0], int(mask.sum()))).copy()
            except (ValueError, TypeError) as error:
                raise ValueError(f"{prefix}.target must be scalar, [band wavelength] or [angle, band wavelength]") from error
            if not np.isfinite(target).all() or np.any(target < 0) or np.any(target > 1):
                raise ValueError(f"{prefix}.target must be finite and within [0, 1]")
            previous = specified[quantity][:, mask]
            if np.any(np.isfinite(previous) & (np.abs(previous-target) > 1e-8)):
                raise ValueError(f"{prefix}: conflicting {quantity} targets at overlapping wavelengths/angles")
            specified[quantity][:, mask] = target
            self.terms.append({"quantity": quantity, "mask": mask, "target": target,
                               "weight": float(weight), "bounds": bounds.tolist()})
        # Reject exact target spectra that already contradict passive R+T+A=1.
        stacked = np.stack(list(specified.values()))
        count = np.isfinite(stacked).sum(axis=0)
        total = np.nansum(stacked, axis=0)
        if np.any(total > 1 + 1e-8) or np.any((count == 3) & (np.abs(total-1) > 1e-8)):
            raise ValueError("Incompatible spectral targets: require passive R+T+A=1 at each wavelength/angle; "
                             "check overlapping bands and shared endpoints")
        # Scale first to avoid overflow for large but finite user weights.
        weights = np.asarray([term["weight"] for term in self.terms])
        weights = weights / weights.max()
        weights /= weights.sum()
        for term, weight in zip(self.terms, weights):
            term["normalized_weight"] = float(weight)
        quantities = {term["quantity"] for term in self.terms}
        if len(quantities) == 1 and all(np.all(term["target"] == 1) for term in self.terms):
            self.evaluation_metric = {"R": "reflection", "T": "transmission", "A": "absorption"}[quantities.pop()]
        else:
            self.evaluation_metric = "score"
        self.legacy_target = self.terms[0]["target"].copy() if self.mode == "target_transmission" else None

    def evaluate(self, reflection, transmission, absorption):
        """Compatibility entry point for callers supplying all three responses."""
        return self.evaluate_spectra({"R": reflection, "T": transmission, "A": absorption})

    def evaluate_spectra(self, spectra):
        """Match S_b to its target, average over each band's wavelengths/angles.

        Accept a mapping of physical response names to [angle, wavelength]
        arrays. Only required responses must be supplied; R-only, T-only and
        A-only objectives follow precisely the same path as mixed objectives.
        Unspecified spectral regions do not contribute. This is a curve error,
        NOT the error between the actual and target band means.
        """
        if not isinstance(spectra, dict) or set(spectra) - {"R", "T", "A"}:
            raise ValueError("Spectra must map physical quantities R/T/A to arrays; S is a generic symbol")
        missing = {term["quantity"] for term in self.terms} - spectra.keys()
        if missing:
            raise ValueError(f"Missing target response spectra: {', '.join(sorted(missing))}")
        for quantity, spectrum in spectra.items():
            if np.asarray(spectrum).shape != self.shape or not np.isfinite(spectrum).all():
                raise ValueError(f"Invalid {quantity} spectrum supplied to target objective")
        errors = []
        for term in self.terms:
            actual_s = np.asarray(spectra[term["quantity"]])[:, term["mask"]]
            target_s = term["target"]
            mae = float(np.mean(np.abs(actual_s-target_s)))
            errors.append({"quantity": term["quantity"], "weight": term["weight"],
                           "normalized_weight": term["normalized_weight"],
                           "wavelength_range_nm": term["bounds"],
                           "wavelengths_nm": self.wave[term["mask"]].tolist(),
                           "target": term["target"].tolist(), "mae": mae,
                           "mean_actual": float(actual_s.mean()), "min_actual": float(actual_s.min()),
                           "max_actual": float(actual_s.max())})
        error = float(sum(term["normalized_weight"] * term["mae"] for term in errors))
        return {"mae": error, "score": 1-error, "objective_mode": self.mode,
                "score_formula": SCORE_FORMULA, "spectral_aggregation": "weighted_band_mae",
                "evaluation_metric": self.evaluation_metric, "target_errors": errors}


def objective_for_config(cfg):
    # Local import avoids a config/target-validation import cycle.
    from .config import grid
    return SpectralObjective(cfg, grid(cfg["wavelengths_nm"], "wavelengths_nm"),
                             grid(cfg["angles_deg"], "angles_deg"))
