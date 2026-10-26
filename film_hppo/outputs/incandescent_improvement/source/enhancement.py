"""OML-PPO Appendix B model, with explicit numerical/thermal closure assumptions.

This is an offline evaluation metric, independent of the PPO training reward.
It is a reflection-only black-emitter model, not a tungsten/material heat model.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.constants import h, c, k, Stefan_Boltzmann
from scipy.integrate import simpson
from scipy.optimize import brentq


DATA_DIR = Path(__file__).with_name("data")
PAPER_URL = "https://arxiv.org/html/2006.11940#A2"


def positive(value, name):
    if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return float(value)


def planck_exitance(wavelengths_nm, temperature_k):
    """Hemispherical blackbody spectral exitance M_lambda, W m^-2 nm^-1.

    M_lambda=pi*B_lambda. The per-metre to per-nm conversion is explicit.
    """
    wave = np.asarray(wavelengths_nm, dtype=float)
    if not np.isfinite(wave).all() or np.any(wave <= 0):
        raise ValueError("Wavelengths must be positive and finite")
    temperature_k = positive(temperature_k, "temperature_k")
    metres = wave * 1e-9
    exponent = h*c/(metres*k*temperature_k)
    # exp(-x)/(1-exp(-x)) avoids overflow in the ultraviolet/Wien tail.
    occupancy = np.exp(-exponent)/(-np.expm1(-exponent))
    return (2*np.pi*h*c*c / metres**5)*occupancy*1e-9


def hemisphere_quadrature(order=64):
    """Gauss-Legendre on u=sin(theta)^2: du=2 sin(theta)cos(theta)dtheta."""
    if isinstance(order, bool) or not isinstance(order, int) or order < 2:
        raise ValueError("angle_order must be an integer >= 2")
    x, weights = np.polynomial.legendre.leggauss(order)
    return np.rad2deg(np.arcsin(np.sqrt((x+1)/2))), weights/2


def eye_response(path=None):
    """Energy-based Sharpe 2005 Eq.5. An explicit alternate CSV is supported."""
    path = Path(path) if path else DATA_DIR/"sharpe2005_vlambda.csv"
    raw = path.read_bytes()
    values = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    if (values.shape[1] != 2 or len(values) < 3 or not np.isfinite(values).all()
            or np.any(np.diff(values[:, 0]) <= 0) or np.any(values[:, 1] < 0)
            or values[:, 1].max() <= 0):
        raise ValueError("Eye CSV must contain a header and ordered wavelength_nm,V_energy columns")
    metadata_path = path.with_suffix(".source.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    digest = hashlib.sha256(raw).hexdigest()
    if metadata.get("data_sha256", digest) != digest:
        raise ValueError("Eye response differs from its recorded source checksum")
    return values[:, 0], values[:, 1], {**metadata, "file": str(path.resolve()), "sha256": digest}


def validate_curve(wavelengths_nm, reflection):
    wave, reflection = np.asarray(wavelengths_nm, float), np.asarray(reflection, float)
    if (wave.ndim != 1 or len(wave) < 3 or reflection.shape != wave.shape
            or not np.isfinite(wave).all() or not np.isfinite(reflection).all()
            or np.any(wave <= 0) or np.any(np.diff(wave) <= 0)
            or np.any(reflection < -1e-9) or np.any(reflection > 1+1e-9)):
        raise ValueError("Require ordered positive wavelengths and finite matching reflectance in [0,1]")
    return wave, np.clip(reflection, 0, 1)


def enhancement(wavelengths_nm, hemispherical_r, *, view_factor=1., power_w=100.,
                area_mm2=20., radiating_sides=2., outside_reflectance=0.,
                visible_range_nm=(400., 780.), eye_csv=None):
    """Compute paper chi at equal power; all unmeasured thermal tails are explicit.

    Default normalization is two radiating faces of area 20 mm^2, giving bare
    T0=2576.81 K, close to the reported 2578 K. Two faces are an implementation
    inference, not an explicitly stated Appendix B geometry. No fit to chi=16.60.
    Outside the measured spectral window R is constant (default zero), never nk
    extrapolation. Integrate the full blackbody tail analytically via sigma*T^4.
    """
    wave, reflection = validate_curve(wavelengths_nm, hemispherical_r)
    for name, value in (("view_factor", view_factor), ("outside_reflectance", outside_reflectance)):
        if isinstance(value, bool) or not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be within [0,1]")
    power_w = positive(power_w, "power_w")
    area = positive(area_mm2, "area_mm2")*1e-6*positive(radiating_sides, "radiating_sides")
    bounds = np.asarray(visible_range_nm, float)
    if bounds.shape != (2,) or not np.isfinite(bounds).all() or not 0 < bounds[0] < bounds[1]:
        raise ValueError("visible_range_nm must contain increasing positive bounds")
    if wave[0] > bounds[0] or wave[-1] < bounds[1]:
        raise ValueError("Optical wavelengths must cover the entire visible integration interval")
    eye_wave, eye_v, eye_source = eye_response(eye_csv)
    if eye_wave[0] > bounds[0] or eye_wave[-1] < bounds[1]:
        raise ValueError("Eye response must cover the visible integration interval")
    f2 = view_factor**2
    t0 = (power_w/(area*Stefan_Boltzmann))**.25

    def net_power(temp):
        # Complement formulation remains stable for high in-window reflection.
        spectrum = planck_exitance(wave, temp)
        in_window = simpson(spectrum, x=wave)
        tail = max(0., Stefan_Boltzmann*temp**4-in_window)
        return area*(simpson(spectrum*(1-f2*reflection), x=wave)
                     + (1-f2*outside_reflectance)*tail)

    lo, hi = t0, t0*2
    while net_power(hi) < power_w and hi < 1e6:
        hi *= 2
    if net_power(hi) < power_w:
        raise ValueError("No finite equilibrium in this perfectly reflecting thermal closure")
    if abs(net_power(t0)-power_w) < 1e-10*power_w:
        temp = t0
    else:
        temp = brentq(lambda t: net_power(t)-power_w, lo, hi, xtol=1e-8)
    # Resample R linearly in wavelength, integrate with the 1 nm eye table and
    # both exact visible endpoints. This is not a new TMM spectral evaluation.
    visible_wave = np.unique(np.r_[bounds, eye_wave[(eye_wave > bounds[0]) & (eye_wave < bounds[1])]])
    visual = np.interp(visible_wave, eye_wave, eye_v)
    effective = 1-f2*np.interp(visible_wave, wave, reflection)
    bare_light = simpson(planck_exitance(visible_wave, t0)*visual, x=visible_wave)
    light = simpson(planck_exitance(visible_wave, temp)*effective*visual, x=visible_wave)
    black_window = simpson(planck_exitance(wave, temp), x=wave)
    return {
        "chi": float(light/bare_light), "view_factor": float(view_factor),
        "temperature_k": float(temp), "bare_temperature_k": float(t0),
        "power_w": power_w, "area_mm2_per_face": float(area_mm2),
        "radiating_sides": float(radiating_sides), "outside_reflectance": float(outside_reflectance),
        "visible_range_nm": bounds.tolist(), "thermal_R_window_nm": [float(wave[0]), float(wave[-1])],
        "power_balance_residual_w": float(net_power(temp)-power_w),
        "blackbody_power_fraction_outside_window": float(1-black_window/(Stefan_Boltzmann*temp**4)),
        "eye_response": eye_source,
    }
