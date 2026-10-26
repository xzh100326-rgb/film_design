"""Lamp application diagnostics and bounded Pareto selection, independent of score."""
import numpy as np


IR_BANDS = ((710., 900.), (910., 1200.), (1210., 1700.),
            (1710., 2300.), (2310., 3000.))


def lamp_metrics(report):
    wave = np.asarray(report['wavelengths_nm'], float)
    angles = np.asarray(report['angles_deg'], float)
    spectra = {q: np.asarray(report[q], float) for q in 'RTA'}
    if any(x.shape != (len(angles), len(wave)) or not np.isfinite(x).all()
           for x in spectra.values()):
        raise ValueError('Lamp metrics require finite angle/wavelength spectra')
    def band(lo, hi):
        mask = (wave >= lo) & (wave <= hi)
        if not mask.any():
            raise ValueError('Lamp metrics require all UV/visible/IR bands')
        return {q: float(x[:, mask].mean()) for q, x in spectra.items()}
    visible = band(480., 700.)
    vis = (wave >= 480) & (wave <= 700)
    near = angles <= 60
    ir = [band(*bounds)['R'] for bounds in IR_BANDS]
    result = {'visible_T': visible['T'], 'visible_R': visible['R'], 'visible_A': visible['A'],
              'visible_T_p10': float(np.quantile(spectra['T'][:, vis], .1)),
              'visible_T_0_60': float(spectra['T'][near][:, vis].mean()),
              'UV_R': band(300., 470.)['R'], 'IR_R': band(710., 3000.)['R'],
              'IR_subband_R': ir, 'IR_min_subband_R': min(ir)}
    # Harmonic attainment penalizes a bad band; this is a selection aid, not a
    # substitute for the full vector or a claim that all sample points pass.
    vector = np.array([visible['T'], 1-visible['R'], result['UV_R'], *ir])
    weights = np.array([3., 1., 1., 2., 1.5, 1., .75, .5])
    result['balanced_harmonic'] = float(weights.sum() / (weights / np.maximum(vector, 1e-9)).sum())
    result['application_targets_met'] = bool(visible['T'] >= .70 and visible['R'] <= .25
        and visible['A'] <= .05 and result['UV_R'] >= .70 and result['IR_R'] >= .85
        and min(ir) >= .70)
    return result


def lamp_key(metrics):
    return (metrics['application_targets_met'], metrics['balanced_harmonic'],
            metrics['visible_T'], -metrics['visible_A'])


def lamp_vector(metrics):
    return np.asarray([metrics['visible_T'], -metrics['visible_R'], -metrics['visible_A'],
                       metrics['UV_R'], *metrics['IR_subband_R']])


def dominates(left, right, tolerance=1e-8):
    a, b = lamp_vector(left), lamp_vector(right)
    return bool(np.all(a >= b-tolerance) and np.any(a > b+tolerance))
