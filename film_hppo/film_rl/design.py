"""Compact persisted designs; full spectra stay available inside FilmOptics."""

from copy import deepcopy

import numpy as np


def design_key(report):
    """Feasible first with bounds, then penalized reward; legacy score otherwise."""
    score = float(report.get('score', report.get('reward', 0.)))
    if report.get('has_constraints', False):
        return (int(report['constraints_satisfied']), float(report['reward']))
    if report.get('reward_objective') == 'enhancement':
        return (1, float(report['reward']))
    return (1, score)


def search_target_reached(report, target_score):
    metric = report['reward'] if report.get('reward_objective') == 'enhancement' else report['score']
    return (metric >= target_score and
            (not report.get('has_constraints', False) or report['constraints_satisfied']))


def _grid_summary(values):
    values = np.asarray(values, dtype=float)
    result = {"start": float(values[0]), "stop": float(values[-1]), "num": len(values)}
    if len(values) > 1 and np.allclose(np.diff(values), np.diff(values)[0], rtol=1e-10, atol=1e-10):
        result["step"] = float(values[1]-values[0])
    return result


def compact_design(report):
    """Summarize export data without changing the report used by training.

    Retain exact score/reward, layer order/thickness and source identity. Full
    spectra can be recomputed with the original config and evaluate CLI.
    """
    if not isinstance(report, dict) or not {"layers", "score", "target_errors"} <= report.keys():
        return report
    if report.get("design_format") == "compact_v1":
        return report
    compact = {key: deepcopy(report[key]) for key in (
        "layer_count", "total_thickness_nm", "score", "reward", "mae", "evaluation_metric"
    ) if key in report}
    compact["design_format"] = "compact_v1"
    compact["value_units"] = ("R/T/A, targets and score are fractions; enhancement_factor is a ratio; reward=chi-penalty"
                              if report.get('reward_objective') == 'enhancement' else
                              "R/T/A, target, score and reward use fractions, not percentages")
    compact["averaging"] = "Uniform mean over the sampled angles and wavelengths within each target band"
    spectra = {q: np.asarray(report[q], dtype=float) for q in "RTA" if q in report}
    wave = np.asarray(report.get("wavelengths_nm", []), dtype=float)
    bands = []
    for term in report["target_errors"]:
        item = {key: deepcopy(term[key]) for key in (
            "quantity", "wavelength_range_nm", "weight", "normalized_weight", "bounds"
        ) if key in term}
        target = np.asarray(term["target"], dtype=float)
        item["target"] = (float(target.flat[0]) if np.all(target == target.flat[0]) else {
            "type": "varying_spectrum", "mean": float(target.mean()),
            "min": float(target.min()), "max": float(target.max()),
            "note": "See original config for the wavelength/angle-dependent target"
        })
        item.update({key: term[key] for key in ("mean_actual", "mae", "min_actual", "max_actual")})
        if wave.size and len(spectra) == 3:
            low, high = term["wavelength_range_nm"]
            mask = (wave >= low) & (wave <= high)
            item["wavelength_count"] = int(mask.sum())
            item["angle_count"] = int(spectra["R"].shape[0])
            item["mean_R"] = float(spectra["R"][:, mask].mean())
            item["mean_T"] = float(spectra["T"][:, mask].mean())
            item["mean_A"] = float(spectra["A"][:, mask].mean())
        bands.append(item)
    compact["target_errors"] = bands
    compact["sampling"] = {key: _grid_summary(report[key]) for key in ("wavelengths_nm", "angles_deg") if key in report}
    for key in (
        "polarization", "objective_mode", "reward_mode", "reward_alpha", "reward_epsilon", "log_reward",
        "reward_objective", "enhancement_factor", "enhancement", "enhancement_settings",
        "score_formula", "reward_formula", "spectral_aggregation",
        "mean_transmission", "min_transmission", "mean_reflection", "min_reflection", "mean_absorption",
        "min_absorption", "absorption_scope", "forbid_adjacent_same_material", "material_source",
        "material_reference", "material_datasets", "substrate", "substrate_thickness_nm", "exit_n", "stack_order",
        "termination_reason", "eval_episode_return", "episode_info", "optimization", "seed_origin", "seed_origin_index",
        "application_metrics", "selection_method",
        "has_constraints", "base_reward", "constraint_penalty", "constraints_satisfied",
        "constraint_violations", "constraint_tolerance",
        "search_origin", "validation",
    ):
        if key in report:
            compact[key] = deepcopy(report[key])
    compact["spectra_note"] = "Full per-wavelength/per-angle spectra and optical-constant arrays omitted. Recompute with original config and evaluate --full-spectrum."
    compact["layers"] = deepcopy(report["layers"])
    return compact
