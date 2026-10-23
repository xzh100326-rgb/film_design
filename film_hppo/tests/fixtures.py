"""Synthetic optical fixtures, independent of mutable experiment configurations."""

import json
from pathlib import Path

import yaml


def base_config(directory):
    return {
        "material_dir": str(directory), "material_source": "fixed_193nm",
        "fixed_materials_193nm": [
            {"name": name, "n": n, "k": 0.}
            for name, n in zip(["MgF2", "LaF3", "AlF3", "GdF3"], [1.42, 1.696, 1.394, 1.676])],
        "substrate_file": "substrate.sub", "incident_n": 1., "wavelengths_nm": [193.],
        "angles_deg": [0., 45.], "polarization": "unpolarized", "target_transmission": 1.,
        "reward_alpha": .5, "reward_epsilon": 1e-6, "min_layers": 1, "max_layers": 20,
        "thickness_min_nm": 5., "thickness_max_nm": 100., "max_total_thickness_nm": 400.,
        "forbid_adjacent_same_material": True, "tmm_device": "cpu",
        "training": {"seed": 0, "cuda": False, "collector_env_num": 4, "evaluator_env_num": 1,
                     "n_evaluator_episode": 8, "n_sample": 1024, "batch_size": 128,
                     "epoch_per_collect": 10, "learning_rate": .0003, "entropy_weight": .001,
                     "target_kl": .02, "max_abs_log_ratio": 10., "eval_freq": 100,
                     "stop_value": .9999, "max_env_step": 1000000, "max_train_iter": 10000,
                     "exp_name": "runs/test_fixture"},
    }


def write_substrate(directory, name="substrate.sub", n=1.5, k=0.):
    path = Path(directory) / name
    path.write_text(json.dumps({"wavelength": [180., 2000.], "n": [n, n], "k": [k, k]}), encoding="utf-8")
    return path


def yaml_config(directory, count=5, optical_constants=None):
    cfg = base_config(directory)
    cfg.pop("fixed_materials_193nm")
    cfg.update(material_source="refractiveindex", refractiveindex_dir=str(Path(directory) / "database"), materials=[])
    values = optical_constants or [(f"M{i}", 1.4 + .05*i, .001) for i in range(count)]
    for name, n, k in values:
        relative = Path("data") / "main" / name / "nk" / "fixture.yml"
        path = Path(cfg["refractiveindex_dir"]) / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump({"COMMENTS": "Synthetic test data; not measured material constants",
                                      "DATA": [{"type": "tabulated nk", "data": f"0.180 {n} {k}\n2.000 {n} {k}\n"}]}),
                        encoding="utf-8")
        cfg["materials"].append({"name": name, "file": relative.as_posix()})
    write_substrate(directory)
    return cfg
