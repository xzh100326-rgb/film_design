"""Explicit check / evaluate / train entry points. No training on import."""

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

from .config import DEFAULT_CONFIG, PROJECT_DIR, load_config
from .design import compact_design


def load_initial_model(cfg, model):
    if not cfg["training"].get("initial_model"):
        return
    from .improve import problem_fingerprint
    checkpoint_path = Path(cfg["training"]["initial_model"])
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_DIR / checkpoint_path
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("problem_fingerprint") != problem_fingerprint(cfg):
        raise ValueError("initial_model physics/material/constraint fingerprint does not match this config")
    model.load_state_dict(checkpoint["model"], strict=True)
    print(f"Warm-start model: {checkpoint_path}")


def check(cfg, output, full_spectrum=False):
    from ding.config import compile_config
    from ding.envs import BaseEnvManager
    from ding.utils import set_pkg_seed
    from ding.worker import SampleSerialCollector
    from .env import ThinFilmEnv
    from .hppo import build_configs, build_model
    from .policy import FilmPPOPolicy

    set_pkg_seed(cfg["training"]["seed"], use_cuda=False)
    main, create = build_configs(cfg)
    main.policy.cuda = False
    # Compile the very same config path used by serial_pipeline_onpolicy.
    create.policy.type += "_command"
    compiled = compile_config(main, seed=cfg["training"]["seed"], auto=True, create_cfg=create, save_cfg=False)
    model = build_model(cfg, main)
    load_initial_model(cfg, model)
    # Initialize training-side shared state for DI-engine's GAE preprocessing;
    # never invoke learn_mode.forward, backward, or optimizer.step.
    policy = FilmPPOPolicy(compiled.policy, model=model, enable_field=["learn", "collect", "eval"])
    env = ThinFilmEnv(cfg)
    observation = env.reset()
    before = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    for _ in range(cfg["max_layers"] + 1):
        result = policy.collect_mode.forward({0: torch.from_numpy(observation)})[0]
        action = {name: value.detach().cpu().numpy() for name, value in result["action"].items()}
        timestep = env.step(action)
        observation = timestep.obs
        if timestep.done:
            break
    else:
        raise AssertionError("Episode did not terminate within its bound")
    assert env.optics.evaluation_count == 1, "TMM must evaluate once per completed stack"
    assert np.isclose(float(timestep.reward[0]), timestep.info["reward"])
    assert timestep.info['eval_episode_return'] == (
        timestep.info['reward'] if env.has_constraints or env.enhancement_mode else timestep.info['score'])
    assert all(torch.equal(tensor, before[name]) for name, tensor in model.state_dict().items())
    summary = deepcopy(timestep.info)
    summary["validation"] = {"config_compiled": True, "policy_forward": True, "training_executed": False,
                             "parameters_unchanged": True, "learning_update_checked": False,
                             "tmm_evaluations": env.optics.evaluation_count}
    env.close()
    # Exercise the DI-engine vector manager with the exact hybrid action dict.
    manager = BaseEnvManager(env_fn=[lambda: ThinFilmEnv(cfg)], cfg=BaseEnvManager.default_config())
    try:
        manager.seed(cfg["training"]["seed"])
        manager.launch()
        actions = policy.collect_mode.forward({i: torch.as_tensor(obs) for i, obs in manager.ready_obs.items()})
        manager.step({i: {k: v.detach().cpu().numpy() for k, v in item["action"].items()} for i, item in actions.items()})
        summary["validation"]["env_manager_step"] = True
    finally:
        manager.close()
    # Verify the real on-policy collector/GAE path without updating any parameter.
    manager = BaseEnvManager(env_fn=[lambda: ThinFilmEnv(cfg)], cfg=BaseEnvManager.default_config())
    collector = None
    previous_dir = Path.cwd()
    try:
        os.chdir(PROJECT_DIR)
        collector = SampleSerialCollector(compiled.policy.collect.collector, env=manager,
                                          policy=policy.collect_mode, exp_name="outputs/check_runtime")
        samples = collector.collect(n_sample=16, train_iter=0)
        assert len(samples) >= 16 and all(torch.isfinite(s["adv"]).all() for s in samples)
        assert all(torch.equal(tensor, before[name]) for name, tensor in model.state_dict().items())
        summary["validation"]["collector_samples"] = len(samples)
        summary["validation"]["gae_finite"] = True
    finally:
        if collector is not None:
            collector.close()
        manager.close()
        os.chdir(previous_dir)
    write_json(output, summary, full_spectrum=full_spectrum)
    print(json.dumps({"status": "check passed; no training", "layers": summary["layer_count"],
                      "T_mean": summary["mean_transmission"], "reward": summary["reward"],
                      "R_mean": summary["mean_reflection"], "A_mean": summary["mean_absorption"],
                      "evaluation_metric": summary["evaluation_metric"],
                      "eval_score": summary["score"], "reward_alpha": summary["reward_alpha"],
                      **{k: summary[k] for k in ('reward_objective', 'enhancement_factor') if k in summary},
                      **{k: summary[k] for k in ('base_reward', 'constraint_penalty', 'constraints_satisfied') if k in summary},
                      "output": str(Path(output).resolve())}, indent=2))


def write_json(path, data, full_spectrum=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data if full_spectrum else compact_design(data), indent=2, ensure_ascii=False), encoding="utf-8")


def evaluate(cfg, design, output, full_spectrum=False):
    from .optics import FilmOptics
    data = json.loads(Path(design).read_text(encoding="utf-8-sig"))
    optics = FilmOptics(cfg)
    layers = []
    for layer in data["layers"]:
        material_id = layer["material_id"]
        if isinstance(material_id, bool) or not isinstance(material_id, int) or not 0 <= material_id < len(optics.names):
            raise ValueError(f"Material ID must be an integer in [0, {len(optics.names)-1}]")
        if "material" in layer and layer["material"] != optics.names[material_id]:
            raise ValueError("Material ID/name mismatch: check the config material ordering")
        layers.append((material_id, layer["thickness_nm"]))
    report = optics.evaluate(layers)
    write_json(output, report, full_spectrum=full_spectrum)
    print(json.dumps({"reward": report["reward"], "mean_transmission": report["mean_transmission"],
                      "mean_reflection": report["mean_reflection"], "mean_absorption": report["mean_absorption"],
                      "evaluation_metric": report["evaluation_metric"],
                      "score": report["score"], "reward_alpha": report["reward_alpha"],
                      **{k: report[k] for k in ('reward_objective', 'enhancement_factor') if k in report},
                      **{k: report[k] for k in ('base_reward', 'constraint_penalty', 'constraints_satisfied') if k in report},
                      "output": str(Path(output).resolve())}, indent=2))


def train(cfg):
    from ding.entry import serial_pipeline_onpolicy
    from ding.utils import set_pkg_seed
    from .hppo import build_configs, build_model
    from .optics import FilmOptics

    optics = FilmOptics(cfg)  # Fail early on missing/incompatible optical data.
    if optics.objective.has_constraints:
        from .bounded_training import serial_pipeline_bounded
        serial_pipeline_onpolicy = serial_pipeline_bounded
    settings = cfg["training"]
    set_pkg_seed(settings["seed"], use_cuda=settings["cuda"])
    main, create = build_configs(cfg, new_run=True)
    model = build_model(cfg, main)
    load_initial_model(cfg, model)
    print(f"Experiment directory: {PROJECT_DIR / main.exp_name}")
    print(f"Best designs directory: {main.env.result_dir}")
    previous_dir = Path.cwd()
    try:
        os.chdir(PROJECT_DIR)
        serial_pipeline_onpolicy((main, create), seed=settings["seed"], model=model,
                                 max_env_step=settings["max_env_step"], max_train_iter=settings["max_train_iter"])
    finally:
        os.chdir(previous_dir)


def main():
    parser = argparse.ArgumentParser(description="193 nm coating design with DI-engine HPPO + TMM-fast")
    parser.add_argument("command", choices=["materials", "check", "evaluate", "train", "improve"])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--design", type=Path, help="Design JSON for evaluate or optional HPPO seed for improve")
    parser.add_argument("--design-dir", type=Path, help="For improve: recursively rank designs only in this directory")
    parser.add_argument("--top-k", type=int, help="For improve --design-dir: number of seeds to refine (default: 5)")
    parser.add_argument("--output", type=Path, default=Path("outputs/evaluation.json"),
                        help="Report JSON, or a new experiment directory for improve")
    parser.add_argument("--non-interactive", action="store_true", help="Require pinned YAML versions; never prompt")
    parser.add_argument("--full-spectrum", action="store_true", help="For check/evaluate: export full spectra instead of compact design summary")
    args = parser.parse_args()
    if args.design_dir is not None and (args.command != "improve" or args.design is not None):
        parser.error("--design-dir applies only to improve and cannot be combined with --design")
    if args.top_k is not None and (args.design_dir is None or args.top_k < 1):
        parser.error("--top-k requires --design-dir and must be positive")
    if args.design_dir is not None and not args.design_dir.is_dir():
        parser.error(f"Design directory does not exist: {args.design_dir}")
    if args.full_spectrum and args.command not in ("check", "evaluate"):
        parser.error("--full-spectrum applies only to check/evaluate")
    if args.command == "evaluate" and args.design is None:
        parser.error("evaluate requires --design")
    try:
        cfg = load_config(args.config, resolve_materials=True,
                          interactive=not args.non_interactive and sys.stdin.isatty(), persist_selection=True)
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        parser.error(str(error))
    if args.command == "materials":
        from .config import grid, material_source
        from .refractiveindex import load_coatings
        wave = grid(cfg["wavelengths_nm"], "wavelengths_nm")
        names, indices, datasets = load_coatings(cfg, wave)
        print(json.dumps({"material_source": material_source(cfg), "materials": cfg.get("materials"),
                          "wavelengths_nm": wave.tolist(),
                          "optical_constants": [{"name": name, "n": index.real.tolist(), "k": index.imag.tolist()}
                                                for name, index in zip(names, indices)],
                          "datasets": datasets, "training_executed": False},
                         ensure_ascii=False, indent=2))
    elif args.command == "train":
        train(cfg)
    elif args.command == "evaluate":
        evaluate(cfg, args.design, args.output, full_spectrum=args.full_spectrum)
    elif args.command == "improve":
        if args.output == Path("outputs/evaluation.json"):
            from datetime import datetime
            args.output = Path("outputs") / datetime.now().strftime("improvement_%Y%m%d_%H%M%S")
        from .improve import improve
        improve(cfg, args.output, args.design, design_dir=args.design_dir, top_k=args.top_k)
    else:
        check(cfg, args.output, full_spectrum=args.full_spectrum)


if __name__ == "__main__":
    main()
