"""Explicit check / evaluate / train entry points. No training on import."""

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path

import numpy as np
import torch

from .config import DEFAULT_CONFIG, PROJECT_DIR, load_config


def check(cfg, output):
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
    # Initialize training-side shared state for DI-engine's GAE preprocessing;
    # never invoke learn_mode.forward, backward, or optimizer.step.
    policy = FilmPPOPolicy(compiled.policy, model=model, enable_field=["learn", "collect", "eval"])
    env = ThinFilmEnv(cfg)
    observation = env.reset()
    before = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    episode_steps = 0
    training_rewards = []
    for _ in range(cfg["max_layers"] + 1):
        result = policy.collect_mode.forward({0: torch.from_numpy(observation)})[0]
        action = {name: value.detach().cpu().numpy() for name, value in result["action"].items()}
        timestep = env.step(action)
        episode_steps += 1
        training_rewards.append(float(timestep.reward[0]))
        assert np.isclose(float(timestep.reward[0]), timestep.info["step_reward"])
        assert env.reward_space.contains(timestep.reward)
        observation = timestep.obs
        if timestep.done:
            break
    else:
        raise AssertionError("Episode did not terminate within its bound")
    assert env.optics.evaluation_count == episode_steps + 1, "One baseline plus one TMM evaluation per valid step"
    assert np.isclose(sum(training_rewards), timestep.info["episode_training_return"])
    assert np.isclose(sum(training_rewards),
                      timestep.info["mixed_score"] - timestep.info["baseline_mixed_score"], atol=2e-7)
    if timestep.info["termination_reason"] == "eos":
        assert float(timestep.reward[0]) == 0.0
    assert timestep.info["eval_episode_return"] == timestep.info["score"]
    assert all(torch.equal(tensor, before[name]) for name, tensor in model.state_dict().items())
    summary = deepcopy(timestep.info)
    summary["validation"] = {"config_compiled": True, "policy_forward": True, "training_executed": False,
                             "parameters_unchanged": True, "learning_update_checked": False,
                             "tmm_evaluations": env.optics.evaluation_count,
                             "episode_steps": episode_steps, "baseline_tmm_evaluations": 1,
                             "per_step_tmm": True, "telescoping_return_checked": True,
                             "step_rewards": training_rewards}
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
    write_json(output, summary)
    print(json.dumps({"status": "check passed; no training", "layers": summary["layer_count"],
                      "T_mean": summary["mean_transmission"], "final_mixed_score": summary["mixed_score"],
                      "last_step_reward": summary["step_reward"],
                      "episode_training_return": summary["episode_training_return"],
                      "eval_score": summary["score"], "reward_alpha": summary["reward_alpha"],
                      "output": str(Path(output).resolve())}, indent=2))


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def evaluate(cfg, design, output):
    from .optics import FilmOptics
    data = json.loads(Path(design).read_text(encoding="utf-8-sig"))
    optics = FilmOptics(cfg)
    layers = []
    for layer in data["layers"]:
        material_id = layer["material_id"]
        if "material" in layer and layer["material"] != optics.names[material_id]:
            raise ValueError("Material ID/name mismatch: check the config material ordering")
        layers.append((material_id, layer["thickness_nm"]))
    report = optics.evaluate(layers)
    write_json(output, report)
    print(json.dumps({"reward": report["reward"], "mean_transmission": report["mean_transmission"],
                      "score": report["score"], "reward_alpha": report["reward_alpha"],
                      "output": str(Path(output).resolve())}, indent=2))


def train(cfg):
    from ding.entry import serial_pipeline_onpolicy
    from ding.utils import set_pkg_seed
    from .hppo import build_configs, build_model
    from .optics import FilmOptics

    FilmOptics(cfg)  # Fail early on missing/incompatible optical data.
    settings = cfg["training"]
    set_pkg_seed(settings["seed"], use_cuda=settings["cuda"])
    main, create = build_configs(cfg, new_run=True)
    model = build_model(cfg, main)
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
    parser.add_argument("command", choices=["check", "evaluate", "train"])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--design", type=Path, help="Design JSON for evaluate")
    parser.add_argument("--output", type=Path, default=Path("outputs/evaluation.json"))
    args = parser.parse_args()
    if args.command == "evaluate" and args.design is None:
        parser.error("evaluate requires --design")
    cfg = load_config(args.config)
    if args.command == "train":
        train(cfg)
    elif args.command == "evaluate":
        evaluate(cfg, args.design, args.output)
    else:
        check(cfg, args.output)


if __name__ == "__main__":
    main()
