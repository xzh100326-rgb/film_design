"""Native HPPO over joint material tokens and continuous slot thicknesses."""

from copy import deepcopy
from datetime import datetime
from pathlib import Path

import torch
from ding.model.template import VAC
from easydict import EasyDict

from .config import PROJECT_DIR
from .actions import observation_size
from .env import ThinFilmEnv  # register the environment
from .policy import FilmPPOPolicy  # register guarded native HPPO and command mode


class FilmVAC(VAC):
    def __init__(self, max_layers, min_layers, material_count, fill_max, **kwargs):
        super().__init__(**kwargs)
        self.max_layers = max_layers
        self.min_layers = min_layers
        self.material_count = material_count
        self.fill_max = fill_max
        base = material_count + 1
        codes = torch.arange(base ** fill_max)
        self.register_buffer("joint_tokens", torch.stack([(codes // base ** i) % base for i in range(fill_max)], -1))

    def forward(self, x, mode):
        result = super().forward(x, mode)
        if "logit" in result:
            logits = result["logit"]["action_type"]
            offset = self.max_layers * (self.material_count + 3)
            pending = x[..., offset:-1].reshape(*x.shape[:-1], self.fill_max, self.max_layers)
            active = pending.sum(-1) > 0.5
            positions = pending.argmax(-1)
            tokens = self.joint_tokens
            invalid_unused = (~active.unsqueeze(-2)) & (tokens != 0)
            invalid_eos = active.unsqueeze(-2) & (positions.unsqueeze(-2) < self.min_layers) & (tokens == self.material_count)
            illegal = (invalid_unused | invalid_eos).any(-1)
            result["logit"]["action_type"] = logits.masked_fill(illegal, -1e9)
            # Bound log-std before exponentiation in the used computation graph.
            # This prevents the unbounded independent Gaussian scale from collapsing
            # or overflowing; keep the same transform in collect/eval/learn modes.
            head = self.actor_head[1]
            args = result["logit"]["action_args"]
            sigma = head.log_sigma_param.clamp(-5.0, 2.0).exp().expand_as(args["mu"])
            # Unselected ranks use the same fixed N(0,1) in collect/eval/learn,
            # so they contribute zero likelihood ratio and zero gradients.
            args["mu"] = torch.where(active, args["mu"], torch.zeros_like(args["mu"]))
            args["sigma"] = torch.where(active, sigma, torch.ones_like(sigma))
        return result


def build_configs(cfg, new_run=False):
    training = cfg["training"]
    exp_path = Path(training["exp_name"])
    if exp_path.is_absolute() or ".." in exp_path.parts:
        raise ValueError("training.exp_name must be a relative path inside this project (for DI-engine Windows logs)")
    if new_run and (PROJECT_DIR / exp_path).exists():
        # Resolve the final name BEFORE deriving env.result_dir. Otherwise
        # DI-engine renames exp_name later, leaving designs in the old run.
        base = exp_path
        suffix = datetime.now().strftime("_%y%m%d_%H%M%S")
        exp_path = base.with_name(base.name + suffix)
        index = 1
        while (PROJECT_DIR / exp_path).exists():
            exp_path = base.with_name(base.name + suffix + f"_{index}")
            index += 1
    exp_name = str(exp_path)
    exp_path = PROJECT_DIR / exp_path
    env_cfg = deepcopy(cfg)
    env_cfg.pop("training", None)
    env_cfg.update(
        collector_env_num=training["collector_env_num"], evaluator_env_num=training["evaluator_env_num"],
        n_evaluator_episode=training["n_evaluator_episode"], stop_value=training["stop_value"],
        result_dir=str(exp_path / "designs"), manager={"shared_memory": False},
    )
    main = EasyDict({
        "exp_name": exp_name, "env": env_cfg,
        "policy": {
            "cuda": training["cuda"], "action_space": "hybrid", "recompute_adv": False,
            "model": {"obs_shape": observation_size(cfg),
                      "action_shape": {"action_type_shape": (len(cfg["material_files"]) + 1) ** cfg["fill_max"],
                                       "action_args_shape": cfg["fill_max"]},
                      "action_space": "hybrid", "encoder_hidden_size_list": [256, 256, 128],
                      "actor_head_hidden_size": 128, "critic_head_hidden_size": 128,
                      "sigma_type": "independent", "bound_type": None},
            "learn": {"epoch_per_collect": training["epoch_per_collect"], "batch_size": training["batch_size"],
                      "learning_rate": training["learning_rate"], "value_weight": 0.5,
                      "entropy_weight": training["entropy_weight"], "clip_ratio": 0.2,
                      "adv_norm": True, "value_norm": False, "ignore_done": False,
                      "target_kl": training.get("target_kl", 0.02),
                      "max_abs_log_ratio": training.get("max_abs_log_ratio", 10.0)},
            # Absolute per-step scores intentionally optimize a sum over the trajectory.
            "collect": {"n_sample": training["n_sample"], "discount_factor": 1.0, "gae_lambda": 0.95},
            "eval": {"evaluator": {"eval_freq": training["eval_freq"]}},
        },
    })
    create = EasyDict({
        "env": {"type": "duv_masked_film", "import_names": ["film_rl.env"]},
        "env_manager": {"type": "base"},
        "policy": {"type": "film_hppo", "import_names": ["film_rl.policy"]},
    })
    return main, create


def build_model(cfg, main=None):
    if main is None:
        main, _ = build_configs(cfg)
    return FilmVAC(max_layers=cfg["max_layers"], min_layers=cfg["min_layers"],
                   material_count=len(cfg["material_files"]), fill_max=cfg["fill_max"],
                   **deepcopy(main.policy.model))
