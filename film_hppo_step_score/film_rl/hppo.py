"""Native HPPO with matching EOS and adjacent-material masks in all forwards."""

from copy import deepcopy
from datetime import datetime
from pathlib import Path

import torch
from ding.model.template import VAC
from easydict import EasyDict

from .config import PROJECT_DIR
from .env import ThinFilmEnv  # register the environment
from .policy import FilmPPOPolicy  # register guarded native HPPO and command mode


class FilmVAC(VAC):
    def __init__(self, max_layers, min_layers, forbid_adjacent_same_material=True, **kwargs):
        super().__init__(**kwargs)
        self.max_layers = max_layers
        self.min_layers = min_layers
        self.forbid_adjacent_same_material = forbid_adjacent_same_material

    def forward(self, x, mode):
        result = super().forward(x, mode)
        if "logit" in result:
            logits = result["logit"]["action_type"]
            material_logits = logits[..., :5]
            if self.forbid_adjacent_same_material:
                encoded = x[..., :-1].reshape(*x.shape[:-1], self.max_layers, 6)
                last_index = (x[..., -1] * self.max_layers).round().long() - 1
                positions = torch.arange(self.max_layers, device=x.device)
                last_row = positions == last_index.unsqueeze(-1)
                # Empty stacks have last_index=-1, hence an all-zero mask.
                previous_material = (encoded[..., :5] * last_row.unsqueeze(-1)).sum(dim=-2)
                material_logits = material_logits.masked_fill(previous_material > 0.5, -1e9)
            legal_eos = x[..., -1] * self.max_layers >= self.min_layers - 1e-5
            eos = logits[..., 5].masked_fill(~legal_eos, -1e9)
            result["logit"]["action_type"] = torch.cat((material_logits, eos.unsqueeze(-1)), dim=-1)
            # Bound log-std before exponentiation in the used computation graph.
            # This prevents the unbounded independent Gaussian scale from collapsing
            # or overflowing; keep the same transform in collect/eval/learn modes.
            head = self.actor_head[1]
            args = result["logit"]["action_args"]
            args["sigma"] = head.log_sigma_param.clamp(-5.0, 2.0).exp().expand_as(args["mu"])
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
            "model": {"obs_shape": cfg["max_layers"] * 6 + 1,
                      "action_shape": {"action_type_shape": 6, "action_args_shape": 5},
                      "action_space": "hybrid", "encoder_hidden_size_list": [256, 256, 128],
                      "actor_head_hidden_size": 128, "critic_head_hidden_size": 128,
                      "sigma_type": "independent", "bound_type": None},
            "learn": {"epoch_per_collect": training["epoch_per_collect"], "batch_size": training["batch_size"],
                      "learning_rate": training["learning_rate"], "value_weight": 0.5,
                      "entropy_weight": training["entropy_weight"], "clip_ratio": 0.2,
                      "adv_norm": True, "value_norm": False, "ignore_done": False,
                      "target_kl": training.get("target_kl", 0.02),
                      "max_abs_log_ratio": training.get("max_abs_log_ratio", 10.0)},
            # gamma=1 avoids favouring early EOS just because terminal reward is delayed.
            "collect": {"n_sample": training["n_sample"], "discount_factor": 1.0, "gae_lambda": 0.95},
            "eval": {"evaluator": {"eval_freq": training["eval_freq"]}},
        },
    })
    create = EasyDict({
        "env": {"type": "duv_film", "import_names": ["film_rl.env"]},
        "env_manager": {"type": "base"},
        "policy": {"type": "film_hppo", "import_names": ["film_rl.policy"]},
    })
    return main, create


def build_model(cfg, main=None):
    if main is None:
        main, _ = build_configs(cfg)
    return FilmVAC(max_layers=cfg["max_layers"], min_layers=cfg["min_layers"],
                   forbid_adjacent_same_material=cfg.get("forbid_adjacent_same_material", True),
                   **deepcopy(main.policy.model))
