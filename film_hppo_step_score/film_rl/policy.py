"""Numerical guards around DI-engine's native hybrid PPO minibatch update.

The native discrete/continuous losses and optimizer are retained. Rollout GAE
and value targets stay fixed; excessive policy drift ends reuse of that rollout.
"""

import math
import warnings

import torch
from torch.distributions import Categorical, Independent, Normal, kl_divergence
from ding.policy import PPOPolicy
from ding.policy.command_mode_policy_instance import DummyCommandModePolicy
from ding.policy.common_utils import default_preprocess_learn
from ding.torch_utils import to_device
from ding.utils import POLICY_REGISTRY


def require_finite(value, name):
    if isinstance(value, dict):
        for key, item in value.items():
            require_finite(item, f"{name}.{key}")
    elif isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
        raise FloatingPointError(f"Nonfinite {name}; aborting before an optimizer update")


def scalar_action_sample(sample):
    """Prevent hybrid categorical actions [B,1] broadcasting to [B,B].

    DI-engine's decollation/collector may leave a one-element action tensor.
    Its hybrid preprocessing does not squeeze that dimension. Make each
    per-transition action a scalar so collating always produces exactly [B].
    Shallow-copy containers to preserve the original rollout.
    """
    action = dict(sample["action"])
    kind = torch.as_tensor(action["action_type"])
    if kind.numel() != 1 or not torch.isfinite(kind).all() or not torch.equal(kind, kind.round()):
        raise ValueError("Each categorical action must contain exactly one finite integer")
    action["action_type"] = kind.reshape(()).long()
    return {**sample, "action": action}


@torch.no_grad()
def policy_drift(new, old, action):
    """Exact KL(old || new) and max absolute sampled log-ratio, in float64.

    Checking *before* the native exp(logp_new-logp_old) avoids float32 overflow.
    Categorical KL and Gaussian KL are checked separately, like the two losses.
    """
    require_finite(new, "new_logits")
    require_finite(old, "rollout_logits")
    require_finite(action, "rollout_actions")
    for label, logits in (("new", new), ("old", old)):
        if (logits["action_args"]["sigma"] <= 0).any():
            raise FloatingPointError(f"{label} Gaussian sigma must be positive")
    discrete_new = Categorical(logits=new["action_type"].double())
    discrete_old = Categorical(logits=old["action_type"].double())
    continuous_new = Independent(Normal(new["action_args"]["mu"].double(), new["action_args"]["sigma"].double()), 1)
    continuous_old = Independent(Normal(old["action_args"]["mu"].double(), old["action_args"]["sigma"].double()), 1)
    discrete_action = action["action_type"].long()
    if discrete_action.shape != old["action_type"].shape[:-1]:
        raise ValueError("Categorical action must have shape [B], never [B, 1]")
    continuous_action = action["action_args"].double()
    differences = (
        discrete_new.log_prob(discrete_action) - discrete_old.log_prob(discrete_action),
        continuous_new.log_prob(continuous_action) - continuous_old.log_prob(continuous_action),
    )
    kl = max(float(kl_divergence(discrete_old, discrete_new).mean()),
             float(kl_divergence(continuous_old, continuous_new).mean()))
    max_log_ratio = max(float(difference.abs().max()) for difference in differences)
    if not math.isfinite(kl) or not math.isfinite(max_log_ratio):
        raise FloatingPointError("Nonfinite policy divergence; aborting before loss evaluation")
    return kl, max_log_ratio


@POLICY_REGISTRY.register("film_hppo")
class FilmPPOPolicy(PPOPolicy):
    config = dict(recompute_adv=False, learn=dict(target_kl=0.02, max_abs_log_ratio=10.0))

    def _init_learn(self):
        if self._cfg.action_space != "hybrid" or self._cfg.recompute_adv:
            raise ValueError("FilmPPOPolicy requires hybrid actions and recompute_adv=False")
        if self._cfg.learn.value_norm or self._cfg.learn.lr_scheduler is not None:
            raise ValueError("FilmPPOPolicy uses fixed rollout targets without value_norm/lr_scheduler")
        if self._cfg.learn.batch_size < 3 or self._cfg.learn.epoch_per_collect < 1:
            raise ValueError("Require batch_size >= 3 and epoch_per_collect >= 1")
        self._target_kl = float(self._cfg.learn.target_kl)
        self._max_log_ratio = float(self._cfg.learn.max_abs_log_ratio)
        if not math.isfinite(self._target_kl) or self._target_kl <= 0:
            raise ValueError("target_kl must be positive and finite")
        if not math.isfinite(self._max_log_ratio) or not 0 < self._max_log_ratio <= 20:
            raise ValueError("Require 0 < max_abs_log_ratio <= 20")
        super()._init_learn()
        # PyTorch pre-hooks run before DI-engine Adam's clipping and update.
        # Reject invalid gradients rather than allowing clipping to spread NaNs.
        self._optimizer.register_step_pre_hook(self._check_gradients)
        self.last_update = {}

    def _check_gradients(self, optimizer, args, kwargs):
        for name, parameter in self._model.named_parameters():
            if parameter.grad is not None:
                require_finite(parameter.grad, f"gradient.{name}")

    def _process_transition(self, obs, policy_output, timestep):
        return scalar_action_sample(super()._process_transition(obs, policy_output, timestep))

    def _forward_learn(self, samples):
        if len(samples) < 2:
            raise ValueError("At least two samples are needed for advantage normalization")
        # Normalize again after collector conversions or when consuming saved data.
        samples = [scalar_action_sample(sample) for sample in samples]
        batch_size = self._cfg.learn.batch_size
        epochs = self._cfg.learn.epoch_per_collect
        logs = []
        self.last_update = {"optimizer_steps": 0, "early_stop": False, "reason": None,
                            "max_kl": 0.0, "max_abs_log_ratio": 0.0}
        self._learn_model.train()
        try:
            # Delegate exactly one minibatch update at a time to the parent.
            # The complete original config is restored even if the update fails.
            self._cfg.learn.epoch_per_collect = 1
            for _ in range(epochs):
                order = torch.randperm(len(samples)).tolist()
                batches = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
                # Avoid std(correction=1) on a singleton tail without dropping data.
                if len(batches) > 1 and len(batches[-1]) == 1:
                    batches[-1].insert(0, batches[-2].pop())
                for indices in batches:
                    batch = [samples[i] for i in indices]
                    data = default_preprocess_learn(batch, ignore_done=self._cfg.learn.ignore_done, use_nstep=False)
                    data = to_device(data, self._device)
                    require_finite(data, "training_batch")
                    with torch.no_grad():
                        output = self._learn_model.forward(data["obs"].float(), mode="compute_actor_critic")
                        require_finite(output, "model_output")
                        kl, max_ratio = policy_drift(output["logit"], data["logit"], data["action"])
                    self.last_update["max_kl"] = max(self.last_update["max_kl"], kl)
                    self.last_update["max_abs_log_ratio"] = max(self.last_update["max_abs_log_ratio"], max_ratio)
                    if kl > self._target_kl or max_ratio > self._max_log_ratio:
                        reason = f"rollout reuse stopped: KL={kl:.6g}, max_abs_log_ratio={max_ratio:.6g}"
                        self.last_update.update(early_stop=True, reason=reason)
                        if not logs:
                            # Fresh on-policy samples should match before the first update.
                            raise FloatingPointError(f"{reason} before any update; stale/mismatched rollout or checkpoint")
                        warnings.warn(reason, RuntimeWarning, stacklevel=2)
                        return logs
                    # DI-engine's splitter uses negative slicing if split_size
                    # exceeds the data length. Pass the actual batch length.
                    self._cfg.learn.batch_size = len(batch)
                    minibatch_logs = super()._forward_learn(batch)
                    for name, parameter in self._model.named_parameters():
                        require_finite(parameter, f"updated_parameter.{name}")
                    if not all(math.isfinite(float(value)) for log in minibatch_logs for value in log.values()):
                        raise FloatingPointError("Nonfinite PPO update metrics")
                    logs.extend(minibatch_logs)
                    self.last_update["optimizer_steps"] += len(minibatch_logs)
            return logs
        finally:
            self._cfg.learn.epoch_per_collect = epochs
            self._cfg.learn.batch_size = batch_size


@POLICY_REGISTRY.register("film_hppo_command")
class FilmPPOCommandPolicy(FilmPPOPolicy, DummyCommandModePolicy):
    pass
