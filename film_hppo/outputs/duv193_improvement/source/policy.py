"""Hybrid PPO with selected-material continuous likelihood and numerical guards.

DI-engine supplies the clipped losses, value loss, optimizer and rollout GAE.
Only the executed material's Gaussian enters the continuous policy ratio;
EOS has no continuous policy term. Entropy retains the original full-vector
definition; optional coefficients separate discrete and continuous exploration.
Historical configurations retain the original loss. Rollout targets stay fixed.
"""

import math
import warnings

import torch
from torch.distributions import Categorical, Normal, kl_divergence
from ding.policy import PPOPolicy
from ding.policy.command_mode_policy_instance import DummyCommandModePolicy
from ding.policy.common_utils import default_preprocess_learn
from ding.torch_utils import to_device
from ding.utils import POLICY_REGISTRY
from ding.rl_utils import (
    ppo_policy_data, ppo_policy_error, ppo_policy_error_continuous,
    ppo_value_data, ppo_value_error,
)


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


def selected_gaussian(logits, action):
    """Gather one Gaussian/action per row; neutralize EOS before log-prob.

    Returns [B,1] parameters and actions, plus a [B] non-EOS mask. Keeping the
    full batch denominator makes an EOS row contribute zero, not rescale the
    remaining rows. Raw latent actions remain unchanged in the stored rollout.
    """
    mu, sigma = logits["mu"], logits["sigma"]
    kind, args = action["action_type"], action["action_args"]
    if mu.ndim != 2 or sigma.shape != mu.shape or args.shape != mu.shape:
        raise ValueError("Gaussian mu/sigma/actions must all have shape [B, M]")
    if kind.shape != mu.shape[:-1]:
        raise ValueError("Categorical action must have shape [B], never [B, 1]")
    count = mu.shape[-1]
    if (kind < 0).any() or (kind > count).any() or not torch.equal(kind, kind.long()):
        raise ValueError("Categorical action must be a material index or EOS=M")
    active = kind < count
    index = kind.long().clamp(max=count-1).unsqueeze(-1)
    selected_mu = mu.gather(-1, index)
    selected_sigma = sigma.gather(-1, index)
    selected_args = args.gather(-1, index)
    mask = active.unsqueeze(-1)
    return {
        "mu": torch.where(mask, selected_mu, torch.zeros_like(selected_mu)),
        "sigma": torch.where(mask, selected_sigma, torch.ones_like(selected_sigma)),
    }, torch.where(mask, selected_args, torch.zeros_like(selected_args)), active


def selected_continuous_policy_error(new, old, action, adv, weight=None, clip_ratio=0.2):
    """DI-engine's scalar Gaussian PPO loss, zero weighted on EOS rows.

    Its returned selected entropy is not used by the learner: the original
    full-vector entropy regularizer is retained as a controlled baseline.
    """
    selected_new, args, active = selected_gaussian(new, action)
    selected_old, _, _ = selected_gaussian(old, action)
    active_weight = active.to(adv.dtype)
    if weight is not None:
        active_weight = active_weight * weight
    data = ppo_policy_data(selected_new, selected_old, args, adv, active_weight)
    return ppo_policy_error_continuous(data, clip_ratio)


@torch.no_grad()
def policy_drift(new, old, action):
    """Categorical KL and executed-material Gaussian KL/log-ratio in float64.

    Gaussian KL is analytic for each selected pair, then averaged over rollout
    rows with zero for EOS. This is a sampled-material diagnostic, not the exact
    KL of the complete hybrid distribution. It matches the policy-loss scope.
    Checking before exp(logp_new-logp_old) avoids float32 ratio overflow.
    """
    require_finite(new, "new_logits")
    require_finite(old, "rollout_logits")
    require_finite(action, "rollout_actions")
    for label, logits in (("new", new), ("old", old)):
        if (logits["action_args"]["sigma"] <= 0).any():
            raise FloatingPointError(f"{label} Gaussian sigma must be positive")
    discrete_new = Categorical(logits=new["action_type"].double())
    discrete_old = Categorical(logits=old["action_type"].double())
    selected_new, continuous_action, _ = selected_gaussian(new["action_args"], action)
    selected_old, _, _ = selected_gaussian(old["action_args"], action)
    continuous_new = Normal(selected_new["mu"].double(), selected_new["sigma"].double())
    continuous_old = Normal(selected_old["mu"].double(), selected_old["sigma"].double())
    discrete_action = action["action_type"].long()
    differences = (
        discrete_new.log_prob(discrete_action) - discrete_old.log_prob(discrete_action),
        continuous_new.log_prob(continuous_action.double()) - continuous_old.log_prob(continuous_action.double()),
    )
    kl = max(float(kl_divergence(discrete_old, discrete_new).mean()),
             float(kl_divergence(continuous_old, continuous_new).mean()))
    max_log_ratio = max(float(difference.abs().max()) for difference in differences)
    if not math.isfinite(kl) or not math.isfinite(max_log_ratio):
        raise FloatingPointError("Nonfinite policy divergence; aborting before loss evaluation")
    return kl, max_log_ratio


@POLICY_REGISTRY.register("film_hppo")
class FilmPPOPolicy(PPOPolicy):
    config = dict(recompute_adv=False, learn=dict(target_kl=0.02, max_abs_log_ratio=10.0,
                                                entropy_weight_discrete=None, entropy_weight_continuous=None))

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
        initial_sigma = self._cfg.learn.get('initial_sigma')
        if initial_sigma is not None:
            if (isinstance(initial_sigma, bool) or not isinstance(initial_sigma, (int, float))
                    or not math.isfinite(initial_sigma) or not math.exp(-5) <= initial_sigma <= math.exp(2)):
                raise ValueError('initial_sigma outside the model scale bounds')
            with torch.no_grad():
                self._model.actor_head[1].log_sigma_param.fill_(math.log(initial_sigma))
        discrete_weight = self._cfg.learn.get("entropy_weight_discrete")
        continuous_weight = self._cfg.learn.get("entropy_weight_continuous")
        self._entropy_weight_discrete = self._entropy_weight if discrete_weight is None else float(discrete_weight)
        self._entropy_weight_continuous = self._entropy_weight if continuous_weight is None else float(continuous_weight)
        if any(not math.isfinite(w) or w < 0 for w in (
                self._entropy_weight_discrete, self._entropy_weight_continuous)):
            raise ValueError("Entropy coefficients must be finite and nonnegative")
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

    def _learn_minibatch(self, data, output):
        """One update using native loss primitives and the native optimizer."""
        adv = data["adv"]
        returns = adv + data["value"]
        if self._adv_norm:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        discrete_loss, discrete_info = ppo_policy_error(
            ppo_policy_data(output["logit"]["action_type"], data["logit"]["action_type"],
                            data["action"]["action_type"], adv, data["weight"]),
            self._clip_ratio,
        )
        continuous_loss, continuous_info = selected_continuous_policy_error(
            output["logit"]["action_args"], data["logit"]["action_args"],
            data["action"], adv, data["weight"], self._clip_ratio,
        )
        # All rows, including EOS, still train the critic with fixed GAE targets.
        value_loss = ppo_value_error(
            ppo_value_data(output["value"], data["value"], returns, data["weight"]),
            self._clip_ratio,
        )
        args = output["logit"]["action_args"]
        continuous_entropy = Normal(args["mu"], args["sigma"]).entropy().sum(-1)
        weight = data["weight"] if data["weight"] is not None else torch.ones_like(adv)
        discrete_entropy = discrete_loss.entropy_loss
        continuous_entropy = (continuous_entropy * weight).mean()
        entropy_loss = discrete_entropy + continuous_entropy
        policy_loss = discrete_loss.policy_loss + continuous_loss.policy_loss
        if self._entropy_weight_discrete == self._entropy_weight_continuous == self._entropy_weight:
            # Preserve the historical arithmetic path as well as its meaning.
            entropy_bonus = self._entropy_weight * entropy_loss
        else:
            entropy_bonus = (self._entropy_weight_discrete * discrete_entropy
                             + self._entropy_weight_continuous * continuous_entropy)
        total_loss = policy_loss + self._value_weight * value_loss - entropy_bonus
        require_finite(total_loss, "total_loss")
        self._optimizer.zero_grad()
        total_loss.backward()
        self._optimizer.step()
        return {
            "cur_lr": self._optimizer.defaults["lr"],
            "total_loss": total_loss.item(), "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(), "entropy_loss": entropy_loss.item(),
            "entropy_discrete": discrete_entropy.item(),
            "entropy_continuous": continuous_entropy.item(), "entropy_bonus": entropy_bonus.item(),
            "adv_max": adv.max().item(), "adv_mean": adv.mean().item(),
            "value_mean": output["value"].mean().item(), "value_max": output["value"].max().item(),
            "approx_kl": max(continuous_info.approx_kl, discrete_info.approx_kl),
            "clipfrac": max(continuous_info.clipfrac, discrete_info.clipfrac),
        }

    def _monitor_vars_learn(self):
        return super()._monitor_vars_learn() + ["entropy_discrete", "entropy_continuous", "entropy_bonus"]

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
                output = self._learn_model.forward(data["obs"].float(), mode="compute_actor_critic")
                require_finite(output, "model_output")
                kl, max_ratio = policy_drift(output["logit"], data["logit"], data["action"])
                self.last_update["max_kl"] = max(self.last_update["max_kl"], kl)
                self.last_update["max_abs_log_ratio"] = max(self.last_update["max_abs_log_ratio"], max_ratio)
                if kl > self._target_kl or max_ratio > self._max_log_ratio:
                    reason = f"rollout reuse stopped: KL={kl:.6g}, max_abs_log_ratio={max_ratio:.6g}"
                    self.last_update.update(early_stop=True, reason=reason)
                    if not logs:
                        raise FloatingPointError(f"{reason} before any update; stale/mismatched rollout or checkpoint")
                    warnings.warn(reason, RuntimeWarning, stacklevel=2)
                    return logs
                log = self._learn_minibatch(data, output)
                for name, parameter in self._model.named_parameters():
                    require_finite(parameter, f"updated_parameter.{name}")
                if not all(math.isfinite(float(value)) for value in log.values()):
                    raise FloatingPointError("Nonfinite PPO update metrics")
                logs.append(log)
                self.last_update["optimizer_steps"] += 1
        return logs


@POLICY_REGISTRY.register("film_hppo_command")
class FilmPPOCommandPolicy(FilmPPOPolicy, DummyCommandModePolicy):
    pass
