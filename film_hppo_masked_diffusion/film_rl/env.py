"""Randomly unmask multiple fixed slots, then score the visible physical stack."""

from copy import deepcopy
import json
from pathlib import Path
import uuid

import gym
import numpy as np
from ding.envs import BaseEnv, BaseEnvTimestep
from ding.utils import ENV_REGISTRY
from easydict import EasyDict

from .actions import decode_tokens, encode_tokens, observation_size
from .config import validate
from .optics import FilmOptics


@ENV_REGISTRY.register("duv_masked_film")
class ThinFilmEnv(BaseEnv):
    MASK = -1

    def __init__(self, cfg):
        self.cfg = deepcopy(dict(cfg))
        validate(self.cfg)
        self.max_layers = self.cfg["max_layers"]
        self.min_layers = self.cfg["min_layers"]
        self.material_count = len(self.cfg["material_files"])
        self.EOS = self.material_count
        self.fill_max = self.cfg["fill_max"]
        self.observation_space = gym.spaces.Box(0, 1, shape=(observation_size(self.cfg),), dtype=np.float32)
        self.action_space = gym.spaces.Dict({
            "action_type": gym.spaces.Discrete((self.material_count + 1) ** self.fill_max),
            "action_args": gym.spaces.Box(-np.inf, np.inf, shape=(self.fill_max,), dtype=np.float32),
        })
        self.reward_space = gym.spaces.Box(0, 1, shape=(1,), dtype=np.float32)
        self.optics = None
        self._rng = np.random.default_rng(0)
        self._seed = 0
        self._dynamic_seed = True
        self._done = True
        self._best_reward = -np.inf
        self._run_id = uuid.uuid4().hex
        self.tokens = np.full(self.max_layers, self.MASK, dtype=np.int64)
        self.thicknesses = np.zeros(self.max_layers, dtype=np.float64)
        self.pending_positions = []

    @classmethod
    def default_config(cls):
        return EasyDict({"type": "duv_masked_film"})

    def seed(self, seed, dynamic_seed=True):
        self._seed = int(seed)
        self._dynamic_seed = dynamic_seed
        self._rng = np.random.default_rng(self._seed)
        self.action_space.seed(self._seed)

    def reset(self):
        if self.optics is None:
            self.optics = FilmOptics(self.cfg)
        if not self._dynamic_seed:
            self._rng = np.random.default_rng(self._seed)
        self.tokens.fill(self.MASK)
        self.thicknesses.fill(0)
        self._training_return = 0.0
        self._step_count = 0
        self._done = False
        self._select_positions()
        return self._observation()

    def _eos_position(self):
        positions = np.flatnonzero(self.tokens == self.EOS)
        return int(positions[0]) if len(positions) else self.max_layers

    def _select_positions(self):
        available = np.flatnonzero(self.tokens[:self._eos_position()] == self.MASK)
        if not len(available):
            self.pending_positions = []
            return
        count = int(self._rng.integers(self.cfg["fill_min"], self.fill_max + 1))
        self.pending_positions = sorted(self._rng.choice(available, min(count, len(available)), replace=False).tolist())

    @property
    def layers(self):
        return [(int(self.tokens[p]), float(self.thicknesses[p]))
                for p in range(self._eos_position()) if self.tokens[p] != self.MASK]

    def _observation(self):
        width = self.material_count + 3
        encoded = np.zeros((self.max_layers, width), dtype=np.float32)
        for position, token in enumerate(self.tokens):
            encoded[position, self.material_count + 1 if token == self.MASK else token] = 1
            if 0 <= token < self.material_count:
                encoded[position, -1] = (self.thicknesses[position] - self.cfg["thickness_min_nm"]) / (
                    self.cfg["thickness_max_nm"] - self.cfg["thickness_min_nm"])
        pending = np.zeros((self.fill_max, self.max_layers), dtype=np.float32)
        for rank, position in enumerate(self.pending_positions):
            pending[rank, position] = 1
        return np.concatenate((encoded.ravel(), pending.ravel(),
                               np.array([self._step_count / self.max_layers], dtype=np.float32)))

    def step(self, action):
        if self._done:
            raise RuntimeError("Call reset() before stepping a new/finished episode")
        tokens = decode_tokens(action["action_type"], self.material_count, self.fill_max)
        args = np.asarray(action["action_args"], dtype=np.float64)
        if args.shape != (self.fill_max,) or not np.isfinite(args).all():
            raise ValueError("action_args must contain fill_max finite Gaussian latent values")
        positions = list(self.pending_positions)
        if any(tokens[rank] != 0 for rank in range(len(positions), self.fill_max)):
            raise ValueError("Unused joint-action digits must be zero")
        if any(tokens[rank] == self.EOS and position < self.min_layers
               for rank, position in enumerate(positions)):
            raise ValueError("EOS cannot occur before min_layers; use the masked FilmVAC")
        # Validate the entire batch before mutating. Every chosen slot is written;
        # the first EOS subsequently determines which writes affect optics.
        low, high = self.cfg["thickness_min_nm"], self.cfg["thickness_max_nm"]
        for rank, position in enumerate(positions):
            self.tokens[position] = tokens[rank]
            self.thicknesses[position] = 0 if tokens[rank] == self.EOS else (
                low + (high - low) * (np.tanh(args[rank]) + 1) / 2)
        self._step_count += 1
        eos = self._eos_position()
        self._done = not np.any(self.tokens[:eos] == self.MASK)
        info = self.optics.evaluate(self.layers)
        reward = info["reward"]
        self._training_return += reward
        info.update(step_reward=reward, training_return=self._training_return,
                    reward_mode="absolute_step_score", step_count=self._step_count,
                    filled_positions=positions, filled_tokens=tokens[:len(positions)],
                    active_positions=[p for p in range(eos) if self.tokens[p] != self.MASK],
                    eos_position=eos if eos < self.max_layers else None,
                    filled_slot_count=int(np.count_nonzero(self.tokens != self.MASK)),
                    slots=self.export_slots(), eligible_design=len(self.layers) >= self.min_layers,
                    terminal=bool(self._done))
        if self._done:
            info.update(termination_reason="eos_prefix_complete" if eos < self.max_layers else "all_slots_filled",
                        eval_episode_return=reward,
                        episode_info={"training_return": self._training_return, "final_score": reward,
                                      "final_mean_transmission": info["mean_transmission"]})
            self.pending_positions = []
        else:
            self._select_positions()
        self._save_best(info)
        return BaseEnvTimestep(self._observation(), np.array([reward], dtype=np.float32), self._done, info)

    def export_slots(self):
        return [None if token == self.MASK else {"token": "EOS"} if token == self.EOS else {
            "material_id": int(token), "material": self.optics.names[token],
            "thickness_nm": float(self.thicknesses[position])}
            for position, token in enumerate(self.tokens)]

    def _save_best(self, info):
        directory = self.cfg.get("result_dir")
        if directory and info["eligible_design"] and info["reward"] > self._best_reward:
            destination = Path(directory)
            destination.mkdir(parents=True, exist_ok=True)
            path = destination / f"best_{self._run_id}.json"
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
            temporary.replace(path)
            self._best_reward = info["reward"]
            # BaseEnvManager runs these environments synchronously. Maintain a
            # run-wide physical best in addition to each environment's best.
            global_path = destination / "best_design.json"
            previous = json.loads(global_path.read_text(encoding="utf-8")) if global_path.exists() else None
            if previous is None or info["reward"] > previous["reward"]:
                temporary = destination / f"best_design_{self._run_id}.tmp"
                temporary.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
                temporary.replace(global_path)

    def random_action(self):
        tokens = [int(self._rng.integers(self.material_count + int(p >= self.min_layers)))
                  for p in self.pending_positions]
        return {"action_type": encode_tokens(tokens, self.material_count, self.fill_max),
                "action_args": self._rng.normal(size=self.fill_max).astype(np.float32)}

    @staticmethod
    def create_collector_env_cfg(cfg):
        return [deepcopy(cfg) for _ in range(cfg["collector_env_num"])]

    @staticmethod
    def create_evaluator_env_cfg(cfg):
        return [deepcopy(cfg) for _ in range(cfg["evaluator_env_num"])]

    def close(self):
        self.optics = None
        self._done = True

    def __repr__(self):
        return f"MaskedFilmEnv(materials={self.material_count}, max_layers={self.max_layers}, fill_max={self.fill_max})"
