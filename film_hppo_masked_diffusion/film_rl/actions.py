"""A scalar joint categorical action keeps native DI-engine hybrid PPO intact."""

import numpy as np


def encode_tokens(tokens, material_count, fill_max):
    tokens = list(tokens)
    if len(tokens) > fill_max:
        raise ValueError("Too many tokens")
    if any(isinstance(t, bool) or not isinstance(t, (int, np.integer))
           or not 0 <= t <= material_count for t in tokens):
        raise ValueError("Token must be a material ID or EOS")
    base = material_count + 1
    return sum(int(token) * base ** index for index, token in enumerate(tokens))


def decode_tokens(action_type, material_count, fill_max):
    value = np.asarray(action_type)
    if value.size != 1 or not np.isfinite(value).all():
        raise ValueError("action_type must contain one finite integer")
    number = float(value.item())
    base = material_count + 1
    if not number.is_integer() or not 0 <= number < base ** fill_max:
        raise ValueError("Joint action_type outside configured range")
    code = int(number)
    return [(code // base ** index) % base for index in range(fill_max)]


def observation_size(cfg):
    return cfg["max_layers"] * (len(cfg["material_files"]) + 3 + cfg["fill_max"]) + 1
