"""Layer helper utilities."""

from __future__ import annotations

from typing import Any


def _get_int(config: dict[str, Any], keys: tuple[str, ...], default: int) -> int:
    for key in keys:
        if key in config and config[key] is not None:
            return int(config[key])
    return default


def get_batch(config: dict[str, Any]) -> int:
    default_batch = 1  # common batch fallback
    return _get_int(config, ("batch", "batch_size"), default_batch)


def get_seq_len(config: dict[str, Any]) -> int:
    default_seq_len = 2048  # typical model context length
    return _get_int(config, ("seq_len", "q_seq_len"), default_seq_len)


def get_kv_seq_len(config: dict[str, Any]) -> int:
    seq_len = get_seq_len(config)
    return _get_int(config, ("kv_seq_len",), seq_len)


def get_hidden_size(config: dict[str, Any]) -> int:
    default_hidden = 4096  # typical hidden size
    return _get_int(config, ("hidden_size", "hidden_dim"), default_hidden)


def get_intermediate_size(config: dict[str, Any]) -> int:
    hidden = get_hidden_size(config)
    expansion = 4  # common FFN expansion ratio
    return _get_int(config, ("intermediate_size", "inter_dim"), hidden * expansion)


def get_moe_intermediate_size(config: dict[str, Any]) -> int:
    return _get_int(config, ("moe_intermediate_size", "moe_inter_dim"), get_intermediate_size(config))


def get_num_heads(config: dict[str, Any]) -> int:
    return _get_int(config, ("num_heads", "n_heads"), 32)


def matmul_flops(m: int, k: int, n: int, groups: int = 1) -> int:
    mul_add = 2  # matmul uses 2 * M * K * N
    return mul_add * m * k * n * groups


def attention_flops(batch: int, heads: int, q_len: int, kv_len: int, q_dim: int, v_dim: int) -> int:
    mul_add = 2  # dot + weight multiply
    return mul_add * batch * heads * q_len * kv_len * (q_dim + v_dim)
