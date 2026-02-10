"""Layer helper utilities."""

from __future__ import annotations

from typing import Any


def _get_int(config: dict[str, Any], keys: tuple[str, ...], source: str = "config") -> int:
    """从配置中获取整数字段（支持多个候选 key），缺失时报错"""
    for key in keys:
        if key in config and config[key] is not None:
            return int(config[key])
    keys_str = "', '".join(keys)
    raise ValueError(f"Missing required field (tried: '{keys_str}') in {source}")


def get_batch(config: dict[str, Any]) -> int:
    return _get_int(config, ("batch", "batch_size"), "layer config")


def get_seq_len(config: dict[str, Any]) -> int:
    return _get_int(config, ("seq_len", "q_seq_len"), "layer config")


def get_kv_seq_len(config: dict[str, Any]) -> int:
    # kv_seq_len 可以回退到 seq_len (prefill 时两者相等)
    try:
        return _get_int(config, ("kv_seq_len",), "layer config")
    except ValueError:
        return get_seq_len(config)


def get_hidden_size(config: dict[str, Any]) -> int:
    return _get_int(config, ("hidden_size", "hidden_dim"), "layer config")


def get_intermediate_size(config: dict[str, Any]) -> int:
    # intermediate_size 可以从 hidden_size 推导（4x expansion 是常见约定）
    try:
        return _get_int(config, ("intermediate_size", "inter_dim"), "layer config")
    except ValueError:
        hidden = get_hidden_size(config)
        return hidden * 4  # 使用常见的 4x expansion ratio


def get_moe_intermediate_size(config: dict[str, Any]) -> int:
    # moe_intermediate_size 可以回退到 intermediate_size
    try:
        return _get_int(config, ("moe_intermediate_size", "moe_inter_dim"), "layer config")
    except ValueError:
        return get_intermediate_size(config)


def get_num_heads(config: dict[str, Any]) -> int:
    return _get_int(config, ("num_heads", "n_heads"), "layer config")


def matmul_flops(m: int, k: int, n: int, groups: int = 1) -> int:
    mul_add = 2  # matmul uses 2 * M * K * N
    return mul_add * m * k * n * groups


def attention_flops(batch: int, heads: int, q_len: int, kv_len: int, q_dim: int, v_dim: int) -> int:
    mul_add = 2  # dot + weight multiply
    return mul_add * batch * heads * q_len * kv_len * (q_dim + v_dim)
