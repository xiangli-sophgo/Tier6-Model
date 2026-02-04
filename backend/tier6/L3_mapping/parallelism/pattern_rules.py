"""Pattern Rules - 预定义的并行切分模板

基于 layer pattern 自动为 op 选择 ParallelSpec。
"""

from tier6.L3_mapping.parallelism.parallel_spec import ParallelSpec, ParallelType

# 预定义的 Pattern 切分模板
PATTERN_TEMPLATES: dict[str, dict[str, ParallelSpec]] = {
    # MLP 层的 TP 切分
    # Gate/Up: 列切分 (TP_COL)，切 N 维度
    # Down: 行切分 (TP_ROW)，切 K 维度，需要 AllReduce
    "mlp": {
        "gate": ParallelSpec(ParallelType.TP_COL, "N"),
        "up": ParallelSpec(ParallelType.TP_COL, "N"),
        "down": ParallelSpec(ParallelType.TP_ROW, "K"),
    },
    # FFN 层 (与 MLP 相同的切分策略)
    "ffn": {
        "gate": ParallelSpec(ParallelType.TP_COL, "N"),
        "up": ParallelSpec(ParallelType.TP_COL, "N"),
        "down": ParallelSpec(ParallelType.TP_ROW, "K"),
    },
    # Attention 层的 TP 切分
    # QKV: 头切分
    # O_proj: 行切分，需要 AllReduce
    "attention": {
        "q_proj": ParallelSpec(ParallelType.TP_HEAD, "head"),
        "k_proj": ParallelSpec(ParallelType.TP_HEAD, "head"),
        "v_proj": ParallelSpec(ParallelType.TP_HEAD, "head"),
        "o_proj": ParallelSpec(ParallelType.TP_ROW, "K"),
    },
    # MLA (Multi-head Latent Attention) 层
    "mla": {
        "q_proj": ParallelSpec(ParallelType.TP_COL, "N"),
        "kv_proj": ParallelSpec(ParallelType.TP_COL, "N"),
        "attn_score": ParallelSpec(ParallelType.TP_COL, "N"),
        "attn_out": ParallelSpec(ParallelType.TP_ROW, "K"),
        "o_proj": ParallelSpec(ParallelType.TP_ROW, "K"),
    },
    # MLA Absorb（对齐 DS_TPU 分解口径）
    "mla_absorb": {
        "q_a": ParallelSpec(ParallelType.REPLICATE, ""),
        "q_b": ParallelSpec(ParallelType.TP_COL, "N"),
        "kv_a": ParallelSpec(ParallelType.REPLICATE, ""),
        "k_compact": ParallelSpec(ParallelType.TP_HEAD, "G"),
        "v_compact": ParallelSpec(ParallelType.TP_HEAD, "G"),
        "attn_score": ParallelSpec(ParallelType.TP_COL, "N"),
        "attn_out": ParallelSpec(ParallelType.TP_ROW, "K"),
        "o_proj": ParallelSpec(ParallelType.TP_ROW, "K"),
    },
    # MoE 层切分
    # Router: 复制
    # Shared Experts: TP 切分
    # Routed Experts: 使用 moe_tp 的 TP 切分 (通信由调度器插入)
    "moe": {
        "router": ParallelSpec(ParallelType.REPLICATE, ""),
        "shared_gate": ParallelSpec(ParallelType.TP_COL, "N"),
        "shared_up": ParallelSpec(ParallelType.TP_COL, "N"),
        "shared_down": ParallelSpec(ParallelType.TP_ROW, "K"),
        "gate": ParallelSpec(ParallelType.TP_COL, "N"),
        "up": ParallelSpec(ParallelType.TP_COL, "N"),
        "down": ParallelSpec(ParallelType.TP_ROW, "K"),
    },
    # Embedding 层
    "embedding": {
        "embed": ParallelSpec(ParallelType.TP_COL, "N"),
    },
    # LMHead 层
    "lmhead": {
        "proj": ParallelSpec(ParallelType.TP_COL, "N"),
    },
}

# Op 类型到默认 ParallelSpec 的映射
DEFAULT_PARALLEL_SPECS: dict[str, ParallelSpec] = {
    "matmul": ParallelSpec(ParallelType.TP_COL, "N"),
    "linear": ParallelSpec(ParallelType.TP_COL, "N"),
    "embedding": ParallelSpec(ParallelType.TP_COL, "N"),
    "layernorm": ParallelSpec(ParallelType.REPLICATE, ""),
    "rmsnorm": ParallelSpec(ParallelType.REPLICATE, ""),
    "softmax": ParallelSpec(ParallelType.REPLICATE, ""),
    "silu": ParallelSpec(ParallelType.REPLICATE, ""),
    "gelu": ParallelSpec(ParallelType.REPLICATE, ""),
    "elementwise": ParallelSpec(ParallelType.REPLICATE, ""),
}


def get_pattern_spec(layer_type: str, op_role: str) -> ParallelSpec | None:
    """根据 layer 类型和 op 角色获取 ParallelSpec

    Args:
        layer_type: 层类型，如 "mlp", "attention", "moe"
        op_role: op 在层中的角色，如 "gate", "up", "down"

    Returns:
        对应的 ParallelSpec，如果未找到则返回 None
    """
    pattern = PATTERN_TEMPLATES.get(layer_type)
    if pattern is None:
        return None
    return pattern.get(op_role)


def get_default_spec(op_type: str) -> ParallelSpec:
    """获取 op 类型的默认 ParallelSpec

    Args:
        op_type: op 类型，如 "matmul", "layernorm"

    Returns:
        默认的 ParallelSpec
    """
    return DEFAULT_PARALLEL_SPECS.get(op_type, ParallelSpec(ParallelType.REPLICATE, ""))
