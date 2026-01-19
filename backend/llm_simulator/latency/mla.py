"""
MLA (Multi-head Latent Attention) 相关延迟计算
DeepSeek V3/R1 专用
"""

from ..types import LLMModelConfig, InferenceConfig, ParallelismStrategy, HardwareConfig, get_bytes_per_element
from .core import calc_gemm_latency, get_arch


def calc_mla_q_projection_latency(
    model: LLMModelConfig,
    inference: InferenceConfig,
    parallelism: ParallelismStrategy,
    hardware: HardwareConfig,
    num_tokens: int,
) -> float:
    """
    MLA Q 投影延迟 (LoRA-style 两阶段)

    1. Down: [B×S, H] → [B×S, q_lora_rank]
    2. Up: [B×S, q_lora_rank] → [B×S, heads×head_dim/TP]
    """
    if model.mla_config is None:
        # 回退到标准 Q 投影
        from .attention import calc_attention_qkv_latency
        return calc_attention_qkv_latency(model, inference, parallelism, hardware, num_tokens) / 3

    B, S, H = inference.batch_size, num_tokens, model.hidden_size
    mla = model.mla_config
    tp = parallelism.tp
    heads = model.num_attention_heads
    head_dim = mla.qk_nope_head_dim + mla.qk_rope_head_dim

    # Down projection
    down_latency = calc_gemm_latency(M=B*S, K=H, N=mla.q_lora_rank)

    # Up projection (分布到 TP)
    up_latency = calc_gemm_latency(M=B*S, K=mla.q_lora_rank, N=heads*head_dim//tp)

    return down_latency + up_latency


def calc_mla_kv_compression_latency(
    model: LLMModelConfig,
    inference: InferenceConfig,
    parallelism: ParallelismStrategy,
    hardware: HardwareConfig,
    num_tokens: int,
) -> float:
    """
    MLA KV 压缩延迟 (Prefill 阶段)

    GEMM: [B×S, H] × [H, kv_lora_rank + qk_rope_head_dim]
    """
    if model.mla_config is None:
        return 0.0

    B, S, H = inference.batch_size, num_tokens, model.hidden_size
    mla = model.mla_config

    # 压缩后的 KV 维度
    compressed_kv_dim = mla.kv_lora_rank + mla.qk_rope_head_dim

    return calc_gemm_latency(M=B*S, K=H, N=compressed_kv_dim)


def calc_mla_kv_decompression_latency(
    model: LLMModelConfig,
    inference: InferenceConfig,
    parallelism: ParallelismStrategy,
    hardware: HardwareConfig,
    context_length: int,
) -> float:
    """
    MLA KV 解压延迟 (Decode 阶段)

    需要将压缩的 KV cache 解压回完整维度
    W_KC: [heads, qk_nope, kv_lora_rank]
    W_VC: [heads, v_head_dim, kv_lora_rank]
    """
    if model.mla_config is None:
        return 0.0

    B = inference.batch_size
    mla = model.mla_config
    tp = parallelism.tp
    heads = model.num_attention_heads

    # K 解压: [B, kv_lora_rank] × [kv_lora_rank, qk_nope] per head
    k_decomp = calc_gemm_latency(
        M=B, K=mla.kv_lora_rank, N=mla.qk_nope_head_dim, G=heads//tp
    )

    # V 解压: [B, kv_lora_rank] × [kv_lora_rank, v_head_dim] per head
    v_decomp = calc_gemm_latency(
        M=B, K=mla.kv_lora_rank, N=mla.v_head_dim, G=heads//tp
    )

    return k_decomp + v_decomp


def calc_mla_attention_score_latency(
    model: LLMModelConfig,
    inference: InferenceConfig,
    parallelism: ParallelismStrategy,
    hardware: HardwareConfig,
    num_tokens: int,
    context_length: int,
) -> float:
    """
    MLA Attention Score 延迟

    MLA 使用 MQA 风格: 所有 head 共享同一个 compressed KV
    Score: [B, heads/TP, S, head_dim] × [B, 1, head_dim, C]
    """
    if model.mla_config is None:
        from .attention import calc_attention_score_latency
        return calc_attention_score_latency(model, inference, parallelism, hardware, num_tokens, context_length)

    B, S = inference.batch_size, num_tokens
    mla = model.mla_config
    tp = parallelism.tp
    heads = model.num_attention_heads

    # 完整 head dim = qk_nope + qk_rope
    head_dim = mla.qk_nope_head_dim + mla.qk_rope_head_dim

    G = B * heads // tp
    return calc_gemm_latency(M=S, K=head_dim, N=context_length, G=G)


def calc_mla_decode_attention_latency(
    model: LLMModelConfig,
    inference: InferenceConfig,
    parallelism: ParallelismStrategy,
    hardware: HardwareConfig,
    context_length: int,
) -> float:
    """
    MLA Decode 阶段完整 Attention 延迟

    包括:
    1. KV 解压 (absorbed into attention)
    2. Score 计算
    3. Softmax @ V
    """
    if model.mla_config is None:
        return 0.0

    B = inference.batch_size
    mla = model.mla_config
    tp = parallelism.tp
    heads = model.num_attention_heads

    # Decode 时 S=1
    num_tokens = 1

    # 使用 absorbed attention 优化
    # W_KC @ c_t: [heads/TP, B, qk_nope, 1] × [B, 1, kv_lora_rank] → [heads/TP, B, qk_nope, kv_lora_rank]
    # 然后 Q @ (W_KC @ c_t): [heads/TP, B, 1, qk_nope] × [heads/TP, B, qk_nope, context]

    # 简化: 直接计算等效 GEMM
    head_dim = mla.qk_nope_head_dim + mla.qk_rope_head_dim

    # Score: [B×heads/TP, 1, context]
    G = B * heads // tp
    score_latency = calc_gemm_latency(M=1, K=head_dim, N=context_length, G=G)

    # Softmax @ V: [B×heads/TP, 1, v_head_dim]
    sv_latency = calc_gemm_latency(M=1, K=context_length, N=mla.v_head_dim, G=G)

    return score_latency + sv_latency


def calc_mla_output_projection_latency(
    model: LLMModelConfig,
    inference: InferenceConfig,
    parallelism: ParallelismStrategy,
    hardware: HardwareConfig,
    num_tokens: int,
) -> float:
    """
    MLA Output Projection 延迟

    GEMM: [B×S, heads×v_head_dim/TP] × [heads×v_head_dim/TP, H]
    """
    if model.mla_config is None:
        from .attention import calc_attention_output_proj_latency
        return calc_attention_output_proj_latency(model, inference, parallelism, hardware, num_tokens)

    B, S, H = inference.batch_size, num_tokens, model.hidden_size
    mla = model.mla_config
    tp = parallelism.tp
    heads = model.num_attention_heads

    input_dim = heads * mla.v_head_dim // tp

    return calc_gemm_latency(M=B*S, K=input_dim, N=H)


def calc_mla_kv_cache_read_latency(
    model: LLMModelConfig,
    inference: InferenceConfig,
    parallelism: ParallelismStrategy,
    hardware: HardwareConfig,
    context_length: int,
) -> float:
    """
    MLA KV Cache 读取延迟

    MLA 的 KV cache 是压缩后的: [B, context, kv_lora_rank + qk_rope_head_dim]
    比标准 MHA 小很多
    """
    if model.mla_config is None:
        return 0.0

    B = inference.batch_size
    mla = model.mla_config
    bytes_per_elem = get_bytes_per_element(model.dtype)

    # 压缩后的 KV 维度
    compressed_dim = mla.kv_lora_rank + mla.qk_rope_head_dim

    # KV cache 大小 (只有一份，不是 K 和 V 分开)
    kv_cache_bytes = B * context_length * compressed_dim * bytes_per_elem

    # 内存带宽
    arch = get_arch()
    data_gb = kv_cache_bytes / 1e9
    effective_bw = arch.dram_bandwidth_bytes / 1e9

    return (data_gb / effective_bw) * 1000 if effective_bw > 0 else 0.0


def calc_mla_kv_cache_write_latency(
    model: LLMModelConfig,
    inference: InferenceConfig,
    parallelism: ParallelismStrategy,
    hardware: HardwareConfig,
    num_tokens: int,
) -> float:
    """MLA KV Cache 写入延迟"""
    if model.mla_config is None:
        return 0.0

    B = inference.batch_size
    mla = model.mla_config
    bytes_per_elem = get_bytes_per_element(model.dtype)

    compressed_dim = mla.kv_lora_rank + mla.qk_rope_head_dim
    kv_cache_bytes = B * num_tokens * compressed_dim * bytes_per_elem

    arch = get_arch()
    data_gb = kv_cache_bytes / 1e9
    effective_bw = arch.dram_bandwidth_bytes / 1e9

    return (data_gb / effective_bw) * 1000 if effective_bw > 0 else 0.0


# MLA Output 别名 (兼容旧接口)
def calc_mla_output_latency(
    model: LLMModelConfig,
    inference: InferenceConfig,
    parallelism: ParallelismStrategy,
    hardware: HardwareConfig,
    num_tokens: int,
    context_length: int,
) -> float:
    """MLA Output 总延迟 (Softmax@V + OutProj)"""
    sv_latency = calc_mla_decode_attention_latency(model, inference, parallelism, hardware, context_length)
    out_latency = calc_mla_output_projection_latency(model, inference, parallelism, hardware, num_tokens)
    return sv_latency + out_latency


# ==================== MLA 细粒度函数 ====================

def calc_rmsnorm_q_lora_latency(
    model: LLMModelConfig,
    inference: InferenceConfig,
    parallelism: ParallelismStrategy,
    hardware: HardwareConfig,
    num_tokens: int,
) -> float:
    """Q LoRA 前的 RMSNorm 延迟"""
    if model.mla_config is None:
        return 0.0

    B, S, H = inference.batch_size, num_tokens, model.hidden_size
    bytes_per_elem = get_bytes_per_element(model.dtype)

    data_bytes = B * S * H * bytes_per_elem * 2
    arch = get_arch()
    data_gb = data_bytes / 1e9
    effective_bw = arch.dram_bandwidth_bytes / 1e9

    return (data_gb / effective_bw) * 1000 if effective_bw > 0 else 0.0


def calc_rmsnorm_kv_lora_latency(
    model: LLMModelConfig,
    inference: InferenceConfig,
    parallelism: ParallelismStrategy,
    hardware: HardwareConfig,
    num_tokens: int,
) -> float:
    """KV LoRA 前的 RMSNorm 延迟"""
    return calc_rmsnorm_q_lora_latency(model, inference, parallelism, hardware, num_tokens)


def calc_mm_q_lora_a_latency(
    model: LLMModelConfig,
    inference: InferenceConfig,
    parallelism: ParallelismStrategy,
    hardware: HardwareConfig,
    num_tokens: int,
) -> float:
    """Q LoRA Down 投影延迟"""
    if model.mla_config is None:
        return 0.0

    B, S, H = inference.batch_size, num_tokens, model.hidden_size
    mla = model.mla_config

    return calc_gemm_latency(M=B*S, K=H, N=mla.q_lora_rank)


def calc_mm_q_lora_b_latency(
    model: LLMModelConfig,
    inference: InferenceConfig,
    parallelism: ParallelismStrategy,
    hardware: HardwareConfig,
    num_tokens: int,
) -> float:
    """Q LoRA Up 投影延迟"""
    if model.mla_config is None:
        return 0.0

    B, S = inference.batch_size, num_tokens
    mla = model.mla_config
    tp = parallelism.tp
    heads = model.num_attention_heads
    head_dim = mla.qk_nope_head_dim + mla.qk_rope_head_dim

    return calc_gemm_latency(M=B*S, K=mla.q_lora_rank, N=heads*head_dim//tp)


def calc_mm_kv_lora_a_latency(
    model: LLMModelConfig,
    inference: InferenceConfig,
    parallelism: ParallelismStrategy,
    hardware: HardwareConfig,
    num_tokens: int,
) -> float:
    """KV LoRA 压缩延迟"""
    return calc_mla_kv_compression_latency(model, inference, parallelism, hardware, num_tokens)


def calc_bmm_qk_latency(
    model: LLMModelConfig,
    inference: InferenceConfig,
    parallelism: ParallelismStrategy,
    hardware: HardwareConfig,
    num_tokens: int,
    context_length: int,
) -> float:
    """Q @ K^T Batch MatMul 延迟"""
    return calc_mla_attention_score_latency(model, inference, parallelism, hardware, num_tokens, context_length)


def calc_bmm_sv_latency(
    model: LLMModelConfig,
    inference: InferenceConfig,
    parallelism: ParallelismStrategy,
    hardware: HardwareConfig,
    num_tokens: int,
    context_length: int,
) -> float:
    """Softmax @ V Batch MatMul 延迟"""
    if model.mla_config is None:
        from .attention import calc_attention_sv_latency
        return calc_attention_sv_latency(model, inference, parallelism, hardware, num_tokens, context_length)

    B, S = inference.batch_size, num_tokens
    mla = model.mla_config
    tp = parallelism.tp
    heads = model.num_attention_heads

    G = B * heads // tp
    return calc_gemm_latency(M=S, K=context_length, N=mla.v_head_dim, G=G)


def calc_attn_fc_latency(
    model: LLMModelConfig,
    inference: InferenceConfig,
    parallelism: ParallelismStrategy,
    hardware: HardwareConfig,
    num_tokens: int,
) -> float:
    """Attention 输出投影延迟"""
    return calc_mla_output_projection_latency(model, inference, parallelism, hardware, num_tokens)
