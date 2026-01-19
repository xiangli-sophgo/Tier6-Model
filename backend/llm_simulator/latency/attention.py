"""
Attention 相关延迟计算
"""

from ..types import LLMModelConfig, InferenceConfig, ParallelismStrategy, HardwareConfig, get_bytes_per_element
from .core import calc_gemm_latency, get_arch


def calc_attention_qkv_latency(
    model: LLMModelConfig,
    inference: InferenceConfig,
    parallelism: ParallelismStrategy,
    hardware: HardwareConfig,
    num_tokens: int,
) -> float:
    """
    QKV 投影延迟

    GEMM: [B×S, H] × [H, H + 2×kv_heads×head_dim] / TP
    """
    B, S, H = inference.batch_size, num_tokens, model.hidden_size
    head_dim = H // model.num_attention_heads
    tp = parallelism.tp

    # Q 投影
    q_latency = calc_gemm_latency(M=B*S, K=H, N=H//tp)

    # K, V 投影
    kv_dim = model.num_kv_heads * head_dim // tp
    k_latency = calc_gemm_latency(M=B*S, K=H, N=kv_dim)
    v_latency = calc_gemm_latency(M=B*S, K=H, N=kv_dim)

    return q_latency + k_latency + v_latency


def calc_attention_score_latency(
    model: LLMModelConfig,
    inference: InferenceConfig,
    parallelism: ParallelismStrategy,
    hardware: HardwareConfig,
    num_tokens: int,
    context_length: int,
) -> float:
    """
    Attention Score (Q @ K^T) 延迟

    GEMM: [B×heads/TP, S, head_dim] × [B×heads/TP, head_dim, C]
    """
    B, S, H = inference.batch_size, num_tokens, model.hidden_size
    heads = model.num_attention_heads
    head_dim = H // heads
    tp = parallelism.tp

    G = B * heads // tp
    return calc_gemm_latency(M=S, K=head_dim, N=context_length, G=G)


def calc_attention_softmax_latency(
    model: LLMModelConfig,
    inference: InferenceConfig,
    parallelism: ParallelismStrategy,
    hardware: HardwareConfig,
    num_tokens: int,
    context_length: int,
) -> float:
    """
    Softmax 延迟 (内存受限操作)

    数据量: B × heads/TP × S × C × 2 (读+写)
    """
    B, S = inference.batch_size, num_tokens
    heads = model.num_attention_heads // parallelism.tp
    bytes_per_elem = get_bytes_per_element(model.dtype)

    data_bytes = B * heads * S * context_length * bytes_per_elem * 2
    data_gb = data_bytes / 1e9
    arch = get_arch()
    effective_bw = arch.dram_bandwidth_bytes / 1e9

    return (data_gb / effective_bw) * 1000 if effective_bw > 0 else 0.0


def calc_attention_sv_latency(
    model: LLMModelConfig,
    inference: InferenceConfig,
    parallelism: ParallelismStrategy,
    hardware: HardwareConfig,
    num_tokens: int,
    context_length: int,
) -> float:
    """
    Softmax @ V 延迟

    GEMM: [B×heads/TP, S, C] × [B×heads/TP, C, head_dim]
    """
    B, S, H = inference.batch_size, num_tokens, model.hidden_size
    heads = model.num_attention_heads
    head_dim = H // heads
    tp = parallelism.tp

    G = B * heads // tp
    return calc_gemm_latency(M=S, K=context_length, N=head_dim, G=G)


def calc_attention_output_proj_latency(
    model: LLMModelConfig,
    inference: InferenceConfig,
    parallelism: ParallelismStrategy,
    hardware: HardwareConfig,
    num_tokens: int,
) -> float:
    """
    Output Projection 延迟

    GEMM: [B×S, H/TP] × [H/TP, H]
    """
    B, S, H = inference.batch_size, num_tokens, model.hidden_size
    tp = parallelism.tp

    return calc_gemm_latency(M=B*S, K=H//tp, N=H)


def calc_attention_output_latency(
    model: LLMModelConfig,
    inference: InferenceConfig,
    parallelism: ParallelismStrategy,
    hardware: HardwareConfig,
    num_tokens: int,
    context_length: int,
) -> float:
    """
    Attention Output 总延迟 (Softmax@V + OutProj)
    """
    sv_latency = calc_attention_sv_latency(
        model, inference, parallelism, hardware, num_tokens, context_length
    )
    out_latency = calc_attention_output_proj_latency(
        model, inference, parallelism, hardware, num_tokens
    )
    return sv_latency + out_latency
