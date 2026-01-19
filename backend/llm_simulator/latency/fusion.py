"""
Kernel Fusion 相关延迟计算

提供融合算子的延迟估算
"""

from ..types import LLMModelConfig, InferenceConfig, ParallelismStrategy, HardwareConfig, get_bytes_per_element
from .core import calc_gemm_latency, get_arch
from .ffn import calc_layernorm_latency, calc_ffn_gate_latency, calc_ffn_up_latency, calc_ffn_down_latency
from .attention import calc_attention_qkv_latency, calc_attention_score_latency, calc_attention_softmax_latency, calc_attention_output_latency


# 重叠系数 (计算与通信重叠)
OVERLAP_COEFFICIENTS = {
    'compute_comm': 0.8,  # 计算与通信重叠率
    'layernorm_gemm': 0.9,  # LayerNorm 与 GEMM 融合收益
    'gate_up': 0.95,  # Gate 和 Up 投影融合收益
}


def calc_fused_layernorm_qkv_latency(
    model: LLMModelConfig,
    inference: InferenceConfig,
    parallelism: ParallelismStrategy,
    hardware: HardwareConfig,
    num_tokens: int,
) -> float:
    """
    融合 LayerNorm + QKV 投影的延迟

    融合后减少内存访问，提高效率
    """
    # 分别计算
    ln_latency = calc_layernorm_latency(model, inference, parallelism, hardware, num_tokens)
    qkv_latency = calc_attention_qkv_latency(model, inference, parallelism, hardware, num_tokens)

    # 融合收益
    fusion_factor = OVERLAP_COEFFICIENTS['layernorm_gemm']

    return ln_latency * (1 - fusion_factor) + qkv_latency


def calc_fused_ffn_gate_up_latency(
    model: LLMModelConfig,
    inference: InferenceConfig,
    parallelism: ParallelismStrategy,
    hardware: HardwareConfig,
    num_tokens: int,
) -> float:
    """
    融合 FFN Gate + Up 投影的延迟 (SwiGLU)

    两个投影可以合并为一个 GEMM
    """
    B, S, H = inference.batch_size, num_tokens, model.hidden_size
    I = model.intermediate_size
    tp = parallelism.tp

    # 融合后: [B×S, H] × [H, 2×I/TP]
    return calc_gemm_latency(M=B*S, K=H, N=2*I//tp)


def calc_single_layer_latency_fused(
    model: LLMModelConfig,
    inference: InferenceConfig,
    parallelism: ParallelismStrategy,
    hardware: HardwareConfig,
    num_tokens: int,
    context_length: int,
) -> float:
    """
    计算单层 Transformer 的融合延迟

    包括:
    - Fused LayerNorm + QKV
    - Attention Score + Softmax + Output
    - Fused LayerNorm + FFN Gate+Up
    - FFN Down
    """
    # Attention 部分
    fused_ln_qkv = calc_fused_layernorm_qkv_latency(model, inference, parallelism, hardware, num_tokens)
    attn_score = calc_attention_score_latency(model, inference, parallelism, hardware, num_tokens, context_length)
    attn_softmax = calc_attention_softmax_latency(model, inference, parallelism, hardware, num_tokens, context_length)
    attn_output = calc_attention_output_latency(model, inference, parallelism, hardware, num_tokens, context_length)

    # FFN 部分
    ln2_latency = calc_layernorm_latency(model, inference, parallelism, hardware, num_tokens)
    fused_gate_up = calc_fused_ffn_gate_up_latency(model, inference, parallelism, hardware, num_tokens)
    ffn_down = calc_ffn_down_latency(model, inference, parallelism, hardware, num_tokens)

    # 融合 LayerNorm + Gate+Up
    ln2_fused_factor = OVERLAP_COEFFICIENTS['layernorm_gemm']
    fused_ln2_gateup = ln2_latency * (1 - ln2_fused_factor) + fused_gate_up

    total = (
        fused_ln_qkv +
        attn_score +
        attn_softmax +
        attn_output +
        fused_ln2_gateup +
        ffn_down
    )

    return total
