"""
FFN (Feed-Forward Network) 相关延迟计算
"""

from ..types import LLMModelConfig, InferenceConfig, ParallelismStrategy, HardwareConfig, get_bytes_per_element
from .core import calc_gemm_latency, get_arch


def calc_ffn_gate_latency(
    model: LLMModelConfig,
    inference: InferenceConfig,
    parallelism: ParallelismStrategy,
    hardware: HardwareConfig,
    num_tokens: int,
) -> float:
    """
    FFN Gate 投影延迟 (SwiGLU 的 gate 部分)

    GEMM: [B×S, H] × [H, I/TP]
    """
    B, S, H = inference.batch_size, num_tokens, model.hidden_size
    I = model.intermediate_size
    tp = parallelism.tp

    return calc_gemm_latency(M=B*S, K=H, N=I//tp)


def calc_ffn_up_latency(
    model: LLMModelConfig,
    inference: InferenceConfig,
    parallelism: ParallelismStrategy,
    hardware: HardwareConfig,
    num_tokens: int,
) -> float:
    """
    FFN Up 投影延迟 (SwiGLU 的 up 部分)

    GEMM: [B×S, H] × [H, I/TP]
    """
    B, S, H = inference.batch_size, num_tokens, model.hidden_size
    I = model.intermediate_size
    tp = parallelism.tp

    return calc_gemm_latency(M=B*S, K=H, N=I//tp)


def calc_ffn_down_latency(
    model: LLMModelConfig,
    inference: InferenceConfig,
    parallelism: ParallelismStrategy,
    hardware: HardwareConfig,
    num_tokens: int,
) -> float:
    """
    FFN Down 投影延迟

    GEMM: [B×S, I/TP] × [I/TP, H]
    """
    B, S, H = inference.batch_size, num_tokens, model.hidden_size
    I = model.intermediate_size
    tp = parallelism.tp

    return calc_gemm_latency(M=B*S, K=I//tp, N=H)


def calc_ffn_activation_latency(
    model: LLMModelConfig,
    inference: InferenceConfig,
    parallelism: ParallelismStrategy,
    hardware: HardwareConfig,
    num_tokens: int,
) -> float:
    """
    FFN 激活函数延迟 (SiLU/GELU)

    内存受限: 读 gate + up, 写 activated
    """
    B, S = inference.batch_size, num_tokens
    I = model.intermediate_size
    tp = parallelism.tp
    bytes_per_elem = get_bytes_per_element(model.dtype)

    # 读 2 × I/TP, 写 1 × I/TP
    data_bytes = B * S * (I // tp) * bytes_per_elem * 3
    data_gb = data_bytes / 1e9
    arch = get_arch()
    effective_bw = arch.dram_bandwidth_bytes / 1e9

    return (data_gb / effective_bw) * 1000 if effective_bw > 0 else 0.0


def calc_layernorm_latency(
    model: LLMModelConfig,
    inference: InferenceConfig,
    parallelism: ParallelismStrategy,
    hardware: HardwareConfig,
    num_tokens: int,
) -> float:
    """
    LayerNorm/RMSNorm 延迟 (内存受限操作)

    数据量: B × S × H × 2 (读+写)
    """
    B, S, H = inference.batch_size, num_tokens, model.hidden_size
    bytes_per_elem = get_bytes_per_element(model.dtype)

    data_bytes = B * S * H * bytes_per_elem * 2
    data_gb = data_bytes / 1e9
    arch = get_arch()
    effective_bw = arch.dram_bandwidth_bytes / 1e9

    return (data_gb / effective_bw) * 1000 if effective_bw > 0 else 0.0


def calc_residual_add_latency(
    model: LLMModelConfig,
    inference: InferenceConfig,
    parallelism: ParallelismStrategy,
    hardware: HardwareConfig,
    num_tokens: int,
) -> float:
    """
    残差连接延迟 (内存受限操作)

    数据量: B × S × H × 3 (读 2, 写 1)
    """
    B, S, H = inference.batch_size, num_tokens, model.hidden_size
    bytes_per_elem = get_bytes_per_element(model.dtype)

    data_bytes = B * S * H * bytes_per_elem * 3
    data_gb = data_bytes / 1e9
    arch = get_arch()
    effective_bw = arch.dram_bandwidth_bytes / 1e9

    return (data_gb / effective_bw) * 1000 if effective_bw > 0 else 0.0
