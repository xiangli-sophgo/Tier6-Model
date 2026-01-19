"""
内存访问延迟计算
"""

from ..types import LLMModelConfig, InferenceConfig, ParallelismStrategy, HardwareConfig, get_bytes_per_element
from .core import get_arch


# 常量定义
PCIE_PROTOCOL_OVERHEAD = 0.15  # PCIe 协议开销
HBM_RANDOM_ACCESS_PENALTY = 1.2  # 随机访问惩罚因子


def calc_hbm_read_latency(
    data_bytes: int | float,
    hardware: HardwareConfig,
    is_sequential: bool = True,
) -> float:
    """
    计算 HBM 读取延迟 (ms)

    Args:
        data_bytes: 数据量 (字节)
        hardware: 硬件配置
        is_sequential: 是否顺序访问

    Returns:
        延迟 (ms)
    """
    if data_bytes <= 0:
        return 0.0

    arch = get_arch()
    data_gb = data_bytes / 1e9
    bandwidth_gb_per_s = arch.dram_bandwidth_bytes / 1e9

    if bandwidth_gb_per_s <= 0:
        return 0.0

    latency_ms = (data_gb / bandwidth_gb_per_s) * 1000

    if not is_sequential:
        latency_ms *= HBM_RANDOM_ACCESS_PENALTY

    return latency_ms


def calc_hbm_write_latency(
    data_bytes: int | float,
    hardware: HardwareConfig,
) -> float:
    """计算 HBM 写入延迟 (ms)"""
    return calc_hbm_read_latency(data_bytes, hardware, is_sequential=True)


def calc_weight_load_latency(
    weight_bytes: int | float,
    hardware: HardwareConfig,
) -> float:
    """
    计算权重加载延迟 (ms)

    权重加载是顺序读取
    """
    return calc_hbm_read_latency(weight_bytes, hardware, is_sequential=True)


def calc_kv_cache_read_latency(
    model: LLMModelConfig,
    inference: InferenceConfig,
    parallelism: ParallelismStrategy,
    hardware: HardwareConfig,
    context_length: int,
) -> float:
    """
    计算 KV Cache 读取延迟 (ms)

    KV Cache 大小 = 2 × B × context × num_kv_heads × head_dim / TP
    """
    head_dim = model.hidden_size // model.num_attention_heads
    bytes_per_elem = get_bytes_per_element(model.dtype)
    tp = parallelism.tp

    kv_cache_bytes = (
        2 * inference.batch_size * context_length *
        model.num_kv_heads * head_dim * bytes_per_elem
    ) // tp

    # KV Cache 访问模式接近随机
    return calc_hbm_read_latency(kv_cache_bytes, hardware, is_sequential=False)


def calc_kv_cache_write_latency(
    model: LLMModelConfig,
    inference: InferenceConfig,
    parallelism: ParallelismStrategy,
    hardware: HardwareConfig,
    num_tokens: int = 1,
) -> float:
    """
    计算 KV Cache 写入延迟 (ms)

    每次写入 num_tokens 个 token 的 KV
    """
    head_dim = model.hidden_size // model.num_attention_heads
    bytes_per_elem = get_bytes_per_element(model.dtype)
    tp = parallelism.tp

    kv_write_bytes = (
        2 * inference.batch_size * num_tokens *
        model.num_kv_heads * head_dim * bytes_per_elem
    ) // tp

    return calc_hbm_write_latency(kv_write_bytes, hardware)


def calc_embedding_latency(
    model: LLMModelConfig,
    inference: InferenceConfig,
    hardware: HardwareConfig,
    num_tokens: int,
) -> float:
    """
    计算 Embedding 层延迟 (ms)

    Embedding 是查表操作，受内存带宽限制
    """
    bytes_per_elem = get_bytes_per_element(model.dtype)
    data_bytes = inference.batch_size * num_tokens * model.hidden_size * bytes_per_elem

    return calc_hbm_read_latency(data_bytes, hardware, is_sequential=False)


def calc_lm_head_latency(
    model: LLMModelConfig,
    inference: InferenceConfig,
    parallelism: ParallelismStrategy,
    hardware: HardwareConfig,
    num_tokens: int,
) -> float:
    """
    计算 LM Head 延迟 (ms)

    GEMM: [B×S, H] × [H, vocab_size]
    """
    from .core import calc_gemm_latency

    B, S, H = inference.batch_size, num_tokens, model.hidden_size
    V = model.vocab_size
    tp = parallelism.tp

    # LM Head 通常按 TP 切分 vocab
    return calc_gemm_latency(M=B*S, K=H, N=V//tp)


def calc_pcie_h2d_latency(
    data_bytes: int | float,
    hardware: HardwareConfig,
) -> float:
    """
    计算 PCIe Host to Device 传输延迟 (ms)
    """
    if data_bytes <= 0:
        return 0.0

    # PCIe 带宽
    pcie_bandwidth_gbps = getattr(hardware.chip, 'pcie_bandwidth_gbps', 64)
    effective_bandwidth = pcie_bandwidth_gbps * (1 - PCIE_PROTOCOL_OVERHEAD)

    data_gb = data_bytes / 1e9
    transfer_ms = (data_gb / effective_bandwidth) * 1000

    # 启动延迟
    startup_us = getattr(hardware.chip, 'pcie_latency_us', 5)
    startup_ms = startup_us / 1000

    return transfer_ms + startup_ms


def calc_pcie_d2h_latency(
    data_bytes: int | float,
    hardware: HardwareConfig,
) -> float:
    """计算 PCIe Device to Host 传输延迟 (ms)"""
    return calc_pcie_h2d_latency(data_bytes, hardware)


def calc_activation_memory_latency(
    model: LLMModelConfig,
    inference: InferenceConfig,
    parallelism: ParallelismStrategy,
    hardware: HardwareConfig,
    num_tokens: int,
) -> float:
    """
    计算激活值内存访问延迟 (ms)

    主要是 Transformer 层间的激活值读写
    """
    B, S, H = inference.batch_size, num_tokens, model.hidden_size
    bytes_per_elem = get_bytes_per_element(model.dtype)

    # 每层的激活值约为 B × S × H × 4 (输入 + QKV + Attention + FFN)
    activation_bytes = B * S * H * bytes_per_elem * 4

    return calc_hbm_read_latency(activation_bytes, hardware, is_sequential=True)
