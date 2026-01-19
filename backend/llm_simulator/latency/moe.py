"""
MoE (Mixture of Experts) 相关延迟计算

包含:
- get_max_expert: 专家负载不均衡查表 (DS_TPU 实测数据)
- MoE Gate/Expert FFN/Shared Expert 延迟计算
- EP Dispatch/Combine 通信延迟
"""

from typing import Optional
from ..types import LLMModelConfig, InferenceConfig, ParallelismStrategy, HardwareConfig, get_bytes_per_element
from .core import calc_gemm_latency, get_arch


# ==================== 专家负载不均衡查表 ====================
# 来自 DS_TPU_1209 实测数据
# 返回: 每个 EP 组内最大激活专家数
# 键: batch (token 数量)
# 值: {chips (EP 数量): max_activated_experts}

EXPERT_LOAD_TABLE = {
    1: {1: 8.0, 2: 5.08915, 4: 3.5096, 8: 2.5613, 16: 2.02795, 32: 1.60595, 64: 1.2973, 128: 1.1086, 256: 1.0},
    2: {1: 15.74335, 2: 9.3783, 4: 5.96155, 8: 4.07395, 16: 2.94035, 32: 2.26275, 64: 1.8285, 128: 1.38245, 256: 1.0},
    3: {1: 23.2566, 2: 13.44475, 4: 8.25905, 8: 5.38985, 16: 3.7346, 32: 2.741, 64: 2.1301, 128: 1.6721, 256: 1.0},
    4: {1: 30.5121, 2: 17.3445, 4: 10.3651, 8: 6.575, 16: 4.44845, 32: 3.18425, 64: 2.326, 128: 1.8603, 256: 1.0},
    5: {1: 37.592, 2: 21.01065, 4: 12.3875, 8: 7.68935, 16: 5.0503, 32: 3.5408, 64: 2.53475, 128: 1.9572, 256: 1.0},
    6: {1: 44.3833, 2: 24.61835, 4: 14.27665, 8: 8.73345, 16: 5.6629, 32: 3.8562, 64: 2.7497, 128: 1.9882, 256: 1.0},
    8: {1: 57.4309, 2: 31.40505, 4: 17.81475, 8: 10.6518, 16: 6.69615, 32: 4.4437, 64: 3.0746, 128: 1.9997, 256: 1.0},
    16: {1: 101.9848, 2: 54.1023, 4: 29.53585, 8: 16.7107, 16: 9.86765, 32: 6.05895, 64: 3.8144, 128: 2.0, 256: 1.0},
    20: {1: 120.3125, 2: 63.3071, 4: 34.1843, 8: 19.05355, 16: 11.00855, 32: 6.5959, 64: 3.9689, 128: 2.0, 256: 1.0},
    24: {1: 136.4634, 2: 71.4372, 4: 38.2351, 8: 21.0515, 16: 11.9747, 32: 7.01615, 64: 3.99745, 128: 2.0, 256: 1.0},
    32: {1: 163.27965, 2: 84.75245, 4: 44.79135, 8: 24.19435, 16: 13.42475, 32: 7.5834, 64: 4.0, 128: 2.0, 256: 1.0},
    40: {1: 184.0948, 2: 94.89465, 4: 49.68275, 8: 26.51455, 16: 14.41565, 32: 7.91665, 64: 4.0, 128: 2.0, 256: 1.0},
    48: {1: 200.20185, 2: 102.74895, 4: 53.3845, 8: 28.1981, 16: 15.1102, 32: 7.9938, 64: 4.0, 128: 2.0, 256: 1.0},
    64: {1: 222.417, 2: 113.352, 4: 58.32715, 8: 30.30105, 16: 15.8369, 32: 8.0, 64: 4.0, 128: 2.0, 256: 1.0},
    128: {1: 251.59985, 2: 126.611, 4: 63.7934, 8: 31.9994, 16: 16.0, 32: 8.0, 64: 4.0, 128: 2.0, 256: 1.0},
    256: {1: 255.9269, 2: 127.9988, 4: 64.0, 8: 32.0, 16: 16.0, 32: 8.0, 64: 4.0, 128: 2.0, 256: 1.0},
}


def get_max_expert(batch: int, chips: int) -> int:
    """
    获取每个 EP 组内最大激活专家数

    基于 DS_TPU 实测数据的查表函数。
    用于估算 MoE 层负载不均衡情况。

    Args:
        batch: Token 数量 (会被截断到 1-256)
        chips: EP 数量 (芯片数)

    Returns:
        最大激活专家数 (整数)

    Raises:
        KeyError: 不支持的 batch/chips 组合
    """
    # 截断 batch 到有效范围
    batch = min(256, max(1, batch))

    # 查找最接近的 batch 值
    valid_batches = sorted(EXPERT_LOAD_TABLE.keys())
    target_batch = batch

    # 如果 batch 不在表中，找最接近的
    if batch not in EXPERT_LOAD_TABLE:
        for b in valid_batches:
            if b >= batch:
                target_batch = b
                break
        else:
            target_batch = valid_batches[-1]

    chip_data = EXPERT_LOAD_TABLE.get(target_batch, {})

    if chips not in chip_data:
        # 查找最接近的 chips 值
        valid_chips = sorted(chip_data.keys())
        for c in valid_chips:
            if c >= chips:
                chips = c
                break
        else:
            chips = valid_chips[-1] if valid_chips else 1

    try:
        return round(chip_data[chips])
    except KeyError:
        # 默认返回 1
        return 1


def get_max_expert_float(batch: int, chips: int) -> float:
    """
    获取每个 EP 组内最大激活专家数 (浮点数版本)

    与 get_max_expert 相同，但返回精确的浮点数值。

    Args:
        batch: Token 数量
        chips: EP 数量

    Returns:
        最大激活专家数 (浮点数)
    """
    batch = min(256, max(1, batch))

    if batch not in EXPERT_LOAD_TABLE:
        valid_batches = sorted(EXPERT_LOAD_TABLE.keys())
        for b in valid_batches:
            if b >= batch:
                batch = b
                break
        else:
            batch = valid_batches[-1]

    chip_data = EXPERT_LOAD_TABLE.get(batch, {})

    if chips not in chip_data:
        valid_chips = sorted(chip_data.keys())
        for c in valid_chips:
            if c >= chips:
                chips = c
                break
        else:
            chips = valid_chips[-1] if valid_chips else 1

    return chip_data.get(chips, 1.0)


def calc_moe_gate_latency(
    model: LLMModelConfig,
    inference: InferenceConfig,
    parallelism: ParallelismStrategy,
    hardware: HardwareConfig,
    num_tokens: int,
) -> float:
    """
    MoE Gate 路由延迟

    GEMM: [B×S, H] × [H, num_experts]
    """
    if model.moe_config is None:
        return 0.0

    B, S, H = inference.batch_size, num_tokens, model.hidden_size
    num_experts = model.moe_config.num_experts

    return calc_gemm_latency(M=B*S, K=H, N=num_experts)


def calc_moe_expert_ffn_latency(
    model: LLMModelConfig,
    inference: InferenceConfig,
    parallelism: ParallelismStrategy,
    hardware: HardwareConfig,
    num_tokens: int,
) -> float:
    """
    MoE Expert FFN 延迟 (考虑负载不均衡)

    每个专家处理的 token 数 = total_tokens × top_k / num_experts × load_factor / EP
    """
    if model.moe_config is None:
        return 0.0

    B, S, H = inference.batch_size, num_tokens, model.hidden_size
    moe = model.moe_config
    ep = parallelism.ep
    tp = parallelism.tp
    moe_tp = getattr(parallelism, 'moe_tp', tp)  # MoE 专用 TP，默认等于 TP

    # 专家 FFN 中间维度
    expert_I = moe.expert_intermediate_size or model.intermediate_size
    top_k = moe.num_experts_per_tok

    # 每个 EP 组内的 token 数和专家数
    total_tokens = B * S
    token_per_ep_group = (total_tokens * top_k + ep - 1) // ep  # ceil_div
    expert_per_ep_group = (moe.num_experts + ep - 1) // ep

    # 使用 get_max_expert 查表获取激活专家数
    activated_expert_per_ep_group = max(1, get_max_expert(total_tokens, ep))
    activated_expert_per_ep_group = min(expert_per_ep_group, activated_expert_per_ep_group)

    # 每个专家处理的 token 数
    m_per_group = (token_per_ep_group + expert_per_ep_group - 1) // expert_per_ep_group

    # Routed Expert FFN (使用 G 维度表示并行专家数)
    # Gate projection: [G, M, K] × [K, N] -> [G, M, N]
    gate_latency = calc_gemm_latency(
        M=m_per_group,
        K=H,
        N=expert_I // moe_tp,
        G=activated_expert_per_ep_group,
    )
    up_latency = calc_gemm_latency(
        M=m_per_group,
        K=H,
        N=expert_I // moe_tp,
        G=activated_expert_per_ep_group,
    )
    down_latency = calc_gemm_latency(
        M=m_per_group,
        K=expert_I // moe_tp,
        N=H,
        G=activated_expert_per_ep_group,
    )

    return gate_latency + up_latency + down_latency


def calc_moe_shared_expert_latency(
    model: LLMModelConfig,
    inference: InferenceConfig,
    parallelism: ParallelismStrategy,
    hardware: HardwareConfig,
    num_tokens: int,
) -> float:
    """
    MoE Shared Expert FFN 延迟 (DeepSeek V3 特有)

    Shared expert 处理所有 token
    """
    if model.moe_config is None:
        return 0.0

    moe = model.moe_config
    if moe.num_shared_experts <= 0:
        return 0.0

    B, S, H = inference.batch_size, num_tokens, model.hidden_size
    tp = parallelism.tp

    # Shared expert 使用完整的 intermediate_size
    shared_I = model.intermediate_size

    # 每个 shared expert 的 FFN
    gate_latency = calc_gemm_latency(M=B*S, K=H, N=shared_I//tp)
    up_latency = calc_gemm_latency(M=B*S, K=H, N=shared_I//tp)
    down_latency = calc_gemm_latency(M=B*S, K=shared_I//tp, N=H)

    single_shared_latency = gate_latency + up_latency + down_latency

    return single_shared_latency * moe.num_shared_experts


def calc_moe_alltoall_latency(
    model: LLMModelConfig,
    inference: InferenceConfig,
    parallelism: ParallelismStrategy,
    hardware: HardwareConfig,
    num_tokens: int,
) -> float:
    """
    MoE All-to-All 通信延迟

    每个 token 发送到对应专家所在的芯片
    """
    if model.moe_config is None:
        return 0.0

    ep = parallelism.ep
    if ep <= 1:
        return 0.0

    B, S, H = inference.batch_size, num_tokens, model.hidden_size
    bytes_per_elem = get_bytes_per_element(model.dtype)

    # 每个 token 的数据量
    token_bytes = H * bytes_per_elem

    # All-to-All: 每个节点发送 (ep-1)/ep 的数据
    total_data_bytes = B * S * token_bytes
    transfer_bytes = total_data_bytes * (ep - 1) / ep

    # 获取网络带宽
    # 使用节点内带宽或集群带宽
    bandwidth_gbps = hardware.node.nvlink_bandwidth_gbps if hasattr(hardware, 'node') else 100
    bandwidth_bytes = bandwidth_gbps * 1e9

    # 启动延迟
    latency_us = hardware.node.nvlink_latency_us if hasattr(hardware, 'node') else 1.0

    transfer_ms = transfer_bytes / bandwidth_bytes * 1000
    startup_ms = latency_us * (ep - 1) / 1000

    return transfer_ms + startup_ms


def calc_ep_dispatch_latency(
    model: LLMModelConfig,
    inference: InferenceConfig,
    parallelism: ParallelismStrategy,
    hardware: HardwareConfig,
    num_tokens: int,
) -> float:
    """EP Dispatch 通信延迟 (All-to-All 发送)"""
    return calc_moe_alltoall_latency(model, inference, parallelism, hardware, num_tokens)


def calc_ep_combine_latency(
    model: LLMModelConfig,
    inference: InferenceConfig,
    parallelism: ParallelismStrategy,
    hardware: HardwareConfig,
    num_tokens: int,
) -> float:
    """EP Combine 通信延迟 (All-to-All 接收)"""
    return calc_moe_alltoall_latency(model, inference, parallelism, hardware, num_tokens)


def is_moe_layer(model: LLMModelConfig, layer_index: int) -> bool:
    """判断是否为 MoE 层"""
    if model.moe_config is None:
        return False

    # 前 K 层使用 Dense FFN
    first_k_dense = model.moe_config.first_k_dense_replace
    return layer_index >= first_k_dense
