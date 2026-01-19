"""
集合通信延迟计算
"""

from ..types import HardwareConfig, LLMModelConfig, InferenceConfig, ParallelismStrategy, get_bytes_per_element


def calc_tp_allreduce_latency(
    data_bytes: int | float,
    hardware: HardwareConfig,
    tp: int,
) -> float:
    """
    TP AllReduce 延迟 (ms)

    使用 Ring AllReduce 算法:
    - 通信量: 2 × (n-1)/n × data_size
    - 延迟: startup × (n-1) + transfer_time

    Args:
        data_bytes: 数据量 (字节)
        hardware: 硬件配置
        tp: 张量并行度

    Returns:
        延迟 (ms)
    """
    if tp <= 1:
        return 0.0

    if data_bytes <= 0:
        return 0.0

    # 获取节点内带宽 (NVLink/NVSwitch)
    bandwidth_gbps = getattr(hardware.node, 'nvlink_bandwidth_gbps', 400)
    latency_us = getattr(hardware.node, 'nvlink_latency_us', 1.0)

    # Ring AllReduce 通信量
    transfer_factor = 2 * (tp - 1) / tp
    data_gb = data_bytes / 1e9
    transfer_ms = (data_gb * transfer_factor / bandwidth_gbps) * 1000

    # 启动延迟
    startup_ms = latency_us * (tp - 1) / 1000

    return transfer_ms + startup_ms


def calc_pp_p2p_latency(
    data_bytes: int | float,
    hardware: HardwareConfig,
    is_cross_node: bool = False,
) -> float:
    """
    PP 点对点通信延迟 (ms)

    Args:
        data_bytes: 数据量 (字节)
        hardware: 硬件配置
        is_cross_node: 是否跨节点

    Returns:
        延迟 (ms)
    """
    if data_bytes <= 0:
        return 0.0

    if is_cross_node:
        # 跨节点使用 InfiniBand/Ethernet
        bandwidth_gbps = getattr(hardware.cluster, 'ib_bandwidth_gbps', 100)
        latency_us = getattr(hardware.cluster, 'ib_latency_us', 5.0)
    else:
        # 节点内使用 NVLink
        bandwidth_gbps = getattr(hardware.node, 'nvlink_bandwidth_gbps', 400)
        latency_us = getattr(hardware.node, 'nvlink_latency_us', 1.0)

    data_gb = data_bytes / 1e9
    transfer_ms = (data_gb / bandwidth_gbps) * 1000
    startup_ms = latency_us / 1000

    return transfer_ms + startup_ms


def calc_ep_alltoall_latency(
    data_bytes: int | float,
    hardware: HardwareConfig,
    ep: int,
    is_cross_node: bool = False,
) -> float:
    """
    EP All-to-All 通信延迟 (ms)

    All-to-All: 每个节点发送 (n-1)/n 的数据给其他节点

    Args:
        data_bytes: 总数据量 (字节)
        hardware: 硬件配置
        ep: Expert 并行度
        is_cross_node: 是否跨节点

    Returns:
        延迟 (ms)
    """
    if ep <= 1:
        return 0.0

    if data_bytes <= 0:
        return 0.0

    if is_cross_node:
        bandwidth_gbps = getattr(hardware.cluster, 'ib_bandwidth_gbps', 100)
        latency_us = getattr(hardware.cluster, 'ib_latency_us', 5.0)
    else:
        bandwidth_gbps = getattr(hardware.node, 'nvlink_bandwidth_gbps', 400)
        latency_us = getattr(hardware.node, 'nvlink_latency_us', 1.0)

    # All-to-All 通信量
    transfer_factor = (ep - 1) / ep
    data_gb = data_bytes / 1e9
    transfer_ms = (data_gb * transfer_factor / bandwidth_gbps) * 1000

    # 启动延迟
    startup_ms = latency_us * (ep - 1) / 1000

    return transfer_ms + startup_ms


def calc_sp_allgather_latency(
    data_bytes: int | float,
    hardware: HardwareConfig,
    sp: int,
) -> float:
    """
    SP AllGather 通信延迟 (ms)

    AllGather: 每个节点收集所有节点的数据

    Args:
        data_bytes: 每个节点的数据量 (字节)
        hardware: 硬件配置
        sp: Sequence 并行度

    Returns:
        延迟 (ms)
    """
    if sp <= 1:
        return 0.0

    if data_bytes <= 0:
        return 0.0

    bandwidth_gbps = getattr(hardware.node, 'nvlink_bandwidth_gbps', 400)
    latency_us = getattr(hardware.node, 'nvlink_latency_us', 1.0)

    # AllGather 通信量
    transfer_factor = (sp - 1) / sp
    data_gb = data_bytes / 1e9
    transfer_ms = (data_gb * transfer_factor / bandwidth_gbps) * 1000

    # 启动延迟
    startup_ms = latency_us * (sp - 1) / 1000

    return transfer_ms + startup_ms


def calc_sp_reduce_scatter_latency(
    data_bytes: int | float,
    hardware: HardwareConfig,
    sp: int,
) -> float:
    """
    SP ReduceScatter 通信延迟 (ms)

    ReduceScatter: AllGather 的逆操作

    Args:
        data_bytes: 总数据量 (字节)
        hardware: 硬件配置
        sp: Sequence 并行度

    Returns:
        延迟 (ms)
    """
    # ReduceScatter 和 AllGather 通信量相同
    return calc_sp_allgather_latency(data_bytes, hardware, sp)


def calc_dp_allreduce_latency(
    data_bytes: int | float,
    hardware: HardwareConfig,
    dp: int,
    is_cross_node: bool = True,
) -> float:
    """
    DP AllReduce 通信延迟 (ms)

    数据并行的梯度同步

    Args:
        data_bytes: 梯度数据量 (字节)
        hardware: 硬件配置
        dp: 数据并行度
        is_cross_node: 是否跨节点

    Returns:
        延迟 (ms)
    """
    if dp <= 1:
        return 0.0

    if data_bytes <= 0:
        return 0.0

    if is_cross_node:
        bandwidth_gbps = getattr(hardware.cluster, 'ib_bandwidth_gbps', 100)
        latency_us = getattr(hardware.cluster, 'ib_latency_us', 5.0)
    else:
        bandwidth_gbps = getattr(hardware.node, 'nvlink_bandwidth_gbps', 400)
        latency_us = getattr(hardware.node, 'nvlink_latency_us', 1.0)

    # Ring AllReduce
    transfer_factor = 2 * (dp - 1) / dp
    data_gb = data_bytes / 1e9
    transfer_ms = (data_gb * transfer_factor / bandwidth_gbps) * 1000

    startup_ms = latency_us * (dp - 1) / 1000

    return transfer_ms + startup_ms


def calc_attention_allreduce_latency(
    model: LLMModelConfig,
    inference: InferenceConfig,
    parallelism: ParallelismStrategy,
    hardware: HardwareConfig,
    num_tokens: int,
) -> float:
    """
    Attention 后的 TP AllReduce 延迟 (ms)

    数据量: B × S × H
    """
    if parallelism.tp <= 1:
        return 0.0

    B, S, H = inference.batch_size, num_tokens, model.hidden_size
    bytes_per_elem = get_bytes_per_element(model.dtype)

    data_bytes = B * S * H * bytes_per_elem

    return calc_tp_allreduce_latency(data_bytes, hardware, parallelism.tp)


def calc_ffn_allreduce_latency(
    model: LLMModelConfig,
    inference: InferenceConfig,
    parallelism: ParallelismStrategy,
    hardware: HardwareConfig,
    num_tokens: int,
) -> float:
    """
    FFN 后的 TP AllReduce 延迟 (ms)

    数据量: B × S × H
    """
    if parallelism.tp <= 1:
        return 0.0

    B, S, H = inference.batch_size, num_tokens, model.hidden_size
    bytes_per_elem = get_bytes_per_element(model.dtype)

    data_bytes = B * S * H * bytes_per_elem

    return calc_tp_allreduce_latency(data_bytes, hardware, parallelism.tp)


def calc_sp_comm_volume_gb(
    model: LLMModelConfig,
    inference: InferenceConfig,
    parallelism: ParallelismStrategy,
    num_tokens: int,
) -> float:
    """计算 SP 通信数据量 (GB)"""
    B, S, H = inference.batch_size, num_tokens, model.hidden_size
    bytes_per_elem = get_bytes_per_element(model.dtype)

    data_bytes = B * S * H * bytes_per_elem
    return data_bytes / 1e9


def calc_ep_tp_combined_latency(
    model: LLMModelConfig,
    inference: InferenceConfig,
    parallelism: ParallelismStrategy,
    hardware: HardwareConfig,
    num_tokens: int,
) -> float:
    """EP+TP 组合通信延迟"""
    if model.moe_config is None:
        return 0.0

    ep = parallelism.ep
    tp = parallelism.tp

    if ep <= 1 and tp <= 1:
        return 0.0

    B, S, H = inference.batch_size, num_tokens, model.hidden_size
    bytes_per_elem = get_bytes_per_element(model.dtype)

    # EP All-to-All
    ep_data_bytes = B * S * H * bytes_per_elem
    ep_latency = calc_ep_alltoall_latency(ep_data_bytes, hardware, ep) if ep > 1 else 0.0

    # TP AllReduce
    tp_data_bytes = B * S * H * bytes_per_elem
    tp_latency = calc_tp_allreduce_latency(tp_data_bytes, hardware, tp) if tp > 1 else 0.0

    return ep_latency + tp_latency


def calc_dp_gradient_sync_latency(
    model: LLMModelConfig,
    inference: InferenceConfig,
    parallelism: ParallelismStrategy,
    hardware: HardwareConfig,
) -> float:
    """DP 梯度同步延迟"""
    dp = parallelism.dp

    if dp <= 1:
        return 0.0

    H = model.hidden_size
    I = model.intermediate_size
    L = model.num_layers

    params_per_layer = 4 * H * H + 3 * H * I

    if model.moe_config:
        num_experts = model.moe_config.num_experts
        expert_I = model.moe_config.expert_intermediate_size or I
        params_per_layer += num_experts * 3 * H * expert_I

    total_params = L * params_per_layer
    bytes_per_elem = get_bytes_per_element(model.dtype)
    gradient_bytes = total_params * bytes_per_elem

    return calc_dp_allreduce_latency(gradient_bytes, hardware, dp, is_cross_node=True)
