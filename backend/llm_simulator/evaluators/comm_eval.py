"""
集合通信评估器

提供精确的通信延迟估算，支持：
- AllReduce: 分层 (Hierarchical) + 扁平 (Flat) 模式
- AllGather: 分层 + 扁平模式
- ReduceScatter: 分层 + 扁平模式
- Dispatch: MoE 专用，EP 通信 + AllGather
- Combine: MoE 专用，EP 通信 + AllGather

分层通信适用于 TP=8/16/32，使用组内+组间两级通信。
"""

from dataclasses import dataclass
from typing import Tuple, Union, Optional

from ..types import ProtocolConfig, NetworkInfraConfig


@dataclass
class CommResult:
    """通信评估结果"""

    latency_us: float
    """延迟 (微秒)"""

    comm_bytes: float
    """实际通信数据量 (字节)"""


# ==================== 通信参数配置说明 ====================
#
# 通信延迟参数来源：
#
# 1. 芯片特定延迟 (从 AcceleratorMicroArch.comm_latency 获取):
#    - chip_to_chip_us: 芯片间物理互联延迟
#    - comm_start_overhead_us: 通信启动开销
#    - memory_read_latency_us: 显存读延迟
#    - memory_write_latency_us: 显存写延迟
#
# 2. 协议参数 (从 ProtocolConfig 获取，可由用户配置):
#    - rtt_tp_us: TP 通信 RTT
#    - rtt_ep_us: EP 通信 RTT
#    - bandwidth_utilization: 带宽利用率
#    - sync_latency_us: 同步延迟
#    (默认值定义在 types.py 的 ProtocolConfig dataclass 中)
#
# 3. 网络基础设施参数 (从 NetworkInfraConfig 获取，可由用户配置):
#    - switch_delay_us: 交换机延迟
#    - cable_delay_us: 线缆延迟
#    (默认值定义在 types.py 的 NetworkInfraConfig dataclass 中)
#
# 4. MoE 参数 (从 model.moe_config 获取):
#    - num_experts_per_tok (topk)

# MoE 默认参数 (应从 moe_config 获取)
DEFAULT_MOE_TOPK = 8            # MoE topk
DEFAULT_PREFILL_FACTOR = 8 / 128  # Prefill 系数


# ==================== AllReduce 评估器 ====================

class AllReduceEval:
    """
    AllReduce 通信评估器

    支持两种模式：
    1. 分层模式 (TP=8/16/32): 组内 ReduceScatter + 组间 AllReduce + 组内 AllGather
    2. 扁平模式 (其他 TP): Ring AllReduce
    """

    def __init__(
        self,
        arch,
        protocol_config: Optional["ProtocolConfig"] = None,
        network_config: Optional["NetworkInfraConfig"] = None,
    ):
        """
        初始化评估器

        Args:
            arch: 架构配置，需要 intra_bw、inter_bw 和 comm_latency 属性
            protocol_config: 协议配置 (可选，使用默认值如果未提供)
            network_config: 网络基础设施配置 (可选，使用默认值如果未提供)
        """
        self.arch = arch
        # 芯片特定的延迟参数 (从 arch.comm_latency 获取)
        self.c2c_lat = arch.comm_latency.chip_to_chip_us
        self.ddr_r_lat = arch.comm_latency.memory_read_latency_us
        self.ddr_w_lat = arch.comm_latency.memory_write_latency_us
        self.start_lat = arch.comm_latency.comm_start_overhead_us

        # 协议参数 (从 protocol_config 获取，或使用 dataclass 默认值)
        proto = protocol_config if protocol_config is not None else ProtocolConfig()
        self.rtt_tp = proto.rtt_tp_us
        self.sync_lat = proto.sync_latency_us
        self.bw_urate = proto.bandwidth_utilization

        self.link_delay = 0  # AllReduce 不使用 link_delay

    def evaluate_raw(
        self,
        tp: int,
        data_bytes: Union[int, float],
        comm_protocol: int = 1,
    ) -> Tuple[float, float]:
        """
        评估 AllReduce 延迟

        Args:
            tp: 张量并行度
            data_bytes: 数据量 (字节)
            comm_protocol: 通信协议 (1/2/3)

        Returns:
            (latency_us, comm_bytes): 延迟 (微秒), 通信量 (字节)
        """
        if tp <= 1:
            return 0.0, 0.0

        # 判断是否使用分层通信
        use_hierarchical = tp in [8, 16, 32]

        if use_hierarchical:
            return self._evaluate_hierarchical(tp, data_bytes)
        else:
            return self._evaluate_flat(tp, data_bytes, comm_protocol)

    def _evaluate_hierarchical(
        self,
        tp: int,
        data_bytes: Union[int, float],
    ) -> Tuple[float, float]:
        """分层 AllReduce"""
        # 组大小: TP=16 用 2，其他用 4
        group_size = 2 if tp == 16 else 4
        num_groups = tp // group_size

        # 阶段1: 组内 ReduceScatter
        comm_size_1 = 2 * (group_size - 1) / group_size * data_bytes
        lat_1 = (comm_size_1 / self.arch.intra_bw / self.bw_urate) * 1e6
        lat_1 += (group_size - 1) * (self.start_lat + self.sync_lat)

        # 阶段2: 组间 AllReduce
        comm_size_2 = 2 * (num_groups - 1) / num_groups * data_bytes
        lat_2 = (comm_size_2 / self.arch.inter_bw / self.bw_urate) * 1e6
        lat_2 += (num_groups - 1) * (self.start_lat + self.sync_lat + self.link_delay)

        # 阶段3: 组内 AllGather (Broadcast)
        comm_size_3 = data_bytes
        lat_3 = (comm_size_3 / self.arch.intra_bw / self.bw_urate) * 1e6
        lat_3 += (group_size - 1) * (self.start_lat + self.sync_lat)

        # 延迟取最大值 (流水线并行)
        latency_us = max(lat_1, lat_2, lat_3)

        # 总通信量
        total_comm = comm_size_1 * num_groups + comm_size_2 + comm_size_3 * num_groups

        return latency_us, total_comm

    def _evaluate_flat(
        self,
        tp: int,
        data_bytes: Union[int, float],
        comm_protocol: int,
    ) -> Tuple[float, float]:
        """扁平 Ring AllReduce"""
        # Ring AllReduce 通信量: 2 * (n-1)/n * data
        comm_size = 2 * (tp - 1) / tp * data_bytes

        # 传输延迟
        lat = (comm_size / self.arch.intra_bw / self.bw_urate) * 1e6

        # 启动延迟
        lat += (tp - 1) * (self.start_lat + self.sync_lat)

        # 协议相关的 RTT 开销
        if comm_protocol == 2:
            lat += self.rtt_tp * 2 * (tp - 1)
        elif comm_protocol == 3:
            lat += self.rtt_tp * min(1, 2 * (tp - 1))

        return lat, comm_size

    def evaluate(self, tp: int, data_bytes: Union[int, float], comm_protocol: int = 1) -> CommResult:
        """评估并返回 CommResult"""
        lat, comm = self.evaluate_raw(tp, data_bytes, comm_protocol)
        return CommResult(latency_us=lat, comm_bytes=comm)


# ==================== AllGather 评估器 ====================

class AllGatherEval:
    """
    AllGather 通信评估器

    支持分层和扁平两种模式。
    """

    def __init__(
        self,
        arch,
        protocol_config: Optional["ProtocolConfig"] = None,
        network_config: Optional["NetworkInfraConfig"] = None,
    ):
        """初始化评估器

        Args:
            arch: AcceleratorMicroArch 对象，包含芯片特定的延迟参数
            protocol_config: 协议配置 (可选)
            network_config: 网络基础设施配置 (可选)
        """
        self.arch = arch

        # 芯片特定的延迟参数 (从 arch.comm_latency 获取)
        self.c2c_lat = arch.comm_latency.chip_to_chip_us
        self.ddr_r_lat = arch.comm_latency.memory_read_latency_us
        self.ddr_w_lat = arch.comm_latency.memory_write_latency_us
        self.start_lat = arch.comm_latency.comm_start_overhead_us

        # 协议参数 (从 protocol_config 获取，或使用 dataclass 默认值)
        proto = protocol_config if protocol_config is not None else ProtocolConfig()
        self.rtt_tp = proto.rtt_tp_us
        self.sync_lat = proto.sync_latency_us
        self.bw_urate = proto.bandwidth_utilization

        self.link_delay = 0

    def evaluate_raw(
        self,
        tp: int,
        data_bytes: Union[int, float],
        comm_protocol: int = 1,
    ) -> Tuple[float, float]:
        """
        评估 AllGather 延迟

        Args:
            tp: 并行度
            data_bytes: 每个节点的数据量 (字节)
            comm_protocol: 通信协议

        Returns:
            (latency_us, comm_bytes)
        """
        if tp <= 1:
            return 0.0, 0.0

        use_hierarchical = tp in [8, 16, 32]

        if use_hierarchical:
            return self._evaluate_hierarchical(tp, data_bytes)
        else:
            return self._evaluate_flat(tp, data_bytes, comm_protocol)

    def _evaluate_hierarchical(
        self,
        tp: int,
        data_bytes: Union[int, float],
    ) -> Tuple[float, float]:
        """分层 AllGather"""
        # 注意: AllGather 所有 tp=8/16/32 都使用 group_size=4
        # 与 AllReduce/ReduceScatter 不同 (它们 tp=16 用 group_size=2)
        group_size = 4
        num_groups = tp // group_size

        # 阶段1: 组内 AllGather
        comm_size_1 = (group_size - 1) * data_bytes
        lat_1 = (comm_size_1 / self.arch.intra_bw / self.bw_urate) * 1e6
        lat_1 += (group_size - 1) * (self.start_lat + self.sync_lat)

        # 阶段2: 组间 AllGather
        comm_size_2 = (num_groups - 1) * data_bytes
        lat_2 = (comm_size_2 / self.arch.inter_bw / self.bw_urate) * 1e6
        lat_2 += (num_groups - 1) * (self.start_lat + self.sync_lat + self.link_delay)

        # 阶段3: 无
        comm_size_3 = 0
        lat_3 = 0

        latency_us = max(lat_1, lat_2, lat_3)
        total_comm = comm_size_1 * num_groups + comm_size_2 + comm_size_3 * num_groups

        return latency_us, total_comm

    def _evaluate_flat(
        self,
        tp: int,
        data_bytes: Union[int, float],
        comm_protocol: int,
    ) -> Tuple[float, float]:
        """扁平 AllGather"""
        # AllGather 通信量: (n-1) * data
        comm_size = (tp - 1) * data_bytes

        lat = (comm_size / self.arch.intra_bw / self.bw_urate) * 1e6
        lat += (tp - 1) * (self.start_lat + self.sync_lat)

        if comm_protocol == 2:
            lat += self.rtt_tp * 2 * (tp - 1)
        elif comm_protocol == 3:
            lat += self.rtt_tp * min(1, 2 * (tp - 1))

        return lat, comm_size

    def evaluate(self, tp: int, data_bytes: Union[int, float], comm_protocol: int = 1) -> CommResult:
        lat, comm = self.evaluate_raw(tp, data_bytes, comm_protocol)
        return CommResult(latency_us=lat, comm_bytes=comm)


# ==================== ReduceScatter 评估器 ====================

class ReduceScatterEval:
    """
    ReduceScatter 通信评估器

    支持分层和扁平两种模式。
    """

    def __init__(
        self,
        arch,
        protocol_config: Optional["ProtocolConfig"] = None,
        network_config: Optional["NetworkInfraConfig"] = None,
    ):
        """初始化评估器

        Args:
            arch: AcceleratorMicroArch 对象，包含芯片特定的延迟参数
            protocol_config: 协议配置 (可选)
            network_config: 网络基础设施配置 (可选)
        """
        self.arch = arch

        # 芯片特定的延迟参数 (从 arch.comm_latency 获取)
        self.c2c_lat = arch.comm_latency.chip_to_chip_us
        self.ddr_r_lat = arch.comm_latency.memory_read_latency_us
        self.ddr_w_lat = arch.comm_latency.memory_write_latency_us
        self.start_lat = arch.comm_latency.comm_start_overhead_us

        # 协议参数 (从 protocol_config 获取，或使用 dataclass 默认值)
        proto = protocol_config if protocol_config is not None else ProtocolConfig()
        self.rtt_tp = proto.rtt_tp_us
        self.sync_lat = proto.sync_latency_us
        self.bw_urate = proto.bandwidth_utilization

        self.link_delay = 0

    def evaluate_raw(
        self,
        tp: int,
        data_bytes: Union[int, float],
        comm_protocol: int = 1,
    ) -> Tuple[float, float]:
        """
        评估 ReduceScatter 延迟

        Args:
            tp: 并行度
            data_bytes: 总数据量 (字节)
            comm_protocol: 通信协议

        Returns:
            (latency_us, comm_bytes)
        """
        if tp <= 1:
            return 0.0, 0.0

        use_hierarchical = tp in [8, 16, 32]

        if use_hierarchical:
            return self._evaluate_hierarchical(tp, data_bytes)
        else:
            return self._evaluate_flat(tp, data_bytes, comm_protocol)

    def _evaluate_hierarchical(
        self,
        tp: int,
        data_bytes: Union[int, float],
    ) -> Tuple[float, float]:
        """分层 ReduceScatter"""
        group_size = 2 if tp == 16 else 4
        num_groups = tp // group_size

        # 阶段1: 组内 ReduceScatter
        comm_size_1 = (group_size - 1) / group_size * data_bytes
        lat_1 = (comm_size_1 / self.arch.intra_bw / self.bw_urate) * 1e6
        lat_1 += (group_size - 1) * (self.start_lat + self.sync_lat)

        # 阶段2: 组间 ReduceScatter
        # 注意: DS_TPU 使用 intra_bw (与 AllReduce 不同)
        comm_size_2 = (num_groups - 1) / num_groups * data_bytes
        lat_2 = (comm_size_2 / self.arch.intra_bw / self.bw_urate) * 1e6
        lat_2 += (num_groups - 1) * (self.start_lat + self.sync_lat + self.link_delay)

        # 阶段3: 无
        comm_size_3 = 0
        lat_3 = 0

        latency_us = max(lat_1, lat_2, lat_3)
        total_comm = comm_size_1 * num_groups + comm_size_2 + comm_size_3 * num_groups

        return latency_us, total_comm

    def _evaluate_flat(
        self,
        tp: int,
        data_bytes: Union[int, float],
        comm_protocol: int,
    ) -> Tuple[float, float]:
        """扁平 ReduceScatter"""
        # ReduceScatter 通信量: (n-1)/n * data
        comm_size = (tp - 1) / tp * data_bytes

        lat = (comm_size / self.arch.intra_bw / self.bw_urate) * 1e6
        lat += (tp - 1) * (self.start_lat + self.sync_lat)

        if comm_protocol == 2:
            lat += self.rtt_tp * 2 * (tp - 1)
        elif comm_protocol == 3:
            lat += self.rtt_tp * min(1, 2 * (tp - 1))

        return lat, comm_size

    def evaluate(self, tp: int, data_bytes: Union[int, float], comm_protocol: int = 1) -> CommResult:
        lat, comm = self.evaluate_raw(tp, data_bytes, comm_protocol)
        return CommResult(latency_us=lat, comm_bytes=comm)


# ==================== Dispatch 评估器 (MoE 专用) ====================

class DispatchEval:
    """
    Dispatch 通信评估器 (MoE 专用)

    Dispatch = EP 跨节点通信 + MoE_TP AllGather
    用于将 token 分发到各个 expert 所在的芯片
    """

    def __init__(
        self,
        arch,
        protocol_config: Optional["ProtocolConfig"] = None,
        network_config: Optional["NetworkInfraConfig"] = None,
        moe_topk: int = DEFAULT_MOE_TOPK,
        prefill_factor: float = DEFAULT_PREFILL_FACTOR,
    ):
        """初始化评估器

        Args:
            arch: AcceleratorMicroArch 对象，包含芯片特定的延迟参数
            protocol_config: 协议配置 (可选)
            network_config: 网络基础设施配置 (可选)
            moe_topk: MoE top-k 值 (通常从 moe_config.num_experts_per_tok 获取)
            prefill_factor: Prefill 系数
        """
        self.arch = arch

        # 芯片特定的延迟参数 (从 arch.comm_latency 获取)
        self.c2c_lat = arch.comm_latency.chip_to_chip_us
        self.ddr_r_lat = arch.comm_latency.memory_read_latency_us
        self.ddr_w_lat = arch.comm_latency.memory_write_latency_us
        self.noc_lat = arch.comm_latency.noc_latency_us
        self.d2d_lat = arch.comm_latency.die_to_die_latency_us

        # 网络基础设施参数 (从 network_config 获取，或使用 dataclass 默认值)
        net = network_config if network_config is not None else NetworkInfraConfig()
        self.switch_delay = net.switch_delay_us
        self.cable_delay = net.cable_delay_us

        # Dispatch/Combine 专用 start_lat (包含 switch_delay 和 cable_delay)
        # start_lat = 2*c2c_lat + ddr_r_lat + ddr_w_lat + noc_lat + 2*d2d_lat + 2*switch_delay + 2*cable_delay
        self.start_lat = arch.comm_latency.dispatch_combine_start_lat(
            self.switch_delay, self.cable_delay
        )

        # 协议参数 (从 protocol_config 获取，或使用 dataclass 默认值)
        proto = protocol_config if protocol_config is not None else ProtocolConfig()
        self.rtt_ep = proto.rtt_ep_us
        self.bw_urate = proto.bandwidth_utilization

        # MoE 参数
        self.topk = moe_topk
        self.prefill_factor = prefill_factor

    def evaluate_raw(
        self,
        moe_tp: int,
        ep: int,
        data_bytes: Union[int, float],
        batch_size: int,
        comm_protocol: int = 1,
        is_prefill: bool = False,
    ) -> Tuple[float, float]:
        """
        评估 Dispatch 延迟

        公式:
        T = (data_size / inter_bw / bw_urate) + start_lat  # dispatch to each EP group
        T += ((moe_tp - 1) * data_size / intra_bw / bw_urate) * 1e6 + (moe_tp - 1) * start_lat  # all-gather in EP group

        Args:
            moe_tp: MoE 张量并行度
            ep: Expert 并行度
            data_bytes: 数据量 (字节)
            batch_size: 批次大小
            comm_protocol: 通信协议
            is_prefill: 是否为 prefill 阶段

        Returns:
            (latency_us, comm_bytes)
        """
        # Step 1: EP 通信 (dispatch to each EP group)
        # T = (data_size / inter_bw / bw_urate) * 1e6 + start_lat
        t_us = (data_bytes / self.arch.inter_bw / self.bw_urate) * 1e6 + self.start_lat

        # Step 2: AllGather in EP group (moe_tp 内的通信)
        # T += ((moe_tp - 1) * data_size / intra_bw / bw_urate) * 1e6 + (moe_tp - 1) * start_lat
        if moe_tp > 1:
            allgather_comm = (moe_tp - 1) * data_bytes
            t_us += (allgather_comm / self.arch.intra_bw / self.bw_urate) * 1e6
            t_us += (moe_tp - 1) * self.start_lat

        # 总通信量: EP 通信 + AllGather 通信
        total_comm = data_bytes + (moe_tp - 1) * data_bytes if moe_tp > 1 else data_bytes

        return t_us, total_comm

    def evaluate(
        self,
        moe_tp: int,
        ep: int,
        data_bytes: Union[int, float],
        batch_size: int,
        comm_protocol: int = 1,
        is_prefill: bool = False,
    ) -> CommResult:
        lat, comm = self.evaluate_raw(moe_tp, ep, data_bytes, batch_size, comm_protocol, is_prefill)
        return CommResult(latency_us=lat, comm_bytes=comm)


# ==================== Combine 评估器 (MoE 专用) ====================

class CombineEval:
    """
    Combine 通信评估器 (MoE 专用)

    Combine = EP 跨节点通信 + MoE_TP AllGather
    用于将 expert 输出收集回原始芯片
    """

    def __init__(
        self,
        arch,
        protocol_config: Optional["ProtocolConfig"] = None,
        network_config: Optional["NetworkInfraConfig"] = None,
        moe_topk: int = DEFAULT_MOE_TOPK,
        prefill_factor: float = DEFAULT_PREFILL_FACTOR,
    ):
        """初始化评估器

        Args:
            arch: AcceleratorMicroArch 对象，包含芯片特定的延迟参数
            protocol_config: 协议配置 (可选)
            network_config: 网络基础设施配置 (可选)
            moe_topk: MoE top-k 值 (通常从 moe_config.num_experts_per_tok 获取)
            prefill_factor: Prefill 系数
        """
        self.arch = arch

        # 芯片特定的延迟参数 (从 arch.comm_latency 获取)
        self.c2c_lat = arch.comm_latency.chip_to_chip_us
        self.ddr_r_lat = arch.comm_latency.memory_read_latency_us
        self.ddr_w_lat = arch.comm_latency.memory_write_latency_us
        self.noc_lat = arch.comm_latency.noc_latency_us
        self.d2d_lat = arch.comm_latency.die_to_die_latency_us

        # 网络基础设施参数 (从 network_config 获取，或使用 dataclass 默认值)
        net = network_config if network_config is not None else NetworkInfraConfig()
        self.switch_delay = net.switch_delay_us
        self.cable_delay = net.cable_delay_us

        # Combine 专用 start_lat (包含 switch_delay 和 cable_delay)
        # start_lat = 2*c2c_lat + ddr_r_lat + ddr_w_lat + noc_lat + 2*d2d_lat + 2*switch_delay + 2*cable_delay
        self.start_lat = arch.comm_latency.dispatch_combine_start_lat(
            self.switch_delay, self.cable_delay
        )

        # 协议参数 (从 protocol_config 获取，或使用 dataclass 默认值)
        proto = protocol_config if protocol_config is not None else ProtocolConfig()
        self.rtt_ep = proto.rtt_ep_us
        self.bw_urate = proto.bandwidth_utilization

        # MoE 参数
        self.topk = moe_topk
        self.prefill_factor = prefill_factor

    def evaluate_raw(
        self,
        moe_tp: int,
        ep: int,
        data_bytes: Union[int, float],
        batch_size: int,
        comm_protocol: int = 1,
        is_prefill: bool = False,
    ) -> Tuple[float, float]:
        """
        评估 Combine 延迟

        公式:
        T = (data_size / inter_bw / bw_urate) + start_lat  # combine from each EP group
        T += ((moe_tp - 1) * data_size / intra_bw / bw_urate) * 1e6 + (moe_tp - 1) * start_lat  # all-gather in EP group

        Args:
            moe_tp: MoE 张量并行度
            ep: Expert 并行度
            data_bytes: 数据量 (字节)
            batch_size: 批次大小
            comm_protocol: 通信协议
            is_prefill: 是否为 prefill 阶段

        Returns:
            (latency_us, comm_bytes)
        """
        # Step 1: EP 通信 (combine from each EP group)
        # T = (data_size / inter_bw / bw_urate) * 1e6 + start_lat
        t_us = (data_bytes / self.arch.inter_bw / self.bw_urate) * 1e6 + self.start_lat

        # Step 2: AllGather in EP group (moe_tp 内的通信)
        # T += ((moe_tp - 1) * data_size / intra_bw / bw_urate) * 1e6 + (moe_tp - 1) * start_lat
        if moe_tp > 1:
            allgather_comm = (moe_tp - 1) * data_bytes
            t_us += (allgather_comm / self.arch.intra_bw / self.bw_urate) * 1e6
            t_us += (moe_tp - 1) * self.start_lat

        # 总通信量: EP 通信 + AllGather 通信
        total_comm = data_bytes + (moe_tp - 1) * data_bytes if moe_tp > 1 else data_bytes

        return t_us, total_comm

    def evaluate(
        self,
        moe_tp: int,
        ep: int,
        data_bytes: Union[int, float],
        batch_size: int,
        comm_protocol: int = 1,
        is_prefill: bool = False,
    ) -> CommResult:
        lat, comm = self.evaluate_raw(moe_tp, ep, data_bytes, batch_size, comm_protocol, is_prefill)
        return CommResult(latency_us=lat, comm_bytes=comm)


# ==================== 模块级别快捷函数 ====================

_allreduce_eval: Optional[AllReduceEval] = None
_allgather_eval: Optional[AllGatherEval] = None
_reducescatter_eval: Optional[ReduceScatterEval] = None
_dispatch_eval: Optional[DispatchEval] = None
_combine_eval: Optional[CombineEval] = None


def init_comm_evaluators(
    arch,
    protocol_config: Optional["ProtocolConfig"] = None,
    network_config: Optional["NetworkInfraConfig"] = None,
    moe_topk: int = DEFAULT_MOE_TOPK,
    prefill_factor: float = DEFAULT_PREFILL_FACTOR,
) -> None:
    """初始化所有通信评估器

    Args:
        arch: 架构配置
        protocol_config: 协议配置 (可选)
        network_config: 网络基础设施配置 (可选)
        moe_topk: MoE top-k 值 (用于 Dispatch/Combine)
        prefill_factor: Prefill 系数 (用于 Dispatch/Combine)
    """
    global _allreduce_eval, _allgather_eval, _reducescatter_eval, _dispatch_eval, _combine_eval
    _allreduce_eval = AllReduceEval(arch, protocol_config, network_config)
    _allgather_eval = AllGatherEval(arch, protocol_config, network_config)
    _reducescatter_eval = ReduceScatterEval(arch, protocol_config, network_config)
    _dispatch_eval = DispatchEval(arch, protocol_config, network_config, moe_topk, prefill_factor)
    _combine_eval = CombineEval(arch, protocol_config, network_config, moe_topk, prefill_factor)


def get_allreduce_eval() -> Optional[AllReduceEval]:
    """获取 AllReduce 评估器"""
    return _allreduce_eval


def get_allgather_eval() -> Optional[AllGatherEval]:
    """获取 AllGather 评估器"""
    return _allgather_eval


def get_reducescatter_eval() -> Optional[ReduceScatterEval]:
    """获取 ReduceScatter 评估器"""
    return _reducescatter_eval


def get_dispatch_eval() -> Optional[DispatchEval]:
    """获取 Dispatch 评估器"""
    return _dispatch_eval


def get_combine_eval() -> Optional[CombineEval]:
    """获取 Combine 评估器"""
    return _combine_eval


def eval_allreduce(
    tp: int,
    data_bytes: Union[int, float],
    intra_bw: float,
    inter_bw: float,
    comm_protocol: int = 1,
) -> Tuple[float, float]:
    """
    快捷函数: 评估 AllReduce

    Args:
        tp: 张量并行度
        data_bytes: 数据量 (字节)
        intra_bw: 组内带宽 (bytes/s)
        inter_bw: 组间带宽 (bytes/s)
        comm_protocol: 通信协议

    Returns:
        (latency_us, comm_bytes)
    """
    from .arch_config import AcceleratorMicroArch, CommunicationLatency

    arch = AcceleratorMicroArch(
        name="TempArch",
        intra_bw=intra_bw,
        inter_bw=inter_bw,
        comm_latency=CommunicationLatency(),
    )

    evaluator = AllReduceEval(arch)
    return evaluator.evaluate_raw(tp, data_bytes, comm_protocol)


def eval_allgather(
    tp: int,
    data_bytes: Union[int, float],
    intra_bw: float,
    inter_bw: float,
    comm_protocol: int = 1,
) -> Tuple[float, float]:
    """快捷函数: 评估 AllGather"""
    from .arch_config import AcceleratorMicroArch, CommunicationLatency

    arch = AcceleratorMicroArch(
        name="TempArch",
        intra_bw=intra_bw,
        inter_bw=inter_bw,
        comm_latency=CommunicationLatency(),
    )

    evaluator = AllGatherEval(arch)
    return evaluator.evaluate_raw(tp, data_bytes, comm_protocol)


def eval_reducescatter(
    tp: int,
    data_bytes: Union[int, float],
    intra_bw: float,
    inter_bw: float,
    comm_protocol: int = 1,
) -> Tuple[float, float]:
    """快捷函数: 评估 ReduceScatter"""
    from .arch_config import AcceleratorMicroArch, CommunicationLatency

    arch = AcceleratorMicroArch(
        name="TempArch",
        intra_bw=intra_bw,
        inter_bw=inter_bw,
        comm_latency=CommunicationLatency(),
    )

    evaluator = ReduceScatterEval(arch)
    return evaluator.evaluate_raw(tp, data_bytes, comm_protocol)
