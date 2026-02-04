"""评估指标数据结构.

定义 L4 Evaluation Engine 的核心数据结构：
    - Granularity: 评估精度层级枚举
    - HardwareSpec: 芯片级硬件参数
    - TopologySpec: chip 间通信拓扑参数
    - StepMetrics: Step 级时延分解
    - Aggregates: 聚合指标
    - EngineResult: 评估结果
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from tier6.L2_arch.topology import TopologySpec


class Granularity(Enum):
    """评估精度层级

    - CHIP: 芯片当黑盒，粗/快，适用于快速筛选 TP/PP/DP 组合
    - CORE: 考虑多核并行与片内路径键，中/中，MVP 主力
    - LANE: 考虑 SIMD/流水线，细/慢，微架构验证
    """

    CHIP = auto()
    CORE = auto()
    LANE = auto()


class BottleneckTag(Enum):
    """瓶颈类型标签"""

    COMPUTE_BOUND = auto()  # 计算瓶颈
    BW_BOUND = auto()  # 带宽瓶颈
    LATENCY_BOUND = auto()  # 延迟瓶颈
    UNKNOWN = auto()  # 未知


@dataclass
class HardwareSpec:
    """芯片级硬件参数

    Attributes:
        compute_tflops: 峰值算力（TFLOPS）
        memory_bandwidth_gbps: 显存带宽（GB/s）
        num_cores: Core 数量（Core 级需要）
        sram_per_core_kb: 每 Core SRAM 容量（KB，Core 级需要）
        noc_bandwidth_gbps: 片内 NoC 带宽（GB/s，Core 级需要）
    """

    compute_tflops: float = 125.0
    memory_bandwidth_gbps: float = 2000.0
    num_cores: int = 64
    sram_per_core_kb: float = 512.0
    noc_bandwidth_gbps: float = 1000.0

    def to_dict(self) -> dict[str, float]:
        """转换为字典"""
        return {
            "compute_tflops": self.compute_tflops,
            "memory_bandwidth_gbps": self.memory_bandwidth_gbps,
            "num_cores": float(self.num_cores),
            "sram_per_core_kb": self.sram_per_core_kb,
            "noc_bandwidth_gbps": self.noc_bandwidth_gbps,
        }


@dataclass
class CommProtocolSpec:
    """通信协议评估参数（L4 口径）"""

    rtt_tp_us: float = 0.35
    rtt_ep_us: float = 0.85
    sync_lat_us: float = 0.0
    bw_utilization: float = 0.95
    cpu_fetch_delay_us: float = 0.0
    moe_topk: float = 8.0
    prefill_topk_factor: float = 8 / 128

    def to_dict(self) -> dict[str, float]:
        """转换为字典"""
        return {
            "rtt_tp_us": self.rtt_tp_us,
            "rtt_ep_us": self.rtt_ep_us,
            "sync_lat_us": self.sync_lat_us,
            "bw_utilization": self.bw_utilization,
            "cpu_fetch_delay_us": self.cpu_fetch_delay_us,
            "moe_topk": self.moe_topk,
            "prefill_topk_factor": self.prefill_topk_factor,
        }


def merge_specs(
    hardware: HardwareSpec | None = None,
    topology: TopologySpec | None = None,
    comm_protocol: CommProtocolSpec | None = None,
) -> dict[str, float]:
    """合并 Hardware/Topology/CommProtocol 为统一参数字典

    Args:
        hardware: 芯片级硬件参数
        topology: chip 间通信拓扑参数
        comm_protocol: 通信协议评估参数

    Returns:
        合并后的参数字典
    """
    hw = hardware or HardwareSpec()
    topo = topology or TopologySpec()
    comm = comm_protocol or CommProtocolSpec()
    return {**hw.to_dict(), **topo.to_dict(), **comm.to_dict()}


@dataclass
class StepMetrics:
    """Step 级时延分解

    所有时间字段单位统一为 ms。

    Attributes:
        op_id: 对应的 op_id
        t_compute: 仅计算核执行时间，不含等待（ms）
        t_comm: 仅通信链路与协议开销，不含计算（ms）
        t_wait: 依赖/资源/队列导致的等待时间（ms）
        t_total: t_compute + t_comm + t_wait（ms）
        bottleneck_tag: 瓶颈类型标签
        flops: 浮点运算量（FLOPs）
        bytes_read: 读取字节数（bytes）
        bytes_write: 写入字节数（bytes）
        meta: 其他元数据
    """

    op_id: str
    t_compute: float = 0.0  # ms
    t_comm: float = 0.0  # ms
    t_wait: float = 0.0  # ms
    t_total: float = 0.0  # ms
    bottleneck_tag: BottleneckTag = BottleneckTag.UNKNOWN
    flops: int = 0
    bytes_read: int = 0
    bytes_write: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """计算 t_total"""
        if self.t_total == 0.0:
            self.t_total = self.t_compute + self.t_comm + self.t_wait


@dataclass
class Aggregates:
    """聚合指标

    Attributes:
        ttft: Time To First Token，首 Token 延迟 = Prefill 时间（ms）
        tpot: Time Per Output Token，平均每 Token 生成时间（ms）
        tps: Tokens Per Second，每秒生成 Token 数
        mfu: Model FLOPS Utilization，Achieved FLOPS / Peak FLOPS
        mbu: Memory Bandwidth Utilization，Required BW / Peak BW
        memory_peak: 内存峰值占用（bytes）
        total_time: 总执行时间（ms）
        total_compute_time: 总计算时间（ms）
        total_comm_time: 总通信时间（ms）
        total_wait_time: 总等待时间（ms）
        total_flops: 总浮点运算量（FLOPs）
        total_bytes: 总访存量（bytes）
        num_steps: Step 数量
        bottleneck_summary: 瓶颈类型统计
    """

    ttft: float = 0.0  # ms
    tpot: float = 0.0  # ms
    tps: float = 0.0  # tokens/s
    mfu: float = 0.0  # ratio
    mbu: float = 0.0  # ratio
    memory_peak: int = 0  # bytes
    total_time: float = 0.0  # ms
    total_compute_time: float = 0.0  # ms
    total_comm_time: float = 0.0  # ms
    total_wait_time: float = 0.0  # ms
    total_flops: int = 0
    total_bytes: int = 0
    num_steps: int = 0
    bottleneck_summary: dict[str, int] = field(default_factory=dict)


@dataclass
class EngineResult:
    """评估引擎结果

    Attributes:
        step_metrics: Step 级时延分解列表
        aggregates: 聚合指标
        granularity: 使用的评估精度
        trace_meta: 追踪与调试信息
    """

    step_metrics: list[StepMetrics] = field(default_factory=list)
    aggregates: Aggregates = field(default_factory=Aggregates)
    granularity: Granularity = Granularity.CHIP
    trace_meta: dict[str, Any] = field(default_factory=dict)

    def get_step(self, op_id: str) -> StepMetrics | None:
        """根据 op_id 获取 StepMetrics"""
        for step in self.step_metrics:
            if step.op_id == op_id:
                return step
        return None

    def summary(self) -> str:
        """生成摘要"""
        agg = self.aggregates
        return (
            f"EngineResult(\n"
            f"  granularity={self.granularity.name}\n"
            f"  num_steps={agg.num_steps}\n"
            f"  total_time={agg.total_time:.2f}ms\n"
            f"  TTFT={agg.ttft:.2f}ms, TPOT={agg.tpot:.2f}ms, TPS={agg.tps:.2f}\n"
            f"  MFU={agg.mfu:.2%}, MBU={agg.mbu:.2%}\n"
            f"  memory_peak={agg.memory_peak / (1024**2):.2f}MB\n"
            f")"
        )
