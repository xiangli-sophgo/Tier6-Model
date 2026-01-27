"""
加速器微架构配置

定义精确建模所需的硬件参数，包括：
- 计算单元参数 (Cube/Tensor Core 维度)
- 内存层次参数 (SRAM 大小、带宽)
- 对齐约束
- 计算-搬运重叠率
- 细粒度通信延迟参数
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CommunicationLatency:
    """细粒度通信延迟参数

    所有参数单位：微秒 (us, microseconds)
    1 微秒 = 1000 纳秒 (ns)

    参考来源：
    - DS_TPU 实测数据 (SG2261/SG2262)
    - NVIDIA 芯片架构推算
    - HBM/DDR 典型延迟值

    通信启动开销计算公式 (对齐 DS_TPU):
    start_lat = 2*c2c_lat + ddr_r_lat + ddr_w_lat + noc_lat + 2*d2d_lat
    """

    chip_to_chip_us: float = 0.2
    """芯片间物理互联延迟 (c2c_lat)

    单位：微秒 (us)

    这是纯粹的物理层互联延迟，不包含启动开销、同步等

    典型值：
    - NVLink 4.0 (H100): ~0.10 us (100 ns)
    - NVLink 3.0 (A100): ~0.15 us (150 ns)
    - SophgoLink (SG2262): ~0.15 us (150 ns)
    - PCIe 4.0/5.0: ~1.0-2.0 us
    """

    memory_read_latency_us: float = 0.15
    """显存读延迟 (ddr_r_lat)

    单位：微秒 (us)

    典型值：
    - HBM3 (H100): ~0.10 us (100 ns)
    - HBM2e (A100): ~0.12 us (120 ns)
    - HBM2 (SG2262): ~0.15 us (150 ns)
    - DDR4/DDR5: ~0.15-0.20 us
    """

    memory_write_latency_us: float = 0.01
    """显存写延迟 (ddr_w_lat)

    单位：微秒 (us)

    写操作通常比读操作快一个数量级（可以缓冲）

    典型值：
    - 各种 HBM/DDR: ~0.01 us (10 ns)
    """

    noc_latency_us: float = 0.05
    """片上网络延迟 (noc_lat, Network-on-Chip)

    单位：微秒 (us)

    芯片内部 NoC 互联的延迟

    典型值：
    - SG2262: ~0.05 us (50 ns)
    - H100: ~0.03-0.05 us
    """

    die_to_die_latency_us: float = 0.04
    """Die-to-Die 延迟 (d2d_lat)

    单位：微秒 (us)

    多 Die 芯片中 Die 间的互联延迟

    典型值：
    - SG2262 (多 Die): ~0.04 us (40 ns)
    - 单 Die 芯片: 0 us
    """

    @property
    def comm_start_overhead_us(self) -> float:
        """AllReduce 通信启动开销 (start_lat)

        动态计算公式:
        start_lat = 2*c2c_lat + ddr_r_lat + ddr_w_lat + noc_lat + 2*d2d_lat

        返回值单位：微秒 (us)
        """
        return (
            2 * self.chip_to_chip_us +
            self.memory_read_latency_us +
            self.memory_write_latency_us +
            self.noc_latency_us +
            2 * self.die_to_die_latency_us
        )

    def dispatch_combine_start_lat(self, switch_delay_us: float, cable_delay_us: float) -> float:
        """Dispatch/Combine 通信启动开销 (start_lat)

        动态计算公式:
        start_lat = 2*c2c_lat + ddr_r_lat + ddr_w_lat + noc_lat + 2*d2d_lat + 2*switch_delay + 2*cable_delay

        Args:
            switch_delay_us: 交换机延迟 (us)
            cable_delay_us: 线缆延迟 (us)

        返回值单位：微秒 (us)
        """
        return (
            2 * self.chip_to_chip_us +
            self.memory_read_latency_us +
            self.memory_write_latency_us +
            self.noc_latency_us +
            2 * self.die_to_die_latency_us +
            2 * switch_delay_us +
            2 * cable_delay_us
        )


@dataclass
class AcceleratorMicroArch:
    """加速器微架构配置"""

    # ========== 基本信息 ==========
    name: str = "unknown"
    """芯片名称"""

    flops_dtype: str = "BF16"
    """算力对应的数据精度 (BF16/FP16/FP8/INT8)"""

    # ========== 计算单元配置 ==========
    num_cores: int = 64
    """核心数量 (TPU cores / CUDA SMs)"""

    cube_m: int = 16
    """矩阵单元 M 维度"""

    cube_k: int = 32
    """矩阵单元 K 维度 (累加维度)"""

    cube_n: int = 8
    """矩阵单元 N 维度"""

    freq_ghz: float = 1.0
    """核心频率 (GHz)"""

    eu_num: int = 512
    """执行单元数量 (用于向量操作估算，如 Softmax)"""

    # ========== 内存配置 ==========
    sram_size_bytes: int = 2 * 1024 * 1024
    """每核 SRAM 大小 (字节)，默认 2MB"""

    sram_utilization: float = 0.45
    """SRAM 可用比例 (预留给系统/编译器)"""

    dram_bandwidth_bytes: float = 3200e9
    """DRAM 总带宽 (字节/秒)"""

    # ========== 对齐约束 ==========
    lane_num: int = 16
    """SIMD lane 数量 (行对齐基数)"""

    align_bytes: int = 32
    """内存对齐字节数 (列对齐基数)"""

    # ========== 执行模型 ==========
    compute_dma_overlap_rate: float = 0.8
    """计算与 DMA 搬运的重叠率 (0-1)"""

    # ========== 通信配置 ==========
    intra_bw: float = 504e9
    """组内通信带宽 (字节/秒)，默认 504 GB/s (NVLink)"""

    inter_bw: float = 100e9
    """组间通信带宽 (字节/秒)，默认 100 GB/s (跨节点)"""

    c2c_bw_unidirectional_gbps: float = 448.0
    """芯片间单向带宽 (GB/s)"""

    intra_latency_us: float = 1.0
    """组内通信延迟 (微秒)，默认 1.0 us (高速互联)

    注：这是粗粒度的端到端延迟，用于快速估算
    更精确的延迟建模请使用 comm_latency 参数
    """

    inter_latency_us: float = 2.0
    """组间通信延迟 (微秒)，默认 2.0 us (跨节点网络)

    注：这是粗粒度的端到端延迟，用于快速估算
    更精确的延迟建模请使用 comm_latency 参数
    """

    comm_latency: CommunicationLatency = field(default_factory=CommunicationLatency)
    """细粒度通信延迟参数（用于精确建模）

    包含芯片间互联、启动开销、显存读写等细粒度延迟
    用于通信算子的精确建模（AllReduce, AllGather 等）
    """

    # ========== 派生属性 ==========
    @property
    def macs_per_cycle(self) -> int:
        """每周期 MAC 操作数"""
        return self.cube_m * self.cube_k * self.cube_n

    @property
    def flops_per_second(self) -> float:
        """峰值 FLOPS"""
        return 2.0 * self.num_cores * self.macs_per_cycle * self.freq_ghz * 1e9

    @property
    def dma_bandwidth_per_core(self) -> float:
        """每核 DMA 带宽 (字节/秒)"""
        return self.dram_bandwidth_bytes / self.num_cores

    @property
    def effective_sram_bytes(self) -> int:
        """有效可用 SRAM 大小"""
        return int(self.sram_size_bytes * self.sram_utilization)

    def compute_freq_from_flops(self, total_flops: float) -> float:
        """从总 FLOPS 反推频率"""
        macs = self.macs_per_cycle
        if macs <= 0:
            return 1.0
        return total_flops / (2.0 * self.num_cores * macs * 1e9)

    def __post_init__(self):
        """初始化后自动计算频率（如果未设置）"""
        # 如果频率为默认值且有足够信息，可以从其他参数推导
        pass
