"""
加速器微架构配置

定义精确建模所需的硬件参数，包括：
- 计算单元参数 (Cube/Tensor Core 维度)
- 内存层次参数 (SRAM 大小、带宽)
- 对齐约束
- 计算-搬运重叠率
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AcceleratorMicroArch:
    """加速器微架构配置"""

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
