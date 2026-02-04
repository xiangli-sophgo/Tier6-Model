"""分析结果模块

定义 OpsBreakdown, MemoryFootprint 数据结构。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class OpsBreakdown:
    """操作数分解

    支持「理论值 vs 实际值」两套统计。

    Attributes:
        total_ops: 总操作数
        cube_ops: Cube 引擎操作数
        vector_ops: Vector 引擎操作数
        scalar_ops: Scalar 引擎操作数
        hau_ops: HAU 引擎操作数
        by_layer: 按层统计的操作数
        actual_*: 分块后的实际值（可选）
        recompute_ops: 重计算开销
        retraffic_bytes: 重搬运开销
    """

    # 理论值（无分块）
    total_ops: int = 0
    cube_ops: int = 0
    vector_ops: int = 0
    scalar_ops: int = 0
    hau_ops: int = 0
    by_layer: dict[str, int] = field(default_factory=dict)

    # 实际值（分块后，可选，L3 填充）
    actual_total_ops: int | None = None
    actual_cube_ops: int | None = None
    actual_vector_ops: int | None = None
    actual_scalar_ops: int | None = None
    actual_hau_ops: int | None = None
    recompute_ops: int | None = None
    retraffic_bytes: int | None = None

    @property
    def recompute_ratio(self) -> float | None:
        """重计算比例"""
        if self.recompute_ops is None or self.total_ops == 0:
            return None
        return self.recompute_ops / self.total_ops

    @property
    def total_flops(self) -> float:
        """总 FLOPs"""
        return float(self.total_ops)

    @property
    def cube_flops(self) -> float:
        """Cube 单元 FLOPs"""
        return float(self.cube_ops)

    @property
    def vector_flops(self) -> float:
        """Vector 单元 FLOPs"""
        return float(self.vector_ops)


@dataclass
class MemoryFootprint:
    """内存占用

    Attributes:
        weights: 权重字节数
        activations: 激活字节数
        kv_cache_bytes: KV Cache 字节数
        temporaries: 临时缓冲区字节数
        peak_lmem: LMEM 峰值（可选）
        peak_l2m: L2M 峰值（可选）
        actual_*: 分块后的实际值（可选）
        spill_to_gmem: 溢出到 GMEM 的字节数
    """

    # 理论值
    weights: int = 0
    activations: int = 0
    kv_cache_bytes: int = 0
    temporaries: int = 0
    peak_lmem: int | None = None
    peak_l2m: int | None = None

    # 实际值（分块后，可选，L3 填充）
    actual_peak_lmem: int | None = None
    actual_peak_l2m: int | None = None
    spill_to_gmem: int | None = None

    @property
    def total_bytes(self) -> int:
        """总内存（权重 + 激活 + KV Cache）"""
        return self.weights + self.activations + self.kv_cache_bytes

    @property
    def weight_bytes(self) -> int:
        """权重大小（字节）"""
        return self.weights

    @property
    def activation_bytes(self) -> int:
        """激活大小（字节）"""
        return self.activations

    @property
    def has_spill(self) -> bool:
        """是否有溢出"""
        return self.spill_to_gmem is not None and self.spill_to_gmem > 0
