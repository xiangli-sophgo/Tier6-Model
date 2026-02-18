"""规格描述模块

定义 Per-Op/Layer 级别的规格数据结构：
- ComputeSpec: 单个 Op 的计算规格
- MemorySpec: 单个 Op 的内存规格
- CommSpec: 单个 Layer 的通信提示

注意：Model-Wide 级别的聚合结果定义在:
- breakdown.py: OpsBreakdown, MemoryFootprint
- comm_pattern.py: CommPattern
"""

from __future__ import annotations

from dataclasses import dataclass, field

from perf_model.L0_entry.types import DataType


@dataclass
class ComputeSpec:
    """计算规格

    Attributes:
        cube_ops: Cube 引擎操作数（MatMul/Conv）
        vector_ops: Vector 引擎操作数（Softmax/LayerNorm）
        scalar_ops: Scalar 引擎操作数
        hau_ops: HAU 引擎操作数（特殊函数）
        precision: 计算精度
    """

    cube_ops: int = 0
    vector_ops: int = 0
    scalar_ops: int = 0
    hau_ops: int = 0
    precision: DataType = DataType.FP16

    @property
    def total_ops(self) -> int:
        """总算力"""
        return self.cube_ops + self.vector_ops + self.scalar_ops + self.hau_ops


@dataclass
class MemorySpec:
    """内存规格

    Attributes:
        weight_bytes: 权重字节数
        activation_bytes: 激活字节数
        temp_bytes: 临时缓冲区字节数
        read_bytes: 读取字节数
        write_bytes: 写入字节数
    """

    weight_bytes: int = 0
    activation_bytes: int = 0
    temp_bytes: int = 0
    read_bytes: int = 0
    write_bytes: int = 0

    @property
    def total_bytes(self) -> int:
        """总内存（不含临时）"""
        return self.weight_bytes + self.activation_bytes

    @property
    def io_bytes(self) -> int:
        """总 IO 字节数"""
        return self.read_bytes + self.write_bytes


@dataclass
class CommSpec:
    """通信规格

    Attributes:
        pattern: 通信模式（allreduce/allgather/p2p/all2all）
        bytes: 通信数据量
        group_size: 通信组大小
        topology_hint: 拓扑提示（ring/tree/mesh）
        overlap_hint: 是否支持计算通信重叠
    """

    pattern: str
    bytes: int
    group_size: int | None = None
    topology_hint: str | None = None
    overlap_hint: bool = False


@dataclass
class PartitionHint:
    """分块提示（已弃用）

    说明：
        根据最新分层设计，L1 不再承担分块提示职责；该结构保留仅为兼容。
    """

    preferred_tile_m: int | None = None
    preferred_tile_n: int | None = None
    preferred_tile_k: int | None = None
    recompute_allowed: bool = True
    memory_bound_hint: bool = False
    overlap_hint: str | None = None


@dataclass
class TileConfig:
    """Tiling 配置

    Attributes:
        tile_m: M 维分块大小（batch * seq_len 方向）
        tile_n: N 维分块大小（输出特征方向）
        tile_k: K 维分块大小（reduction 方向）
        extra: 其他维度的分块配置
    """

    tile_m: int | None = None
    tile_n: int | None = None
    tile_k: int | None = None
    extra: dict[str, int] = field(default_factory=dict)

    def get(self, dim: str, default: int | None = None) -> int | None:
        """获取指定维度的 tile 大小"""
        if dim == "m":
            return self.tile_m
        elif dim == "n":
            return self.tile_n
        elif dim == "k":
            return self.tile_k
        return self.extra.get(dim, default)


@dataclass
class TiledMemoryInfo:
    """Tiling 后的内存信息（供 Layer Group 搜索使用）

    Attributes:
        input_buffer: 输入 buffer 大小（bytes）
        output_buffer: 输出 buffer 大小（bytes）
        weight_buffer: 权重 buffer 大小（bytes，如果需要切片）
        intermediate: 中间结果大小（bytes）
        peak_lmem: 峰值 LMEM 需求（bytes）
        num_tiles: tile 数量
        recompute_flops: 因 tiling 导致的重计算 FLOPs
    """

    input_buffer: int = 0
    output_buffer: int = 0
    weight_buffer: int = 0
    intermediate: int = 0
    peak_lmem: int = 0
    num_tiles: int = 1
    recompute_flops: int = 0

    @property
    def total_buffer(self) -> int:
        """总 buffer 大小"""
        return (
            self.input_buffer
            + self.output_buffer
            + self.weight_buffer
            + self.intermediate
        )

    def fits_in(self, lmem_size: int) -> bool:
        """检查是否能放入指定大小的 LMEM"""
        return self.peak_lmem <= lmem_size


@dataclass
class TileableDim:
    """可 tiling 的维度描述

    Attributes:
        name: 维度名称（m/n/k/seq/head 等）
        size: 维度原始大小
        min_tile: 最小 tile 大小
        alignment: 对齐要求
        is_reduction: 是否是 reduction 维度（影响重计算）
    """

    name: str
    size: int
    min_tile: int = 1
    alignment: int = 1
    is_reduction: bool = False

    def valid_tile_sizes(self, max_tile: int | None = None) -> list[int]:
        """返回有效的 tile 大小列表"""
        max_t = max_tile or self.size
        tiles = []
        t = self.min_tile
        while t <= min(self.size, max_t):
            if t % self.alignment == 0:
                tiles.append(t)
            t *= 2  # 通常按 2 的幂次增长
        # 确保包含完整维度
        if self.size not in tiles and self.size <= max_t:
            tiles.append(self.size)
        return sorted(tiles)
