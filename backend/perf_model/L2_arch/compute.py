"""计算单元实现

实现 Cube/Vector/HAU 等计算单元规格。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Union


@dataclass
class CubeSpec:
    """Cube 计算单元规格

    用于矩阵乘法等密集计算。

    Attributes:
        name: 单元名称
        dim_m: 矩阵单元 M 维度
        dim_k: 矩阵单元 K 维度 (累加维度)
        dim_n: 矩阵单元 N 维度
        mac_per_lane: 每 lane MAC 数 (按数据类型)
        frequency_ghz: 工作频率
        lane_count: lane 数量
        core_count: 核心数量
    """

    name: str = "Cube"
    dim_m: int = 16
    dim_k: int = 32
    dim_n: int = 8
    mac_per_lane: dict[str, int] = field(default_factory=dict)
    frequency_ghz: float = 1.0
    lane_count: int = 64
    core_count: int = 1

    def peak_flops(self, dtype: str) -> float:
        """计算峰值算力

        FLOPs = mac_per_lane * lane_count * core_count * frequency * 2

        Args:
            dtype: 数据类型 (BF16/FP16/INT8 等)

        Returns:
            峰值算力 (FLOPS)
        """
        mac = self.mac_per_lane.get(dtype.upper(), 0)
        # MAC = Multiply-Accumulate, 计为 2 FLOPs
        return mac * self.lane_count * self.core_count * self.frequency_ghz * 1e9 * 2

    def eu_count(self, dtype: str) -> int:
        """获取执行单元数量

        Args:
            dtype: 数据类型

        Returns:
            EU 数量
        """
        return (
            self.mac_per_lane.get(dtype.upper(), 0) * self.lane_count * self.core_count
        )

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        frequency_ghz: float,
        lane_count: int,
        core_count: int,
    ) -> "CubeSpec":
        """从配置创建

        Args:
            config: 配置字典，包含 m/k/n 维度和 mac_per_lane
            frequency_ghz: 工作频率
            lane_count: lane 数量
            core_count: 核心数量

        Returns:
            CubeSpec 实例

        Raises:
            ValueError: 缺少必需的 m/k/n 维度参数
        """
        if "m" not in config:
            raise ValueError("Missing 'compute_units.cube.m' in chip config")
        if "k" not in config:
            raise ValueError("Missing 'compute_units.cube.k' in chip config")
        if "n" not in config:
            raise ValueError("Missing 'compute_units.cube.n' in chip config")

        # mac_per_lane 必需（否则 peak_flops 为 0）
        if "mac_per_lane" not in config or not config["mac_per_lane"]:
            raise ValueError("Missing or empty 'compute_units.cube.mac_per_lane' in chip config")

        mac_per_lane = {}
        for dtype, mac in config["mac_per_lane"].items():
            mac_per_lane[dtype.upper()] = mac

        return cls(
            name="Cube",
            dim_m=int(config["m"]),
            dim_k=int(config["k"]),
            dim_n=int(config["n"]),
            mac_per_lane=mac_per_lane,
            frequency_ghz=frequency_ghz,
            lane_count=lane_count,
            core_count=core_count,
        )


@dataclass
class VectorSpec:
    """Vector 计算单元规格

    用于激活函数、归一化等向量计算。

    Attributes:
        name: 单元名称
        eu_per_lane: 每 lane EU 数 (按数据类型)
        frequency_ghz: 工作频率
        lane_count: lane 数量
        core_count: 核心数量
    """

    name: str = "Vector"
    eu_per_lane: dict[str, int] = field(default_factory=dict)
    frequency_ghz: float = 1.0
    lane_count: int = 64
    core_count: int = 1

    def peak_flops(self, dtype: str) -> float:
        """计算峰值算力

        FLOPs = eu_per_lane * lane_count * core_count * frequency

        Args:
            dtype: 数据类型

        Returns:
            峰值算力 (FLOPS)
        """
        eu = self.eu_per_lane.get(dtype.upper(), 0)
        return eu * self.lane_count * self.core_count * self.frequency_ghz * 1e9

    def eu_count(self, dtype: str) -> int:
        """获取执行单元数量

        Args:
            dtype: 数据类型

        Returns:
            EU 数量
        """
        return (
            self.eu_per_lane.get(dtype.upper(), 0) * self.lane_count * self.core_count
        )

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        frequency_ghz: float,
        lane_count: int,
        core_count: int,
    ) -> "VectorSpec":
        """从配置创建

        Args:
            config: 配置字典
            frequency_ghz: 工作频率
            lane_count: lane 数量
            core_count: 核心数量

        Returns:
            VectorSpec 实例
        """
        eu_per_lane = {}
        for dtype, eu in config.get("eu_per_lane", {}).items():
            eu_per_lane[dtype.upper()] = eu

        return cls(
            name="Vector",
            eu_per_lane=eu_per_lane,
            frequency_ghz=frequency_ghz,
            lane_count=lane_count,
            core_count=core_count,
        )


@dataclass
class HAUSpec:
    """HAU 计算单元规格

    硬件加速器单元，用于特殊计算（如 Softmax、LayerNorm 等）。

    Attributes:
        name: 单元名称
        throughput_ops: 吞吐量 (ops/cycle)
        frequency_ghz: 工作频率
        supported_ops: 支持的操作列表
    """

    name: str = "HAU"
    throughput_ops: dict[str, float] = field(default_factory=dict)
    frequency_ghz: float = 1.0
    supported_ops: list[str] = field(default_factory=list)

    def peak_flops(self, dtype: str) -> float:
        """计算峰值算力

        Args:
            dtype: 数据类型

        Returns:
            峰值算力 (FLOPS)
        """
        throughput = self.throughput_ops.get(dtype.upper(), 0)
        return throughput * self.frequency_ghz * 1e9

    def eu_count(self, dtype: str) -> int:
        """获取执行单元数量

        Args:
            dtype: 数据类型

        Returns:
            EU 数量（对于 HAU 返回 1）
        """
        return 1 if dtype.upper() in self.throughput_ops else 0

    @classmethod
    def from_config(cls, config: dict[str, Any], frequency_ghz: float) -> "HAUSpec":
        """从配置创建

        Args:
            config: 配置字典
            frequency_ghz: 工作频率

        Returns:
            HAUSpec 实例
        """
        return cls(
            name="HAU",
            throughput_ops=config.get("throughput_ops", {}),
            frequency_ghz=frequency_ghz,
            supported_ops=config.get("supported_ops", []),
        )


# 类型别名
ComputeSpec = Union[CubeSpec, VectorSpec, HAUSpec]


def create_compute_unit(
    unit_type: str,
    config: dict[str, Any],
    frequency_ghz: float = 1.0,
    lane_count: int = 64,
    core_count: int = 1,
) -> ComputeSpec:
    """创建计算单元

    Args:
        unit_type: 单元类型 (cube/vector/hau)
        config: 配置字典
        frequency_ghz: 工作频率
        lane_count: lane 数量
        core_count: 核心数量

    Returns:
        计算单元实例

    Raises:
        ValueError: 不支持的单元类型
    """
    unit_type = unit_type.lower()
    if unit_type == "cube":
        return CubeSpec.from_config(config, frequency_ghz, lane_count, core_count)
    elif unit_type == "vector":
        return VectorSpec.from_config(config, frequency_ghz, lane_count, core_count)
    elif unit_type == "hau":
        return HAUSpec.from_config(config, frequency_ghz)
    else:
        raise ValueError(f"Unsupported compute unit type: {unit_type}")
