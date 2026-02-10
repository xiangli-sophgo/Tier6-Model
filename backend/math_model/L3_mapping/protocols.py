"""L3 映射层协议定义

定义并行规划、切片、调度的接口协议。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from math_model.L1_workload.ir import Model
    from math_model.L2_arch.protocols import ClusterSpec


class ParallelMode(str, Enum):
    """并行模式"""

    TP = "tp"  # Tensor Parallelism
    PP = "pp"  # Pipeline Parallelism
    DP = "dp"  # Data Parallelism
    EP = "ep"  # Expert Parallelism
    SP = "sp"  # Sequence Parallelism


@dataclass
class ParallelismConfig:
    """并行配置

    Attributes:
        tp: Tensor Parallelism 度
        pp: Pipeline Parallelism 度
        dp: Data Parallelism 度
        ep: Expert Parallelism 度
        sp: Sequence Parallelism 度
    """

    tp: int = 1
    pp: int = 1
    dp: int = 1
    ep: int = 1
    sp: int = 1

    @property
    def total_devices(self) -> int:
        """总设备数 = TP × PP × DP × EP"""
        return self.tp * self.pp * self.dp * self.ep

    def to_dict(self) -> dict[str, int]:
        """转换为字典"""
        return {
            "tp": self.tp,
            "pp": self.pp,
            "dp": self.dp,
            "ep": self.ep,
            "sp": self.sp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ParallelismConfig":
        """从字典创建

        注意: 并行度字段都是可选的，未提供时默认为 1（不并行）
        """
        return cls(
            tp=data.get("tp", 1),  # 可选，默认 1
            pp=data.get("pp", 1),  # 可选，默认 1
            dp=data.get("dp", 1),  # 可选，默认 1
            ep=data.get("ep", 1),  # 可选，默认 1
            sp=data.get("sp", 1),  # 可选，默认 1
        )


@dataclass
class DeviceAssignment:
    """设备分配

    Attributes:
        device_id: 设备 ID
        global_rank: 全局 rank
        tp_rank: TP rank
        pp_rank: PP rank
        dp_rank: DP rank
        ep_rank: EP rank
        sp_rank: SP rank
    """

    device_id: str
    global_rank: int
    tp_rank: int = 0
    pp_rank: int = 0
    dp_rank: int = 0
    ep_rank: int = 0
    sp_rank: int = 0


@dataclass
class ParallelGroup:
    """并行组

    Attributes:
        mode: 并行模式
        group_id: 组 ID
        device_ids: 组内设备 ID 列表
        bandwidth_gbps: 组内带宽 (Gbps)
        latency_us: 组内延迟 (us)
    """

    mode: ParallelMode
    group_id: int
    device_ids: list[str]
    bandwidth_gbps: float = 0.0
    latency_us: float = 0.0


@dataclass
class ParallelGroupAssignment:
    """并行组分配结果

    Attributes:
        config: 并行配置
        assignments: 设备分配列表
        tp_groups: TP 组列表
        pp_groups: PP 组列表
        dp_groups: DP 组列表
        ep_groups: EP 组列表
    """

    config: ParallelismConfig
    assignments: list[DeviceAssignment] = field(default_factory=list)
    tp_groups: list[ParallelGroup] = field(default_factory=list)
    pp_groups: list[ParallelGroup] = field(default_factory=list)
    dp_groups: list[ParallelGroup] = field(default_factory=list)
    ep_groups: list[ParallelGroup] = field(default_factory=list)


@runtime_checkable
class ParallelismPlanner(Protocol):
    """并行规划器协议"""

    def plan(
        self,
        model: "Model",
        cluster: "ClusterSpec",
        config: ParallelismConfig,
    ) -> ParallelGroupAssignment:
        """规划并行策略

        Args:
            model: 模型 IR
            cluster: 集群规格
            config: 并行配置

        Returns:
            ParallelGroupAssignment: 并行组分配结果
        """
        ...


@runtime_checkable
class TilingPlanner(Protocol):
    """切片规划器协议"""

    def plan(
        self,
        model: "Model",
        cluster: "ClusterSpec",
        parallel_assignment: ParallelGroupAssignment,
    ) -> "TilePlan":
        """规划切片策略

        Args:
            model: 模型 IR
            cluster: 集群规格
            parallel_assignment: 并行组分配

        Returns:
            TilePlan: 切片计划
        """
        ...


@runtime_checkable
class Scheduler(Protocol):
    """调度器协议"""

    def schedule(
        self,
        model: "Model",
        cluster: "ClusterSpec",
        parallel_assignment: ParallelGroupAssignment,
        tile_plan: "TilePlan | None" = None,
    ) -> "ExecPlan":
        """生成执行计划

        Args:
            model: 模型 IR
            cluster: 集群规格
            parallel_assignment: 并行组分配
            tile_plan: 切片计划 (可选)

        Returns:
            ExecPlan: 执行计划
        """
        ...


@dataclass
class TileConfig:
    """切片配置

    Attributes:
        layer_name: 层名称
        tile_m: M 维度切片大小
        tile_k: K 维度切片大小
        tile_n: N 维度切片大小
        attrs: 扩展属性
    """

    layer_name: str
    tile_m: int = 1
    tile_k: int = 1
    tile_n: int = 1
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass
class TilePlan:
    """切片计划

    Attributes:
        tiles: 各层的切片配置
        memory_budget_mb: 内存预算 (MB)
        attrs: 扩展属性
    """

    tiles: list[TileConfig] = field(default_factory=list)
    memory_budget_mb: float = 0.0
    attrs: dict[str, Any] = field(default_factory=dict)

    def get_tile(self, layer_name: str) -> TileConfig | None:
        """获取指定层的切片配置"""
        for tile in self.tiles:
            if tile.layer_name == layer_name:
                return tile
        return None
