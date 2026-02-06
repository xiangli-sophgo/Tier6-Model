"""通信模式模块

定义 CommPattern, DataDependencyGraph 数据结构。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from math_model.L1_workload.specs import CommSpec


@dataclass
class DataDependencyGraph:
    """数据依赖图

    Attributes:
        nodes: 节点 ID 列表
        edges: 边列表（src, dst）
        edge_types: 边类型映射（data/control）
        branches: 分支路径（可选）
    """

    nodes: list[str] = field(default_factory=list)
    edges: list[tuple[str, str]] = field(default_factory=list)
    edge_types: dict[tuple[str, str], str] = field(default_factory=dict)
    branches: list[list[str]] = field(default_factory=list)

    def add_edge(self, src: str, dst: str, edge_type: str = "data") -> None:
        """添加边"""
        self.edges.append((src, dst))
        self.edge_types[(src, dst)] = edge_type

    def get_predecessors(self, node: str) -> list[str]:
        """获取前驱节点"""
        return [src for src, dst in self.edges if dst == node]

    def get_successors(self, node: str) -> list[str]:
        """获取后继节点"""
        return [dst for src, dst in self.edges if src == node]


@dataclass
class CommPattern:
    """通信模式

    Attributes:
        total_bytes: 总通信字节数
        by_layer: 按层的通信规格
        critical_path: 关键路径上的层
    """

    total_bytes: int = 0
    by_layer: dict[str, CommSpec] = field(default_factory=dict)
    critical_path: list[str] = field(default_factory=list)

    def add_layer_comm(self, layer_name: str, comm: CommSpec) -> None:
        """添加层的通信规格"""
        self.by_layer[layer_name] = comm
        self.total_bytes += comm.bytes

    def get_comm_by_pattern(self, pattern: str) -> list[CommSpec]:
        """按模式筛选通信"""
        return [c for c in self.by_layer.values() if c.pattern == pattern]

    @property
    def has_allreduce(self) -> bool:
        """是否有 AllReduce 通信"""
        return any(c.pattern == "allreduce" for c in self.by_layer.values())

    @property
    def has_all2all(self) -> bool:
        """是否有 All2All 通信"""
        return any(c.pattern in {"all2all", "alltoall"} for c in self.by_layer.values())

    @property
    def allreduce_size(self) -> int:
        """AllReduce 数据量（字节）"""
        return sum(c.bytes for c in self.by_layer.values() if c.pattern == "allreduce")

    @property
    def alltoall_size(self) -> int:
        """AllToAll 数据量（字节）"""
        return sum(
            c.bytes for c in self.by_layer.values() if c.pattern in {"all2all", "alltoall"}
        )

    @property
    def p2p_size(self) -> int:
        """点对点通信数据量（字节）"""
        return sum(c.bytes for c in self.by_layer.values() if c.pattern == "p2p")
