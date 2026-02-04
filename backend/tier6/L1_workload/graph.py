"""计算图模块

定义 WorkloadGraph, GraphNode, GraphEdge 数据结构。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator

from tier6.L1_workload.tensor import TensorDesc


class NodeKind(Enum):
    """节点粒度"""

    LAYER = "layer"
    OP = "op"


class NodeRole(Enum):
    """节点角色"""

    COMPUTE = "compute"  # 计算密集型
    MEMORY = "memory"  # 内存密集型
    COMM = "comm"  # 通信型


class EdgeType(Enum):
    """边类型"""

    DATA = "data"  # 数据依赖
    CONTROL = "control"  # 控制依赖


@dataclass
class GraphNode:
    """计算图节点

    Attributes:
        node_id: 节点唯一标识
        kind: 节点粒度（layer/op）
        role: 节点角色（compute/memory/comm）
        ref: 引用的 Layer.name 或 Op.name
        attrs: 扩展属性
    """

    node_id: str
    kind: NodeKind | str
    role: NodeRole | str
    ref: str
    attrs: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # 支持字符串自动转换为枚举
        if isinstance(self.kind, str):
            self.kind = NodeKind(self.kind)
        if isinstance(self.role, str):
            self.role = NodeRole(self.role)


@dataclass
class GraphEdge:
    """计算图边

    Attributes:
        src: 源节点 ID
        dst: 目标节点 ID
        edge_type: 边类型（data/control）
        tensor: 传输的张量描述（控制依赖时为 None）
        attrs: 扩展属性
    """

    src: str
    dst: str
    edge_type: EdgeType | str
    tensor: TensorDesc | None = None
    attrs: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.edge_type, str):
            self.edge_type = EdgeType(self.edge_type)


@dataclass
class WorkloadGraph:
    """工作负载计算图

    支持多分支/汇合的 DAG 结构。

    Attributes:
        nodes: 节点字典（key = node_id）
        edges: 边列表
        entry_nodes: 入口节点 ID 列表
        exit_nodes: 出口节点 ID 列表
    """

    nodes: dict[str, GraphNode] = field(default_factory=dict)
    edges: list[GraphEdge] = field(default_factory=list)
    entry_nodes: list[str] = field(default_factory=list)
    exit_nodes: list[str] = field(default_factory=list)

    def add_node(self, node: GraphNode) -> None:
        """添加节点"""
        self.nodes[node.node_id] = node

    def add_edge(self, edge: GraphEdge) -> None:
        """添加边"""
        self.edges.append(edge)

    def get_predecessors(self, node_id: str) -> list[GraphNode]:
        """获取前驱节点"""
        return [self.nodes[e.src] for e in self.edges if e.dst == node_id]

    def get_successors(self, node_id: str) -> list[GraphNode]:
        """获取后继节点"""
        return [self.nodes[e.dst] for e in self.edges if e.src == node_id]

    def get_input_edges(self, node_id: str) -> list[GraphEdge]:
        """获取流入该节点的边"""
        return [e for e in self.edges if e.dst == node_id]

    def get_output_edges(self, node_id: str) -> list[GraphEdge]:
        """获取流出该节点的边"""
        return [e for e in self.edges if e.src == node_id]

    def topological_sort(self) -> list[str]:
        """拓扑排序 (Kahn 算法)

        Returns:
            按拓扑序排列的节点 ID 列表

        Raises:
            ValueError: 存在环
        """
        # 计算入度
        in_degree: dict[str, int] = {node_id: 0 for node_id in self.nodes}
        for edge in self.edges:
            in_degree[edge.dst] += 1

        # 初始化队列
        queue = [node_id for node_id, degree in in_degree.items() if degree == 0]
        result: list[str] = []

        while queue:
            node_id = queue.pop(0)
            result.append(node_id)

            for edge in self.edges:
                if edge.src == node_id:
                    in_degree[edge.dst] -= 1
                    if in_degree[edge.dst] == 0:
                        queue.append(edge.dst)

        if len(result) != len(self.nodes):
            raise ValueError("Graph contains a cycle")

        return result

    def iter_nodes(self) -> Iterator[GraphNode]:
        """迭代所有节点"""
        yield from self.nodes.values()

    def iter_nodes_by_role(self, role: NodeRole | str) -> Iterator[GraphNode]:
        """按角色迭代节点"""
        if isinstance(role, str):
            role = NodeRole(role)
        for node in self.nodes.values():
            if node.role == role:
                yield node

    def __len__(self) -> int:
        return len(self.nodes)
