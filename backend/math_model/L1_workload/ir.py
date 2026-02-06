"""工作负载 IR 模块

定义 WorkloadIR Protocol, Model 顶层结构。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from math_model.L1_workload.breakdown import MemoryFootprint, OpsBreakdown
from math_model.L1_workload.comm_pattern import CommPattern, DataDependencyGraph
from math_model.L1_workload.graph import GraphNode, NodeRole, WorkloadGraph
from math_model.L1_workload.layer import Layer, Module
from math_model.L1_workload.metadata import ModelMetadata


@runtime_checkable
class WorkloadIR(Protocol):
    """工作负载 IR 协议

    定义 L1 对外提供的统一接口，所有 Loader 和 ModelBuilder 都应返回此协议的实现。
    """

    @property
    def name(self) -> str:
        """工作负载名称"""
        ...

    # ======================== 分析接口 ========================

    def get_ops_breakdown(self) -> OpsBreakdown:
        """获取操作数分解"""
        ...

    def get_memory_footprint(self) -> MemoryFootprint:
        """获取内存占用"""
        ...

    def get_data_dependencies(self) -> DataDependencyGraph:
        """获取数据依赖图"""
        ...

    def get_communication_pattern(self) -> CommPattern:
        """获取通信模式"""
        ...

    # ======================== 结构访问 ========================

    def get_layers(self) -> list[Layer]:
        """获取所有层"""
        ...

    def get_graph(self) -> WorkloadGraph:
        """获取计算图"""
        ...

    def get_metadata(self) -> ModelMetadata:
        """获取模型元数据"""
        ...

    # ======================== 统一聚合 ========================

    def get_all_nodes(self) -> list[GraphNode]:
        """获取所有节点（按拓扑序）"""
        ...

    def get_nodes_by_role(self, role: str | NodeRole) -> list[GraphNode]:
        """按角色筛选节点"""
        ...

    def get_layer_infos(self) -> list[dict[str, Any]]:
        """批量获取所有 Layer 的 get_info() 输出"""
        ...


@dataclass
class Model:
    """模型 IR 实现

    WorkloadIR 的具体实现，由 Loader 或 ModelBuilder 生成。

    Attributes:
        modules: 模块列表
        graph: 计算图
        metadata: 模型元数据
    """

    modules: list[Module] = field(default_factory=list)
    graph: WorkloadGraph = field(default_factory=WorkloadGraph)
    metadata: ModelMetadata | None = None

    # ======================== 分析接口 ========================

    @property
    def name(self) -> str:
        """工作负载名称"""
        if self.metadata is None:
            raise ValueError("Model metadata not set")
        return self.metadata.name

    def get_ops_breakdown(self) -> OpsBreakdown:
        """获取操作数分解"""
        layers = self.get_layers()
        total_ops = 0
        cube_ops = 0
        vector_ops = 0
        scalar_ops = 0
        hau_ops = 0
        by_layer: dict[str, int] = {}

        for layer in layers:
            info = layer.get_info()
            flops = info.get("flops", 0)
            total_ops += flops
            cube_ops += info.get("cube_ops", 0)
            vector_ops += info.get("vector_ops", 0)
            by_layer[layer.name] = flops

        return OpsBreakdown(
            total_ops=total_ops,
            cube_ops=cube_ops,
            vector_ops=vector_ops,
            scalar_ops=scalar_ops,
            hau_ops=hau_ops,
            by_layer=by_layer,
        )

    def get_memory_footprint(self) -> MemoryFootprint:
        """获取内存占用"""
        layers = self.get_layers()
        weights = 0
        activations = 0

        for layer in layers:
            info = layer.get_info()
            weights += info.get("weight_bytes", 0)
            activations += info.get("activation_bytes", 0)

        return MemoryFootprint(
            weights=weights,
            activations=activations,
        )

    def get_data_dependencies(self) -> DataDependencyGraph:
        """获取数据依赖图"""
        dep_graph = DataDependencyGraph()
        dep_graph.nodes = list(self.graph.nodes.keys())

        for edge in self.graph.edges:
            dep_graph.add_edge(edge.src, edge.dst, edge.edge_type.value)

        return dep_graph

    def get_communication_pattern(self) -> CommPattern:
        """获取通信模式"""
        pattern = CommPattern()
        for layer in self.get_layers():
            if layer.comm is not None:
                pattern.add_layer_comm(layer.name, layer.comm)
        return pattern

    # ======================== 结构访问 ========================

    def get_layers(self) -> list[Layer]:
        """获取所有层"""
        result: list[Layer] = []
        for module in self.modules:
            result.extend(module.get_all_layers())
        return result

    def get_graph(self) -> WorkloadGraph:
        """获取计算图"""
        return self.graph

    def get_metadata(self) -> ModelMetadata:
        """获取模型元数据"""
        if self.metadata is None:
            raise ValueError("Model metadata not set")
        return self.metadata

    # ======================== 统一聚合 ========================

    def get_all_nodes(self) -> list[GraphNode]:
        """获取所有节点（按拓扑序）"""
        return list(self.graph.nodes.values())

    def get_nodes_by_role(self, role: str | NodeRole) -> list[GraphNode]:
        """按角色筛选节点"""
        return list(self.graph.iter_nodes_by_role(role))

    def get_layer_infos(self) -> list[dict[str, Any]]:
        """批量获取所有 Layer 的 get_info() 输出"""
        return [layer.get_info() for layer in self.get_layers()]
