"""模型基类模块

定义 ModelBase 抽象基类。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Iterator

from math_model.L1_workload.graph import GraphEdge, GraphNode, NodeKind, NodeRole, WorkloadGraph
from math_model.L1_workload.layer import Layer, Module
from math_model.L1_workload.metadata import ModelMetadata
from math_model.L1_workload.tensor import TensorDesc

if TYPE_CHECKING:
    from math_model.L1_workload.ir import WorkloadIR
    from math_model.L1_workload.layers.base import LayerBase


class ModelBase(ABC):
    """模型基类

    定义模型构建与访问接口。

    使用场景:
        - 从 YAML 配置构建模型（模型结构固定，参数可配置）
        - 支持层级遍历与统计

    子类需实现:
        - name: 模型名称
        - build: 构建模型层结构
        - _build_metadata: 构建模型元数据

    Example:
        >>> @model_registry.register("llama2_7b")
        ... class Llama2_7BModel(ModelBase):
        ...     @property
        ...     def name(self) -> str:
        ...         return "llama2_7b"
        ...
        ...     def build(self) -> None:
        ...         for i in range(self._config["num_layers"]):
        ...             self._layers.append(AttentionLayer(f"attn_{i}", self._config))
        ...             self._layers.append(FFNLayer(f"ffn_{i}", self._config))
    """

    def __init__(self, config: dict[str, Any]):
        """初始化模型

        Args:
            config: 模型配置字典，包含 hidden_size/num_layers/num_heads 等
        """
        self._config = config
        self._layers: list[LayerBase] = []
        self._metadata: ModelMetadata | None = None

    @property
    @abstractmethod
    def name(self) -> str:
        """模型名称，如 'llama2_7b'"""
        ...

    @property
    def config(self) -> dict[str, Any]:
        """返回模型配置"""
        return self._config

    @abstractmethod
    def build(self) -> None:
        """构建模型层结构

        子类实现此方法，填充 self._layers
        """
        ...

    def get_layers(self) -> list["LayerBase"]:
        """获取所有层"""
        if not self._layers:
            self.build()
        return self._layers

    def iter_layers(self) -> Iterator["LayerBase"]:
        """迭代所有层"""
        yield from self.get_layers()

    def get_metadata(self) -> ModelMetadata:
        """获取模型元数据"""
        if self._metadata is None:
            self._metadata = self._build_metadata()
        return self._metadata

    @abstractmethod
    def _build_metadata(self) -> ModelMetadata:
        """构建模型元数据，子类实现"""
        ...

    def get_info(self) -> dict[str, Any]:
        """获取模型汇总信息

        Returns:
            dict: 包含 name/num_layers/total_flops/total_params 等
        """
        layers = self.get_layers()
        total_flops = sum(layer.get_info().get("flops", 0) for layer in layers)
        total_params = sum(layer.get_info().get("weight_bytes", 0) for layer in layers)
        return {
            "name": self.name,
            "num_layers": len(layers),
            "total_flops": total_flops,
            "total_params": total_params,
            "config": self._config,
        }

    def to_ir(self) -> "WorkloadIR":
        """转换为 WorkloadIR

        Returns:
            WorkloadIR: 可供 L2/L3/L4 使用的中间表示
        """
        from math_model.L1_workload.ir import Model

        # 将所有层包装为单一 Module
        layers = [layer.to_layer() for layer in self.get_layers()]
        return Model(
            modules=[Module(name="main", type="sequential", layers=layers, submodules=[], attrs={})],
            graph=self._build_graph(layers),
            metadata=self.get_metadata(),
        )

    def _build_graph(self, layers: list[Layer]) -> WorkloadGraph:
        """构建顺序执行图

        默认实现为顺序连接，并可拼接层内 OP 级子图。
        子类可覆盖实现复杂拓扑。

        Args:
            layers: 层列表

        Returns:
            WorkloadGraph: 计算图
        """
        graph = WorkloadGraph()
        base_layers = {layer.name: layer for layer in self.get_layers()}
        layer_node_id_by_name: dict[str, str] = {}

        # 创建节点
        for i, layer in enumerate(layers):
            layer_node_id = f"n_{i}"
            node = GraphNode(
                node_id=layer_node_id,
                kind=NodeKind.LAYER,
                role=NodeRole(layer.attrs.get("role", "compute")),
                ref=layer.name,
            )
            graph.add_node(node)
            layer_node_id_by_name[layer.name] = layer_node_id

        # 拼接层内 OP 级子图（若层提供）
        for layer in layers:
            base_layer = base_layers.get(layer.name)
            if base_layer is None:
                continue
            layer_node_id = layer_node_id_by_name[layer.name]
            intra_nodes, intra_edges, intra_entries, _ = base_layer.build_intra_graph(layer_node_id)
            for intra_node in intra_nodes:
                graph.add_node(intra_node)
            for intra_edge in intra_edges:
                graph.add_edge(intra_edge)
            for entry_id in intra_entries:
                # 控制边表达“层包含这些入口 OP”，不改变层间数据依赖主干
                graph.add_edge(
                    GraphEdge(src=layer_node_id, dst=entry_id, edge_type="control")
                )

        # 创建边（顺序连接）
        for i in range(len(layers) - 1):
            src_layer = layers[i]
            dst_layer = layers[i + 1]
            tensor = None
            if src_layer.outputs:
                output = src_layer.outputs[0]
                tensor = TensorDesc(
                    name=output.name,
                    shape=output.shape,
                    dtype=output.dtype,
                    is_weight=output.is_weight,
                    layout=output.layout,
                    producer_id=src_layer.name,
                    consumer_id=dst_layer.name,
                    layout_signature=output.layout_signature,
                )
            edge = GraphEdge(
                src=f"n_{i}",
                dst=f"n_{i+1}",
                edge_type="data",
                tensor=tensor,
            )
            graph.add_edge(edge)

        # 设置入口出口
        if layers:
            graph.entry_nodes = ["n_0"]
            graph.exit_nodes = [f"n_{len(layers)-1}"]

        return graph
