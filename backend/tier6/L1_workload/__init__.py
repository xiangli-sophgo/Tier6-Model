"""L1: 工作负载表示层

定义模型的层级结构、算子、计算图等。

核心类型:
    - Model: 模型 IR 实现
    - Module: 模块（逻辑分组）
    - Layer: 层（算子级抽象）
    - Op: 算子（最小计算单元）
    - TensorDesc: 张量描述
    - WorkloadGraph: 计算图
"""

from tier6.L1_workload.ir import Model, WorkloadIR
from tier6.L1_workload.layer import Layer, Module
from tier6.L1_workload.op import Op, AtomicInstruction
from tier6.L1_workload.tensor import TensorDesc, TensorShape, LayoutSignature
from tier6.L1_workload.graph import (
    WorkloadGraph,
    GraphNode,
    GraphEdge,
    NodeKind,
    NodeRole,
    EdgeType,
)
from tier6.L1_workload.specs import (
    ComputeSpec,
    MemorySpec,
    CommSpec,
    TileConfig,
    TiledMemoryInfo,
)
from tier6.L1_workload.breakdown import OpsBreakdown, MemoryFootprint
from tier6.L1_workload.comm_pattern import CommPattern, DataDependencyGraph
from tier6.L1_workload.metadata import ModelMetadata, MLAConfig, MoEConfig

__all__ = [
    # IR
    "Model",
    "WorkloadIR",
    # 结构
    "Module",
    "Layer",
    "Op",
    "AtomicInstruction",
    # 张量
    "TensorDesc",
    "TensorShape",
    "LayoutSignature",
    # 计算图
    "WorkloadGraph",
    "GraphNode",
    "GraphEdge",
    "NodeKind",
    "NodeRole",
    "EdgeType",
    # 规格
    "ComputeSpec",
    "MemorySpec",
    "CommSpec",
    "TileConfig",
    "TiledMemoryInfo",
    # 分析结果
    "OpsBreakdown",
    "MemoryFootprint",
    # 通信模式
    "CommPattern",
    "DataDependencyGraph",
    # 元数据
    "ModelMetadata",
    "MLAConfig",
    "MoEConfig",
]
