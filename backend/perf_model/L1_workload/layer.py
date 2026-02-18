"""层数据结构模块

定义 Layer 数据类。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from perf_model.L1_workload.op import Op
from perf_model.L1_workload.specs import CommSpec, ComputeSpec, MemorySpec
from perf_model.L1_workload.tensor import TensorDesc


@dataclass
class Layer:
    """层数据结构

    算子级抽象，为性能建模的基本单元。

    Attributes:
        name: 层名称（唯一标识）
        op_type: 层类型（attention/ffn/layernorm 等）
        inputs: 输入张量描述
        outputs: 输出张量描述
        params: 层参数（hidden_size/intermediate_size 等）
        ops: 包含的算子列表
        comm: 通信提示（可选，仅用于统计/可视化）
        attrs: 扩展属性
    """

    name: str
    op_type: str
    inputs: list[TensorDesc] = field(default_factory=list)
    outputs: list[TensorDesc] = field(default_factory=list)
    params: dict[str, float | int | str] = field(default_factory=dict)
    ops: list[Op] = field(default_factory=list)
    comm: CommSpec | None = None
    attrs: dict[str, str] = field(default_factory=dict)

    @property
    def role(self) -> str:
        """获取层角色（从 attrs 或推断）"""
        return self.attrs.get("role", "compute")

    def get_info(self) -> dict[str, Any]:
        """获取层汇总信息

        Returns:
            dict: 包含 name/op_type/flops/weight_bytes/activation_bytes/comm_hint_bytes
        """
        compute = self._aggregate_compute()
        memory = self._aggregate_memory()
        return {
            "name": self.name,
            "op_type": self.op_type,
            "role": self.role,
            "flops": compute.total_ops,
            "cube_ops": compute.cube_ops,
            "vector_ops": compute.vector_ops,
            "weight_bytes": memory.weight_bytes,
            "activation_bytes": memory.activation_bytes,
            "comm_hint_bytes": self.comm.bytes if self.comm else 0,
        }

    def _aggregate_compute(self) -> ComputeSpec:
        """聚合所有算子的计算规格"""
        if not self.ops:
            return ComputeSpec()
        return ComputeSpec(
            cube_ops=sum(op.compute.cube_ops for op in self.ops),
            vector_ops=sum(op.compute.vector_ops for op in self.ops),
            scalar_ops=sum(op.compute.scalar_ops for op in self.ops),
            hau_ops=sum(op.compute.hau_ops for op in self.ops),
        )

    def _aggregate_memory(self) -> MemorySpec:
        """聚合所有算子的内存规格"""
        if not self.ops:
            return MemorySpec()
        return MemorySpec(
            weight_bytes=sum(op.memory.weight_bytes for op in self.ops),
            activation_bytes=sum(op.memory.activation_bytes for op in self.ops),
            temp_bytes=sum(op.memory.temp_bytes for op in self.ops),
            read_bytes=sum(op.memory.read_bytes for op in self.ops),
            write_bytes=sum(op.memory.write_bytes for op in self.ops),
        )

    def get_flops(self) -> int:
        """获取总 FLOPs"""
        return self._aggregate_compute().total_ops

    def get_weight_bytes(self) -> int:
        """获取权重字节数"""
        return sum(t.bytes for t in self.inputs if t.is_weight)

    def get_activation_bytes(self) -> int:
        """获取激活字节数"""
        return sum(t.bytes for t in self.inputs if not t.is_weight) + sum(
            t.bytes for t in self.outputs
        )


@dataclass
class Module:
    """模块数据结构

    逻辑分组（Embedding/Attention/FFN 等），可复用/复合。

    Attributes:
        name: 模块名称
        type: 模块类型
        layers: 包含的层列表
        submodules: 子模块列表（支持嵌套）
        attrs: 扩展属性
    """

    name: str
    type: str
    layers: list[Layer] = field(default_factory=list)
    submodules: list["Module"] = field(default_factory=list)
    attrs: dict[str, str] = field(default_factory=dict)

    def get_all_layers(self) -> list[Layer]:
        """递归获取所有层"""
        result = list(self.layers)
        for submodule in self.submodules:
            result.extend(submodule.get_all_layers())
        return result
