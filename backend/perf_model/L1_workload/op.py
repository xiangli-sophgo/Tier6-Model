"""算子数据结构模块

定义 Op 数据类。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from perf_model.L1_workload.specs import ComputeSpec, MemorySpec
from perf_model.L1_workload.tensor import TensorDesc


@dataclass
class AtomicInstruction:
    """原子指令（可选，用于 Trace/指令级分析）

    Attributes:
        name: 指令名称
        op_type: 指令类型
        cycles: 时钟周期数
        attrs: 扩展属性
    """

    name: str
    op_type: str
    cycles: int = 0
    attrs: dict[str, str] = field(default_factory=dict)


@dataclass
class Op:
    """算子数据结构

    IR 中可计算的最小单元，可对应后端算子或 Fusion 后的算子。

    Attributes:
        name: 算子名称（唯一标识）
        op_type: 算子类型（matmul/conv/softmax 等）
        inputs: 输入张量描述
        outputs: 输出张量描述
        compute: 计算规格
        memory: 内存规格
        attrs: 扩展属性
        atomic_instructions: 原子指令列表（可选）
    """

    name: str
    op_type: str
    inputs: list[TensorDesc] = field(default_factory=list)
    outputs: list[TensorDesc] = field(default_factory=list)
    compute: ComputeSpec = field(default_factory=ComputeSpec)
    memory: MemorySpec = field(default_factory=MemorySpec)
    attrs: dict[str, str] = field(default_factory=dict)
    atomic_instructions: list[AtomicInstruction] = field(default_factory=list)

    @property
    def role(self) -> str:
        """获取算子角色（从 attrs 或推断）"""
        return self.attrs.get("role", "compute")

    def get_info(self) -> dict[str, Any]:
        """获取算子汇总信息

        Returns:
            dict: 包含 name/op_type/role/ops/weight_bytes/activation_bytes
        """
        return {
            "name": self.name,
            "op_type": self.op_type,
            "role": self.role,
            "total_ops": self.compute.total_ops,
            "cube_ops": self.compute.cube_ops,
            "vector_ops": self.compute.vector_ops,
            "weight_bytes": self.memory.weight_bytes,
            "activation_bytes": self.memory.activation_bytes,
        }

    def get_input_bytes(self) -> int:
        """获取输入字节数"""
        return sum(t.bytes for t in self.inputs)

    def get_output_bytes(self) -> int:
        """获取输出字节数"""
        return sum(t.bytes for t in self.outputs)

    def get_weight_bytes(self) -> int:
        """获取权重字节数"""
        return sum(t.bytes for t in self.inputs if t.is_weight)
