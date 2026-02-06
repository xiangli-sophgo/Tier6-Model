"""算子基类模块

定义 OpBase 抽象基类和 OpRole 枚举。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

from math_model.core.types import DataType
from math_model.L1_workload.op import Op
from math_model.L1_workload.specs import ComputeSpec, MemorySpec
from math_model.L1_workload.tensor import TensorDesc


class OpRole(Enum):
    """算子角色"""

    CUBE = "cube"  # Cube 引擎（MatMul/Conv）
    VECTOR = "vector"  # Vector 引擎（Softmax/LayerNorm）
    SCALAR = "scalar"  # Scalar 引擎（标量运算）
    HAU = "hau"  # HAU 引擎（特殊函数）
    MEMORY = "memory"  # 内存操作（Reshape/Transpose）
    COMM = "comm"  # 通信操作（AllReduce/AllGather）


class OpBase(ABC):
    """算子基类

    定义算子的计算与内存特征接口。

    设计原则:
        - 算子是最小的性能计算单元
        - 每个算子对应特定的硬件执行引擎
        - 算子应能独立计算 OPs / Memory Access

    子类需实现:
        - op_type: 算子类型
        - role: 算子角色
        - get_inputs: 获取输入张量描述
        - get_outputs: 获取输出张量描述
        - compute_ops: 计算操作数
        - compute_memory_access: 计算内存访问量

    Example:
        >>> @op_registry.register("matmul")
        ... class MatMulOp(OpBase):
        ...     @property
        ...     def op_type(self) -> str:
        ...         return "matmul"
        ...
        ...     @property
        ...     def role(self) -> OpRole:
        ...         return OpRole.CUBE
        ...
        ...     def compute_ops(self) -> int:
        ...         return 2 * self._m * self._n * self._k
    """

    def __init__(
        self,
        name: str,
        *,
        dtype: DataType = DataType.FP16,
    ):
        """初始化算子

        Args:
            name: 算子名称
            dtype: 数据类型
        """
        self._name = name
        self._dtype = dtype

    @property
    def name(self) -> str:
        """算子名称"""
        return self._name

    @property
    @abstractmethod
    def op_type(self) -> str:
        """算子类型，如 'matmul' / 'softmax' / 'allreduce'"""
        ...

    @property
    @abstractmethod
    def role(self) -> OpRole:
        """算子角色，对应硬件引擎"""
        ...

    @abstractmethod
    def get_inputs(self) -> list[TensorDesc]:
        """获取输入张量描述"""
        ...

    @abstractmethod
    def get_outputs(self) -> list[TensorDesc]:
        """获取输出张量描述"""
        ...

    @abstractmethod
    def compute_ops(self) -> int:
        """计算操作数

        Returns:
            int: 操作次数（乘加算两次）
        """
        ...

    @abstractmethod
    def compute_memory_access(self) -> tuple[int, int]:
        """计算内存访问量

        Returns:
            tuple[int, int]: (read_bytes, write_bytes)
        """
        ...

    def get_compute_spec(self) -> ComputeSpec:
        """获取计算规格"""
        ops = self.compute_ops()
        return ComputeSpec(
            cube_ops=ops if self.role == OpRole.CUBE else 0,
            vector_ops=ops if self.role == OpRole.VECTOR else 0,
            scalar_ops=ops if self.role == OpRole.SCALAR else 0,
            hau_ops=ops if self.role == OpRole.HAU else 0,
            precision=self._dtype,
        )

    def get_memory_spec(self) -> MemorySpec:
        """获取内存规格"""
        read_bytes, write_bytes = self.compute_memory_access()
        weight_bytes = sum(t.bytes for t in self.get_inputs() if t.is_weight)
        activation_bytes = read_bytes - weight_bytes + write_bytes
        return MemorySpec(
            weight_bytes=weight_bytes,
            activation_bytes=activation_bytes,
            read_bytes=read_bytes,
            write_bytes=write_bytes,
        )

    def get_info(self) -> dict[str, Any]:
        """获取算子汇总信息

        Returns:
            dict: 包含 name/op_type/role/ops/read_bytes/write_bytes
        """
        read_bytes, write_bytes = self.compute_memory_access()
        return {
            "name": self.name,
            "op_type": self.op_type,
            "role": self.role.value,
            "ops": self.compute_ops(),
            "read_bytes": read_bytes,
            "write_bytes": write_bytes,
        }

    def to_op(self) -> Op:
        """转换为 Op 数据结构"""
        return Op(
            name=self.name,
            op_type=self.op_type,
            inputs=self.get_inputs(),
            outputs=self.get_outputs(),
            compute=self.get_compute_spec(),
            memory=self.get_memory_spec(),
            attrs={"role": self.role.value},
        )
