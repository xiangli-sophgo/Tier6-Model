"""张量描述模块

定义 TensorDesc 和 TensorShape 类型。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import reduce
from operator import mul
from typing import Any, TypeAlias

from perf_model.L0_entry.types import DataType

# 张量形状类型别名
TensorShape: TypeAlias = list[int]


@dataclass
class LayoutSignature:
    """并行/切分签名

    Attributes:
        parallel_type: 并行类型（TP/PP/DP/EP/SP/NONE）
        split_dim: 切分维度（heads/hidden/sequence/expert 等）
        split_factor: 切分因子
        replica_group_id: 副本分组标识
        extras: 兼容扩展字段
    """

    parallel_type: str
    split_dim: str
    split_factor: int
    replica_group_id: str
    extras: dict[str, str | int] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LayoutSignature":
        required_keys = {"parallel_type", "split_dim", "split_factor", "replica_group_id"}
        missing = required_keys - set(data.keys())
        if missing:
            raise ValueError(
                f"layout_signature missing fields: {sorted(missing)}; "
                "required: parallel_type/split_dim/split_factor/replica_group_id"
            )
        extras = {k: v for k, v in data.items() if k not in required_keys}
        return cls(
            parallel_type=str(data["parallel_type"]),
            split_dim=str(data["split_dim"]),
            split_factor=int(data["split_factor"]),
            replica_group_id=str(data["replica_group_id"]),
            extras=extras,
        )


@dataclass
class TensorDesc:
    """张量描述

    Attributes:
        name: 张量名称
        shape: 张量形状
        dtype: 数据类型
        is_weight: 是否为权重（用于区分权重/激活）
        layout: 数据布局（NCHW/NHWC/NC1HWC0 等）
        producer_id: 生产者标识（可选）
        consumer_id: 消费者标识（可选）
        layout_signature: 并行/切分签名（可选）
    """

    name: str
    shape: TensorShape
    dtype: DataType
    is_weight: bool = False
    layout: str | None = None
    producer_id: str | None = None
    consumer_id: str | None = None
    layout_signature: LayoutSignature | None = None

    def __post_init__(self) -> None:
        if isinstance(self.layout_signature, dict):
            self.layout_signature = LayoutSignature.from_dict(self.layout_signature)

    @property
    def bytes(self) -> int:
        """计算字节数"""
        return self.elements * self.dtype.bytes

    @property
    def elements(self) -> int:
        """元素数量"""
        if not self.shape:
            return 0
        return reduce(mul, self.shape, 1)

    @property
    def ndim(self) -> int:
        """维度数"""
        return len(self.shape)

    def __repr__(self) -> str:
        weight_str = ", weight" if self.is_weight else ""
        return f"TensorDesc({self.name}: {self.shape}, {self.dtype.value}{weight_str})"
