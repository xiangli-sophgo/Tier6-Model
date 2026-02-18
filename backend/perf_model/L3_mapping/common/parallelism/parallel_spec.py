"""ParallelSpec - 并行规格定义

定义 op 的 chip 级切分方式。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto


class ParallelType(Enum):
    """并行类型枚举

    定义 chip 间的并行切分方式：
    - TP_COL: 列切分（切 N 维度），用于 Gate/Up 投影
    - TP_ROW: 行切分（切 K 维度），用于 Down 投影
    - TP_HEAD: 头切分，用于 Attention 的 QKV
    - EP_EXPERT: 专家切分，用于 MoE
    - REPLICATE: 复制，不切分
    """

    TP_COL = auto()  # 列切分 (切 N 维度)
    TP_ROW = auto()  # 行切分 (切 K 维度)
    TP_HEAD = auto()  # 头切分 (Attention)
    EP_EXPERT = auto()  # 专家切分 (MoE)
    REPLICATE = auto()  # 复制，不切分


@dataclass
class ParallelSpec:
    """并行规格

    描述单个 op 的 chip 级切分方式。

    Attributes:
        parallel_type: 并行类型
        split_dim: 切分的维度名称 (如 "N", "K", "head", "expert")
        split_factor: 切分因子，默认为 1 表示按 TP 数切分
        layout_signature: 布局签名 (parallel_type/split_dim/split_factor/replica_group_id)
    """

    parallel_type: ParallelType
    split_dim: str
    split_factor: int = 1
    layout_signature: dict[str, str | int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """补齐布局签名"""
        if not self.layout_signature:
            self.layout_signature = {
                "parallel_type": self.parallel_type.name,
                "split_dim": self.split_dim,
                "split_factor": self.split_factor,
                "replica_group_id": "default",
            }

    def get_local_shape(
        self, original_shape: dict[str, int], tp: int
    ) -> dict[str, int]:
        """计算切分后的 local shape

        Args:
            original_shape: 原始 shape，如 {"M": 4096, "N": 11008, "K": 4096}
            tp: TP 并行度

        Returns:
            切分后的 shape
        """
        if self.parallel_type == ParallelType.REPLICATE:
            return original_shape.copy()

        local_shape = original_shape.copy()
        if self.split_dim in local_shape:
            local_shape[self.split_dim] = local_shape[self.split_dim] // tp
        return local_shape


@dataclass
class OpParallelDecl:
    """Op 的并行声明

    在 L1 的 Op 定义中声明该 op 支持的并行方式。

    Attributes:
        allowed_types: 允许的并行类型列表
        splittable_dims: 可切分的维度列表
        default_type: 默认的并行类型
    """

    allowed_types: list[ParallelType] = field(default_factory=list)
    splittable_dims: list[str] = field(default_factory=list)
    default_type: ParallelType = ParallelType.REPLICATE
