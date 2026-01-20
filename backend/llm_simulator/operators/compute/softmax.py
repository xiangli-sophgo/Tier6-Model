"""
Softmax 算子

Softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
"""

from dataclasses import dataclass, field
from typing import Dict, Any

from ..base import ComputeOperator, ComputeOpType


@dataclass
class SoftmaxOperator(ComputeOperator):
    """
    Softmax 算子

    parallel_params 必须包含:
        - QS: int, Query 序列长度
        - KS: int, Key 序列长度
        - num_heads: int, 注意力头数
    """
    name: str = ""
    op_type: ComputeOpType = ComputeOpType.SOFTMAX
    parallel_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """初始化后计算基础指标"""
        params = self.parallel_params
        QS = params.get('QS', 1)
        KS = params.get('KS', 1)
        num_heads = params.get('num_heads', 1)

        # Softmax 是纯向量操作，无持久化权重
        self.param = 0
        self.dram_occupy = 0
        # FLOPs 估算 (reduce_max + exp + reduce_sum + div)
        self.flops = num_heads * QS * KS * 5

    @property
    def QS(self) -> int:
        return self.parallel_params.get('QS', 1)

    @property
    def KS(self) -> int:
        return self.parallel_params.get('KS', 1)

    @property
    def num_heads(self) -> int:
        return self.parallel_params.get('num_heads', 1)
