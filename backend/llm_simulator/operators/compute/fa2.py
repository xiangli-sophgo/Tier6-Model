"""
Flash Attention 2 算子

FA2 = Softmax(Q @ K^T / sqrt(d)) @ V
"""

from dataclasses import dataclass, field
from typing import Dict, Any

from ..base import ComputeOperator, ComputeOpType


@dataclass
class FA2Operator(ComputeOperator):
    """
    Flash Attention 2 算子

    parallel_params 必须包含:
        - B: int, Batch 维度 (通常是 num_heads)
        - QS: int, Query 序列长度
        - KS: int, Key/Value 序列长度
        - QD: int, Query/Key 维度 (head_dim)
        - VD: int, Value 维度 (head_dim)
    """
    name: str = ""
    op_type: ComputeOpType = ComputeOpType.FA2
    parallel_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """初始化后计算基础指标"""
        params = self.parallel_params
        B = params.get('B', 1)
        QS = params.get('QS', 1)
        KS = params.get('KS', 1)
        QD = params.get('QD', 1)
        VD = params.get('VD', 1)

        # FLOPs 计算:
        # Q @ K^T: 2 * B * QS * KS * QD
        # Softmax: ~5 * B * QS * KS (近似)
        # P @ V: 2 * B * QS * KS * VD
        qk_flops = 2 * B * QS * KS * QD
        pv_flops = 2 * B * QS * KS * VD
        softmax_flops = 5 * B * QS * KS
        self.flops = qk_flops + pv_flops + softmax_flops

        # FA2 没有持久化权重
        self.param = 0
        self.dram_occupy = 0

    @property
    def B(self) -> int:
        return self.parallel_params.get('B', 1)

    @property
    def QS(self) -> int:
        return self.parallel_params.get('QS', 1)

    @property
    def KS(self) -> int:
        return self.parallel_params.get('KS', 1)

    @property
    def QD(self) -> int:
        return self.parallel_params.get('QD', 1)

    @property
    def VD(self) -> int:
        return self.parallel_params.get('VD', 1)
