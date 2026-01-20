"""
Multi-Query Attention 算子

MQA 用于 MLAAbsorb/MLAAbsorbv32 变体 (absorbed KV 优化)
与 MHA 的区别: QS 作为 heads 维度，而不是 H

参数说明:
- B: batch size (seqs)
- QS: heads 维度 (num_heads)
- KS: key/value 序列长度
- QD: query/key 维度
- VD: value 维度
"""

from dataclasses import dataclass, field
from typing import Dict, Any

from ..base import ComputeOperator, ComputeOpType


@dataclass
class MQAOperator(ComputeOperator):
    """
    Multi-Query Attention 算子

    对齐 DS_TPU 的 MQAOperator，用于 MLAAbsorb/MLAAbsorbv32 变体

    parallel_params 必须包含:
        - B: int, Batch 维度 (seqs)
        - QS: int, heads 维度 (作为 query 序列长度)
        - KS: int, Key/Value 序列长度
        - QD: int, Query/Key 维度
        - VD: int, Value 维度
    """
    name: str = ""
    op_type: ComputeOpType = ComputeOpType.MQA
    parallel_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """初始化后计算基础指标"""
        params = self.parallel_params
        B = params.get('B', 1)
        QS = params.get('QS', 1)  # 在 MQA 中 QS 代表 heads
        KS = params.get('KS', 1)
        QD = params.get('QD', 1)
        VD = params.get('VD', 1)

        # FLOPs 计算 (对齐 DS_TPU):
        # 注意: MQA 没有 H 维度，QS 充当 heads
        # Q @ K^T: 2 * B * QS * KS * QD
        # P @ V: 2 * B * QS * KS * VD
        # Softmax: ~5 * B * QS * KS
        qk_flops = 2 * B * QS * KS * QD
        pv_flops = 2 * B * QS * KS * VD
        softmax_flops = 5 * B * QS * KS
        self.flops = qk_flops + pv_flops + softmax_flops

        # MQA 没有持久化权重
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
