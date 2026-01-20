"""
Multi-Head Attention 算子

MHA = Softmax(Q @ K^T / sqrt(d)) @ V
用于 MLA/MLAv32 变体 (kv_b_proj 解压缩方式)

参数说明:
- B: batch size
- H: num_heads (注意力头数)
- QS: query 序列长度
- KS: key/value 序列长度
- QD: query/key 维度 (qk_nope + qk_rope)
- VD: value 维度 (v_head_dim)
"""

from dataclasses import dataclass, field
from typing import Dict, Any

from ..base import ComputeOperator, ComputeOpType


@dataclass
class MHAOperator(ComputeOperator):
    """
    Multi-Head Attention 算子

    对齐 DS_TPU 的 MHAOperator，用于 MLA/MLAv32 变体

    parallel_params 必须包含:
        - B: int, Batch 维度 (batch_size)
        - H: int, 注意力头数 (num_heads)
        - QS: int, Query 序列长度
        - KS: int, Key/Value 序列长度
        - QD: int, Query/Key 维度 (qk_nope + qk_rope)
        - VD: int, Value 维度 (v_head_dim)
    """
    name: str = ""
    op_type: ComputeOpType = ComputeOpType.MHA
    parallel_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """初始化后计算基础指标"""
        params = self.parallel_params
        B = params.get('B', 1)
        H = params.get('H', 1)
        QS = params.get('QS', 1)
        KS = params.get('KS', 1)
        QD = params.get('QD', 1)
        VD = params.get('VD', 1)

        # FLOPs 计算 (对齐 DS_TPU):
        # Q @ K^T: 2 * B * H * QS * KS * QD
        # P @ V: 2 * B * H * QS * KS * VD
        # Softmax: ~5 * B * H * QS * KS (近似)
        qk_flops = 2 * B * H * QS * KS * QD
        pv_flops = 2 * B * H * QS * KS * VD
        softmax_flops = 5 * B * H * QS * KS
        self.flops = qk_flops + pv_flops + softmax_flops

        # MHA 没有持久化权重
        self.param = 0
        self.dram_occupy = 0

    @property
    def B(self) -> int:
        return self.parallel_params.get('B', 1)

    @property
    def H(self) -> int:
        return self.parallel_params.get('H', 1)

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

    @property
    def effective_B(self) -> int:
        """等效 Batch 维度 (用于评估器)"""
        return self.B * self.H
