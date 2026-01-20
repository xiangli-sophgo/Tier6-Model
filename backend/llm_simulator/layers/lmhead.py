"""
LMHead 层

输出层: 将隐藏状态映射到词表 logits
"""

from dataclasses import dataclass, field
from typing import Dict, Any

from .base import BaseLayer
from ..operators import (
    MatMulOperator,
    RMSNormOperator,
    AllGatherOperator,
)


@dataclass
class LMHeadLayer(BaseLayer):
    """
    LMHead (Language Model Head) 层

    结构: RMSNorm -> Linear (hidden_dim -> vocab_size)

    config 必须包含:
        - hidden_dim: int, 隐藏维度
        - vocab_size: int, 词表大小
        - batch_size: int, 批次大小
        - seq_len: int, 序列长度
        - tp: int, 张量并行度
        - comm_protocol: int, 通信协议
    """
    name: str = "lmhead"
    layer_type: str = "LMHead"
    config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """构建层内算子"""
        self.comp_ops = []
        self.comm_ops = []
        self.operator_categories = {}
        self._build_operators()

    def _build_operators(self):
        """构建 LMHead 层的算子"""
        cfg = self.config
        hidden_dim = cfg.get('hidden_dim', 7168)
        vocab_size = cfg.get('vocab_size', 151936)
        batch_size = cfg.get('batch_size', 1)
        seq_len = cfg.get('seq_len', 1)
        tp = cfg.get('tp', 1)
        comm_protocol = cfg.get('comm_protocol', 1)

        tokens = batch_size * seq_len
        dtype_bytes = 2  # bf16

        # 1. RMSNorm
        rmsnorm_op = RMSNormOperator(
            name=f"{self.name}_rmsnorm",
            parallel_params={
                'batch_size': tokens,
                'hidden_dim': hidden_dim,
                'has_scale': True,
                'has_bias': False,
            }
        )
        self.add_operator(rmsnorm_op)

        # 2. Linear: hidden_dim -> vocab_size / tp
        lm_linear_op = MatMulOperator(
            name=f"{self.name}_linear",
            parallel_params={
                'G': 1,
                'M': tokens,
                'K': hidden_dim,
                'N': vocab_size // tp,
                'input_dtype': 'bf16',
                'output_dtype': 'bf16',
            }
        )
        self.add_operator(lm_linear_op)

        # 3. TP > 1 时需要 AllGather (收集完整 logits)
        if tp > 1:
            # AllGather: 每个 rank 有 vocab_size/tp，收集后得到完整 vocab_size
            comm_size = tokens * vocab_size * dtype_bytes
            allgather_op = AllGatherOperator(
                name=f"{self.name}_allgather",
                parallel_params={
                    'tp': tp,
                    'comm_size': comm_size,
                    'comm_protocol': comm_protocol,
                }
            )
            self.add_operator(allgather_op)
