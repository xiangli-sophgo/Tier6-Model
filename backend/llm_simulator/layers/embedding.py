"""
Embedding 层

包含词嵌入查表和可选的 AllReduce 通信
"""

from dataclasses import dataclass, field
from typing import Dict, Any

from .base import BaseLayer
from ..operators import MatMulOperator, AllReduceOperator


@dataclass
class EmbeddingLayer(BaseLayer):
    """
    Embedding 层

    config 必须包含:
        - vocab_size: int, 词表大小
        - hidden_dim: int, 隐藏维度
        - batch_size: int, 批次大小
        - seq_len: int, 序列长度
        - tp: int, 张量并行度
        - comm_protocol: int, 通信协议
    """
    name: str = "embedding"
    layer_type: str = "Embedding"
    config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """构建层内算子"""
        self.comp_ops = []
        self.comm_ops = []
        self.operator_categories = {}
        self._build_operators()

    def _build_operators(self):
        """构建 Embedding 层的算子"""
        cfg = self.config
        vocab_size = cfg.get('vocab_size', 151936)
        hidden_dim = cfg.get('hidden_dim', 7168)
        batch_size = cfg.get('batch_size', 1)
        seq_len = cfg.get('seq_len', 1)
        tp = cfg.get('tp', 1)
        comm_protocol = cfg.get('comm_protocol', 1)

        # Embedding 本质是一个查表操作，可以建模为 GEMM
        # Input: (batch_size, seq_len) one-hot -> (batch_size, seq_len, vocab_size/tp)
        # Weight: (vocab_size/tp, hidden_dim)
        # Output: (batch_size, seq_len, hidden_dim)
        embed_op = MatMulOperator(
            name=f"{self.name}_embed",
            parallel_params={
                'G': 1,
                'M': batch_size * seq_len,
                'K': vocab_size // tp,
                'N': hidden_dim,
                'input_dtype': 'bf16',
                'output_dtype': 'bf16',
            }
        )
        self.add_operator(embed_op)

        # TP > 1 时需要 AllReduce
        if tp > 1:
            # 通信数据量 = batch_size * seq_len * hidden_dim * dtype_bytes
            dtype_bytes = 2  # bf16
            comm_size = batch_size * seq_len * hidden_dim * dtype_bytes
            allreduce_op = AllReduceOperator(
                name=f"{self.name}_allreduce",
                parallel_params={
                    'tp': tp,
                    'comm_size': comm_size,
                    'comm_protocol': comm_protocol,
                }
            )
            self.add_operator(allreduce_op)
