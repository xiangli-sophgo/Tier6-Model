"""
FFN (Feed-Forward Network) 层

标准的 Dense MLP: Gate + Up + Down 三个投影
"""

from dataclasses import dataclass, field
from typing import Dict, Any

from .base import BaseLayer
from ..operators import (
    MatMulOperator,
    RMSNormOperator,
    AllReduceOperator,
)


@dataclass
class MLPLayer(BaseLayer):
    """
    MLP (Dense FFN) 层

    结构: RMSNorm -> Gate -> Up -> SiLU -> Down

    config 必须包含:
        - hidden_dim: int, 隐藏维度
        - inter_dim: int, 中间层维度 (通常是 hidden_dim * 4 或 * 8/3)
        - batch_size: int, 全局批次大小 (对齐 DS_TPU)
        - seq_len: int, 序列长度
        - tp: int, 张量并行度
        - dp: int, 数据并行度 (用于计算 local_batch)
        - comm_protocol: int, 通信协议
    """
    name: str = "mlp"
    layer_type: str = "MLP"
    config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """构建层内算子"""
        self.comp_ops = []
        self.comm_ops = []
        self.operator_categories = {}
        self._build_operators()

    def _build_operators(self):
        """构建 MLP 层的算子"""
        cfg = self.config
        hidden_dim = cfg.get('hidden_dim', 7168)
        inter_dim = cfg.get('inter_dim', 18432)
        batch_size = cfg.get('batch_size', 1)  # 全局 batch (对齐 DS_TPU)
        seq_len = cfg.get('seq_len', 1)
        tp = cfg.get('tp', 1)
        dp = cfg.get('dp', 1)  # 数据并行度
        comm_protocol = cfg.get('comm_protocol', 1)

        # 计算本地 batch (对齐 DS_TPU: local_batch = batch_size // dp)
        local_batch = batch_size // dp if dp > 0 else batch_size
        tokens = local_batch * seq_len
        inter_dim_per_tp = inter_dim // tp

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

        # 2. Gate 投影: hidden_dim -> inter_dim / tp
        gate_op = MatMulOperator(
            name=f"{self.name}_gate_proj",
            parallel_params={
                'G': 1,
                'M': tokens,
                'K': hidden_dim,
                'N': inter_dim_per_tp,
                'input_dtype': 'bf16',
                'output_dtype': 'bf16',
            }
        )
        self.add_operator(gate_op)

        # 3. Up 投影: hidden_dim -> inter_dim / tp
        up_op = MatMulOperator(
            name=f"{self.name}_up_proj",
            parallel_params={
                'G': 1,
                'M': tokens,
                'K': hidden_dim,
                'N': inter_dim_per_tp,
                'input_dtype': 'bf16',
                'output_dtype': 'bf16',
            }
        )
        self.add_operator(up_op)

        # 4. Down 投影: inter_dim / tp -> hidden_dim
        down_op = MatMulOperator(
            name=f"{self.name}_down_proj",
            parallel_params={
                'G': 1,
                'M': tokens,
                'K': inter_dim_per_tp,
                'N': hidden_dim,
                'input_dtype': 'bf16',
                'output_dtype': 'bf16',
            }
        )
        self.add_operator(down_op)

        # 5. TP > 1 时需要 AllReduce
        if tp > 1:
            dtype_bytes = 2
            comm_size = tokens * hidden_dim * dtype_bytes
            allreduce_op = AllReduceOperator(
                name=f"{self.name}_allreduce",
                parallel_params={
                    'tp': tp,
                    'comm_size': comm_size,
                    'comm_protocol': comm_protocol,
                }
            )
            self.add_operator(allreduce_op)
