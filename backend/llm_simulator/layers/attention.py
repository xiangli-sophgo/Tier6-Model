"""
Attention 层

支持 MLA (Multi-head Latent Attention) 和 MHA (Multi-Head Attention)
"""

from dataclasses import dataclass, field
from typing import Dict, Any

from .base import BaseLayer
from ..operators import (
    MatMulOperator,
    FA2Operator,
    RMSNormOperator,
    AllReduceOperator,
)


@dataclass
class MLALayer(BaseLayer):
    """
    MLA (Multi-head Latent Attention) 层 - DeepSeek V3 风格

    使用低秩压缩减少 KV Cache 大小

    config 必须包含:
        - hidden_dim: int, 隐藏维度
        - num_heads: int, 注意力头数
        - head_dim: int, 每头维度
        - kv_lora_rank: int, KV 压缩维度
        - q_lora_rank: int, Q 压缩维度 (可选)
        - batch_size: int, 批次大小
        - seq_len: int, 序列长度
        - tp: int, 张量并行度
        - comm_protocol: int, 通信协议
        - is_prefill: bool, 是否为 prefill 阶段
    """
    name: str = "mla"
    layer_type: str = "MLA"
    config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """构建层内算子"""
        self.comp_ops = []
        self.comm_ops = []
        self.operator_categories = {}
        self._build_operators()

    def _build_operators(self):
        """构建 MLA 层的算子"""
        cfg = self.config
        hidden_dim = cfg.get('hidden_dim', 7168)
        num_heads = cfg.get('num_heads', 128)
        head_dim = cfg.get('head_dim', 128)
        kv_lora_rank = cfg.get('kv_lora_rank', 512)
        q_lora_rank = cfg.get('q_lora_rank', 1536)
        batch_size = cfg.get('batch_size', 1)
        seq_len = cfg.get('seq_len', 1)
        tp = cfg.get('tp', 1)
        comm_protocol = cfg.get('comm_protocol', 1)

        # 每个 TP rank 的头数
        heads_per_tp = num_heads // tp
        tokens = batch_size * seq_len

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

        # 2. Q 投影 (两阶段: down + up)
        # Q down: hidden_dim -> q_lora_rank
        q_down_op = MatMulOperator(
            name=f"{self.name}_q_down",
            parallel_params={
                'G': 1,
                'M': tokens,
                'K': hidden_dim,
                'N': q_lora_rank,
                'input_dtype': 'bf16',
                'output_dtype': 'bf16',
            }
        )
        self.add_operator(q_down_op)

        # Q up: q_lora_rank -> num_heads * head_dim / tp
        q_up_op = MatMulOperator(
            name=f"{self.name}_q_up",
            parallel_params={
                'G': 1,
                'M': tokens,
                'K': q_lora_rank,
                'N': heads_per_tp * head_dim,
                'input_dtype': 'bf16',
                'output_dtype': 'bf16',
            }
        )
        self.add_operator(q_up_op)

        # 3. KV 压缩: hidden_dim -> kv_lora_rank
        kv_compress_op = MatMulOperator(
            name=f"{self.name}_kv_compress",
            parallel_params={
                'G': 1,
                'M': tokens,
                'K': hidden_dim,
                'N': kv_lora_rank,
                'input_dtype': 'bf16',
                'output_dtype': 'bf16',
            }
        )
        self.add_operator(kv_compress_op)

        # 4. Flash Attention (在压缩空间)
        fa2_op = FA2Operator(
            name=f"{self.name}_fa2",
            parallel_params={
                'B': heads_per_tp,
                'QS': seq_len,
                'KS': seq_len,
                'QD': head_dim,
                'VD': head_dim,
            }
        )
        self.add_operator(fa2_op)

        # 5. Output 投影: num_heads * head_dim / tp -> hidden_dim
        o_proj_op = MatMulOperator(
            name=f"{self.name}_o_proj",
            parallel_params={
                'G': 1,
                'M': tokens,
                'K': heads_per_tp * head_dim,
                'N': hidden_dim,
                'input_dtype': 'bf16',
                'output_dtype': 'bf16',
            }
        )
        self.add_operator(o_proj_op)

        # 6. TP > 1 时需要 AllReduce
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


@dataclass
class MHALayer(BaseLayer):
    """
    MHA (Multi-Head Attention) 层 - 标准 Transformer 风格

    config 必须包含:
        - hidden_dim: int, 隐藏维度
        - num_heads: int, 注意力头数
        - num_kv_heads: int, KV 头数 (GQA)
        - head_dim: int, 每头维度
        - batch_size: int, 批次大小
        - seq_len: int, 序列长度
        - tp: int, 张量并行度
        - comm_protocol: int, 通信协议
    """
    name: str = "mha"
    layer_type: str = "MHA"
    config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """构建层内算子"""
        self.comp_ops = []
        self.comm_ops = []
        self.operator_categories = {}
        self._build_operators()

    def _build_operators(self):
        """构建 MHA 层的算子"""
        cfg = self.config
        hidden_dim = cfg.get('hidden_dim', 4096)
        num_heads = cfg.get('num_heads', 32)
        num_kv_heads = cfg.get('num_kv_heads', num_heads)
        head_dim = cfg.get('head_dim', 128)
        batch_size = cfg.get('batch_size', 1)
        seq_len = cfg.get('seq_len', 1)
        tp = cfg.get('tp', 1)
        comm_protocol = cfg.get('comm_protocol', 1)

        heads_per_tp = num_heads // tp
        kv_heads_per_tp = num_kv_heads // tp
        tokens = batch_size * seq_len

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

        # 2. QKV 投影 (合并)
        qkv_dim = (heads_per_tp + 2 * kv_heads_per_tp) * head_dim
        qkv_op = MatMulOperator(
            name=f"{self.name}_qkv_proj",
            parallel_params={
                'G': 1,
                'M': tokens,
                'K': hidden_dim,
                'N': qkv_dim,
                'input_dtype': 'bf16',
                'output_dtype': 'bf16',
            }
        )
        self.add_operator(qkv_op)

        # 3. Flash Attention
        fa2_op = FA2Operator(
            name=f"{self.name}_fa2",
            parallel_params={
                'B': heads_per_tp,
                'QS': seq_len,
                'KS': seq_len,
                'QD': head_dim,
                'VD': head_dim,
            }
        )
        self.add_operator(fa2_op)

        # 4. Output 投影
        o_proj_op = MatMulOperator(
            name=f"{self.name}_o_proj",
            parallel_params={
                'G': 1,
                'M': tokens,
                'K': heads_per_tp * head_dim,
                'N': hidden_dim,
                'input_dtype': 'bf16',
                'output_dtype': 'bf16',
            }
        )
        self.add_operator(o_proj_op)

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
