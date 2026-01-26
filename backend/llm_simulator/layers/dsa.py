"""
DSA (Deep Sparse Attention) 层

DeepSeek V3.2 特有的稀疏注意力层，用于 index-based token selection。
包含 wq_b, wk, weights_proj MatMul 算子和 fp8_index MQA 算子。

对齐 DS_TPU: model/layers/dsa.py
"""

from dataclasses import dataclass, field
from typing import Dict, Any

from .base import BaseLayer
from ..operators import (
    MatMulOperator,
    MQAOperator,
    AllReduceOperator,
)


@dataclass
class DSALayer(BaseLayer):
    """
    DSA (Deep Sparse Attention) 层 - DeepSeek V3.2 稀疏注意力

    用于 index-based token selection 和 weighting。
    这不是主要的注意力计算层，而是用于选择稀疏 token 的辅助层。

    结构 (对齐 DS_TPU):
    - wq_b: [seqs, q_lora_rank] → [seqs, heads * index_head_dim]
    - wk: [seqs, hidden_dim] → [seqs, index_head_dim]
    - weights_proj: [seqs, hidden_dim] → [seqs, heads]
    - fp8_index: MQA 算子，执行稀疏注意力
    - allreduce (if tp > 1)

    config 必须包含:
        - hidden_dim: int, 隐藏维度
        - q_lora_rank: int, Q 压缩维度
        - n_index_heads: int, index 头数
        - index_head_dim: int, index 头维度
        - topk_index: int, 稀疏 topk 值
        - batch_size: int, 全局批次大小 (对齐 DS_TPU)
        - seq_len: int, 序列长度 (prefill) 或 1 (decode)
        - kv_seq_len: int, KV 序列长度
        - tp: int, 张量并行度
        - dp: int, 数据并行度 (用于计算 local_batch)
        - enable_tp_sp: bool, 是否启用 TP 序列并行
        - comm_protocol: int, 通信协议
        - is_prefill: bool, 是否为 prefill 阶段
    """
    name: str = "dsa"
    layer_type: str = "DSA"
    config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """构建层内算子"""
        self.comp_ops = []
        self.comm_ops = []
        self.operator_categories = {}
        self._build_operators()

    def _build_operators(self):
        """构建 DSA 层的算子 (对齐 DS_TPU)"""
        cfg = self.config
        hidden_dim = cfg.get('hidden_dim', 7168)
        q_lora_rank = cfg.get('q_lora_rank', 1536)
        n_index_heads = cfg.get('n_index_heads', 128)
        index_head_dim = cfg.get('index_head_dim', 128)
        topk_index = cfg.get('topk_index', 2048)
        batch_size = cfg.get('batch_size', 1)  # 全局 batch (对齐 DS_TPU)
        seq_len = cfg.get('seq_len', 1)
        kv_seq_len = cfg.get('kv_seq_len', 4096)
        tp = cfg.get('tp', 1)
        dp = cfg.get('dp', 1)  # 数据并行度
        enable_tp_sp = cfg.get('enable_tp_sp', False)
        comm_protocol = cfg.get('comm_protocol', 1)
        is_prefill = cfg.get('is_prefill', False)

        # 计算本地 batch (对齐 DS_TPU: local_batch = batch_size // dp)
        local_batch = batch_size // dp if dp > 0 else batch_size

        # 每个 TP rank 的 index heads
        heads_per_tp = n_index_heads // tp

        # q_seq_len: prefill 时使用 seq_len，decode 时为 1
        q_seq_len = seq_len if is_prefill else 1
        seqs = local_batch * q_seq_len

        dtype_bytes = 2  # bf16

        # 1. wq_b: q_lora_rank → heads * index_head_dim
        # 对齐 DS_TPU: parallel_params={'G': 1, 'M': seqs, 'K': q_lora_rank, 'N': heads * index_head_dim}
        wq_b = MatMulOperator(
            name=f"{self.name}_wq_b",
            parallel_params={
                'G': 1,
                'M': seqs,
                'K': q_lora_rank,
                'N': heads_per_tp * index_head_dim,
                'input_dtype': 'bf16',
                'output_dtype': 'bf16',
            }
        )
        self.add_operator(wq_b)

        # 2. wk: hidden_dim → index_head_dim
        # 对齐 DS_TPU: parallel_params={'G': 1, 'M': seqs, 'K': hidden_dim, 'N': index_head_dim}
        wk = MatMulOperator(
            name=f"{self.name}_wk",
            parallel_params={
                'G': 1,
                'M': seqs,
                'K': hidden_dim,
                'N': index_head_dim,
                'input_dtype': 'bf16',
                'output_dtype': 'bf16',
            }
        )
        self.add_operator(wk)

        # 3. weights_proj: hidden_dim → heads
        # 对齐 DS_TPU: parallel_params={'G': 1, 'M': seqs, 'K': hidden_dim, 'N': heads}
        weights_proj = MatMulOperator(
            name=f"{self.name}_weights_proj",
            parallel_params={
                'G': 1,
                'M': seqs,
                'K': hidden_dim,
                'N': heads_per_tp,
                'input_dtype': 'bf16',
                'output_dtype': 'bf16',
            }
        )
        self.add_operator(weights_proj)

        # 4. fp8_index: MQA 算子
        # 对齐 DS_TPU:
        # - 如果 tp > 1 and enable_tp_sp: KS = kv_seq_len // tp
        # - 否则: KS = kv_seq_len
        # - QD = index_head_dim // 2 (FP8 quantization)
        # - VD = 64 (固定)
        if tp > 1 and enable_tp_sp:
            ks_effective = kv_seq_len // tp
        else:
            ks_effective = kv_seq_len

        fp8_index = MQAOperator(
            name=f"{self.name}_fp8_index",
            parallel_params={
                'B': local_batch,
                'QS': heads_per_tp,
                'KS': ks_effective,
                'QD': index_head_dim // 2,  # FP8 量化
                'VD': 64,  # 固定输出维度
            }
        )
        self.add_operator(fp8_index)

        # 5. AllReduce (if tp > 1)
        # 对齐 DS_TPU: comm_size = seqs * hidden_dim * BF16
        if tp > 1:
            comm_size = seqs * hidden_dim * dtype_bytes
            allreduce_op = AllReduceOperator(
                name=f"{self.name}_allreduce",
                parallel_params={
                    'tp': tp,
                    'comm_size': comm_size,
                    'comm_protocol': comm_protocol,
                }
            )
            self.add_operator(allreduce_op)

    def get_info(self) -> Dict[str, Any]:
        """获取 DSA 层信息 (使用基类方法，确保包含 perf 字典)"""
        # 调用基类的 get_info 获取标准格式
        info = super().get_info()

        # 添加 DSA 特有的配置信息
        cfg = self.config
        info['config'] = {
            'hidden_dim': cfg.get('hidden_dim', 7168),
            'q_lora_rank': cfg.get('q_lora_rank', 1536),
            'n_index_heads': cfg.get('n_index_heads', 128),
            'index_head_dim': cfg.get('index_head_dim', 128),
            'topk_index': cfg.get('topk_index', 2048),
            'batch_size': cfg.get('batch_size', 1),
            'batch_size_local': cfg.get('batch_size', 1) // cfg.get('dp', 1),
            'seq_len': cfg.get('seq_len', 1),
            'kv_seq_len': cfg.get('kv_seq_len', 4096),
            'tp': cfg.get('tp', 1),
            'dp': cfg.get('dp', 1),
        }

        return info
