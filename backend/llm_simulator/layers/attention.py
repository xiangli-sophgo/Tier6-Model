"""
Attention 层

支持 4 种 MLA 变体 (对齐 DS_TPU):
- MLALayer: 基础版本，使用 kv_b_proj 解压缩
- MLAv32Layer: V3.2 DSA 稀疏注意力
- MLAAbsorbLayer: absorbed KV 优化，w_kc/w_vc 替代 kv_b_proj
- MLAAbsorbv32Layer: absorbed + DSA，支持 SP

以及标准 MHA:
- MHALayer: Multi-Head Attention (标准 Transformer)
"""

from dataclasses import dataclass, field
from typing import Dict, Any

from .base import BaseLayer
from ..operators import (
    MatMulOperator,
    FA2Operator,
    MHAOperator,
    MQAOperator,
    RMSNormOperator,
    AllReduceOperator,
    AllGatherOperator,
    ReduceScatterOperator,
)


@dataclass
class MLALayer(BaseLayer):
    """
    MLA 基础版 - 对齐 DS_TPU MLA

    使用 kv_b_proj 解压缩 KV Cache

    算子列表:
    - q_a_proj: [seqs, hidden_dim] → [seqs, q_lora_rank]
    - q_b_proj: [seqs, q_lora_rank] → [seqs, heads * (qk_nope + qk_rope)]
    - kv_a_proj: [seqs, hidden_dim] → [seqs, kv_lora_rank + qk_rope]
    - kv_b_proj: [seqs, kv_lora_rank] → [seqs, heads * (qk_nope + v_head)]
    - mha: MHAOperator
    - o_proj: [seqs, heads * v_head] → [seqs, hidden_dim]
    - allreduce (if tp > 1)

    config 必须包含:
        - hidden_dim: int, 隐藏维度
        - num_heads: int, 注意力头数
        - qk_nope_dim: int, QK non-positional 维度
        - qk_rope_dim: int, QK RoPE 维度
        - v_head_dim: int, V 头维度
        - kv_lora_rank: int, KV 压缩维度
        - q_lora_rank: int, Q 压缩维度
        - batch_size: int, 全局批次大小 (对齐 DS_TPU)
        - seq_len: int, 序列长度
        - kv_seq_len: int, KV 序列长度
        - tp: int, 张量并行度
        - dp: int, 数据并行度 (用于计算 local_batch)
        - comm_protocol: int, 通信协议
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
        qk_nope_dim = cfg.get('qk_nope_dim', 128)
        qk_rope_dim = cfg.get('qk_rope_dim', 64)
        v_head_dim = cfg.get('v_head_dim', 128)
        kv_lora_rank = cfg.get('kv_lora_rank', 512)
        q_lora_rank = cfg.get('q_lora_rank', 1536)
        batch_size = cfg.get('batch_size', 1)  # 全局 batch (对齐 DS_TPU)
        seq_len = cfg.get('seq_len', 1)
        kv_seq_len = cfg.get('kv_seq_len', seq_len)
        tp = cfg.get('tp', 1)
        dp = cfg.get('dp', 1)  # 数据并行度
        comm_protocol = cfg.get('comm_protocol', 1)

        # 计算本地 batch (对齐 DS_TPU: local_batch = batch_size // dp)
        local_batch = batch_size // dp if dp > 0 else batch_size

        # 每个 TP rank 的头数
        heads_per_tp = num_heads // tp
        seqs = local_batch * seq_len

        # QK 总维度
        qk_dim = qk_nope_dim + qk_rope_dim

        # 1. Q_A projection: hidden_dim → q_lora_rank
        q_a_proj = MatMulOperator(
            name=f"{self.name}_q_a_proj",
            parallel_params={
                'G': 1,
                'M': seqs,
                'K': hidden_dim,
                'N': q_lora_rank,
                'input_dtype': 'bf16',
                'output_dtype': 'bf16',
            }
        )
        self.add_operator(q_a_proj)

        # 2. Q_B projection: q_lora_rank → heads * (qk_nope + qk_rope)
        q_b_proj = MatMulOperator(
            name=f"{self.name}_q_b_proj",
            parallel_params={
                'G': 1,
                'M': seqs,
                'K': q_lora_rank,
                'N': heads_per_tp * qk_dim,
                'input_dtype': 'bf16',
                'output_dtype': 'bf16',
            }
        )
        self.add_operator(q_b_proj)

        # 3. KV_A projection: hidden_dim → kv_lora_rank + qk_rope
        kv_a_proj = MatMulOperator(
            name=f"{self.name}_kv_a_proj",
            parallel_params={
                'G': 1,
                'M': seqs,
                'K': hidden_dim,
                'N': kv_lora_rank + qk_rope_dim,
                'input_dtype': 'bf16',
                'output_dtype': 'bf16',
            }
        )
        self.add_operator(kv_a_proj)

        # 4. KV_B projection: kv_lora_rank → heads * (qk_nope + v_head)
        # 关键差异: MLA 使用 kv_b_proj 解压缩
        kv_b_proj = MatMulOperator(
            name=f"{self.name}_kv_b_proj",
            parallel_params={
                'G': 1,
                'M': seqs,
                'K': kv_lora_rank,
                'N': heads_per_tp * (qk_nope_dim + v_head_dim),
                'input_dtype': 'bf16',
                'output_dtype': 'bf16',
            }
        )
        self.add_operator(kv_b_proj)

        # 5. MHA (Multi-Head Attention)
        mha_op = MHAOperator(
            name=f"{self.name}_mha",
            parallel_params={
                'B': local_batch,  # 使用本地 batch
                'H': heads_per_tp,
                'QS': seq_len,
                'KS': kv_seq_len,
                'QD': qk_dim,
                'VD': v_head_dim,
            }
        )
        self.add_operator(mha_op)

        # 6. Output projection: heads * v_head → hidden_dim
        o_proj = MatMulOperator(
            name=f"{self.name}_o_proj",
            parallel_params={
                'G': 1,
                'M': seqs,
                'K': heads_per_tp * v_head_dim,
                'N': hidden_dim,
                'input_dtype': 'bf16',
                'output_dtype': 'bf16',
            }
        )
        self.add_operator(o_proj)

        # 7. AllReduce (if tp > 1)
        if tp > 1:
            dtype_bytes = 2
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


@dataclass
class MLAv32Layer(BaseLayer):
    """
    MLA V3.2 - DSA 稀疏注意力

    与 MLALayer 相同，但:
    - 新增参数: topk_index (默认 2048)
    - mha 的 KS = min(kv_seq_len, topk_index)
    - dram_occupy 仍基于完整 kv_seq_len
    """
    name: str = "mla_v32"
    layer_type: str = "MLAv32"
    config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """构建层内算子"""
        self.comp_ops = []
        self.comm_ops = []
        self.operator_categories = {}
        self._build_operators()

    def _build_operators(self):
        """构建 MLAv32 层的算子"""
        cfg = self.config
        hidden_dim = cfg.get('hidden_dim', 7168)
        num_heads = cfg.get('num_heads', 128)
        qk_nope_dim = cfg.get('qk_nope_dim', 128)
        qk_rope_dim = cfg.get('qk_rope_dim', 64)
        v_head_dim = cfg.get('v_head_dim', 128)
        kv_lora_rank = cfg.get('kv_lora_rank', 512)
        q_lora_rank = cfg.get('q_lora_rank', 1536)
        batch_size = cfg.get('batch_size', 1)  # 全局 batch (对齐 DS_TPU)
        seq_len = cfg.get('seq_len', 1)
        kv_seq_len = cfg.get('kv_seq_len', seq_len)
        topk_index = cfg.get('topk_index', 2048)  # DSA 稀疏注意力
        tp = cfg.get('tp', 1)
        dp = cfg.get('dp', 1)  # 数据并行度
        comm_protocol = cfg.get('comm_protocol', 1)

        # 计算本地 batch (对齐 DS_TPU: local_batch = batch_size // dp)
        local_batch = batch_size // dp if dp > 0 else batch_size

        heads_per_tp = num_heads // tp
        seqs = local_batch * seq_len
        qk_dim = qk_nope_dim + qk_rope_dim

        # DSA: 实际计算的 KS 是 min(kv_seq_len, topk_index)
        effective_kv_seq = min(kv_seq_len, topk_index)

        # 1. Q_A projection
        q_a_proj = MatMulOperator(
            name=f"{self.name}_q_a_proj",
            parallel_params={
                'G': 1, 'M': seqs, 'K': hidden_dim, 'N': q_lora_rank,
                'input_dtype': 'bf16', 'output_dtype': 'bf16',
            }
        )
        self.add_operator(q_a_proj)

        # 2. Q_B projection
        q_b_proj = MatMulOperator(
            name=f"{self.name}_q_b_proj",
            parallel_params={
                'G': 1, 'M': seqs, 'K': q_lora_rank, 'N': heads_per_tp * qk_dim,
                'input_dtype': 'bf16', 'output_dtype': 'bf16',
            }
        )
        self.add_operator(q_b_proj)

        # 3. KV_A projection
        kv_a_proj = MatMulOperator(
            name=f"{self.name}_kv_a_proj",
            parallel_params={
                'G': 1, 'M': seqs, 'K': hidden_dim, 'N': kv_lora_rank + qk_rope_dim,
                'input_dtype': 'bf16', 'output_dtype': 'bf16',
            }
        )
        self.add_operator(kv_a_proj)

        # 4. KV_B projection
        kv_b_proj = MatMulOperator(
            name=f"{self.name}_kv_b_proj",
            parallel_params={
                'G': 1, 'M': seqs, 'K': kv_lora_rank,
                'N': heads_per_tp * (qk_nope_dim + v_head_dim),
                'input_dtype': 'bf16', 'output_dtype': 'bf16',
            }
        )
        self.add_operator(kv_b_proj)

        # 5. MHA - 使用 effective_kv_seq (DSA 稀疏)
        mha_op = MHAOperator(
            name=f"{self.name}_mha",
            parallel_params={
                'B': local_batch,  # 使用本地 batch
                'H': heads_per_tp,
                'QS': seq_len,
                'KS': effective_kv_seq,  # DSA 稀疏注意力
                'QD': qk_dim,
                'VD': v_head_dim,
            }
        )
        self.add_operator(mha_op)

        # 6. Output projection
        o_proj = MatMulOperator(
            name=f"{self.name}_o_proj",
            parallel_params={
                'G': 1, 'M': seqs, 'K': heads_per_tp * v_head_dim, 'N': hidden_dim,
                'input_dtype': 'bf16', 'output_dtype': 'bf16',
            }
        )
        self.add_operator(o_proj)

        # 7. AllReduce (if tp > 1)
        if tp > 1:
            dtype_bytes = 2
            comm_size = seqs * hidden_dim * dtype_bytes
            allreduce_op = AllReduceOperator(
                name=f"{self.name}_allreduce",
                parallel_params={
                    'tp': tp, 'comm_size': comm_size, 'comm_protocol': comm_protocol,
                }
            )
            self.add_operator(allreduce_op)


@dataclass
class MLAAbsorbLayer(BaseLayer):
    """
    MLA absorbed - 在线计算 K/V

    用 w_kc/w_vc 替代 kv_b_proj，使用 MQA 而非 MHA

    算子列表:
    - q_a_proj, q_b_proj, kv_a_proj (同 MLALayer)
    - w_kc: G=heads, [seqs, qk_nope] × [qk_nope, kv_lora] (替代 kv_b_proj)
    - w_vc: G=heads, [seqs, v_head] × [v_head, kv_lora]
    - mqa: MQAOperator (不是 MHA)
    - o_proj
    - allgather (if tp > 1 and enable_tp_sp)
    - reducescatter (if tp > 1 and enable_tp_sp)
    - allreduce (if tp > 1)
    """
    name: str = "mla_absorb"
    layer_type: str = "MLAAbsorb"
    config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """构建层内算子"""
        self.comp_ops = []
        self.comm_ops = []
        self.operator_categories = {}
        self._build_operators()

    def _build_operators(self):
        """构建 MLAAbsorb 层的算子"""
        cfg = self.config
        hidden_dim = cfg.get('hidden_dim', 7168)
        num_heads = cfg.get('num_heads', 128)
        qk_nope_dim = cfg.get('qk_nope_dim', 128)
        qk_rope_dim = cfg.get('qk_rope_dim', 64)
        v_head_dim = cfg.get('v_head_dim', 128)
        kv_lora_rank = cfg.get('kv_lora_rank', 512)
        q_lora_rank = cfg.get('q_lora_rank', 1536)
        batch_size = cfg.get('batch_size', 1)  # 全局 batch (对齐 DS_TPU)
        seq_len = cfg.get('seq_len', 1)
        kv_seq_len = cfg.get('kv_seq_len', seq_len)
        tp = cfg.get('tp', 1)
        dp = cfg.get('dp', 1)  # 数据并行度
        comm_protocol = cfg.get('comm_protocol', 1)
        enable_tp_sp = cfg.get('enable_tp_sp', False)

        # 计算本地 batch (对齐 DS_TPU: local_batch = batch_size // dp)
        local_batch = batch_size // dp if dp > 0 else batch_size

        heads_per_tp = num_heads // tp
        seqs = local_batch * seq_len
        qk_dim = qk_nope_dim + qk_rope_dim

        # 1. Q_A projection
        q_a_proj = MatMulOperator(
            name=f"{self.name}_q_a_proj",
            parallel_params={
                'G': 1, 'M': seqs, 'K': hidden_dim, 'N': q_lora_rank,
                'input_dtype': 'bf16', 'output_dtype': 'bf16',
            }
        )
        self.add_operator(q_a_proj)

        # 2. Q_B projection
        q_b_proj = MatMulOperator(
            name=f"{self.name}_q_b_proj",
            parallel_params={
                'G': 1, 'M': seqs, 'K': q_lora_rank, 'N': heads_per_tp * qk_dim,
                'input_dtype': 'bf16', 'output_dtype': 'bf16',
            }
        )
        self.add_operator(q_b_proj)

        # 3. KV_A projection
        kv_a_proj = MatMulOperator(
            name=f"{self.name}_kv_a_proj",
            parallel_params={
                'G': 1, 'M': seqs, 'K': hidden_dim, 'N': kv_lora_rank + qk_rope_dim,
                'input_dtype': 'bf16', 'output_dtype': 'bf16',
            }
        )
        self.add_operator(kv_a_proj)

        # 4. W_KC: 分组矩阵乘 G=heads
        # 在线计算 K: [seqs, kv_lora] × [kv_lora, qk_nope]
        w_kc = MatMulOperator(
            name=f"{self.name}_w_kc",
            parallel_params={
                'G': heads_per_tp,  # 分组
                'M': seqs,
                'K': kv_lora_rank,
                'N': qk_nope_dim,
                'input_dtype': 'bf16',
                'output_dtype': 'bf16',
            }
        )
        self.add_operator(w_kc)

        # 5. W_VC: 分组矩阵乘 G=heads
        # 在线计算 V: [seqs, kv_lora] × [kv_lora, v_head]
        w_vc = MatMulOperator(
            name=f"{self.name}_w_vc",
            parallel_params={
                'G': heads_per_tp,  # 分组
                'M': seqs,
                'K': kv_lora_rank,
                'N': v_head_dim,
                'input_dtype': 'bf16',
                'output_dtype': 'bf16',
            }
        )
        self.add_operator(w_vc)

        # 6. MQA (注意: absorbed 使用 MQA 而非 MHA)
        # MQA 中 QS 是 heads 维度
        mqa_op = MQAOperator(
            name=f"{self.name}_mqa",
            parallel_params={
                'B': seqs,  # batch 是 seqs
                'QS': heads_per_tp,  # QS 是 heads
                'KS': kv_seq_len,
                'QD': qk_dim,
                'VD': v_head_dim,
            }
        )
        self.add_operator(mqa_op)

        # 7. Output projection
        o_proj = MatMulOperator(
            name=f"{self.name}_o_proj",
            parallel_params={
                'G': 1, 'M': seqs, 'K': heads_per_tp * v_head_dim, 'N': hidden_dim,
                'input_dtype': 'bf16', 'output_dtype': 'bf16',
            }
        )
        self.add_operator(o_proj)

        # 8. 通信算子
        if tp > 1:
            dtype_bytes = 2

            # SP 模式需要 AllGather 和 ReduceScatter
            if enable_tp_sp:
                # AllGather: 收集序列数据
                ag_size = seqs * hidden_dim * dtype_bytes
                allgather_op = AllGatherOperator(
                    name=f"{self.name}_allgather",
                    parallel_params={
                        'tp': tp, 'comm_size': ag_size, 'comm_protocol': comm_protocol,
                    }
                )
                self.add_operator(allgather_op)

                # ReduceScatter: 分散序列数据
                rs_size = seqs * hidden_dim * dtype_bytes
                reducescatter_op = ReduceScatterOperator(
                    name=f"{self.name}_reducescatter",
                    parallel_params={
                        'tp': tp, 'comm_size': rs_size, 'comm_protocol': comm_protocol,
                    }
                )
                self.add_operator(reducescatter_op)

            # AllReduce
            ar_size = seqs * hidden_dim * dtype_bytes
            allreduce_op = AllReduceOperator(
                name=f"{self.name}_allreduce",
                parallel_params={
                    'tp': tp, 'comm_size': ar_size, 'comm_protocol': comm_protocol,
                }
            )
            self.add_operator(allreduce_op)


@dataclass
class MLAAbsorbv32Layer(BaseLayer):
    """
    MLA absorbed V3.2 - 结合 absorbed KV 和 DSA

    与 MLAAbsorbLayer 相同，但:
    - 新增参数: topk_index
    - mqa 的 KS = min(kv_seq_len, topk_index)
    - 使用 batch_size_local 而非 seqs
    """
    name: str = "mla_absorb_v32"
    layer_type: str = "MLAAbsorbv32"
    config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """构建层内算子"""
        self.comp_ops = []
        self.comm_ops = []
        self.operator_categories = {}
        self._build_operators()

    def _build_operators(self):
        """构建 MLAAbsorbv32 层的算子"""
        cfg = self.config
        hidden_dim = cfg.get('hidden_dim', 7168)
        num_heads = cfg.get('num_heads', 128)
        qk_nope_dim = cfg.get('qk_nope_dim', 128)
        qk_rope_dim = cfg.get('qk_rope_dim', 64)
        v_head_dim = cfg.get('v_head_dim', 128)
        kv_lora_rank = cfg.get('kv_lora_rank', 512)
        q_lora_rank = cfg.get('q_lora_rank', 1536)
        batch_size = cfg.get('batch_size', 1)  # 全局 batch (对齐 DS_TPU)
        seq_len = cfg.get('seq_len', 1)
        kv_seq_len = cfg.get('kv_seq_len', seq_len)
        topk_index = cfg.get('topk_index', 2048)  # DSA 稀疏注意力
        tp = cfg.get('tp', 1)
        dp = cfg.get('dp', 1)  # 数据并行度
        comm_protocol = cfg.get('comm_protocol', 1)
        enable_tp_sp = cfg.get('enable_tp_sp', False)

        # 计算本地 batch (对齐 DS_TPU: local_batch = batch_size // dp)
        local_batch = batch_size // dp if dp > 0 else batch_size

        heads_per_tp = num_heads // tp
        seqs = local_batch * seq_len
        qk_dim = qk_nope_dim + qk_rope_dim

        # DSA: 实际计算的 KS 是 min(kv_seq_len, topk_index)
        effective_kv_seq = min(kv_seq_len, topk_index)

        # SP 模式: 进一步除以 tp
        if enable_tp_sp:
            batch_size_sp = local_batch // tp
            seqs_local = batch_size_sp * seq_len
        else:
            seqs_local = seqs

        # 1. Q_A projection
        q_a_proj = MatMulOperator(
            name=f"{self.name}_q_a_proj",
            parallel_params={
                'G': 1, 'M': seqs_local, 'K': hidden_dim, 'N': q_lora_rank,
                'input_dtype': 'bf16', 'output_dtype': 'bf16',
            }
        )
        self.add_operator(q_a_proj)

        # 2. Q_B projection
        q_b_proj = MatMulOperator(
            name=f"{self.name}_q_b_proj",
            parallel_params={
                'G': 1, 'M': seqs_local, 'K': q_lora_rank, 'N': heads_per_tp * qk_dim,
                'input_dtype': 'bf16', 'output_dtype': 'bf16',
            }
        )
        self.add_operator(q_b_proj)

        # 3. KV_A projection
        kv_a_proj = MatMulOperator(
            name=f"{self.name}_kv_a_proj",
            parallel_params={
                'G': 1, 'M': seqs_local, 'K': hidden_dim, 'N': kv_lora_rank + qk_rope_dim,
                'input_dtype': 'bf16', 'output_dtype': 'bf16',
            }
        )
        self.add_operator(kv_a_proj)

        # 4. W_KC: 分组矩阵乘
        w_kc = MatMulOperator(
            name=f"{self.name}_w_kc",
            parallel_params={
                'G': heads_per_tp, 'M': seqs_local, 'K': kv_lora_rank, 'N': qk_nope_dim,
                'input_dtype': 'bf16', 'output_dtype': 'bf16',
            }
        )
        self.add_operator(w_kc)

        # 5. W_VC: 分组矩阵乘
        w_vc = MatMulOperator(
            name=f"{self.name}_w_vc",
            parallel_params={
                'G': heads_per_tp, 'M': seqs_local, 'K': kv_lora_rank, 'N': v_head_dim,
                'input_dtype': 'bf16', 'output_dtype': 'bf16',
            }
        )
        self.add_operator(w_vc)

        # 6. MQA - 使用 effective_kv_seq (DSA 稀疏)
        mqa_op = MQAOperator(
            name=f"{self.name}_mqa",
            parallel_params={
                'B': seqs_local,
                'QS': heads_per_tp,
                'KS': effective_kv_seq,  # DSA 稀疏注意力
                'QD': qk_dim,
                'VD': v_head_dim,
            }
        )
        self.add_operator(mqa_op)

        # 7. Output projection
        o_proj = MatMulOperator(
            name=f"{self.name}_o_proj",
            parallel_params={
                'G': 1, 'M': seqs_local, 'K': heads_per_tp * v_head_dim, 'N': hidden_dim,
                'input_dtype': 'bf16', 'output_dtype': 'bf16',
            }
        )
        self.add_operator(o_proj)

        # 8. 通信算子
        if tp > 1:
            dtype_bytes = 2

            if enable_tp_sp:
                # AllGather
                ag_size = seqs_local * hidden_dim * dtype_bytes
                allgather_op = AllGatherOperator(
                    name=f"{self.name}_allgather",
                    parallel_params={
                        'tp': tp, 'comm_size': ag_size, 'comm_protocol': comm_protocol,
                    }
                )
                self.add_operator(allgather_op)

                # ReduceScatter
                rs_size = seqs_local * hidden_dim * dtype_bytes
                reducescatter_op = ReduceScatterOperator(
                    name=f"{self.name}_reducescatter",
                    parallel_params={
                        'tp': tp, 'comm_size': rs_size, 'comm_protocol': comm_protocol,
                    }
                )
                self.add_operator(reducescatter_op)

            # AllReduce
            ar_size = seqs_local * hidden_dim * dtype_bytes
            allreduce_op = AllReduceOperator(
                name=f"{self.name}_allreduce",
                parallel_params={
                    'tp': tp, 'comm_size': ar_size, 'comm_protocol': comm_protocol,
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
        - batch_size: int, 全局批次大小 (对齐 DS_TPU)
        - seq_len: int, 序列长度
        - tp: int, 张量并行度
        - dp: int, 数据并行度 (用于计算 local_batch)
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
        batch_size = cfg.get('batch_size', 1)  # 全局 batch (对齐 DS_TPU)
        seq_len = cfg.get('seq_len', 1)
        tp = cfg.get('tp', 1)
        dp = cfg.get('dp', 1)  # 数据并行度
        comm_protocol = cfg.get('comm_protocol', 1)

        # 计算本地 batch (对齐 DS_TPU: local_batch = batch_size // dp)
        local_batch = batch_size // dp if dp > 0 else batch_size

        heads_per_tp = num_heads // tp
        kv_heads_per_tp = num_kv_heads // tp
        tokens = local_batch * seq_len

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
