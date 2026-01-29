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


# ==================== MLA 基类（共用配置提取和算子构建逻辑）====================

@dataclass
class MLALayerBase(BaseLayer):
    """
    MLA 层基类 - 提取公共配置和算子构建逻辑

    子类需要实现:
    - _build_kv_projections(): 构建 KV 投影算子
    - _build_attention(): 构建注意力算子
    """
    name: str = "mla_base"
    layer_type: str = "MLABase"
    config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """构建层内算子"""
        self.comp_ops = []
        self.comm_ops = []
        self.operator_categories = {}
        self._extract_config()
        self._build_operators()

    def _extract_config(self):
        """提取公共配置参数"""
        cfg = self.config
        self.hidden_dim = cfg.get('hidden_dim', 7168)
        self.num_heads = cfg.get('num_heads', 128)
        self.qk_nope_dim = cfg.get('qk_nope_dim', 128)
        self.qk_rope_dim = cfg.get('qk_rope_dim', 64)
        self.v_head_dim = cfg.get('v_head_dim', 128)
        self.kv_lora_rank = cfg.get('kv_lora_rank', 512)
        self.q_lora_rank = cfg.get('q_lora_rank', 1536)
        self.batch_size = cfg.get('batch_size', 1)
        self.seq_len = cfg.get('seq_len', 1)
        self.kv_seq_len = cfg.get('kv_seq_len', self.seq_len)
        self.topk_index = cfg.get('topk_index', 2048)  # DSA 稀疏注意力
        self.tp = cfg.get('tp', 1)
        self.dp = cfg.get('dp', 1)
        self.comm_protocol = cfg.get('comm_protocol', 1)
        self.enable_tp_sp = cfg.get('enable_tp_sp', False)

        # 计算派生值
        self.local_batch = self.batch_size // self.dp if self.dp > 0 else self.batch_size
        self.heads_per_tp = self.num_heads // self.tp
        self.seqs = self.local_batch * self.seq_len
        self.qk_dim = self.qk_nope_dim + self.qk_rope_dim

    def _build_operators(self):
        """模板方法：构建所有算子"""
        self._build_q_projections()
        self._build_kv_a_projection()
        self._build_kv_projections()  # 子类实现
        self._build_attention()       # 子类实现
        self._build_output_projection()
        self._build_communication()

    def _build_q_projections(self):
        """构建 Q 投影算子（所有 MLA 变体共用）"""
        # Q_A projection
        q_a_proj = MatMulOperator(
            name=f"{self.name}_q_a_proj",
            parallel_params={
                'G': 1, 'M': self._get_seqs(), 'K': self.hidden_dim, 'N': self.q_lora_rank,
                'input_dtype': 'bf16', 'output_dtype': 'bf16',
            }
        )
        self.add_operator(q_a_proj)

        # Q_B projection
        q_b_proj = MatMulOperator(
            name=f"{self.name}_q_b_proj",
            parallel_params={
                'G': 1, 'M': self._get_seqs(), 'K': self.q_lora_rank, 'N': self.heads_per_tp * self.qk_dim,
                'input_dtype': 'bf16', 'output_dtype': 'bf16',
            }
        )
        self.add_operator(q_b_proj)

    def _build_kv_a_projection(self):
        """构建 KV_A 投影算子（所有 MLA 变体共用）"""
        kv_a_proj = MatMulOperator(
            name=f"{self.name}_kv_a_proj",
            parallel_params={
                'G': 1, 'M': self._get_seqs(), 'K': self.hidden_dim, 'N': self.kv_lora_rank + self.qk_rope_dim,
                'input_dtype': 'bf16', 'output_dtype': 'bf16',
            }
        )
        self.add_operator(kv_a_proj)

    def _build_kv_projections(self):
        """构建 KV 投影算子（子类实现）"""
        raise NotImplementedError("子类需要实现 _build_kv_projections()")

    def _build_attention(self):
        """构建注意力算子（子类实现）"""
        raise NotImplementedError("子类需要实现 _build_attention()")

    def _build_output_projection(self):
        """构建输出投影算子（所有 MLA 变体共用）"""
        o_proj = MatMulOperator(
            name=f"{self.name}_o_proj",
            parallel_params={
                'G': 1, 'M': self._get_seqs(), 'K': self.heads_per_tp * self.v_head_dim, 'N': self.hidden_dim,
                'input_dtype': 'bf16', 'output_dtype': 'bf16',
            }
        )
        self.add_operator(o_proj)

    def _build_communication(self):
        """构建通信算子（所有 MLA 变体共用）"""
        if self.tp > 1:
            dtype_bytes = 2
            comm_size = self._get_seqs() * self.hidden_dim * dtype_bytes

            # SP 模式需要 AllGather 和 ReduceScatter
            if self.enable_tp_sp:
                self._build_sp_communication(comm_size)

            # AllReduce
            allreduce_op = AllReduceOperator(
                name=f"{self.name}_allreduce",
                parallel_params={
                    'tp': self.tp, 'comm_size': comm_size, 'comm_protocol': self.comm_protocol,
                }
            )
            self.add_operator(allreduce_op)

    def _build_sp_communication(self, comm_size: int):
        """构建 SP 通信算子（可选，仅 Absorb 变体使用）"""
        allgather_op = AllGatherOperator(
            name=f"{self.name}_allgather",
            parallel_params={
                'tp': self.tp, 'comm_size': comm_size, 'comm_protocol': self.comm_protocol,
            }
        )
        self.add_operator(allgather_op)

        reducescatter_op = ReduceScatterOperator(
            name=f"{self.name}_reducescatter",
            parallel_params={
                'tp': self.tp, 'comm_size': comm_size, 'comm_protocol': self.comm_protocol,
            }
        )
        self.add_operator(reducescatter_op)

    def _get_seqs(self) -> int:
        """获取序列数（子类可覆盖，SP 模式下除以 tp）"""
        return self.seqs

    def _get_effective_kv_seq(self) -> int:
        """获取有效 KV 序列长度（DSA 变体覆盖）"""
        return self.kv_seq_len


# ==================== MLA 具体实现 ====================

@dataclass
class MLALayer(MLALayerBase):
    """
    MLA 基础版 - 对齐 DS_TPU MLA

    使用 kv_b_proj 解压缩 KV Cache
    """
    name: str = "mla"
    layer_type: str = "MLA"

    def _build_kv_projections(self):
        """使用 kv_b_proj 解压缩"""
        kv_b_proj = MatMulOperator(
            name=f"{self.name}_kv_b_proj",
            parallel_params={
                'G': 1,
                'M': self.seqs,
                'K': self.kv_lora_rank,
                'N': self.heads_per_tp * (self.qk_nope_dim + self.v_head_dim),
                'input_dtype': 'bf16',
                'output_dtype': 'bf16',
            }
        )
        self.add_operator(kv_b_proj)

    def _build_attention(self):
        """使用 MHA"""
        mha_op = MHAOperator(
            name=f"{self.name}_mha",
            parallel_params={
                'B': self.local_batch,
                'H': self.heads_per_tp,
                'QS': self.seq_len,
                'KS': self._get_effective_kv_seq(),
                'QD': self.qk_dim,
                'VD': self.v_head_dim,
            }
        )
        self.add_operator(mha_op)


@dataclass
class MLAv32Layer(MLALayer):
    """
    MLA V3.2 - DSA 稀疏注意力

    与 MLALayer 相同，但 mha 的 KS = min(kv_seq_len, topk_index)
    """
    name: str = "mla_v32"
    layer_type: str = "MLAv32"

    def _get_effective_kv_seq(self) -> int:
        """DSA: 使用稀疏注意力"""
        return min(self.kv_seq_len, self.topk_index)


@dataclass
class MLAAbsorbLayer(MLALayerBase):
    """
    MLA absorbed - 在线计算 K/V

    用 w_kc/w_vc 替代 kv_b_proj，使用 MQA 而非 MHA
    """
    name: str = "mla_absorb"
    layer_type: str = "MLAAbsorb"

    def _build_kv_projections(self):
        """使用 w_kc/w_vc 在线计算 K/V"""
        # W_KC: 分组矩阵乘 G=heads
        w_kc = MatMulOperator(
            name=f"{self.name}_w_kc",
            parallel_params={
                'G': self.heads_per_tp,
                'M': self._get_seqs(),
                'K': self.kv_lora_rank,
                'N': self.qk_nope_dim,
                'input_dtype': 'bf16',
                'output_dtype': 'bf16',
            }
        )
        self.add_operator(w_kc)

        # W_VC: 分组矩阵乘 G=heads
        w_vc = MatMulOperator(
            name=f"{self.name}_w_vc",
            parallel_params={
                'G': self.heads_per_tp,
                'M': self._get_seqs(),
                'K': self.kv_lora_rank,
                'N': self.v_head_dim,
                'input_dtype': 'bf16',
                'output_dtype': 'bf16',
            }
        )
        self.add_operator(w_vc)

    def _build_attention(self):
        """使用 MQA"""
        mqa_op = MQAOperator(
            name=f"{self.name}_mqa",
            parallel_params={
                'B': self._get_seqs(),
                'QS': self.heads_per_tp,
                'KS': self._get_effective_kv_seq(),
                'QD': self.qk_dim,
                'VD': self.v_head_dim,
            }
        )
        self.add_operator(mqa_op)


@dataclass
class MLAAbsorbv32Layer(MLAAbsorbLayer):
    """
    MLA absorbed V3.2 - 结合 absorbed KV 和 DSA

    与 MLAAbsorbLayer 相同，但支持 DSA 和 SP 模式
    """
    name: str = "mla_absorb_v32"
    layer_type: str = "MLAAbsorbv32"

    def _get_seqs(self) -> int:
        """SP 模式: 序列数除以 tp"""
        if self.enable_tp_sp:
            batch_size_sp = self.local_batch // self.tp
            return batch_size_sp * self.seq_len
        return self.seqs

    def _get_effective_kv_seq(self) -> int:
        """DSA: 使用稀疏注意力"""
        return min(self.kv_seq_len, self.topk_index)


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
