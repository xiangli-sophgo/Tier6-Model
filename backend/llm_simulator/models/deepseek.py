"""
DeepSeek 模型定义

支持 DeepSeek V3/V3.2 架构:
- MLA (Multi-head Latent Attention)
- MoE (Mixture of Experts)
"""

from dataclasses import dataclass, field
from typing import Dict, Any

from .base import BaseModel
from ..layers import (
    EmbeddingLayer,
    MLALayer,
    MLPLayer,
    MoELayer,
    LMHeadLayer,
)


@dataclass
class DeepSeekModel(BaseModel):
    """
    DeepSeek V3/V3.2 模型

    架构:
    - Embedding (1层)
    - MLA + Dense MLP (n_dense_layers 层)
    - MLA + MoE (n_moe_layers 层)
    - LMHead (1层)

    config 必须包含:
        # 模型结构
        - hidden_dim: int, 隐藏维度 (7168)
        - inter_dim: int, FFN 中间维度 (18432)
        - vocab_size: int, 词表大小 (151936)
        - n_layers: int, 总层数 (61)
        - n_dense_layers: int, Dense 层数 (3)
        - n_moe_layers: int, MoE 层数 (58)
        - num_heads: int, 注意力头数 (128)
        - head_dim: int, 每头维度 (128)
        - kv_lora_rank: int, KV 压缩维度 (512)
        - q_lora_rank: int, Q 压缩维度 (1536)

        # MoE 配置
        - num_experts: int, 专家数 (256)
        - num_activated_experts: int, 激活专家数 (8)
        - num_shared_experts: int, 共享专家数 (1)
        - expert_inter_dim: int, 专家 FFN 维度 (2048)

        # 部署配置
        - batch_size: int, 批次大小
        - seq_len: int, 序列长度
        - tp: int, 张量并行度
        - moe_tp: int, MoE 张量并行度
        - ep: int, 专家并行度
        - comm_protocol: int, 通信协议
        - is_prefill: bool, 是否为 prefill 阶段
    """
    name: str = "deepseek-v3"
    model_type: str = "DeepSeek"
    config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """构建模型"""
        self.layers = []
        self.layer_counts = {}
        self.operator_map = {}
        self.operator_types = set()
        self._build_model()

    def _build_model(self):
        """构建 DeepSeek 模型的所有层"""
        cfg = self.config

        # 模型结构参数
        hidden_dim = cfg.get('hidden_dim', 7168)
        inter_dim = cfg.get('inter_dim', 18432)
        vocab_size = cfg.get('vocab_size', 151936)
        n_layers = cfg.get('n_layers', 61)
        n_dense_layers = cfg.get('n_dense_layers', 3)
        n_moe_layers = cfg.get('n_moe_layers', 58)
        num_heads = cfg.get('num_heads', 128)
        head_dim = cfg.get('head_dim', 128)
        kv_lora_rank = cfg.get('kv_lora_rank', 512)
        q_lora_rank = cfg.get('q_lora_rank', 1536)

        # MoE 参数
        num_experts = cfg.get('num_experts', 256)
        num_activated = cfg.get('num_activated_experts', 8)
        num_shared = cfg.get('num_shared_experts', 1)
        expert_inter_dim = cfg.get('expert_inter_dim', 2048)

        # 部署参数
        batch_size = cfg.get('batch_size', 1)
        seq_len = cfg.get('seq_len', 1)
        tp = cfg.get('tp', 1)
        moe_tp = cfg.get('moe_tp', 1)
        ep = cfg.get('ep', 1)
        comm_protocol = cfg.get('comm_protocol', 1)
        is_prefill = cfg.get('is_prefill', False)

        # 1. Embedding 层
        embedding_layer = EmbeddingLayer(
            name="embedding",
            config={
                'vocab_size': vocab_size,
                'hidden_dim': hidden_dim,
                'batch_size': batch_size,
                'seq_len': seq_len,
                'tp': tp,
                'comm_protocol': comm_protocol,
            }
        )
        self.add_layer(embedding_layer, count=1)

        # 2. MLA 层 (用于所有 transformer 层)
        mla_layer = MLALayer(
            name="mla",
            config={
                'hidden_dim': hidden_dim,
                'num_heads': num_heads,
                'head_dim': head_dim,
                'kv_lora_rank': kv_lora_rank,
                'q_lora_rank': q_lora_rank,
                'batch_size': batch_size,
                'seq_len': seq_len,
                'tp': tp,
                'comm_protocol': comm_protocol,
                'is_prefill': is_prefill,
            }
        )
        self.add_layer(mla_layer, count=n_layers)

        # 3. Dense MLP 层 (前 n_dense_layers 层)
        if n_dense_layers > 0:
            dense_mlp_layer = MLPLayer(
                name="dense_mlp",
                config={
                    'hidden_dim': hidden_dim,
                    'inter_dim': inter_dim,
                    'batch_size': batch_size,
                    'seq_len': seq_len,
                    'tp': tp,
                    'comm_protocol': comm_protocol,
                }
            )
            self.add_layer(dense_mlp_layer, count=n_dense_layers)

        # 4. MoE 层 (后 n_moe_layers 层)
        if n_moe_layers > 0:
            moe_layer = MoELayer(
                name="moe",
                config={
                    'hidden_dim': hidden_dim,
                    'inter_dim': expert_inter_dim,
                    'num_experts': num_experts,
                    'num_activated_experts': num_activated,
                    'num_shared_experts': num_shared,
                    'batch_size': batch_size,
                    'seq_len': seq_len,
                    'tp': tp,
                    'moe_tp': moe_tp,
                    'ep': ep,
                    'comm_protocol': comm_protocol,
                    'is_prefill': is_prefill,
                }
            )
            self.add_layer(moe_layer, count=n_moe_layers)

        # 5. LMHead 层
        lmhead_layer = LMHeadLayer(
            name="lmhead",
            config={
                'hidden_dim': hidden_dim,
                'vocab_size': vocab_size,
                'batch_size': batch_size,
                'seq_len': seq_len,
                'tp': tp,
                'comm_protocol': comm_protocol,
            }
        )
        self.add_layer(lmhead_layer, count=1)


def create_deepseek_v3(batch_size: int = 1, seq_len: int = 1,
                       tp: int = 1, moe_tp: int = 1, ep: int = 1,
                       comm_protocol: int = 1, is_prefill: bool = False) -> DeepSeekModel:
    """
    创建 DeepSeek V3 模型实例

    DeepSeek V3 配置:
    - 61 层 (3 Dense + 58 MoE)
    - 256 个专家，激活 8 个
    - MLA 注意力
    """
    return DeepSeekModel(
        name="deepseek-v3",
        config={
            # 模型结构
            'hidden_dim': 7168,
            'inter_dim': 18432,
            'vocab_size': 151936,
            'n_layers': 61,
            'n_dense_layers': 3,
            'n_moe_layers': 58,
            'num_heads': 128,
            'head_dim': 128,
            'kv_lora_rank': 512,
            'q_lora_rank': 1536,
            # MoE
            'num_experts': 256,
            'num_activated_experts': 8,
            'num_shared_experts': 1,
            'expert_inter_dim': 2048,
            # 部署
            'batch_size': batch_size,
            'seq_len': seq_len,
            'tp': tp,
            'moe_tp': moe_tp,
            'ep': ep,
            'comm_protocol': comm_protocol,
            'is_prefill': is_prefill,
        }
    )
