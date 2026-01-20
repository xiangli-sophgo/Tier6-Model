"""
Llama 模型定义

支持标准 Llama 架构:
- MHA/GQA (Multi-Head Attention / Grouped Query Attention)
- Dense FFN
"""

from dataclasses import dataclass, field
from typing import Dict, Any

from .base import BaseModel
from ..layers import (
    EmbeddingLayer,
    MHALayer,
    MLPLayer,
    LMHeadLayer,
)


@dataclass
class LlamaModel(BaseModel):
    """
    Llama 模型

    架构:
    - Embedding (1层)
    - MHA/GQA + Dense FFN (n_layers 层)
    - LMHead (1层)

    config 必须包含:
        # 模型结构
        - hidden_dim: int, 隐藏维度
        - inter_dim: int, FFN 中间维度
        - vocab_size: int, 词表大小
        - n_layers: int, 层数
        - num_heads: int, 注意力头数
        - num_kv_heads: int, KV 头数 (GQA)
        - head_dim: int, 每头维度

        # 部署配置
        - batch_size: int, 批次大小
        - seq_len: int, 序列长度
        - tp: int, 张量并行度
        - comm_protocol: int, 通信协议
    """
    name: str = "llama"
    model_type: str = "Llama"
    config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """构建模型"""
        self.layers = []
        self.layer_counts = {}
        self.operator_map = {}
        self.operator_types = set()
        self._build_model()

    def _build_model(self):
        """构建 Llama 模型的所有层"""
        cfg = self.config

        # 模型结构参数
        hidden_dim = cfg.get('hidden_dim', 4096)
        inter_dim = cfg.get('inter_dim', 11008)
        vocab_size = cfg.get('vocab_size', 32000)
        n_layers = cfg.get('n_layers', 32)
        num_heads = cfg.get('num_heads', 32)
        num_kv_heads = cfg.get('num_kv_heads', num_heads)
        head_dim = cfg.get('head_dim', 128)

        # 部署参数
        batch_size = cfg.get('batch_size', 1)
        seq_len = cfg.get('seq_len', 1)
        tp = cfg.get('tp', 1)
        comm_protocol = cfg.get('comm_protocol', 1)

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

        # 2. MHA 层
        mha_layer = MHALayer(
            name="mha",
            config={
                'hidden_dim': hidden_dim,
                'num_heads': num_heads,
                'num_kv_heads': num_kv_heads,
                'head_dim': head_dim,
                'batch_size': batch_size,
                'seq_len': seq_len,
                'tp': tp,
                'comm_protocol': comm_protocol,
            }
        )
        self.add_layer(mha_layer, count=n_layers)

        # 3. Dense MLP 层
        mlp_layer = MLPLayer(
            name="mlp",
            config={
                'hidden_dim': hidden_dim,
                'inter_dim': inter_dim,
                'batch_size': batch_size,
                'seq_len': seq_len,
                'tp': tp,
                'comm_protocol': comm_protocol,
            }
        )
        self.add_layer(mlp_layer, count=n_layers)

        # 4. LMHead 层
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


def create_llama_7b(batch_size: int = 1, seq_len: int = 1,
                    tp: int = 1, comm_protocol: int = 1) -> LlamaModel:
    """创建 Llama 7B 模型实例"""
    return LlamaModel(
        name="llama-7b",
        config={
            'hidden_dim': 4096,
            'inter_dim': 11008,
            'vocab_size': 32000,
            'n_layers': 32,
            'num_heads': 32,
            'num_kv_heads': 32,
            'head_dim': 128,
            'batch_size': batch_size,
            'seq_len': seq_len,
            'tp': tp,
            'comm_protocol': comm_protocol,
        }
    )


def create_llama_70b(batch_size: int = 1, seq_len: int = 1,
                     tp: int = 1, comm_protocol: int = 1) -> LlamaModel:
    """创建 Llama 70B 模型实例 (GQA)"""
    return LlamaModel(
        name="llama-70b",
        config={
            'hidden_dim': 8192,
            'inter_dim': 28672,
            'vocab_size': 32000,
            'n_layers': 80,
            'num_heads': 64,
            'num_kv_heads': 8,  # GQA
            'head_dim': 128,
            'batch_size': batch_size,
            'seq_len': seq_len,
            'tp': tp,
            'comm_protocol': comm_protocol,
        }
    )
