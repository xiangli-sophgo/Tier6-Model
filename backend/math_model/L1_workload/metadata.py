"""模型元数据模块

定义 ModelMetadata 数据结构。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from math_model.core.types import DataType


@dataclass
class ModelMetadata:
    """模型元数据

    Attributes:
        name: 模型名称
        dtype: 默认数据类型
        hidden_size: 隐藏层大小
        num_layers: 层数
        num_heads: 注意力头数
        num_kv_heads: KV 头数（GQA/MQA）
        intermediate_size: FFN 中间层大小
        vocab_size: 词表大小
        seq_len: 序列长度
        batch: 批次大小
        tags: 扩展字段
    """

    name: str
    dtype: DataType
    hidden_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int | None = None
    intermediate_size: int | None = None
    vocab_size: int | None = None
    seq_len: int | None = None
    batch: int | None = None
    tags: dict[str, str] = field(default_factory=dict)

    @property
    def head_dim(self) -> int:
        """每个注意力头的维度"""
        return self.hidden_size // self.num_heads if self.num_heads > 0 else 0

    @property
    def effective_kv_heads(self) -> int:
        """有效 KV 头数"""
        return self.num_kv_heads if self.num_kv_heads is not None else self.num_heads

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelMetadata":
        """从字典创建"""
        dtype = data.get("dtype", "fp16")
        if isinstance(dtype, str):
            dtype = DataType.from_string(dtype)

        return cls(
            name=data.get("name", "unknown"),
            dtype=dtype,
            hidden_size=data.get("hidden_size", 0),
            num_layers=data.get("num_layers", 0),
            num_heads=data.get("num_heads", 0),
            num_kv_heads=data.get("num_kv_heads"),
            intermediate_size=data.get("intermediate_size"),
            vocab_size=data.get("vocab_size"),
            seq_len=data.get("seq_len"),
            batch=data.get("batch"),
            tags=data.get("tags", {}),
        )


@dataclass
class MLAConfig:
    """MLA (Multi-head Latent Attention) 配置

    Attributes:
        enabled: 是否启用 MLA
        kv_lora_rank: KV 压缩秩
        q_lora_rank: Q LoRA 秩
        qk_rope_dim: RoPE 维度
        qk_nope_dim: 非 RoPE 维度
        v_head_dim: V 头维度
    """

    enabled: bool = False
    kv_lora_rank: int = 512
    q_lora_rank: int = 1536
    qk_rope_dim: int = 64
    qk_nope_dim: int = 128
    v_head_dim: int = 128

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MLAConfig":
        """从字典创建"""
        return cls(
            enabled=data.get("enabled", False),
            kv_lora_rank=data.get("kv_lora_rank", 512),
            q_lora_rank=data.get("q_lora_rank", 1536),
            qk_rope_dim=data.get("qk_rope_dim", 64),
            qk_nope_dim=data.get("qk_nope_dim", 128),
            v_head_dim=data.get("v_head_dim", 128),
        )


@dataclass
class MoEConfig:
    """MoE (Mixture of Experts) 配置

    Attributes:
        enabled: 是否启用 MoE
        num_experts: 专家总数
        num_shared_experts: 共享专家数
        experts_per_token: 每个 token 激活的专家数
        router_topk_policy: 路由策略
    """

    enabled: bool = False
    num_experts: int = 256
    num_shared_experts: int = 1
    experts_per_token: int = 8
    router_topk_policy: str = "greedy"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MoEConfig":
        """从字典创建"""
        return cls(
            enabled=data.get("enabled", False),
            num_experts=data.get("num_experts", 256),
            num_shared_experts=data.get("num_shared_experts", 1),
            experts_per_token=data.get("experts_per_token", 8),
            router_topk_policy=data.get("router_topk_policy", "greedy"),
        )
