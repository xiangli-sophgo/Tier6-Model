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
        # 必需字段检查
        if "name" not in data:
            raise ValueError("Missing required field 'name' in ModelMetadata")
        if "hidden_size" not in data or data["hidden_size"] <= 0:
            raise ValueError("Missing or invalid 'hidden_size' in ModelMetadata")
        if "num_layers" not in data or data["num_layers"] <= 0:
            raise ValueError("Missing or invalid 'num_layers' in ModelMetadata")
        if "num_heads" not in data or data["num_heads"] <= 0:
            raise ValueError("Missing or invalid 'num_heads' in ModelMetadata")

        # dtype 可选，默认 fp16
        dtype = data.get("dtype", "fp16")
        if isinstance(dtype, str):
            dtype = DataType.from_string(dtype)

        return cls(
            name=data["name"],
            dtype=dtype,
            hidden_size=data["hidden_size"],
            num_layers=data["num_layers"],
            num_heads=data["num_heads"],
            num_kv_heads=data.get("num_kv_heads"),  # 可选，None 表示 MHA
            intermediate_size=data.get("intermediate_size"),  # 可选，可从 hidden_size 推导
            vocab_size=data.get("vocab_size"),  # 可选
            seq_len=data.get("seq_len"),  # 可选
            batch=data.get("batch"),  # 可选
            tags=data.get("tags", {}),  # 可选，空字典
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
        """从字典创建

        注意: enabled=False 时其他字段可选；enabled=True 时必须提供所有参数
        """
        enabled = data.get("enabled", False)
        if not enabled:
            # MLA 未启用，使用 dataclass 默认值
            return cls(enabled=False)

        # MLA 启用时，所有参数必须显式提供
        required_fields = ["kv_lora_rank", "q_lora_rank", "qk_rope_dim", "qk_nope_dim", "v_head_dim"]
        missing = [f for f in required_fields if f not in data]
        if missing:
            raise ValueError(f"MLA enabled but missing required fields: {missing}")

        return cls(
            enabled=True,
            kv_lora_rank=data["kv_lora_rank"],
            q_lora_rank=data["q_lora_rank"],
            qk_rope_dim=data["qk_rope_dim"],
            qk_nope_dim=data["qk_nope_dim"],
            v_head_dim=data["v_head_dim"],
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
        """从字典创建

        注意: enabled=False 时其他字段可选；enabled=True 时必须提供所有参数
        """
        enabled = data.get("enabled", False)
        if not enabled:
            # MoE 未启用，使用 dataclass 默认值
            return cls(enabled=False)

        # MoE 启用时，所有参数必须显式提供
        required_fields = ["num_experts", "num_shared_experts", "experts_per_token"]
        missing = [f for f in required_fields if f not in data]
        if missing:
            raise ValueError(f"MoE enabled but missing required fields: {missing}")

        return cls(
            enabled=True,
            num_experts=data["num_experts"],
            num_shared_experts=data["num_shared_experts"],
            experts_per_token=data["experts_per_token"],
            router_topk_policy=data.get("router_topk_policy", "greedy"),  # 路由策略可选
        )
