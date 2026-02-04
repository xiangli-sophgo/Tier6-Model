"""内存分析模块

生成内存占用分解报告。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tier6.L1_workload.ir import Model
    from tier6.L1_workload.metadata import ModelMetadata
    from tier6.L3_mapping.protocols import ParallelismConfig


@dataclass
class MemoryBreakdown:
    """内存占用分解

    Attributes:
        total_bytes: 总内存占用 (bytes)
        weights_bytes: 权重内存 (bytes)
        kv_cache_bytes: KV Cache 内存 (bytes)
        activations_bytes: 激活内存 (bytes)
        optimizer_bytes: 优化器状态 (bytes, 训练时)
        gradient_bytes: 梯度内存 (bytes, 训练时)
        temp_bytes: 临时缓冲区 (bytes)
        per_layer: 每层内存占用
    """

    total_bytes: int = 0
    weights_bytes: int = 0
    kv_cache_bytes: int = 0
    activations_bytes: int = 0
    optimizer_bytes: int = 0
    gradient_bytes: int = 0
    temp_bytes: int = 0
    per_layer: dict[str, dict[str, int]] = field(default_factory=dict)

    @property
    def total_gb(self) -> float:
        """总内存 (GB)"""
        return self.total_bytes / (1024**3)

    @property
    def weights_gb(self) -> float:
        """权重内存 (GB)"""
        return self.weights_bytes / (1024**3)

    @property
    def kv_cache_gb(self) -> float:
        """KV Cache 内存 (GB)"""
        return self.kv_cache_bytes / (1024**3)

    @property
    def activations_gb(self) -> float:
        """激活内存 (GB)"""
        return self.activations_bytes / (1024**3)

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "totalBytes": self.total_bytes,
            "totalGb": self.total_gb,
            "weights": {
                "bytes": self.weights_bytes,
                "gb": self.weights_gb,
                "percentage": self.weights_bytes / self.total_bytes * 100 if self.total_bytes > 0 else 0,
            },
            "kvCache": {
                "bytes": self.kv_cache_bytes,
                "gb": self.kv_cache_gb,
                "percentage": self.kv_cache_bytes / self.total_bytes * 100 if self.total_bytes > 0 else 0,
            },
            "activations": {
                "bytes": self.activations_bytes,
                "gb": self.activations_gb,
                "percentage": self.activations_bytes / self.total_bytes * 100 if self.total_bytes > 0 else 0,
            },
            "optimizer": {
                "bytes": self.optimizer_bytes,
                "gb": self.optimizer_bytes / (1024**3),
            },
            "gradient": {
                "bytes": self.gradient_bytes,
                "gb": self.gradient_bytes / (1024**3),
            },
            "temp": {
                "bytes": self.temp_bytes,
                "gb": self.temp_bytes / (1024**3),
            },
        }


class MemoryAnalyzer:
    """内存分析器

    计算模型在推理/训练时的内存占用。
    """

    def __init__(self, dtype_bytes: int = 2) -> None:
        """初始化

        Args:
            dtype_bytes: 数据类型字节数 (默认 FP16 = 2)
        """
        self.dtype_bytes = dtype_bytes

    def calculate_weights_memory(
        self,
        hidden_size: int,
        num_layers: int,
        intermediate_size: int,
        vocab_size: int,
        num_kv_heads: int | None = None,
        num_heads: int = 1,
        tp_degree: int = 1,
    ) -> int:
        """计算权重内存

        Args:
            hidden_size: 隐藏层大小
            num_layers: 层数
            intermediate_size: FFN 中间层大小
            vocab_size: 词表大小
            num_kv_heads: KV 头数 (GQA)
            num_heads: 注意力头数
            tp_degree: TP 并行度

        Returns:
            int: 权重内存 (bytes)
        """
        head_dim = hidden_size // num_heads if num_heads > 0 else hidden_size
        effective_kv_heads = num_kv_heads if num_kv_heads else num_heads

        # Attention 权重: Q, K, V, O projections
        # Q: hidden_size -> hidden_size
        # K: hidden_size -> kv_heads * head_dim
        # V: hidden_size -> kv_heads * head_dim
        # O: hidden_size -> hidden_size
        attn_weights_per_layer = (
            hidden_size * hidden_size  # Q
            + hidden_size * effective_kv_heads * head_dim  # K
            + hidden_size * effective_kv_heads * head_dim  # V
            + hidden_size * hidden_size  # O
        )

        # FFN 权重: gate, up, down
        ffn_weights_per_layer = (
            hidden_size * intermediate_size  # gate
            + hidden_size * intermediate_size  # up
            + intermediate_size * hidden_size  # down
        )

        # LayerNorm 权重
        ln_weights_per_layer = hidden_size * 2  # 2 LayerNorms

        # 每层总权重
        weights_per_layer = (
            attn_weights_per_layer + ffn_weights_per_layer + ln_weights_per_layer
        )

        # 所有层 + Embedding + LM Head
        embedding_weights = vocab_size * hidden_size
        lm_head_weights = hidden_size * vocab_size  # 通常与 embedding 共享

        total_params = num_layers * weights_per_layer + embedding_weights

        # TP 切分
        return (total_params // tp_degree) * self.dtype_bytes

    def calculate_kv_cache_memory(
        self,
        batch_size: int,
        seq_len: int,
        num_layers: int,
        hidden_size: int,
        num_kv_heads: int | None = None,
        num_heads: int = 1,
        tp_degree: int = 1,
    ) -> int:
        """计算 KV Cache 内存

        Args:
            batch_size: 批次大小
            seq_len: 序列长度
            num_layers: 层数
            hidden_size: 隐藏层大小
            num_kv_heads: KV 头数 (GQA)
            num_heads: 注意力头数
            tp_degree: TP 并行度

        Returns:
            int: KV Cache 内存 (bytes)
        """
        head_dim = hidden_size // num_heads if num_heads > 0 else hidden_size
        effective_kv_heads = num_kv_heads if num_kv_heads else num_heads

        # KV Cache 大小: 2 (K+V) × batch × seq × kv_heads × head_dim × layers
        kv_cache_per_layer = (
            2  # K and V
            * batch_size
            * seq_len
            * (effective_kv_heads // tp_degree)
            * head_dim
        )

        return num_layers * kv_cache_per_layer * self.dtype_bytes

    def calculate_activation_memory(
        self,
        batch_size: int,
        seq_len: int,
        hidden_size: int,
        intermediate_size: int,
        num_layers: int,
        tp_degree: int = 1,
    ) -> int:
        """计算激活内存

        Args:
            batch_size: 批次大小
            seq_len: 序列长度
            hidden_size: 隐藏层大小
            intermediate_size: FFN 中间层大小
            num_layers: 层数
            tp_degree: TP 并行度

        Returns:
            int: 激活内存 (bytes)
        """
        # 主要激活：hidden states
        hidden_activation = batch_size * seq_len * hidden_size

        # FFN 激活 (最大)
        ffn_activation = batch_size * seq_len * (intermediate_size // tp_degree)

        # Attention 激活
        attn_activation = batch_size * seq_len * hidden_size

        # 取最大激活 (流水线重用)
        peak_activation = max(hidden_activation, ffn_activation, attn_activation)

        return peak_activation * self.dtype_bytes

    def analyze(
        self,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        intermediate_size: int,
        vocab_size: int,
        batch_size: int = 1,
        seq_len: int = 1024,
        num_kv_heads: int | None = None,
        tp_degree: int = 1,
    ) -> MemoryBreakdown:
        """分析内存占用

        Args:
            hidden_size: 隐藏层大小
            num_layers: 层数
            num_heads: 注意力头数
            intermediate_size: FFN 中间层大小
            vocab_size: 词表大小
            batch_size: 批次大小
            seq_len: 序列长度
            num_kv_heads: KV 头数
            tp_degree: TP 并行度

        Returns:
            MemoryBreakdown: 内存分解
        """
        weights = self.calculate_weights_memory(
            hidden_size=hidden_size,
            num_layers=num_layers,
            intermediate_size=intermediate_size,
            vocab_size=vocab_size,
            num_kv_heads=num_kv_heads,
            num_heads=num_heads,
            tp_degree=tp_degree,
        )

        kv_cache = self.calculate_kv_cache_memory(
            batch_size=batch_size,
            seq_len=seq_len,
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_kv_heads=num_kv_heads,
            num_heads=num_heads,
            tp_degree=tp_degree,
        )

        activations = self.calculate_activation_memory(
            batch_size=batch_size,
            seq_len=seq_len,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_layers=num_layers,
            tp_degree=tp_degree,
        )

        total = weights + kv_cache + activations

        return MemoryBreakdown(
            total_bytes=total,
            weights_bytes=weights,
            kv_cache_bytes=kv_cache,
            activations_bytes=activations,
        )

    def analyze_from_model(
        self,
        model: "Model",
        batch_size: int = 1,
        seq_len: int = 1024,
        tp_degree: int = 1,
    ) -> MemoryBreakdown:
        """从模型分析内存占用

        Args:
            model: 模型 IR
            batch_size: 批次大小
            seq_len: 序列长度
            tp_degree: TP 并行度

        Returns:
            MemoryBreakdown: 内存分解
        """
        metadata = model.get_metadata()

        return self.analyze(
            hidden_size=metadata.hidden_size,
            num_layers=metadata.num_layers,
            num_heads=metadata.num_heads,
            intermediate_size=metadata.intermediate_size or metadata.hidden_size * 4,
            vocab_size=metadata.vocab_size or 32000,
            batch_size=batch_size,
            seq_len=seq_len,
            num_kv_heads=metadata.num_kv_heads,
            tp_degree=tp_degree,
        )
