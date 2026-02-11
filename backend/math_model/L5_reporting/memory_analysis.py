"""内存分析模块

生成内存占用分解报告。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from math_model.L1_workload.ir import Model
    from math_model.L1_workload.metadata import ModelMetadata
    from math_model.L3_mapping.protocols import ParallelismConfig


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

    def to_dict(self, chip_memory_capacity_gb: float | None = None) -> dict[str, Any]:
        """转换为字典（匹配前端 MemoryAnalysis 接口）

        Args:
            chip_memory_capacity_gb: 芯片显存容量（GB），用于计算 is_sufficient 和 utilization

        Returns:
            dict: 前端 MemoryAnalysis 格式
                {
                    "model_memory_gb": float,
                    "kv_cache_memory_gb": float,
                    "activation_memory_gb": float,
                    "overhead_gb": float,
                    "total_per_chip_gb": float,
                    "is_memory_sufficient": bool,
                    "memory_utilization": float,
                }
        """
        total_gb = self.total_bytes / (1024 ** 3)

        result = {
            "model_memory_gb": self.weights_bytes / (1024 ** 3),
            "kv_cache_memory_gb": self.kv_cache_bytes / (1024 ** 3),
            "activation_memory_gb": self.activations_bytes / (1024 ** 3),
            "overhead_gb": self.temp_bytes / (1024 ** 3),  # 映射 temp_bytes 为 overhead
            "total_per_chip_gb": total_gb,
        }

        if chip_memory_capacity_gb is not None:
            result["is_memory_sufficient"] = total_gb <= chip_memory_capacity_gb
            result["memory_utilization"] = total_gb / chip_memory_capacity_gb if chip_memory_capacity_gb > 0 else 0.0
        else:
            result["is_memory_sufficient"] = True
            result["memory_utilization"] = 0.0

        return result


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
        pp_degree: int = 1,
        mla_enabled: bool = False,
        kv_lora_rank: int = 0,
        qk_rope_dim: int = 0,
    ) -> int:
        """计算 KV Cache 内存（支持 MLA 压缩）

        Args:
            batch_size: 批次大小
            seq_len: 序列长度（完整上下文）
            num_layers: 层数
            hidden_size: 隐藏层大小
            num_kv_heads: KV 头数 (GQA)
            num_heads: 注意力头数
            tp_degree: TP 并行度
            pp_degree: PP 并行度
            mla_enabled: 是否启用 MLA（DeepSeek-V2/V3/R1）
            kv_lora_rank: MLA 压缩秩
            qk_rope_dim: MLA RoPE 维度

        Returns:
            int: 每芯片 KV Cache 内存 (bytes)

        Examples:
            # LLaMA-3 70B (TP=4, PP=1, batch=32, seq=4096, BF16)
            kv_bytes = analyzer.calculate_kv_cache_memory(
                batch_size=32, seq_len=4096, num_layers=80,
                hidden_size=8192, num_kv_heads=8, num_heads=64,
                tp_degree=4, pp_degree=1
            )
            # 结果: ~10.7 GB

            # DeepSeek-V3 (TP=8, PP=1, batch=128, seq=4096, FP8, MLA)
            kv_bytes = analyzer.calculate_kv_cache_memory(
                batch_size=128, seq_len=4096, num_layers=61,
                hidden_size=7168, num_kv_heads=128, num_heads=128,
                tp_degree=8, pp_degree=1,
                mla_enabled=True, kv_lora_rank=512, qk_rope_dim=64
            )
            # 结果: ~2.3 GB (节省 96%)
        """
        layers_per_chip = num_layers // pp_degree if pp_degree > 0 else num_layers

        if mla_enabled:
            # MLA: K 和 V 共享压缩向量 c_t (无因子 2)
            # c_t 大小 = (kv_lora_rank + qk_rope_dim) / tp_degree
            # DeepSeek-V3 (TP=8): (512/8 + 64/8) * 1 byte = 72 bytes/token
            # MLA 的压缩向量也按 TP 切分
            kv_lora_rank_per_chip = kv_lora_rank // tp_degree if tp_degree > 0 else kv_lora_rank
            qk_rope_dim_per_chip = qk_rope_dim // tp_degree if tp_degree > 0 else qk_rope_dim
            kv_per_token_per_layer = (kv_lora_rank_per_chip + qk_rope_dim_per_chip) * self.dtype_bytes
            kv_cache_bytes = batch_size * seq_len * layers_per_chip * kv_per_token_per_layer
        else:
            # 标准 GQA: 2 * (K + V) * kv_heads * head_dim
            # TP 切分 kv_heads
            head_dim = hidden_size // num_heads if num_heads > 0 else hidden_size
            effective_kv_heads = num_kv_heads if num_kv_heads else num_heads
            kv_heads_per_chip = effective_kv_heads // tp_degree if tp_degree > 0 else effective_kv_heads
            kv_per_token_per_layer = 2 * kv_heads_per_chip * head_dim * self.dtype_bytes
            kv_cache_bytes = batch_size * seq_len * layers_per_chip * kv_per_token_per_layer

        return kv_cache_bytes

    def calculate_activation_memory(
        self,
        batch_size: int,
        seq_len: int,
        hidden_size: int,
        intermediate_size: int,
        num_layers: int,
        tp_degree: int = 1,
        pp_degree: int = 1,
    ) -> int:
        """计算激活内存（推理峰值）

        推理时只存储一层的激活（流水线重用），不像训练需要存储所有层。

        Args:
            batch_size: 批次大小
            seq_len: 当前阶段序列长度（prefill=prompt_length, decode=1）
            hidden_size: 隐藏层大小
            intermediate_size: FFN 中间层大小
            num_layers: 模型总层数
            tp_degree: TP 并行度
            pp_degree: PP 并行度

        Returns:
            int: 每芯片激活内存 (bytes)
        """
        # 计算不同操作的激活值大小
        hidden_activation = batch_size * seq_len * hidden_size
        ffn_activation = batch_size * seq_len * (intermediate_size // tp_degree if tp_degree > 0 else intermediate_size)
        attn_activation = batch_size * seq_len * hidden_size

        # 取峰值（同时只有一个操作的激活在内存中）
        peak_activation_per_layer = max(hidden_activation, ffn_activation, attn_activation)

        # 推理时只需一层的激活（不是 PP stage 中所有层）
        activation_bytes = peak_activation_per_layer * self.dtype_bytes

        return activation_bytes

    def calculate_overhead(
        self,
        weight_bytes: int,
        kv_cache_bytes: int,
    ) -> int:
        """计算系统开销

        开销包括：
        - CUDA 上下文：300-800 MB
        - NCCL 通信缓冲区：256 MB - 1 GB
        - 内存分配器碎片：5-10%

        公式：15% of (weights + KV cache)，限制在 [500MB, 4GB]

        Args:
            weight_bytes: 权重内存
            kv_cache_bytes: KV Cache 内存

        Returns:
            int: 开销内存 (bytes)
        """
        overhead = int(0.15 * (weight_bytes + kv_cache_bytes))

        # 边界：最小 500 MB，最大 4 GB
        MIN_OVERHEAD = 500 * 1024 * 1024
        MAX_OVERHEAD = 4 * 1024 * 1024 * 1024

        return max(MIN_OVERHEAD, min(overhead, MAX_OVERHEAD))

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
            intermediate_size=(
                metadata.intermediate_size
                if metadata.intermediate_size is not None
                else metadata.hidden_size * 4
            ),  # 可从 hidden_size 推导
            vocab_size=(
                metadata.vocab_size if metadata.vocab_size is not None else 32000
            ),  # 常见默认值
            batch_size=batch_size,
            seq_len=seq_len,
            num_kv_heads=metadata.num_kv_heads,
            tp_degree=tp_degree,
        )
