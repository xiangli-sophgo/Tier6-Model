"""Llama 模型实现

定义 Llama2 系列模型。
"""

from __future__ import annotations

from typing import Any

from math_model.core.types import DataType
from math_model.L1_workload.layers.attention import AttentionLayer
from math_model.L1_workload.layers.ffn import FFNLayer
from math_model.L1_workload.metadata import ModelMetadata
from math_model.L1_workload.models import model_registry
from math_model.L1_workload.models.base import ModelBase


@model_registry.register("llama2")
class Llama2Model(ModelBase):
    """Llama2 模型

    支持 Llama2-7B/13B/70B 等配置。

    Config 参数:
        - hidden_size: 隐藏层大小（4096 for 7B）
        - num_layers: 层数（32 for 7B）
        - num_heads: 注意力头数（32 for 7B）
        - intermediate_size: FFN 中间层大小（11008 for 7B）
        - vocab_size: 词表大小（32000）
        - seq_len: 序列长度
        - batch: 批次大小

    Example:
        >>> model = Llama2Model({
        ...     "hidden_size": 4096,
        ...     "num_layers": 32,
        ...     "num_heads": 32,
        ...     "intermediate_size": 11008,
        ...     "vocab_size": 32000,
        ...     "seq_len": 2048,
        ...     "batch": 1,
        ... })
        >>> ir = model.to_ir()
        >>> print(ir.get_ops_breakdown())
    """

    @property
    def name(self) -> str:
        """模型名称"""
        return "llama2"

    def build(self) -> None:
        """构建模型层结构

        Llama2 结构:
            - Embedding
            - N x (Attention + FFN)
            - RMSNorm
            - LM Head
        """
        # 从 config 获取层数（必需）
        if "num_layers" not in self._config:
            raise ValueError("Missing required field 'num_layers' in Llama model config")
        num_layers = self._config["num_layers"]

        # 构建 Transformer 层
        for i in range(num_layers):
            # Attention 层
            self._layers.append(AttentionLayer(f"layers.{i}.attention", self._config))

            # FFN 层
            self._layers.append(FFNLayer(f"layers.{i}.ffn", self._config))

    def _build_metadata(self) -> ModelMetadata:
        """构建模型元数据"""
        # 必需字段检查（与 ModelMetadata.from_dict 对齐）
        required_fields = ["hidden_size", "num_layers", "num_heads"]
        missing = [f for f in required_fields if f not in self._config]
        if missing:
            raise ValueError(f"Missing required fields in Llama model config: {missing}")

        return ModelMetadata(
            name=self.name,
            dtype=DataType.from_string(self._config.get("dtype", "fp16")),  # dtype 可选
            hidden_size=self._config["hidden_size"],
            num_layers=self._config["num_layers"],
            num_heads=self._config["num_heads"],
            vocab_size=self._config.get("vocab_size"),  # 可选
            seq_len=self._config.get("seq_len"),  # 可选
            batch=self._config.get("batch"),  # 可选
        )
