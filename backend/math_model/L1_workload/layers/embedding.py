"""Embedding layer implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from math_model.L0_entry.types import DataType
from math_model.L1_workload.graph import GraphEdge, GraphNode
from math_model.L1_workload.layers import layer_registry
from math_model.L1_workload.layers.base import LayerBase, LayerRole
from math_model.L1_workload.layers.utils import get_batch, get_hidden_size, get_seq_len
from math_model.L1_workload.tensor import TensorDesc

if TYPE_CHECKING:
    from math_model.L1_workload.operators.base import OpBase


@layer_registry.register("embedding")
class EmbeddingLayer(LayerBase):
    """Token Embedding layer.

    计算流程:
        1. token_ids -> embedding table lookup
        2. 输出 embedding 向量

    FLOPs:
        0 (查表)

    Memory:
        weights = vocab_size * hidden_size
        activations = batch * seq_len * hidden_size

    PyTorch 示例:
        ```python
        import torch
        #token_ids: [batch, seq_len]
        token_ids = torch.randint(0, vocab_size, (batch, seq_len))
        #emb.weight: [vocab_size, hidden_size]
        emb = torch.nn.Embedding(vocab_size, hidden_size)
        out = emb(token_ids)  #out: [batch, seq_len, hidden_size]
        ```

    Config 参数:
        - vocab_size: 词表大小
        - hidden_size / hidden_dim: embedding 维度
        - seq_len / q_seq_len: 序列长度
        - batch / batch_size: 批次大小
    """

    @property
    def op_type(self) -> str:
        return "embedding"

    @property
    def role(self) -> LayerRole:
        return LayerRole.MEMORY

    def get_inputs(self) -> list[TensorDesc]:
        batch = get_batch(self._config)
        seq_len = get_seq_len(self._config)
        return [
            self._tensor(name="token_ids", shape=[batch, seq_len], dtype=DataType.INT32),
        ]

    def get_outputs(self) -> list[TensorDesc]:
        batch = get_batch(self._config)
        seq_len = get_seq_len(self._config)
        hidden = get_hidden_size(self._config)
        return [
            self._tensor(name="embeddings", shape=[batch, seq_len, hidden], is_output=True),
        ]

    def compute_flops(self) -> int:
        zero_flops = 0  # embedding lookup has no FLOPs
        return zero_flops

    def compute_memory(self) -> tuple[int, int]:
        default_vocab = 0  # unknown vocab size
        vocab = int(self._config.get("vocab_size", default_vocab))
        hidden = get_hidden_size(self._config)
        batch = get_batch(self._config)
        seq_len = get_seq_len(self._config)
        act_bytes = self._activation_dtype.bytes
        weight_bytes_per_elem = self._weight_dtype.bytes

        weight_bytes = vocab * hidden * weight_bytes_per_elem
        activation_bytes = batch * seq_len * hidden * act_bytes
        return weight_bytes, activation_bytes

    def _build_ops(self) -> list["OpBase"]:
        """构建内部 Op 列表

        计算流程伪代码::

            # 输入: token_ids [B, S]
            # 权重: embedding table [V, H]
            # output = embedding[token_ids]          # 查表

        Op 分解:
            - 暂无（Embedding 查表/内存访问未建模）

        Returns:
            list[OpBase]: Op 列表
        """
        return []

    def build_intra_graph(
        self, layer_node_id: str
    ) -> tuple[list[GraphNode], list[GraphEdge], list[str], list[str]]:
        """Embedding 查表未建模 OP，层内图为空。"""
        return ([], [], [], [])
