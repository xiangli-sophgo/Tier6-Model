"""LM Head layer implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from math_model.L1_workload.graph import GraphEdge, GraphNode, NodeKind, NodeRole
from math_model.L1_workload.layers import layer_registry
from math_model.L1_workload.layers.base import LayerBase, LayerRole
from math_model.L1_workload.layers.utils import get_batch, get_hidden_size, get_seq_len, matmul_flops
from math_model.L1_workload.operators.compute.matmul import MatMulOp
from math_model.L1_workload.tensor import TensorDesc

if TYPE_CHECKING:
    from math_model.L1_workload.operators.base import OpBase


@layer_registry.register("lmhead")
class LMHeadLayer(LayerBase):
    """LM Head 层（输出到词表）。

    计算流程:
        1. hidden_states @ W_vocab -> logits

    FLOPs:
        2 * (batch * seq_len) * hidden_size * vocab_size

    Memory:
        weights = hidden_size * vocab_size
        activations = batch * seq_len * (hidden_size + vocab_size)

    PyTorch 示例:
        ```python
        import torch
        #x: [batch, seq_len, hidden_size]
        x = torch.randn(batch, seq_len, hidden_size)
        #w: [hidden_size, vocab_size]
        w = torch.randn(hidden_size, vocab_size)
        logits = x @ w  #logits: [batch, seq_len, vocab_size]
        ```

    Config 参数:
        - hidden_size / hidden_dim: 隐藏维度
        - vocab_size: 词表大小
        - seq_len / q_seq_len: 序列长度
        - batch / batch_size: 批次大小
    """

    @property
    def op_type(self) -> str:
        return "lmhead"

    @property
    def role(self) -> LayerRole:
        return LayerRole.COMPUTE

    def get_inputs(self) -> list[TensorDesc]:
        batch = get_batch(self._config)
        seq_len = get_seq_len(self._config)
        hidden = get_hidden_size(self._config)
        return [
            self._tensor(name="hidden_states", shape=[batch, seq_len, hidden]),
        ]

    def get_outputs(self) -> list[TensorDesc]:
        batch = get_batch(self._config)
        seq_len = get_seq_len(self._config)
        default_vocab = 0  # unknown vocab size
        vocab = int(self._config.get("vocab_size", default_vocab))
        return [
            self._tensor(name="logits", shape=[batch, seq_len, vocab], is_output=True),
        ]

    def compute_flops(self) -> int:
        batch = get_batch(self._config)
        seq_len = get_seq_len(self._config)
        hidden = get_hidden_size(self._config)
        default_vocab = 0  # unknown vocab size
        vocab = int(self._config.get("vocab_size", default_vocab))
        return matmul_flops(batch * seq_len, hidden, vocab)

    def compute_memory(self) -> tuple[int, int]:
        batch = get_batch(self._config)
        seq_len = get_seq_len(self._config)
        hidden = get_hidden_size(self._config)
        default_vocab = 0  # unknown vocab size
        vocab = int(self._config.get("vocab_size", default_vocab))
        act_bytes = self._activation_dtype.bytes
        weight_bytes_per_elem = self._weight_dtype.bytes

        weight_bytes = hidden * vocab * weight_bytes_per_elem
        activation_bytes = batch * seq_len * (hidden + vocab) * act_bytes
        return weight_bytes, activation_bytes

    def _build_ops(self) -> list["OpBase"]:
        """构建内部 Op 列表

        计算流程伪代码::

            # 输入: x [B, S, H]
            # 权重: W_vocab [H, V]
            # logits = x.reshape(B*S, H) @ W_vocab   # [B*S, H] @ [H, V] -> [B*S, V]
            # logits = logits.reshape(B, S, V)

        Op 分解:
            1. lm_head: MatMul [B*S, H] @ [H, V] -> [B*S, V]

        注: Reshape 等 Memory Op 暂未建模。

        Returns:
            list[OpBase]: Op 列表
        """
        batch = get_batch(self._config)
        seq_len = get_seq_len(self._config)
        hidden = get_hidden_size(self._config)
        default_vocab = 0  # unknown vocab size
        vocab = int(self._config.get("vocab_size", default_vocab))
        tokens = batch * seq_len

        return [
            self._matmul_op(
                f"{self.name}_lm_head",
                {"M": tokens, "K": hidden, "N": vocab},
            )
        ]

    def build_intra_graph(
        self, layer_node_id: str
    ) -> tuple[list[GraphNode], list[GraphEdge], list[str], list[str]]:
        """构建 LMHead 的层内 OP 级计算图。"""
        ops = self.get_ops()
        if not ops:
            return ([], [], [], [])

        op = ops[0]
        op_id = f"{layer_node_id}::op::{op.name}"
        node = GraphNode(node_id=op_id, kind=NodeKind.OP, role=NodeRole.COMPUTE, ref=op.name)

        batch = get_batch(self._config)
        seq_len = get_seq_len(self._config)
        vocab = int(self._config.get("vocab_size", 0))
        tokens = batch * seq_len

        logits = self._tensor(
            name=f"{op.name}_out",
            shape=[tokens, vocab],  # shape: [tokens, vocab]
            producer_id=op.name,
        )
        # LMHead 为单 OP，层内无依赖边；返回空 edges，但保留出口语义
        _ = logits
        return ([node], [], [op_id], [op_id])
