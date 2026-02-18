"""MLP layer implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from perf_model.L1_workload.graph import GraphEdge, GraphNode, NodeKind, NodeRole
from perf_model.L1_workload.layers import layer_registry
from perf_model.L1_workload.layers.base import LayerBase, LayerRole
from perf_model.L1_workload.layers.utils import (
    get_batch,
    get_hidden_size,
    get_intermediate_size,
    get_seq_len,
    matmul_flops,
)
from perf_model.L1_workload.operators.compute.matmul import MatMulOp
from perf_model.L1_workload.tensor import TensorDesc

if TYPE_CHECKING:
    from perf_model.L1_workload.operators.base import OpBase


@layer_registry.register("mlp")
class MLPLayer(LayerBase):
    """MLP 层（SwiGLU 风格）。

    计算流程:
        1. Gate Projection: x @ W_gate
        2. Up Projection: x @ W_up
        3. Activation: silu(gate) * up
        4. Down Projection: hidden @ W_down

    FLOPs:
        Gate + Up + Down = 2 * (batch * seq_len) * hidden_size * intermediate_size * 3

    Memory:
        weights = (hidden_size * intermediate_size * 2 + intermediate_size * hidden_size)
        activations = batch * seq_len * (hidden_size + intermediate_size * 2 + hidden_size)

    PyTorch 示例:
        ```python
        import torch

        x = torch.randn(batch, seq_len, hidden_size)
        #w_gate: [hidden_size, intermediate_size]
        gate = x @ w_gate #gate: [batch, seq_len, intermediate_size]
        #w_up: [hidden_size, intermediate_size]
        up = x @ w_up #up: [batch, seq_len, intermediate_size]
        hidden = torch.nn.functional.silu(gate) * up #hidden: [batch, seq_len, intermediate_size]
        #w_down: [intermediate_size, hidden_size]
        out = hidden @ w_down #out: [batch, seq_len, hidden_size]
        ```

    Config 参数:
        - hidden_size / hidden_dim: 隐藏维度
        - intermediate_size / inter_dim: 中间层维度
        - seq_len / q_seq_len: 序列长度
        - batch / batch_size: 批次大小
    """

    @property
    def op_type(self) -> str:
        return "mlp"

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
        hidden = get_hidden_size(self._config)
        return [
            self._tensor(name="mlp_out", shape=[batch, seq_len, hidden], is_output=True),
        ]

    def compute_flops(self) -> int:
        batch = get_batch(self._config)
        seq_len = get_seq_len(self._config)
        hidden = get_hidden_size(self._config)
        inter = get_intermediate_size(self._config)
        m = batch * seq_len
        gate_up_count = 2  # gate + up projections
        down_count = 1  # down projection
        return (
            matmul_flops(m, hidden, inter) * gate_up_count
            + matmul_flops(m, inter, hidden) * down_count
        )

    def compute_memory(self) -> tuple[int, int]:
        batch = get_batch(self._config)
        seq_len = get_seq_len(self._config)
        hidden = get_hidden_size(self._config)
        inter = get_intermediate_size(self._config)
        act_bytes = self._activation_dtype.bytes
        weight_bytes_per_elem = self._weight_dtype.bytes
        two = 2  # gate/up projection count

        weight_bytes = (hidden * inter * two + inter * hidden) * weight_bytes_per_elem
        activation_bytes = batch * seq_len * (hidden + inter * two + hidden) * act_bytes
        return weight_bytes, activation_bytes

    def _build_ops(self) -> list["OpBase"]:
        """构建内部 Op 列表

        计算流程伪代码::

            # 输入: x [B, S, H]
            # 权重: W_gate [H, I], W_up [H, I], W_down [I, H]

            # Step 1: Gate Projection
            # gate = x.reshape(B*S, H) @ W_gate     # [B*S, H] @ [H, I] -> [B*S, I]

            # Step 2: Up Projection
            # up = x.reshape(B*S, H) @ W_up         # [B*S, H] @ [H, I] -> [B*S, I]

            # Step 3: Activation (SwiGLU)
            # hidden = silu(gate) * up              # (Vector 引擎，此处未建模)

            # Step 4: Down Projection
            # out = hidden @ W_down                 # [B*S, I] @ [I, H] -> [B*S, H]
            # out = out.reshape(B, S, H)

        Op 分解:
            1. gate_proj: MatMul [B*S, H] @ [H, I] -> [B*S, I]
            2. up_proj:   MatMul [B*S, H] @ [H, I] -> [B*S, I]
            3. down_proj: MatMul [B*S, I] @ [I, H] -> [B*S, H]

        注: Activation/Reshape 等 Vector/Memory Op 暂未建模。

        Returns:
            list[OpBase]: Op 列表
        """
        batch = get_batch(self._config)
        seq_len = get_seq_len(self._config)
        hidden = get_hidden_size(self._config)
        inter = get_intermediate_size(self._config)
        tokens = batch * seq_len

        ops: list[OpBase] = []

        # 1. Gate Projection: [B*S, H] @ [H, I]
        ops.append(
            self._matmul_op(
                f"{self.name}_gate_proj",
                {"M": tokens, "K": hidden, "N": inter},
            )
        )

        # 2. Up Projection: [B*S, H] @ [H, I]
        ops.append(
            self._matmul_op(
                f"{self.name}_up_proj",
                {"M": tokens, "K": hidden, "N": inter},
            )
        )

        # 3. Down Projection: [B*S, I] @ [I, H]
        ops.append(
            self._matmul_op(
                f"{self.name}_down_proj",
                {"M": tokens, "K": inter, "N": hidden},
            )
        )

        return ops

    def build_intra_graph(
        self, layer_node_id: str
    ) -> tuple[list[GraphNode], list[GraphEdge], list[str], list[str]]:
        """构建 MLP 的层内 OP 级计算图（gate/up 并行，down 汇合）。"""
        ops = self.get_ops()
        if len(ops) < 3:
            return ([], [], [], [])

        gate_op, up_op, down_op = ops[0], ops[1], ops[2]
        gate_id = f"{layer_node_id}::op::{gate_op.name}"
        up_id = f"{layer_node_id}::op::{up_op.name}"
        down_id = f"{layer_node_id}::op::{down_op.name}"

        nodes = [
            GraphNode(node_id=gate_id, kind=NodeKind.OP, role=NodeRole.COMPUTE, ref=gate_op.name),
            GraphNode(node_id=up_id, kind=NodeKind.OP, role=NodeRole.COMPUTE, ref=up_op.name),
            GraphNode(node_id=down_id, kind=NodeKind.OP, role=NodeRole.COMPUTE, ref=down_op.name),
        ]

        batch = get_batch(self._config)
        seq_len = get_seq_len(self._config)
        inter = get_intermediate_size(self._config)
        tokens = batch * seq_len

        gate_tensor = self._tensor(
            name=f"{gate_op.name}_out",
            shape=[tokens, inter],  # shape: [tokens, intermediate]
            producer_id=gate_op.name,
            consumer_id=down_op.name,
        )
        up_tensor = self._tensor(
            name=f"{up_op.name}_out",
            shape=[tokens, inter],  # shape: [tokens, intermediate]
            producer_id=up_op.name,
            consumer_id=down_op.name,
        )

        edges = [
            GraphEdge(src=gate_id, dst=down_id, edge_type="data", tensor=gate_tensor),
            GraphEdge(src=up_id, dst=down_id, edge_type="data", tensor=up_tensor),
        ]

        return (nodes, edges, [gate_id, up_id], [down_id])
