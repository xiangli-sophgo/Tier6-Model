"""FFN 层实现

定义 Feed-Forward Network / MLP / MoE 等层。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from math_model.L1_workload.graph import GraphEdge, GraphNode, NodeKind, NodeRole
from math_model.L1_workload.layers import layer_registry
from math_model.L1_workload.layers.base import LayerBase, LayerRole
from math_model.L1_workload.layers.utils import (
    get_batch,
    get_hidden_size,
    get_intermediate_size,
    get_seq_len,
    matmul_flops,
)
from math_model.L1_workload.tensor import TensorDesc

if TYPE_CHECKING:
    from math_model.L1_workload.operators.base import OpBase


@layer_registry.register("ffn")
class FFNLayer(LayerBase):
    """Feed-Forward Network 层

    计算流程 (SwiGLU):
        1. Gate Projection: x @ Wgate -> gate
        2. Up Projection: x @ Wup -> up
        3. Activation: silu(gate) * up
        4. Down Projection: hidden @ Wdown -> output

    Config 参数:
        - hidden_size / hidden_dim: 隐藏层大小
        - intermediate_size / inter_dim: 中间层大小
        - seq_len / q_seq_len: 序列长度
        - batch / batch_size: 批次大小

    Example:
        >>> layer = FFNLayer("ffn_0", {
        ...     "hidden_size": 4096,
        ...     "intermediate_size": 11008,
        ...     "q_seq_len": 1,
        ...     "batch": 1,
        ... })
        >>> print(layer.compute_flops())
    """

    @property
    def op_type(self) -> str:
        return "ffn"

    @property
    def role(self) -> LayerRole:
        return LayerRole.COMPUTE

    def get_inputs(self) -> list[TensorDesc]:
        """获取输入张量: hidden_states"""
        b = get_batch(self._config)
        s = get_seq_len(self._config)
        h = get_hidden_size(self._config)
        return [self._tensor(name="hidden_states", shape=[b, s, h])]

    def get_outputs(self) -> list[TensorDesc]:
        """获取输出张量: ffn_output"""
        b = get_batch(self._config)
        s = get_seq_len(self._config)
        h = get_hidden_size(self._config)
        return [self._tensor(name="ffn_output", shape=[b, s, h], is_output=True)]

    def compute_flops(self) -> int:
        """计算 FLOPs

        FLOPs 分解 (SwiGLU):
            - Gate Projection: 2 * B * S * H * I
            - Up Projection: 2 * B * S * H * I
            - Down Projection: 2 * B * S * I * H
            Total: 6 * B * S * H * I
        """
        b = get_batch(self._config)
        s = get_seq_len(self._config)
        h = get_hidden_size(self._config)
        i = get_intermediate_size(self._config)
        tokens = b * s
        gate_up_count = 2  # gate + up projections
        down_count = 1  # down projection

        return (
            matmul_flops(tokens, h, i) * gate_up_count
            + matmul_flops(tokens, i, h) * down_count
        )

    def compute_memory(self) -> tuple[int, int]:
        """计算内存占用

        Returns:
            tuple[int, int]: (weight_bytes, activation_bytes)
        """
        b = get_batch(self._config)
        s = get_seq_len(self._config)
        h = get_hidden_size(self._config)
        i = get_intermediate_size(self._config)
        act_bytes = self._activation_dtype.bytes
        weight_bytes_per_elem = self._weight_dtype.bytes
        gate_up_count = 2  # gate + up weights

        weight_bytes = (h * i * gate_up_count + i * h) * weight_bytes_per_elem
        activation_bytes = (b * s * h + b * s * i * gate_up_count + b * s * h) * act_bytes

        return weight_bytes, activation_bytes

    def _build_ops(self) -> list["OpBase"]:
        """构建内部 Op 列表

        计算流程伪代码::

            # 输入: x [B, S, H]
            # 权重: Wgate [H, I], Wup [H, I], Wdown [I, H]

            # Step 1: Gate Projection
            # gate = x.reshape(B*S, H) @ Wgate      # [B*S, H] @ [H, I] -> [B*S, I]

            # Step 2: Up Projection
            # up = x.reshape(B*S, H) @ Wup          # [B*S, H] @ [H, I] -> [B*S, I]

            # Step 3: Activation (SwiGLU)
            # hidden = silu(gate) * up              # (Vector 引擎，此处未建模)

            # Step 4: Down Projection
            # out = hidden @ Wdown                  # [B*S, I] @ [I, H] -> [B*S, H]
            # out = out.reshape(B, S, H)

        Op 分解:
            1. gate_proj: MatMul [B*S, H] @ [H, I] -> [B*S, I]
            2. up_proj:   MatMul [B*S, H] @ [H, I] -> [B*S, I]
            3. down_proj: MatMul [B*S, I] @ [I, H] -> [B*S, H]

        注: Activation/Reshape 等 Vector/Memory Op 暂未建模。

        Returns:
            list[OpBase]: Op 列表
        """
        b = get_batch(self._config)
        s = get_seq_len(self._config)
        h = get_hidden_size(self._config)
        i = get_intermediate_size(self._config)
        tokens = b * s

        ops: list[OpBase] = []

        # 1. Gate Projection: [B*S, H] @ [H, I]
        ops.append(
            self._matmul_op(
                f"{self.name}_gate_proj",
                {"M": tokens, "K": h, "N": i},
            )
        )

        # 2. Up Projection: [B*S, H] @ [H, I]
        ops.append(
            self._matmul_op(
                f"{self.name}_up_proj",
                {"M": tokens, "K": h, "N": i},
            )
        )

        # 3. Down Projection: [B*S, I] @ [I, H]
        ops.append(
            self._matmul_op(
                f"{self.name}_down_proj",
                {"M": tokens, "K": i, "N": h},
            )
        )

        return ops

    def build_intra_graph(
        self, layer_node_id: str
    ) -> tuple[list[GraphNode], list[GraphEdge], list[str], list[str]]:
        """构建 FFN 的层内 OP 级计算图

        计算流程 (SwiGLU 简化):
            1. gate_proj 与 up_proj 并行
            2. down_proj 依赖 gate_proj 与 up_proj 的结果

        参数说明:
            - layer_node_id: 模型级 Layer 节点 ID，用于生成唯一的 OP 节点 ID
        """
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

        b = get_batch(self._config)
        s = get_seq_len(self._config)
        i = get_intermediate_size(self._config)
        tokens = b * s

        # gate/up 输出张量: [B*S, I]
        gate_tensor = self._tensor(
            name=f"{gate_op.name}_out",
            shape=[tokens, i],  # shape: [tokens, intermediate]
            producer_id=gate_op.name,
            consumer_id=down_op.name,
        )
        up_tensor = self._tensor(
            name=f"{up_op.name}_out",
            shape=[tokens, i],  # shape: [tokens, intermediate]
            producer_id=up_op.name,
            consumer_id=down_op.name,
        )

        edges = [
            GraphEdge(src=gate_id, dst=down_id, edge_type="data", tensor=gate_tensor),
            GraphEdge(src=up_id, dst=down_id, edge_type="data", tensor=up_tensor),
        ]

        # gate/up 为层内入口，down 为层内出口
        return (nodes, edges, [gate_id, up_id], [down_id])
