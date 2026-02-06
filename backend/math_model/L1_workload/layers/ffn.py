"""FFN 层实现

定义 Feed-Forward Network / MLP / MoE 等层。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from math_model.core.types import DataType
from math_model.L1_workload.graph import GraphEdge, GraphNode, NodeKind, NodeRole
from math_model.L1_workload.layers import layer_registry
from math_model.L1_workload.layers.base import LayerBase, LayerRole
from math_model.L1_workload.operators.compute.matmul import MatMulOp
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
        - hidden_size: 隐藏层大小
        - intermediate_size: 中间层大小
        - seq_len: 序列长度
        - batch: 批次大小

    Example:
        >>> layer = FFNLayer("ffn_0", {
        ...     "hidden_size": 4096,
        ...     "intermediate_size": 11008,
        ...     "seq_len": 2048,
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
        default_batch = 1  # common batch fallback
        default_seq_len = 2048  # typical model context length
        default_hidden = 4096  # typical hidden size
        b = self._config.get("batch", default_batch)
        s = self._config.get("seq_len", default_seq_len)
        h = self._config.get("hidden_size", default_hidden)
        return [self._tensor(name="hidden_states", shape=[b, s, h])]

    def get_outputs(self) -> list[TensorDesc]:
        """获取输出张量: ffn_output"""
        default_batch = 1  # common batch fallback
        default_seq_len = 2048  # typical model context length
        default_hidden = 4096  # typical hidden size
        b = self._config.get("batch", default_batch)
        s = self._config.get("seq_len", default_seq_len)
        h = self._config.get("hidden_size", default_hidden)
        return [self._tensor(name="ffn_output", shape=[b, s, h], is_output=True)]

    def compute_flops(self) -> int:
        """计算 FLOPs

        FLOPs 分解 (SwiGLU):
            - Gate Projection: 2 * B * S * H * I
            - Up Projection: 2 * B * S * H * I
            - Down Projection: 2 * B * S * I * H
            Total: 6 * B * S * H * I
        """
        default_batch = 1  # common batch fallback
        default_seq_len = 2048  # typical model context length
        default_hidden = 4096  # typical hidden size
        expansion = 4  # intermediate_size = hidden_size * 4
        b = self._config.get("batch", default_batch)
        s = self._config.get("seq_len", default_seq_len)
        h = self._config.get("hidden_size", default_hidden)
        i = self._config.get("intermediate_size", h * expansion)
        mul_add = 2  # matmul uses 2 * M * K * N
        proj_count = 3  # gate + up + down projections

        # Gate + Up + Down projections
        return mul_add * b * s * h * i * proj_count

    def compute_memory(self) -> tuple[int, int]:
        """计算内存占用

        Returns:
            tuple[int, int]: (weight_bytes, activation_bytes)
        """
        default_batch = 1  # common batch fallback
        default_seq_len = 2048  # typical model context length
        default_hidden = 4096  # typical hidden size
        expansion = 4  # intermediate_size = hidden_size * 4
        h = self._config.get("hidden_size", default_hidden)
        i = self._config.get("intermediate_size", h * expansion)
        b = self._config.get("batch", default_batch)
        s = self._config.get("seq_len", default_seq_len)
        act_bytes = self._activation_dtype.bytes
        weight_bytes_per_elem = self._weight_dtype.bytes
        up_down_count = 2  # gate + up activations

        # 权重: Wgate, Wup, Wdown
        weight_bytes = (h * i * up_down_count + i * h) * weight_bytes_per_elem

        # 激活: input, gate, up, hidden, output
        activation_bytes = (b * s * h + b * s * i * up_down_count + b * s * h) * act_bytes

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
        default_batch = 1  # common batch fallback
        default_seq_len = 2048  # typical model context length
        default_hidden = 4096  # typical hidden size
        expansion = 4  # intermediate_size = hidden_size * 4
        b = self._config.get("batch", default_batch)
        s = self._config.get("seq_len", default_seq_len)
        h = self._config.get("hidden_size", default_hidden)
        i = self._config.get("intermediate_size", h * expansion)
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

        default_batch = 1
        default_seq_len = 2048
        default_hidden = 4096
        expansion = 4
        b = self._config.get("batch", default_batch)
        s = self._config.get("seq_len", default_seq_len)
        h = self._config.get("hidden_size", default_hidden)
        i = self._config.get("intermediate_size", h * expansion)
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
