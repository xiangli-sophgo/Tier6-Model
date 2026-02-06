"""MoE layer implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from math_model.L1_workload.graph import GraphEdge, GraphNode, NodeKind, NodeRole
from math_model.L1_workload.layers import layer_registry
from math_model.L1_workload.layers.base import LayerBase, LayerRole
from math_model.L1_workload.layers.utils import (
    get_batch,
    get_hidden_size,
    get_moe_intermediate_size,
    get_seq_len,
    matmul_flops,
)
from math_model.L1_workload.operators.compute.matmul import MatMulOp
from math_model.L1_workload.tensor import TensorDesc

if TYPE_CHECKING:
    from math_model.L1_workload.operators.base import OpBase


@layer_registry.register("moe")
class MoELayer(LayerBase):
    """MoE 层（路由专家）。

    计算流程:
        1. Gate: hidden -> expert scores
        2. Routed experts: Top-K 选择专家，执行 MLP
        3. Shared experts: 所有 token 执行共享 MLP（不经路由）
        4. Combine: 汇聚专家输出

    FLOPs:
        gate = 2 * (batch * seq_len) * hidden_size * n_routed_experts
        routed_experts = n_activated_experts * 2 * (batch * seq_len) * hidden_size * moe_intermediate_size * 3
        shared_experts = n_shared_experts * 2 * (batch * seq_len) * hidden_size * moe_intermediate_size * 3

    Memory:
        weights = gate_weights + expert_weights + shared_weights
        gate_weights = hidden_size * n_routed_experts
        expert_weights = n_routed_experts * (hidden_size * moe_intermediate_size * 2 + moe_intermediate_size * hidden_size)
        shared_weights = n_shared_experts * (hidden_size * moe_intermediate_size * 2 + moe_intermediate_size * hidden_size)
        activations = batch * seq_len * (hidden_size + moe_intermediate_size * 2 + hidden_size + n_shared_experts * (moe_intermediate_size * 2 + hidden_size))

    PyTorch 示例:
        ```python
        import torch
        #x: [batch, seq_len, hidden_size]
        x = torch.randn(batch, seq_len, hidden_size)
        #w_gate: [hidden_size, n_routed_experts]
        gate_logits = x @ w_gate  #gate_logits: [batch, seq_len, n_routed_experts]
        topk_vals, topk_idx = gate_logits.topk(k=n_activated_experts, dim=-1)
        #w_gate_e: [n_routed_experts, hidden_size, moe_intermediate_size]
        #w_up_e: [n_routed_experts, hidden_size, moe_intermediate_size]
        #w_down_e: [n_routed_experts, moe_intermediate_size, hidden_size]
        #pseudo: per-expert MLP + combine -> [batch, seq_len, hidden_size]
        ```

    Config 参数:
        - hidden_size / hidden_dim: 隐藏维度
        - moe_intermediate_size / moe_inter_dim: MoE 中间层维度
        - n_routed_experts: 专家总数
        - n_shared_experts: 共享专家数
        - n_activated_experts: 每 token 激活专家数
        - seq_len / q_seq_len: 序列长度
        - batch / batch_size: 批次大小
    """

    @property
    def op_type(self) -> str:
        return "moe"

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
            self._tensor(name="moe_out", shape=[batch, seq_len, hidden], is_output=True),
        ]

    def compute_flops(self) -> int:
        batch = get_batch(self._config)
        seq_len = get_seq_len(self._config)
        hidden = get_hidden_size(self._config)
        moe_inter = get_moe_intermediate_size(self._config)
        default_routed = 1  # fallback when MoE disabled
        default_activated = 1  # fallback activation count
        default_shared = 0  # fallback shared expert count
        n_routed = int(self._config.get("n_routed_experts", default_routed))
        n_activated = int(self._config.get("n_activated_experts", default_activated))
        n_shared = int(self._config.get("n_shared_experts", default_shared))
        gate_up_count = 2  # gate + up projections per expert
        down_count = 1  # down projection per expert

        tokens = batch * seq_len
        gate_flops = matmul_flops(tokens, hidden, n_routed)

        per_expert_flops = (
            matmul_flops(tokens, hidden, moe_inter) * gate_up_count
            + matmul_flops(tokens, moe_inter, hidden) * down_count
        )
        shared_flops = per_expert_flops * n_shared if n_shared > 0 else 0
        return gate_flops + per_expert_flops * n_activated + shared_flops

    def compute_memory(self) -> tuple[int, int]:
        batch = get_batch(self._config)
        seq_len = get_seq_len(self._config)
        hidden = get_hidden_size(self._config)
        moe_inter = get_moe_intermediate_size(self._config)
        default_routed = 1  # fallback when MoE disabled
        default_shared = 0  # fallback shared expert count
        n_routed = int(self._config.get("n_routed_experts", default_routed))
        n_shared = int(self._config.get("n_shared_experts", default_shared))
        act_bytes = self._activation_dtype.bytes
        weight_bytes_per_elem = self._weight_dtype.bytes
        gate_up_count = 2  # gate + up weights per expert

        gate_weights = hidden * n_routed
        expert_weights = n_routed * (hidden * moe_inter * gate_up_count + moe_inter * hidden)
        shared_weights = n_shared * (hidden * moe_inter * gate_up_count + moe_inter * hidden)
        weight_bytes = (gate_weights + expert_weights + shared_weights) * weight_bytes_per_elem

        shared_activations = n_shared * (moe_inter * gate_up_count + hidden)
        activation_bytes = (
            batch
            * seq_len
            * (hidden + moe_inter * gate_up_count + hidden + shared_activations)
            * act_bytes
        )
        return weight_bytes, activation_bytes

    def _build_ops(self) -> list["OpBase"]:
        """构建内部 Op 列表

        计算流程伪代码::

            # 输入: x [B, S, H]
            # 权重: W_gate [H, E], W_up_e [H, I], W_gate_e [H, I], W_down_e [I, H]
            # 共享专家权重: W_shared_gate [H, I], W_shared_up [H, I], W_shared_down [I, H]
            # 其中: E = n_routed_experts, I = moe_intermediate_size

            # Step 1: Router Gate
            # gate_logits = x.reshape(B*S, H) @ W_gate     # [B*S, H] @ [H, E] -> [B*S, E]
            # topk = topk(gate_logits, k=n_activated)      # (Vector 引擎，此处未建模)

            # Step 2: Per-Expert MLP (Top-K 专家)
            # gate_e = x @ W_gate_e                        # [E*, B*S, H] @ [E*, H, I] -> [E*, B*S, I]
            # up_e = x @ W_up_e                            # [E*, B*S, H] @ [E*, H, I] -> [E*, B*S, I]
            # hidden = silu(gate_e) * up_e                 # (Vector 引擎)
            # out_e = hidden @ W_down_e                    # [E*, B*S, I] @ [E*, I, H] -> [E*, B*S, H]
            # out = combine(out_e)                         # (Router/Combine 未建模)

            # Step 3: Shared MLP (Shared 专家)
            # shared_gate = x @ W_shared_gate              # [B*S, H] @ [H, I] -> [B*S, I]
            # shared_up = x @ W_shared_up                  # [B*S, H] @ [H, I] -> [B*S, I]
            # shared_hidden = silu(shared_gate) * shared_up# (Vector 引擎)
            # shared_out = shared_hidden @ W_shared_down   # [B*S, I] @ [I, H] -> [B*S, H]

        Op 分解:
            1. gate_proj:    MatMul [B*S, H] @ [H, E] -> [B*S, E]
            2. shared_gate:  MatMul [B*S, H] @ [H, I] -> [B*S, I] (可选, n_shared_experts > 0)
            3. shared_up:    MatMul [B*S, H] @ [H, I] -> [B*S, I] (可选, n_shared_experts > 0)
            4. shared_down:  MatMul [B*S, I] @ [I, H] -> [B*S, H] (可选, n_shared_experts > 0)
            5. expert_gate:  MatMul [E*, B*S, H] @ [E*, H, I] -> [E*, B*S, I]
            6. expert_up:    MatMul [E*, B*S, H] @ [E*, H, I] -> [E*, B*S, I]
            7. expert_down:  MatMul [E*, B*S, I] @ [E*, I, H] -> [E*, B*S, H]

        注: Top-K 选择/Router/Combine 等 Vector/Memory Op 暂未建模。

        Returns:
            list[OpBase]: Op 列表
        """
        batch = get_batch(self._config)
        seq_len = get_seq_len(self._config)
        hidden = get_hidden_size(self._config)
        moe_inter = get_moe_intermediate_size(self._config)
        default_routed = 1  # fallback when MoE disabled
        default_activated = 1  # fallback activation count
        default_shared = 0  # fallback shared expert count
        n_routed = int(self._config.get("n_routed_experts", default_routed))
        n_activated = int(self._config.get("n_activated_experts", default_activated))
        n_shared = int(self._config.get("n_shared_experts", default_shared))

        tokens = batch * seq_len
        ops: list[OpBase] = []

        # 1. Router gate: [B*S, H] @ [H, E]
        ops.append(
            self._matmul_op(
                f"{self.name}_gate_proj",
                {"M": tokens, "K": hidden, "N": n_routed},
            )
        )

        if n_shared > 0:
            shared_params = {"M": tokens, "K": hidden, "N": moe_inter}
            if n_shared > 1:
                shared_params["G"] = n_shared
            ops.append(
                self._matmul_op(
                    f"{self.name}_shared_gate_proj",
                    shared_params,
                )
            )
            ops.append(
                self._matmul_op(
                    f"{self.name}_shared_up_proj",
                    shared_params,
                )
            )
            shared_down_params = {"M": tokens, "K": moe_inter, "N": hidden}
            if n_shared > 1:
                shared_down_params["G"] = n_shared
            ops.append(
                self._matmul_op(
                    f"{self.name}_shared_down_proj",
                    shared_down_params,
                )
            )

        # 2. Expert gate: [E*, B*S, H] @ [E*, H, I]
        ops.append(
            self._matmul_op(
                f"{self.name}_expert_gate",
                {"G": n_activated, "M": tokens, "K": hidden, "N": moe_inter},
            )
        )

        # 3. Expert up: [E*, B*S, H] @ [E*, H, I]
        ops.append(
            self._matmul_op(
                f"{self.name}_expert_up",
                {"G": n_activated, "M": tokens, "K": hidden, "N": moe_inter},
            )
        )

        # 4. Expert down: [E*, B*S, I] @ [E*, I, H]
        ops.append(
            self._matmul_op(
                f"{self.name}_expert_down",
                {"G": n_activated, "M": tokens, "K": moe_inter, "N": hidden},
            )
        )

        return ops

    def build_intra_graph(
        self, layer_node_id: str
    ) -> tuple[list[GraphNode], list[GraphEdge], list[str], list[str]]:
        """构建 MoE 的层内 OP 级计算图。"""
        ops = self.get_ops()
        if len(ops) < 4:
            return ([], [], [], [])

        ops_by_name = {op.name: op for op in ops}
        gate_op = ops_by_name.get(f"{self.name}_gate_proj")
        shared_gate_op = ops_by_name.get(f"{self.name}_shared_gate_proj")
        shared_up_op = ops_by_name.get(f"{self.name}_shared_up_proj")
        shared_down_op = ops_by_name.get(f"{self.name}_shared_down_proj")
        expert_gate_op = ops_by_name.get(f"{self.name}_expert_gate")
        expert_up_op = ops_by_name.get(f"{self.name}_expert_up")
        expert_down_op = ops_by_name.get(f"{self.name}_expert_down")

        if not gate_op or not expert_gate_op or not expert_up_op or not expert_down_op:
            return ([], [], [], [])

        gate_id = f"{layer_node_id}::op::{gate_op.name}"
        expert_gate_id = f"{layer_node_id}::op::{expert_gate_op.name}"
        expert_up_id = f"{layer_node_id}::op::{expert_up_op.name}"
        expert_down_id = f"{layer_node_id}::op::{expert_down_op.name}"
        shared_gate_id = (
            f"{layer_node_id}::op::{shared_gate_op.name}" if shared_gate_op else ""
        )
        shared_up_id = f"{layer_node_id}::op::{shared_up_op.name}" if shared_up_op else ""
        shared_down_id = (
            f"{layer_node_id}::op::{shared_down_op.name}" if shared_down_op else ""
        )

        nodes = [
            GraphNode(node_id=gate_id, kind=NodeKind.OP, role=NodeRole.COMPUTE, ref=gate_op.name),
            GraphNode(
                node_id=expert_gate_id,
                kind=NodeKind.OP,
                role=NodeRole.COMPUTE,
                ref=expert_gate_op.name,
            ),
            GraphNode(
                node_id=expert_up_id,
                kind=NodeKind.OP,
                role=NodeRole.COMPUTE,
                ref=expert_up_op.name,
            ),
            GraphNode(
                node_id=expert_down_id,
                kind=NodeKind.OP,
                role=NodeRole.COMPUTE,
                ref=expert_down_op.name,
            ),
        ]

        if shared_gate_op and shared_up_op and shared_down_op:
            nodes.extend(
                [
                    GraphNode(
                        node_id=shared_gate_id,
                        kind=NodeKind.OP,
                        role=NodeRole.COMPUTE,
                        ref=shared_gate_op.name,
                    ),
                    GraphNode(
                        node_id=shared_up_id,
                        kind=NodeKind.OP,
                        role=NodeRole.COMPUTE,
                        ref=shared_up_op.name,
                    ),
                    GraphNode(
                        node_id=shared_down_id,
                        kind=NodeKind.OP,
                        role=NodeRole.COMPUTE,
                        ref=shared_down_op.name,
                    ),
                ]
            )

        batch = get_batch(self._config)
        seq_len = get_seq_len(self._config)
        hidden = get_hidden_size(self._config)
        moe_inter = get_moe_intermediate_size(self._config)
        n_activated = int(self._config.get("n_activated_experts", 1))
        n_shared = int(self._config.get("n_shared_experts", 0))
        tokens = batch * seq_len

        expert_gate_tensor = self._tensor(
            name=f"{expert_gate_op.name}_out",
            shape=[n_activated, tokens, moe_inter],  # shape: [experts, tokens, inter]
            producer_id=expert_gate_op.name,
            consumer_id=expert_down_op.name,
        )
        expert_up_tensor = self._tensor(
            name=f"{expert_up_op.name}_out",
            shape=[n_activated, tokens, moe_inter],  # shape: [experts, tokens, inter]
            producer_id=expert_up_op.name,
            consumer_id=expert_down_op.name,
        )
        expert_down_tensor = self._tensor(
            name=f"{expert_down_op.name}_out",
            shape=[n_activated, tokens, hidden],  # shape: [experts, tokens, hidden]
            producer_id=expert_down_op.name,
        )

        edges = [
            # gate 影响专家选择，使用 control 边表达路由依赖
            GraphEdge(src=gate_id, dst=expert_gate_id, edge_type="control"),
            GraphEdge(src=gate_id, dst=expert_up_id, edge_type="control"),
            GraphEdge(
                src=expert_gate_id, dst=expert_down_id, edge_type="data", tensor=expert_gate_tensor
            ),
            GraphEdge(
                src=expert_up_id, dst=expert_down_id, edge_type="data", tensor=expert_up_tensor
            ),
        ]

        if shared_gate_op and shared_up_op and shared_down_op:
            shared_gate_tensor_shape = (
                [n_shared, tokens, moe_inter] if n_shared > 1 else [tokens, moe_inter]
            )
            shared_up_tensor_shape = (
                [n_shared, tokens, moe_inter] if n_shared > 1 else [tokens, moe_inter]
            )
            shared_down_tensor_shape = (
                [n_shared, tokens, hidden] if n_shared > 1 else [tokens, hidden]
            )

            shared_gate_tensor = self._tensor(
                name=f"{shared_gate_op.name}_out",
                shape=shared_gate_tensor_shape,
                producer_id=shared_gate_op.name,
                consumer_id=shared_down_op.name,
            )
            shared_up_tensor = self._tensor(
                name=f"{shared_up_op.name}_out",
                shape=shared_up_tensor_shape,
                producer_id=shared_up_op.name,
                consumer_id=shared_down_op.name,
            )
            _ = self._tensor(
                name=f"{shared_down_op.name}_out",
                shape=shared_down_tensor_shape,
                producer_id=shared_down_op.name,
            )

            edges.extend(
                [
                    GraphEdge(
                        src=shared_gate_id,
                        dst=shared_down_id,
                        edge_type="data",
                        tensor=shared_gate_tensor,
                    ),
                    GraphEdge(
                        src=shared_up_id,
                        dst=shared_down_id,
                        edge_type="data",
                        tensor=shared_up_tensor,
                    ),
                ]
            )

        _ = expert_down_tensor
        input_nodes = [gate_id]
        output_nodes = [expert_down_id]

        if shared_gate_op and shared_up_op and shared_down_op:
            input_nodes.extend([shared_gate_id, shared_up_id])
            output_nodes.append(shared_down_id)

        return (nodes, edges, input_nodes, output_nodes)
