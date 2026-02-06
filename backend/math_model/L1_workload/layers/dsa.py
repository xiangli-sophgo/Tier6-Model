"""DSA layer implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from math_model.L1_workload.graph import GraphEdge, GraphNode, NodeKind, NodeRole
from math_model.L1_workload.layers import layer_registry
from math_model.L1_workload.layers.base import LayerBase, LayerRole
from math_model.L1_workload.layers.utils import (
    attention_flops,
    get_batch,
    get_hidden_size,
    get_kv_seq_len,
    get_seq_len,
    matmul_flops,
)
from math_model.L1_workload.operators.compute.matmul import MatMulOp
from math_model.L1_workload.tensor import TensorDesc

if TYPE_CHECKING:
    from math_model.L1_workload.operators.base import OpBase


@layer_registry.register("dsa")
class DSALayer(LayerBase):
    """Deep Sparse Attention (DSA) 层。

    计算流程:
        1. Index head projection
        2. Sparse attention over topk_index KV
        3. 输出加权聚合

    FLOPs:
        wq_b = 2 * tokens * q_lora_rank * (n_index_heads * index_head_dim)
        wk = 2 * tokens * hidden_size * index_head_dim
        weights_proj = 2 * tokens * hidden_size * n_index_heads
        attn = 2 * batch * n_index_heads * seq_len * effective_kv * (q_dim + v_dim)

    Memory:
        weights = q_lora_rank * n_index_heads * index_head_dim + hidden_size * index_head_dim + hidden_size * n_index_heads
        activations = batch * seq_len * (hidden_size + n_index_heads * (index_head_dim + v_dim) + hidden_size)

    Config 参数:
        - hidden_size / hidden_dim
        - n_index_heads
        - index_head_dim
        - q_lora_rank
        - seq_len / q_seq_len
        - kv_seq_len (可选)
        - topk_index: 稀疏注意力截断长度
        - batch / batch_size
    """

    @property
    def op_type(self) -> str:
        return "dsa"

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
            self._tensor(name="dsa_out", shape=[batch, seq_len, hidden], is_output=True),
        ]

    def compute_flops(self) -> int:
        batch = get_batch(self._config)
        seq_len = get_seq_len(self._config)
        kv_seq_len = get_kv_seq_len(self._config)
        hidden = get_hidden_size(self._config)
        default_index_heads = 1  # fallback to single head
        n_index_heads = int(self._config.get("n_index_heads", default_index_heads))
        min_head_dim = 1  # avoid zero dim
        index_head_dim = int(
            self._config.get(
                "index_head_dim", max(min_head_dim, hidden // n_index_heads)
            )
        )
        topk = int(self._config.get("topk_index", kv_seq_len))
        effective_kv = min(kv_seq_len, topk)

        q_lora_rank = int(self._config.get("q_lora_rank", hidden))
        tokens = batch * seq_len

        flops = 0
        flops += matmul_flops(tokens, q_lora_rank, n_index_heads * index_head_dim)
        flops += matmul_flops(tokens, hidden, index_head_dim)
        flops += matmul_flops(tokens, hidden, n_index_heads)

        min_q_dim = 1  # avoid zero dim
        q_dim = max(min_q_dim, index_head_dim // 2)
        default_value_dim = 64  # typical sparse attention value dim
        v_dim = int(self._config.get("value_dim", default_value_dim))
        flops += attention_flops(
            batch, n_index_heads, seq_len, effective_kv, q_dim, v_dim
        )
        return flops

    def compute_memory(self) -> tuple[int, int]:
        batch = get_batch(self._config)
        seq_len = get_seq_len(self._config)
        hidden = get_hidden_size(self._config)
        default_index_heads = 1  # fallback to single head
        n_index_heads = int(self._config.get("n_index_heads", default_index_heads))
        min_head_dim = 1  # avoid zero dim
        index_head_dim = int(
            self._config.get(
                "index_head_dim", max(min_head_dim, hidden // n_index_heads)
            )
        )
        q_lora_rank = int(self._config.get("q_lora_rank", hidden))
        default_value_dim = 64  # typical sparse attention value dim
        v_dim = int(self._config.get("value_dim", default_value_dim))

        act_bytes = self._activation_dtype.bytes
        weight_bytes_per_elem = self._weight_dtype.bytes
        weight_bytes = (
            q_lora_rank * n_index_heads * index_head_dim
            + hidden * index_head_dim
            + hidden * n_index_heads
        ) * weight_bytes_per_elem

        activation_bytes = (
            batch
            * seq_len
            * (hidden + n_index_heads * (index_head_dim + v_dim) + hidden)
            * act_bytes
        )
        return weight_bytes, activation_bytes

    def _build_ops(self) -> list["OpBase"]:
        """构建内部 Op 列表

        计算流程伪代码::

            # 输入: x [B, S, H], x_compressed [B, S, q_lora_rank] (由上游 LoRA 压缩)
            # 权重: Wq_b [q_lora_rank, nH*d]  (LoRA up-projection)
            #       Wk   [H, d]               (Key projection)
            #       Ww   [H, nH]              (Weight/Value projection)
            # 其中: nH = n_index_heads, d = index_head_dim

            # Step 1: Query projection (LoRA up-projection)
            # 注: 完整 LoRA 为 x @ Wq_a @ Wq_b，此处假设 x_compressed = x @ Wq_a 已由上游完成
            # q_proj = x_compressed @ Wq_b          # [B, S, q_lora_rank] @ [q_lora_rank, nH*d]
            #                                        # -> [B, S, nH*d]
            # q = q_proj.reshape(B, S, nH, d).transpose(0, 2, 1, 3)  # [B, nH, S, d]

            # Step 2: Key/Weight projection
            # k_proj = x @ Wk                        # [B, S, H] @ [H, d] -> [B, S, d]
            # w_proj = x @ Ww                        # [B, S, H] @ [H, nH] -> [B, S, nH]
            # k = k_proj.unsqueeze(1)                # [B, 1, S, d] (broadcast across heads)
            # w = w_proj.transpose(-1, -2)           # [B, nH, S] (attention weights)

            # Step 3: Sparse attention
            # q_index = q[..., :q_dim]               # 取前 q_dim 维作为 index
            # k_index = k[..., :q_dim]               # [B, 1, S, q_dim]
            # scores = q_index @ k_index.T           # [B, nH, S, q_dim] @ [B, 1, q_dim, K]
            #                                        # -> [B, nH, S, K]
            # attn_weights = softmax(scores)         # (Vector 引擎)
            # attn_out = attn_weights @ v            # [B, nH, S, K] @ [B, nH, K, v_dim]
            #                                        # -> [B, nH, S, v_dim]

        Op 分解:
            1. q_proj:     MatMul [B*S, q_lora_rank] @ [q_lora_rank, nH*d] -> [B*S, nH*d]
            2. k_proj:     MatMul [B*S, H] @ [H, d] -> [B*S, d]
            3. w_proj:     MatMul [B*S, H] @ [H, nH] -> [B*S, nH]
            4. attn_score: BatchMatMul [B*nH, S, q_dim] @ [B*nH, q_dim, K] -> [B*nH, S, K]
            5. attn_out:   BatchMatMul [B*nH, S, K] @ [B*nH, K, v_dim] -> [B*nH, S, v_dim]

        注: Softmax/Reshape/Transpose 等 Vector/Memory Op 暂未建模。

        Returns:
            list[OpBase]: Op 列表
        """
        batch = get_batch(self._config)
        seq_len = get_seq_len(self._config)
        kv_seq_len = get_kv_seq_len(self._config)
        hidden = get_hidden_size(self._config)
        default_index_heads = 1  # fallback to single head
        n_index_heads = int(self._config.get("n_index_heads", default_index_heads))
        min_head_dim = 1  # avoid zero dim
        index_head_dim = int(
            self._config.get(
                "index_head_dim", max(min_head_dim, hidden // n_index_heads)
            )
        )
        topk = int(self._config.get("topk_index", kv_seq_len))
        effective_kv = min(kv_seq_len, topk)
        q_lora_rank = int(self._config.get("q_lora_rank", hidden))

        min_q_dim = 1  # avoid zero dim
        q_dim = max(min_q_dim, index_head_dim // 2)
        default_value_dim = 64  # typical sparse attention value dim
        v_dim = int(self._config.get("value_dim", default_value_dim))

        tokens = batch * seq_len
        ops: list[OpBase] = []

        # 1. Index head projection: [B*S, q_lora_rank] @ [q_lora_rank, n_index_heads*index_head_dim]
        ops.append(
            self._matmul_op(
                f"{self.name}_q_proj",
                {"M": tokens, "K": q_lora_rank, "N": n_index_heads * index_head_dim},
            )
        )

        # 2. Key projection: [B*S, hidden] @ [hidden, index_head_dim]
        ops.append(
            self._matmul_op(
                f"{self.name}_k_proj",
                {"M": tokens, "K": hidden, "N": index_head_dim},
            )
        )

        # 3. Weight projection: [B*S, hidden] @ [hidden, n_index_heads]
        ops.append(
            self._matmul_op(
                f"{self.name}_weights_proj",
                {"M": tokens, "K": hidden, "N": n_index_heads},
            )
        )

        # 4. Sparse attention score: [B*H, S, q_dim] @ [B*H, q_dim, K]
        ops.append(
            self._matmul_op(
                f"{self.name}_attn_score",
                {
                    "G": batch * n_index_heads,
                    "M": seq_len,
                    "K": q_dim,
                    "N": effective_kv,
                },
            )
        )

        # 5. Sparse attention output: [B*H, S, K] @ [B*H, K, v_dim]
        ops.append(
            self._matmul_op(
                f"{self.name}_attn_out",
                {
                    "G": batch * n_index_heads,
                    "M": seq_len,
                    "K": effective_kv,
                    "N": v_dim,
                },
            )
        )

        return ops

    def build_intra_graph(
        self, layer_node_id: str
    ) -> tuple[list[GraphNode], list[GraphEdge], list[str], list[str]]:
        """构建 DSA 的层内 OP 级计算图。"""
        ops = self.get_ops()
        if len(ops) < 5:
            return ([], [], [], [])

        q_op, k_op, w_op, score_op, out_op = ops[0], ops[1], ops[2], ops[3], ops[4]
        q_id = f"{layer_node_id}::op::{q_op.name}"
        k_id = f"{layer_node_id}::op::{k_op.name}"
        w_id = f"{layer_node_id}::op::{w_op.name}"
        score_id = f"{layer_node_id}::op::{score_op.name}"
        out_id = f"{layer_node_id}::op::{out_op.name}"

        nodes = [
            GraphNode(node_id=q_id, kind=NodeKind.OP, role=NodeRole.COMPUTE, ref=q_op.name),
            GraphNode(node_id=k_id, kind=NodeKind.OP, role=NodeRole.COMPUTE, ref=k_op.name),
            GraphNode(node_id=w_id, kind=NodeKind.OP, role=NodeRole.COMPUTE, ref=w_op.name),
            GraphNode(node_id=score_id, kind=NodeKind.OP, role=NodeRole.COMPUTE, ref=score_op.name),
            GraphNode(node_id=out_id, kind=NodeKind.OP, role=NodeRole.COMPUTE, ref=out_op.name),
        ]

        batch = get_batch(self._config)
        seq_len = get_seq_len(self._config)
        kv_seq_len = get_kv_seq_len(self._config)
        hidden = get_hidden_size(self._config)
        n_index_heads = int(self._config.get("n_index_heads", 1))
        index_head_dim = int(self._config.get("index_head_dim", max(1, hidden // n_index_heads)))
        effective_kv = min(kv_seq_len, int(self._config.get("topk_index", kv_seq_len)))
        q_dim = max(1, index_head_dim // 2)
        v_dim = int(self._config.get("value_dim", 64))
        tokens = batch * seq_len

        q_tensor = self._tensor(
            name=f"{q_op.name}_out",
            shape=[tokens, n_index_heads * index_head_dim],  # shape: [tokens, heads*dim]
            producer_id=q_op.name,
            consumer_id=score_op.name,
        )
        k_tensor = self._tensor(
            name=f"{k_op.name}_out",
            shape=[tokens, index_head_dim],  # shape: [tokens, dim]
            producer_id=k_op.name,
            consumer_id=score_op.name,
        )
        score_tensor = self._tensor(
            name=f"{score_op.name}_out",
            shape=[batch * n_index_heads, seq_len, effective_kv],  # shape: [batch*heads, seq, k]
            producer_id=score_op.name,
            consumer_id=out_op.name,
        )
        out_tensor = self._tensor(
            name=f"{out_op.name}_out",
            shape=[batch * n_index_heads, seq_len, v_dim],  # shape: [batch*heads, seq, v_dim]
            producer_id=out_op.name,
        )

        edges = [
            GraphEdge(src=q_id, dst=score_id, edge_type="data", tensor=q_tensor),
            GraphEdge(src=k_id, dst=score_id, edge_type="data", tensor=k_tensor),
            GraphEdge(src=score_id, dst=out_id, edge_type="data", tensor=score_tensor),
            # 权重投影影响注意力加权，使用 control 边表达依赖
            GraphEdge(src=w_id, dst=out_id, edge_type="control"),
        ]

        _ = out_tensor, q_dim
        return (nodes, edges, [q_id, k_id, w_id], [out_id])
