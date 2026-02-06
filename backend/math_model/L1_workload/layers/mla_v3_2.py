"""MLA v3.2 layer implementation."""

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
    get_num_heads,
    get_seq_len,
    matmul_flops,
)
from math_model.L1_workload.operators.compute.matmul import MatMulOp
from math_model.L1_workload.tensor import TensorDesc

if TYPE_CHECKING:
    from math_model.L1_workload.operators.base import OpBase


@layer_registry.register("mla_v3_2")
class MLAv32Layer(LayerBase):
    """MLA v3.2 层（Top-K 稀疏注意力）。

    计算流程:
        1. Q/KV 低秩投影
        2. 注意力仅对 topk_index 的 KV 做计算
        3. 输出投影

    FLOPs:
        与 MLA 相同，但 KV 长度取 min(kv_seq_len, topk_index)

    Memory:
        weights = 同 MLA
        activations = batch * seq_len * (hidden_size + num_heads * q_dim + num_heads * v_head_dim + hidden_size)

    PyTorch 示例:
        ```python
        import torch
        #x: [batch, seq_len, hidden_size]
        x = torch.randn(batch, seq_len, hidden_size)
        #w_q_a: [hidden_size, q_lora_rank], w_q_b: [q_lora_rank, heads * q_dim]
        q = (x @ w_q_a) @ w_q_b  #q: [batch, seq_len, heads * q_dim]
        #w_kv_a: [hidden_size, kv_lora_rank + qk_rope_head_dim]
        #w_kv_b: [kv_lora_rank, heads * (q_dim + v_dim)]
        kv = (x @ w_kv_a) @ w_kv_b  #kv: [batch, seq_len, heads * (q_dim + v_dim)]
        q = q.view(batch, seq_len, heads, q_dim).transpose(1, 2)
        k = kv[..., : heads * q_dim].view(batch, seq_len, heads, q_dim).transpose(1, 2)
        v = kv[..., heads * q_dim :].view(batch, seq_len, heads, v_dim).transpose(1, 2)
        k = k[..., :effective_kv, :]
        v = v[..., :effective_kv, :]
        attn = torch.softmax(q @ k.transpose(-1, -2), dim=-1) @ v  #attn: [batch, heads, seq, v_dim]
        #w_o: [heads * v_dim, hidden_size]
        out = attn.transpose(1, 2).reshape(batch, seq_len, -1) @ w_o  #out: [batch, seq_len, hidden_size]
        ```

    Config 参数:
        - hidden_size / hidden_dim
        - num_heads / n_heads
        - q_lora_rank / kv_lora_rank
        - qk_nope_head_dim / qk_rope_head_dim / v_head_dim
        - seq_len / q_seq_len
        - kv_seq_len (可选)
        - topk_index: 稀疏注意力截断长度
        - batch / batch_size
    """

    @property
    def op_type(self) -> str:
        return "mla_v3_2"

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
            self._tensor(name="mla_v3_2_out", shape=[batch, seq_len, hidden], is_output=True),
        ]

    def compute_flops(self) -> int:
        batch = get_batch(self._config)
        seq_len = get_seq_len(self._config)
        kv_seq_len = get_kv_seq_len(self._config)
        hidden = get_hidden_size(self._config)
        heads = get_num_heads(self._config)
        topk = int(self._config.get("topk_index", kv_seq_len))
        effective_kv = min(kv_seq_len, topk)

        q_lora_rank = int(self._config.get("q_lora_rank", hidden))
        kv_lora_rank = int(self._config.get("kv_lora_rank", hidden))
        qk_nope = int(self._config.get("qk_nope_head_dim", hidden // heads))
        default_qk_rope = 0  # allow no RoPE dims
        qk_rope = int(self._config.get("qk_rope_head_dim", default_qk_rope))
        v_dim = int(self._config.get("v_head_dim", hidden // heads))
        q_dim = qk_nope + qk_rope

        tokens = batch * seq_len
        flops = 0
        flops += matmul_flops(tokens, hidden, q_lora_rank)
        flops += matmul_flops(tokens, q_lora_rank, heads * q_dim)
        flops += matmul_flops(tokens, hidden, kv_lora_rank + qk_rope)
        flops += matmul_flops(tokens, kv_lora_rank, heads * (qk_nope + v_dim))
        flops += matmul_flops(tokens, heads * v_dim, hidden)
        flops += attention_flops(batch, heads, seq_len, effective_kv, q_dim, v_dim)
        return flops

    def compute_memory(self) -> tuple[int, int]:
        batch = get_batch(self._config)
        seq_len = get_seq_len(self._config)
        hidden = get_hidden_size(self._config)
        heads = get_num_heads(self._config)

        q_lora_rank = int(self._config.get("q_lora_rank", hidden))
        kv_lora_rank = int(self._config.get("kv_lora_rank", hidden))
        qk_nope = int(self._config.get("qk_nope_head_dim", hidden // heads))
        default_qk_rope = 0  # allow no RoPE dims
        qk_rope = int(self._config.get("qk_rope_head_dim", default_qk_rope))
        v_dim = int(self._config.get("v_head_dim", hidden // heads))
        q_dim = qk_nope + qk_rope

        act_bytes = self._activation_dtype.bytes
        weight_bytes_per_elem = self._weight_dtype.bytes
        weight_bytes = (
            hidden * q_lora_rank
            + q_lora_rank * heads * q_dim
            + hidden * (kv_lora_rank + qk_rope)
            + kv_lora_rank * heads * (qk_nope + v_dim)
            + heads * v_dim * hidden
        ) * weight_bytes_per_elem

        activation_bytes = batch * seq_len * (hidden + heads * q_dim + heads * v_dim + hidden) * act_bytes
        return weight_bytes, activation_bytes

    def _build_ops(self) -> list["OpBase"]:
        """构建内部 Op 列表

        计算流程伪代码::

            # 输入: x [B, S, H]
            # 权重: Wq_a [H, q_rank], Wq_b [q_rank, heads*q_dim]
            #       Wkv_a [H, kv_rank + qk_rope], Wkv_b [kv_rank, heads*(qk_nope+v_dim)]
            #       Wo [heads*v_dim, H]

            # Step 1: Q/KV 低秩投影
            # q = (x @ Wq_a) @ Wq_b
            # kv = (x @ Wkv_a) @ Wkv_b

            # Step 2: Top-K Attention
            # k/v 仅保留前 effective_kv
            # score = q @ k^T
            # attn_out = score @ v

            # Step 3: Output Projection
            # out = attn_out.reshape(B*S, heads*v_dim) @ Wo

        Op 分解:
            1. q_a:        MatMul [B*S, H] @ [H, q_rank] -> [B*S, q_rank]
            2. q_b:        MatMul [B*S, q_rank] @ [q_rank, heads*q_dim] -> [B*S, heads*q_dim]
            3. kv_a:       MatMul [B*S, H] @ [H, kv_rank + qk_rope] -> [B*S, kv_rank + qk_rope]
            4. kv_b:       MatMul [B*S, kv_rank] @ [kv_rank, heads*(qk_nope+v_dim)] -> [B*S, heads*(qk_nope+v_dim)]
            5. attn_score: BatchMatMul [B*heads, S, q_dim] @ [B*heads, q_dim, K] -> [B*heads, S, K]
            6. attn_out:   BatchMatMul [B*heads, S, K] @ [B*heads, K, v_dim] -> [B*heads, S, v_dim]
            7. out_proj:   MatMul [B*S, heads*v_dim] @ [heads*v_dim, H] -> [B*S, H]

        注: Softmax/Reshape/Transpose 等 Vector/Memory Op 暂未建模。

        Returns:
            list[OpBase]: Op 列表
        """
        batch = get_batch(self._config)
        seq_len = get_seq_len(self._config)
        kv_seq_len = get_kv_seq_len(self._config)
        hidden = get_hidden_size(self._config)
        heads = get_num_heads(self._config)
        topk = int(self._config.get("topk_index", kv_seq_len))
        effective_kv = min(kv_seq_len, topk)

        q_lora_rank = int(self._config.get("q_lora_rank", hidden))
        kv_lora_rank = int(self._config.get("kv_lora_rank", hidden))
        qk_nope = int(self._config.get("qk_nope_head_dim", hidden // heads))
        default_qk_rope = 0  # allow no RoPE dims
        qk_rope = int(self._config.get("qk_rope_head_dim", default_qk_rope))
        v_dim = int(self._config.get("v_head_dim", hidden // heads))
        q_dim = qk_nope + qk_rope

        tokens = batch * seq_len
        ops: list[OpBase] = []

        # 1. Q LoRA projection: [B*S, H] @ [H, q_rank]
        ops.append(
            self._matmul_op(
                f"{self.name}_q_a",
                {"M": tokens, "K": hidden, "N": q_lora_rank},
            )
        )

        # 2. Q projection: [B*S, q_rank] @ [q_rank, heads*q_dim]
        ops.append(
            self._matmul_op(
                f"{self.name}_q_b",
                {"M": tokens, "K": q_lora_rank, "N": heads * q_dim},
            )
        )

        # 3. KV LoRA projection: [B*S, H] @ [H, kv_rank + qk_rope]
        ops.append(
            self._matmul_op(
                f"{self.name}_kv_a",
                {"M": tokens, "K": hidden, "N": kv_lora_rank + qk_rope},
            )
        )

        # 4. KV projection: [B*S, kv_rank] @ [kv_rank, heads*(qk_nope+v_dim)]
        ops.append(
            self._matmul_op(
                f"{self.name}_kv_b",
                {"M": tokens, "K": kv_lora_rank, "N": heads * (qk_nope + v_dim)},
            )
        )

        # 5. Attention score: [B*heads, S, q_dim] @ [B*heads, q_dim, K]
        ops.append(
            self._matmul_op(
                f"{self.name}_attn_score",
                {"G": batch * heads, "M": seq_len, "K": q_dim, "N": effective_kv},
            )
        )

        # 6. Attention output: [B*heads, S, K] @ [B*heads, K, v_dim]
        ops.append(
            self._matmul_op(
                f"{self.name}_attn_out",
                {"G": batch * heads, "M": seq_len, "K": effective_kv, "N": v_dim},
            )
        )

        # 7. Output projection: [B*S, heads*v_dim] @ [heads*v_dim, H]
        ops.append(
            self._matmul_op(
                f"{self.name}_out_proj",
                {"M": tokens, "K": heads * v_dim, "N": hidden},
            )
        )

        return ops

    def build_intra_graph(
        self, layer_node_id: str
    ) -> tuple[list[GraphNode], list[GraphEdge], list[str], list[str]]:
        """构建 MLA v3.2 的层内 OP 级计算图。"""
        ops = self.get_ops()
        if len(ops) < 7:
            return ([], [], [], [])

        q_a_op, q_b_op, kv_a_op, kv_b_op, score_op, out_op, proj_op = ops
        q_a_id = f"{layer_node_id}::op::{q_a_op.name}"
        q_b_id = f"{layer_node_id}::op::{q_b_op.name}"
        kv_a_id = f"{layer_node_id}::op::{kv_a_op.name}"
        kv_b_id = f"{layer_node_id}::op::{kv_b_op.name}"
        score_id = f"{layer_node_id}::op::{score_op.name}"
        out_id = f"{layer_node_id}::op::{out_op.name}"
        proj_id = f"{layer_node_id}::op::{proj_op.name}"

        nodes = [
            GraphNode(node_id=q_a_id, kind=NodeKind.OP, role=NodeRole.COMPUTE, ref=q_a_op.name),
            GraphNode(node_id=q_b_id, kind=NodeKind.OP, role=NodeRole.COMPUTE, ref=q_b_op.name),
            GraphNode(node_id=kv_a_id, kind=NodeKind.OP, role=NodeRole.COMPUTE, ref=kv_a_op.name),
            GraphNode(node_id=kv_b_id, kind=NodeKind.OP, role=NodeRole.COMPUTE, ref=kv_b_op.name),
            GraphNode(node_id=score_id, kind=NodeKind.OP, role=NodeRole.COMPUTE, ref=score_op.name),
            GraphNode(node_id=out_id, kind=NodeKind.OP, role=NodeRole.COMPUTE, ref=out_op.name),
            GraphNode(node_id=proj_id, kind=NodeKind.OP, role=NodeRole.COMPUTE, ref=proj_op.name),
        ]

        batch = get_batch(self._config)
        seq_len = get_seq_len(self._config)
        kv_seq_len = get_kv_seq_len(self._config)
        hidden = get_hidden_size(self._config)
        heads = get_num_heads(self._config)
        q_lora_rank = int(self._config.get("q_lora_rank", hidden))
        kv_lora_rank = int(self._config.get("kv_lora_rank", hidden))
        qk_nope = int(self._config.get("qk_nope_head_dim", hidden // heads))
        qk_rope = int(self._config.get("qk_rope_head_dim", 0))
        v_dim = int(self._config.get("v_head_dim", hidden // heads))
        q_dim = qk_nope + qk_rope
        effective_kv = min(kv_seq_len, int(self._config.get("topk_index", kv_seq_len)))
        tokens = batch * seq_len

        q_a_tensor = self._tensor(
            name=f"{q_a_op.name}_out",
            shape=[tokens, q_lora_rank],  # shape: [tokens, q_rank]
            producer_id=q_a_op.name,
            consumer_id=q_b_op.name,
        )
        q_b_tensor = self._tensor(
            name=f"{q_b_op.name}_out",
            shape=[batch * heads, seq_len, q_dim],  # shape: [batch*heads, seq, q_dim]
            producer_id=q_b_op.name,
            consumer_id=score_op.name,
        )
        kv_a_tensor = self._tensor(
            name=f"{kv_a_op.name}_out",
            shape=[tokens, kv_lora_rank + qk_rope],  # shape: [tokens, kv_rank+rope]
            producer_id=kv_a_op.name,
            consumer_id=kv_b_op.name,
        )
        kv_b_tensor = self._tensor(
            name=f"{kv_b_op.name}_out",
            shape=[batch * heads, effective_kv, q_dim],  # shape: [batch*heads, kv, q_dim]
            producer_id=kv_b_op.name,
            consumer_id=score_op.name,
        )
        score_tensor = self._tensor(
            name=f"{score_op.name}_out",
            shape=[batch * heads, seq_len, effective_kv],  # shape: [batch*heads, seq, kv]
            producer_id=score_op.name,
            consumer_id=out_op.name,
        )
        out_tensor = self._tensor(
            name=f"{out_op.name}_out",
            shape=[batch * heads, seq_len, v_dim],  # shape: [batch*heads, seq, v_dim]
            producer_id=out_op.name,
            consumer_id=proj_op.name,
        )

        edges = [
            GraphEdge(src=q_a_id, dst=q_b_id, edge_type="data", tensor=q_a_tensor),
            GraphEdge(src=kv_a_id, dst=kv_b_id, edge_type="data", tensor=kv_a_tensor),
            GraphEdge(src=q_b_id, dst=score_id, edge_type="data", tensor=q_b_tensor),
            GraphEdge(src=kv_b_id, dst=score_id, edge_type="data", tensor=kv_b_tensor),
            GraphEdge(src=score_id, dst=out_id, edge_type="data", tensor=score_tensor),
            GraphEdge(src=out_id, dst=proj_id, edge_type="data", tensor=out_tensor),
        ]

        _ = qk_nope
        return (nodes, edges, [q_a_id, kv_a_id], [proj_id])
