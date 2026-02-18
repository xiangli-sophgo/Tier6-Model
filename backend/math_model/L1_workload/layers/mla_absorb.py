"""MLA absorb layer implementation."""

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
from math_model.L1_workload.operators.compute.fa2 import FA2Op
from math_model.L1_workload.tensor import TensorDesc

if TYPE_CHECKING:
    from math_model.L1_workload.operators.base import OpBase


@layer_registry.register("mla_absorb")
class MLAAbsorbLayer(LayerBase):
    """MLA Absorb 层（分解式 KV 投影）。

    计算流程:
        1. Q 低秩投影
        2. KV 分解投影 (w_kc / w_vc)
        3. 注意力计算与输出投影

    FLOPs:
        q_a = 2 * tokens * hidden_size * q_lora_rank
        q_b = 2 * tokens * q_lora_rank * (num_heads * (qk_nope_head_dim + qk_rope_head_dim))
        kv_a = 2 * tokens * hidden_size * (kv_lora_rank + qk_rope_head_dim)
        w_kc = 2 * num_heads * tokens * qk_nope_head_dim * kv_lora_rank
        w_vc = 2 * num_heads * tokens * v_head_dim * kv_lora_rank
        attn = 2 * batch * num_heads * seq_len * kv_seq_len * (q_dim + kv_lora_rank)
        out = 2 * tokens * (num_heads * v_head_dim) * hidden_size

    Memory:
        weights = sum of projection matrices above
        activations = batch * seq_len * (hidden_size + num_heads * (qk_nope_head_dim + qk_rope_head_dim) + num_heads * v_head_dim + hidden_size)

    PyTorch 示例:
        ```python
        import torch
        #x: [batch, seq_len, hidden_size]
        x = torch.randn(batch, seq_len, hidden_size)
        #w_q_a: [hidden_size, q_lora_rank], w_q_b: [q_lora_rank, heads * (qk_nope_head_dim + qk_rope_head_dim)]
        q = (x @ w_q_a) @ w_q_b  #q: [batch, seq_len, heads * (qk_nope_head_dim + qk_rope_head_dim)]
        #w_kv_a: [hidden_size, kv_lora_rank + qk_rope_head_dim]
        kv_a = x @ w_kv_a  #kv_a: [batch, seq_len, kv_lora_rank + qk_rope_head_dim]
        #w_kc: [qk_nope_head_dim, kv_lora_rank], w_vc: [v_head_dim, kv_lora_rank]
        k = kv_a[..., :qk_nope_head_dim] @ w_kc  #k: [batch, seq_len, kv_lora_rank]
        v = kv_a[..., :v_head_dim] @ w_vc  #v: [batch, seq_len, kv_lora_rank]
        q = q.view(batch, seq_len, heads, qk_nope_head_dim + qk_rope_head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, heads, kv_lora_rank).transpose(1, 2)
        v = v.view(batch, seq_len, heads, kv_lora_rank).transpose(1, 2)
        attn = torch.softmax(q @ k.transpose(-1, -2), dim=-1) @ v  #attn: [batch, heads, seq, kv_lora_rank]
        #w_o: [heads * kv_lora_rank, hidden_size]
        out = attn.transpose(1, 2).reshape(batch, seq_len, -1) @ w_o  #out: [batch, seq_len, hidden_size]
        ```

    Config 参数:
        - hidden_size / hidden_dim
        - num_heads / n_heads
        - q_lora_rank / kv_lora_rank
        - qk_nope_head_dim / qk_rope_head_dim / v_head_dim
        - seq_len / q_seq_len
        - kv_seq_len (可选)
        - batch / batch_size
    """

    @property
    def op_type(self) -> str:
        return "mla_absorb"

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
            self._tensor(name="mla_absorb_out", shape=[batch, seq_len, hidden], is_output=True),
        ]

    def compute_flops(self) -> int:
        batch = get_batch(self._config)
        seq_len = get_seq_len(self._config)
        kv_seq_len = get_kv_seq_len(self._config)
        hidden = get_hidden_size(self._config)
        heads = get_num_heads(self._config)

        q_lora_rank = int(self._config.get("q_lora_rank", hidden))
        kv_lora_rank = int(self._config.get("kv_lora_rank", hidden))
        qk_nope = int(self._config.get("qk_nope_head_dim", hidden // heads))
        default_qk_rope = 0  # allow no RoPE dims
        qk_rope = int(self._config.get("qk_rope_head_dim", default_qk_rope))
        v_dim = int(self._config.get("v_head_dim", hidden // heads))
        q_dim = kv_lora_rank + qk_rope

        tokens = batch * seq_len
        flops = 0
        flops += matmul_flops(tokens, hidden, q_lora_rank)
        flops += matmul_flops(tokens, q_lora_rank, heads * (qk_nope + qk_rope))
        flops += matmul_flops(tokens, hidden, kv_lora_rank + qk_rope)
        flops += matmul_flops(tokens, qk_nope, kv_lora_rank, groups=heads)
        flops += matmul_flops(tokens, v_dim, kv_lora_rank, groups=heads)
        flops += matmul_flops(tokens, heads * v_dim, hidden)
        flops += attention_flops(batch, heads, seq_len, kv_seq_len, q_dim, kv_lora_rank)
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

        act_bytes = self._activation_dtype.bytes
        weight_bytes_per_elem = self._weight_dtype.bytes
        weight_bytes = (
            hidden * q_lora_rank
            + q_lora_rank * heads * (qk_nope + qk_rope)
            + hidden * (kv_lora_rank + qk_rope)
            + heads * qk_nope * kv_lora_rank
            + heads * v_dim * kv_lora_rank
            + heads * v_dim * hidden
        ) * weight_bytes_per_elem

        activation_bytes = batch * seq_len * (hidden + heads * (qk_nope + qk_rope) + heads * v_dim + hidden) * act_bytes
        return weight_bytes, activation_bytes

    def _build_ops(self) -> list["OpBase"]:
        """构建内部 Op 列表

        计算流程伪代码::

            # 输入: x [B, S, H]
            # 权重: Wq_a [H, q_rank], Wq_b [q_rank, heads*(qk_nope+qk_rope)]
            #       Wkv_a [H, kv_rank + qk_rope], W_kc [qk_nope, kv_rank], W_vc [v_dim, kv_rank]
            #       Wo [heads*v_dim, H]

            # Step 1: Q 低秩投影
            # q_a = x.reshape(B*S, H) @ Wq_a                 # [B*S, H] @ [H, q_rank]
            # q = q_a @ Wq_b                                 # [B*S, q_rank] @ [q_rank, heads*(qk_nope+qk_rope)]

            # Step 2: KV 分解投影
            # kv_a = x.reshape(B*S, H) @ Wkv_a               # [B*S, H] @ [H, kv_rank + qk_rope]
            # k = kv_a[..., :qk_nope] @ W_kc                 # [heads, B*S, qk_nope] @ [heads, qk_nope, kv_rank]
            # v = kv_a[..., :v_dim] @ W_vc                   # [heads, B*S, v_dim] @ [heads, v_dim, kv_rank]

            # Step 3: Attention
            # q = reshape/transpose -> [B, heads, S, q_dim]
            # k/v = reshape/transpose -> [B, heads, K, kv_rank]
            # score = q @ k^T                                # [B*heads, S, q_dim] @ [B*heads, q_dim, K]
            # attn_out = score @ v                           # [B*heads, S, K] @ [B*heads, K, kv_rank]

            # Step 4: Output Projection
            # out = attn_out.reshape(B*S, heads*v_dim) @ Wo  # [B*S, heads*v_dim] @ [heads*v_dim, H]

        Op 分解:
            1. q_a:        MatMul [B*S, H] @ [H, q_rank] -> [B*S, q_rank]
            2. q_b:        MatMul [B*S, q_rank] @ [q_rank, heads*(qk_nope+qk_rope)] -> [B*S, heads*(qk_nope+qk_rope)]
            3. kv_a:       MatMul [B*S, H] @ [H, kv_rank + qk_rope] -> [B*S, kv_rank + qk_rope]
            4. k_compact:  MatMul [heads, B*S, qk_nope] @ [heads, qk_nope, kv_rank] -> [heads, B*S, kv_rank]
            5. v_compact:  MatMul [heads, B*S, v_dim] @ [heads, v_dim, kv_rank] -> [heads, B*S, kv_rank]
            6. attn_score: BatchMatMul [B*heads, S, q_dim] @ [B*heads, q_dim, K] -> [B*heads, S, K]
            7. attn_out:   BatchMatMul [B*heads, S, K] @ [B*heads, K, kv_rank] -> [B*heads, S, kv_rank]
            8. out_proj:   MatMul [B*S, heads*v_dim] @ [heads*v_dim, H] -> [B*S, H]

        注: Softmax/Reshape/Transpose 等 Vector/Memory Op 暂未建模。

        Returns:
            list[OpBase]: Op 列表
        """
        batch = get_batch(self._config)
        seq_len = get_seq_len(self._config)
        kv_seq_len = get_kv_seq_len(self._config)
        hidden = get_hidden_size(self._config)
        heads = get_num_heads(self._config)

        q_lora_rank = int(self._config.get("q_lora_rank", hidden))
        kv_lora_rank = int(self._config.get("kv_lora_rank", hidden))
        qk_nope = int(self._config.get("qk_nope_head_dim", hidden // heads))
        default_qk_rope = 0  # allow no RoPE dims
        qk_rope = int(self._config.get("qk_rope_head_dim", default_qk_rope))
        v_dim = int(self._config.get("v_head_dim", hidden // heads))
        q_dim = kv_lora_rank + qk_rope

        tokens = batch * seq_len
        ops: list[OpBase] = []

        # 1. Q LoRA projection: [B*S, H] @ [H, q_rank]
        ops.append(
            self._matmul_op(
                f"{self.name}_q_a",
                {"M": tokens, "K": hidden, "N": q_lora_rank},
            )
        )

        # 2. Q projection: [B*S, q_rank] @ [q_rank, heads*(qk_nope+qk_rope)]
        ops.append(
            self._matmul_op(
                f"{self.name}_q_b",
                {"M": tokens, "K": q_lora_rank, "N": heads * (qk_nope + qk_rope)},
            )
        )

        # 3. KV LoRA projection: [B*S, H] @ [H, kv_rank + qk_rope]
        ops.append(
            self._matmul_op(
                f"{self.name}_kv_a",
                {"M": tokens, "K": hidden, "N": kv_lora_rank + qk_rope},
            )
        )

        # 4. K compact projection: [heads, B*S, qk_nope] @ [heads, qk_nope, kv_rank]
        ops.append(
            self._matmul_op(
                f"{self.name}_k_compact",
                {"G": heads, "M": tokens, "K": qk_nope, "N": kv_lora_rank},
            )
        )

        # 5. V compact projection: [heads, B*S, v_dim] @ [heads, v_dim, kv_rank]
        ops.append(
            self._matmul_op(
                f"{self.name}_v_compact",
                {"G": heads, "M": tokens, "K": v_dim, "N": kv_lora_rank},
            )
        )

        # 6. Fused FA2 attention: QK^T + softmax + PV
        ops.append(
            self._fa2_op(
                f"{self.name}_attn",
                {
                    "B": batch * heads,
                    "QS": seq_len,
                    "KS": kv_seq_len,
                    "QD": q_dim,
                    "VD": kv_lora_rank,
                },
            )
        )

        # 8. Output projection: [B*S, heads*v_dim] @ [heads*v_dim, H]
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
        """构建 MLA-Absorb 的层内 OP 级计算图。"""
        ops = self.get_ops()
        if len(ops) < 7:
            return ([], [], [], [])

        q_a_op, q_b_op, kv_a_op, k_comp_op, v_comp_op, attn_op, proj_op = ops
        q_a_id = f"{layer_node_id}::op::{q_a_op.name}"
        q_b_id = f"{layer_node_id}::op::{q_b_op.name}"
        kv_a_id = f"{layer_node_id}::op::{kv_a_op.name}"
        k_comp_id = f"{layer_node_id}::op::{k_comp_op.name}"
        v_comp_id = f"{layer_node_id}::op::{v_comp_op.name}"
        attn_id = f"{layer_node_id}::op::{attn_op.name}"
        proj_id = f"{layer_node_id}::op::{proj_op.name}"

        nodes = [
            GraphNode(node_id=q_a_id, kind=NodeKind.OP, role=NodeRole.COMPUTE, ref=q_a_op.name),
            GraphNode(node_id=q_b_id, kind=NodeKind.OP, role=NodeRole.COMPUTE, ref=q_b_op.name),
            GraphNode(node_id=kv_a_id, kind=NodeKind.OP, role=NodeRole.COMPUTE, ref=kv_a_op.name),
            GraphNode(node_id=k_comp_id, kind=NodeKind.OP, role=NodeRole.COMPUTE, ref=k_comp_op.name),
            GraphNode(node_id=v_comp_id, kind=NodeKind.OP, role=NodeRole.COMPUTE, ref=v_comp_op.name),
            GraphNode(node_id=attn_id, kind=NodeKind.OP, role=NodeRole.COMPUTE, ref=attn_op.name),
            GraphNode(node_id=proj_id, kind=NodeKind.OP, role=NodeRole.COMPUTE, ref=proj_op.name),
        ]

        batch = get_batch(self._config)
        seq_len = get_seq_len(self._config)
        hidden = get_hidden_size(self._config)
        heads = get_num_heads(self._config)
        q_lora_rank = int(self._config.get("q_lora_rank", hidden))
        kv_lora_rank = int(self._config.get("kv_lora_rank", hidden))
        qk_rope = int(self._config.get("qk_rope_head_dim", 0))
        v_dim = int(self._config.get("v_head_dim", hidden // heads))
        tokens = batch * seq_len

        q_a_tensor = self._tensor(
            name=f"{q_a_op.name}_out",
            shape=[tokens, q_lora_rank],
            producer_id=q_a_op.name,
            consumer_id=q_b_op.name,
        )
        q_b_tensor = self._tensor(
            name=f"{q_b_op.name}_out",
            shape=[batch * heads, seq_len, kv_lora_rank + qk_rope],
            producer_id=q_b_op.name,
            consumer_id=attn_op.name,
        )
        kv_a_tensor = self._tensor(
            name=f"{kv_a_op.name}_out",
            shape=[tokens, kv_lora_rank + qk_rope],
            producer_id=kv_a_op.name,
        )
        k_comp_tensor = self._tensor(
            name=f"{k_comp_op.name}_out",
            shape=[batch * heads, seq_len, kv_lora_rank],
            producer_id=k_comp_op.name,
            consumer_id=attn_op.name,
        )
        v_comp_tensor = self._tensor(
            name=f"{v_comp_op.name}_out",
            shape=[batch * heads, seq_len, kv_lora_rank],
            producer_id=v_comp_op.name,
            consumer_id=attn_op.name,
        )
        attn_tensor = self._tensor(
            name=f"{attn_op.name}_out",
            shape=[batch * heads, seq_len, kv_lora_rank],
            producer_id=attn_op.name,
            consumer_id=proj_op.name,
        )

        edges = [
            GraphEdge(src=q_a_id, dst=q_b_id, edge_type="data", tensor=q_a_tensor),
            GraphEdge(src=kv_a_id, dst=k_comp_id, edge_type="data", tensor=kv_a_tensor),
            GraphEdge(src=kv_a_id, dst=v_comp_id, edge_type="data", tensor=kv_a_tensor),
            GraphEdge(src=q_b_id, dst=attn_id, edge_type="data", tensor=q_b_tensor),
            GraphEdge(src=k_comp_id, dst=attn_id, edge_type="data", tensor=k_comp_tensor),
            GraphEdge(src=v_comp_id, dst=attn_id, edge_type="data", tensor=v_comp_tensor),
            GraphEdge(src=attn_id, dst=proj_id, edge_type="data", tensor=attn_tensor),
        ]

        _ = v_dim
        return (nodes, edges, [q_a_id, kv_a_id], [proj_id])
