"""Attention 层实现

定义 Multi-Head Attention / GQA / MQA 等注意力层。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tier6.L1_workload.graph import GraphEdge, GraphNode, NodeKind, NodeRole
from tier6.L1_workload.layers import layer_registry
from tier6.L1_workload.layers.base import LayerBase, LayerRole
from tier6.L1_workload.operators.compute.matmul import MatMulOp
from tier6.L1_workload.specs import TileableDim, TileConfig, TiledMemoryInfo
from tier6.L1_workload.tensor import TensorDesc

if TYPE_CHECKING:
    from tier6.L1_workload.operators.base import OpBase


@layer_registry.register("attention")
class AttentionLayer(LayerBase):
    """Multi-Head Attention 层

    计算流程:
        1. QKV Projection: hidden_states @ Wqkv -> Q, K, V
        2. Attention: softmax(Q @ K^T / sqrt(d)) @ V
        3. Output Projection: attn_output @ Wo -> output

    Config 参数:
        - hidden_size: 隐藏层大小
        - num_heads: 注意力头数
        - num_kv_heads: KV 头数（GQA/MQA）
        - seq_len: 序列长度
        - batch: 批次大小

    Example:
        >>> layer = AttentionLayer("attn_0", {
        ...     "hidden_size": 4096,
        ...     "num_heads": 32,
        ...     "seq_len": 2048,
        ...     "batch": 1,
        ... })
        >>> print(layer.compute_flops())
    """

    @property
    def op_type(self) -> str:
        return "attention"

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
        """获取输出张量: attn_output"""
        default_batch = 1  # common batch fallback
        default_seq_len = 2048  # typical model context length
        default_hidden = 4096  # typical hidden size
        b = self._config.get("batch", default_batch)
        s = self._config.get("seq_len", default_seq_len)
        h = self._config.get("hidden_size", default_hidden)
        return [self._tensor(name="attn_output", shape=[b, s, h], is_output=True)]

    def compute_flops(self) -> int:
        """计算 FLOPs

        FLOPs 分解:
            - QKV Projection: 2 * B * S * H * 3H = 6 * B * S * H^2
            - Attention Score: 2 * B * num_heads * S * S * head_dim = 2 * B * S^2 * H
            - Attention Output: 2 * B * num_heads * S * S * head_dim = 2 * B * S^2 * H
            - Output Projection: 2 * B * S * H * H = 2 * B * S * H^2
        """
        default_batch = 1  # common batch fallback
        default_seq_len = 2048  # typical model context length
        default_hidden = 4096  # typical hidden size
        b = self._config.get("batch", default_batch)
        s = self._config.get("seq_len", default_seq_len)
        h = self._config.get("hidden_size", default_hidden)
        mul_add = 2  # matmul uses 2 * M * K * N
        proj_mats = 4  # Wq, Wk, Wv, Wo
        attn_stages = 2  # score + weighted sum

        # QKV + Output projection
        proj_flops = mul_add * b * s * h * h * proj_mats  # Wq, Wk, Wv, Wo

        # Attention score + output
        attn_flops = mul_add * attn_stages * b * s * s * h  # score + weighted sum

        return proj_flops + attn_flops

    def compute_memory(self) -> tuple[int, int]:
        """计算内存占用

        Returns:
            tuple[int, int]: (weight_bytes, activation_bytes)
        """
        default_batch = 1  # common batch fallback
        default_seq_len = 2048  # typical model context length
        default_hidden = 4096  # typical hidden size
        h = self._config.get("hidden_size", default_hidden)
        b = self._config.get("batch", default_batch)
        s = self._config.get("seq_len", default_seq_len)
        act_bytes = self._activation_dtype.bytes
        weight_bytes_per_elem = self._weight_dtype.bytes
        proj_mats = 4  # Wq, Wk, Wv, Wo
        activation_terms = 7  # input + qkv + scores + out + output

        # 权重: Wq, Wk, Wv, Wo
        weight_bytes = proj_mats * h * h * weight_bytes_per_elem

        # 激活: input, Q, K, V, attn_scores, attn_output, output
        activation_bytes = b * s * h * act_bytes * activation_terms

        return weight_bytes, activation_bytes

    # =========================================================================
    # Op 级别分解（可选，用于精细分析）
    # =========================================================================

    def _build_ops(self) -> list["OpBase"]:
        """构建内部 Op 列表

        计算流程伪代码::

            # 输入: x [B, S, H]
            # 权重: Wq [H, H], Wk [H, H], Wv [H, H], Wo [H, H]
            # 或合并: Wqkv [H, 3H]

            # Step 1: QKV Projection
            # x_flat = x.reshape(B*S, H)
            # qkv = x_flat @ Wqkv                    # [B*S, H] @ [H, 3H] -> [B*S, 3H]
            # Q, K, V = split(qkv, 3)                # 各 [B*S, H]
            # Q = Q.reshape(B, S, num_heads, d).transpose(0, 2, 1, 3)  # [B, num_heads, S, d]
            # K = K.reshape(B, S, num_heads, d).transpose(0, 2, 1, 3)  # [B, num_heads, S, d]
            # V = V.reshape(B, S, num_heads, d).transpose(0, 2, 1, 3)  # [B, num_heads, S, d]

            # Step 2: Attention Score
            # scores = Q @ K.T / sqrt(d)             # [B, num_heads, S, d] @ [B, num_heads, d, S]
            #                                        # -> [B, num_heads, S, S]
            # scores = softmax(scores, dim=-1)       # (Vector 引擎，此处未建模)

            # Step 3: Attention Output
            # attn_out = scores @ V                  # [B, num_heads, S, S] @ [B, num_heads, S, d]
            #                                        # -> [B, num_heads, S, d]
            # attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B*S, H)

            # Step 4: Output Projection
            # output = attn_out @ Wo                 # [B*S, H] @ [H, H] -> [B*S, H]
            # output = output.reshape(B, S, H)

        Op 分解:
            1. qkv_proj:   MatMul [B*S, H] @ [H, 3H] -> [B*S, 3H]
            2. attn_score: BatchMatMul [B*num_heads, S, d] @ [B*num_heads, d, S] -> [B*num_heads, S, S]
            3. attn_out:   BatchMatMul [B*num_heads, S, S] @ [B*num_heads, S, d] -> [B*num_heads, S, d]
            4. out_proj:   MatMul [B*S, H] @ [H, H] -> [B*S, H]

        注: Softmax/Scale/Reshape/Transpose 等 Vector/Memory Op 暂未建模。

        Returns:
            list[OpBase]: Op 列表
        """
        b = self._config.get("batch", 1)
        s = self._config.get("seq_len", 2048)
        h = self._config.get("hidden_size", 4096)
        num_heads = self._config.get("num_heads", 32)
        head_dim = h // num_heads

        ops: list[OpBase] = []

        # 1. QKV Projection: [B*S, H] @ [H, 3H]
        ops.append(
            self._matmul_op(
                f"{self.name}_qkv_proj",
                {"M": b * s, "K": h, "N": 3 * h},
            )
        )

        # 2. Attention Score: [B*num_heads, S, head_dim] @ [B*num_heads, head_dim, S]
        ops.append(
            self._matmul_op(
                f"{self.name}_attn_score",
                {"G": b * num_heads, "M": s, "K": head_dim, "N": s},
            )
        )

        # 3. Attention Output: [B*num_heads, S, S] @ [B*num_heads, S, head_dim]
        ops.append(
            self._matmul_op(
                f"{self.name}_attn_out",
                {"G": b * num_heads, "M": s, "K": s, "N": head_dim},
            )
        )

        # 4. Output Projection: [B*S, H] @ [H, H]
        ops.append(
            self._matmul_op(
                f"{self.name}_out_proj",
                {"M": b * s, "K": h, "N": h},
            )
        )

        return ops

    def compute_flops_from_ops(self) -> int:
        """从 Op 聚合计算 FLOPs（验证用）"""
        return sum(op.compute_ops() for op in self.get_ops())

    def build_intra_graph(
        self, layer_node_id: str
    ) -> tuple[list[GraphNode], list[GraphEdge], list[str], list[str]]:
        """构建 Attention 的层内 OP 级计算图。"""
        ops = self.get_ops()
        if len(ops) < 4:
            return ([], [], [], [])

        qkv_op, score_op, out_op, proj_op = ops[0], ops[1], ops[2], ops[3]
        qkv_id = f"{layer_node_id}::op::{qkv_op.name}"
        score_id = f"{layer_node_id}::op::{score_op.name}"
        out_id = f"{layer_node_id}::op::{out_op.name}"
        proj_id = f"{layer_node_id}::op::{proj_op.name}"

        nodes = [
            GraphNode(node_id=qkv_id, kind=NodeKind.OP, role=NodeRole.COMPUTE, ref=qkv_op.name),
            GraphNode(node_id=score_id, kind=NodeKind.OP, role=NodeRole.COMPUTE, ref=score_op.name),
            GraphNode(node_id=out_id, kind=NodeKind.OP, role=NodeRole.COMPUTE, ref=out_op.name),
            GraphNode(node_id=proj_id, kind=NodeKind.OP, role=NodeRole.COMPUTE, ref=proj_op.name),
        ]

        b = self._config.get("batch", 1)
        s = self._config.get("seq_len", 2048)
        h = self._config.get("hidden_size", 4096)
        num_heads = self._config.get("num_heads", 32)
        head_dim = h // num_heads

        qkv_tensor = self._tensor(
            name=f"{qkv_op.name}_out",
            shape=[b * s, 3 * h],  # shape: [tokens, 3*hidden]
            producer_id=qkv_op.name,
            consumer_id=score_op.name,
        )
        score_tensor = self._tensor(
            name=f"{score_op.name}_out",
            shape=[b * num_heads, s, s],  # shape: [batch*heads, seq, seq]
            producer_id=score_op.name,
            consumer_id=out_op.name,
        )
        out_tensor = self._tensor(
            name=f"{out_op.name}_out",
            shape=[b * num_heads, s, head_dim],  # shape: [batch*heads, seq, head_dim]
            producer_id=out_op.name,
            consumer_id=proj_op.name,
        )

        edges = [
            GraphEdge(src=qkv_id, dst=score_id, edge_type="data", tensor=qkv_tensor),
            GraphEdge(src=score_id, dst=out_id, edge_type="data", tensor=score_tensor),
            GraphEdge(src=out_id, dst=proj_id, edge_type="data", tensor=out_tensor),
        ]

        return (nodes, edges, [qkv_id], [proj_id])

    # =========================================================================
    # Tile-Aware 接口
    # =========================================================================

    def get_tileable_dims(self) -> list[TileableDim]:
        """返回可 tiling 的维度"""
        b = self._config.get("batch", 1)
        s = self._config.get("seq_len", 2048)
        h = self._config.get("hidden_size", 4096)

        return [
            TileableDim(name="m", size=b * s, min_tile=64, alignment=64),
            TileableDim(name="n", size=h, min_tile=64, alignment=64),
            TileableDim(
                name="k", size=h, min_tile=256, alignment=256, is_reduction=True
            ),
        ]

    def get_tiled_memory(self, tile_config: TileConfig) -> TiledMemoryInfo:
        """计算 tiling 后的内存需求"""
        b = self._config.get("batch", 1)
        s = self._config.get("seq_len", 2048)
        h = self._config.get("hidden_size", 4096)
        act_bytes = self._activation_dtype.bytes
        weight_bytes_per_elem = self._weight_dtype.bytes

        tile_m = tile_config.tile_m or (b * s)
        tile_n = tile_config.tile_n or h
        tile_k = tile_config.tile_k or h

        # QKV Projection 为主要内存瓶颈
        input_buffer = tile_m * tile_k * act_bytes
        weight_buffer = tile_k * tile_n * weight_bytes_per_elem
        output_buffer = tile_m * tile_n * act_bytes

        num_tiles = ((b * s + tile_m - 1) // tile_m) * ((h + tile_k - 1) // tile_k)

        return TiledMemoryInfo(
            input_buffer=input_buffer,
            output_buffer=output_buffer,
            weight_buffer=weight_buffer,
            peak_lmem=input_buffer + weight_buffer + output_buffer,
            num_tiles=num_tiles,
        )
