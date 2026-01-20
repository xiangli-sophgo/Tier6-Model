"""
MoE (Mixture of Experts) 层

包含共享专家和路由专家，支持专家并行 (EP)
"""

from dataclasses import dataclass, field
from typing import Dict, Any

from .base import BaseLayer
from ..operators import (
    MatMulOperator,
    RMSNormOperator,
    AllReduceOperator,
    DispatchOperator,
    CombineOperator,
)


@dataclass
class MoELayer(BaseLayer):
    """
    MoE 层 - DeepSeek V3 风格

    结构:
    - Gate Router (选择 top-k 专家)
    - Dispatch (EP > 1 时分发 token)
    - Routed Experts (Gate + Up + Down)
    - Combine (EP > 1 时汇集结果)
    - Shared Experts (可选)

    config 必须包含:
        - hidden_dim: int, 隐藏维度
        - inter_dim: int, 专家中间层维度
        - num_experts: int, 总专家数
        - num_activated_experts: int, 激活的专家数 (top-k)
        - num_shared_experts: int, 共享专家数 (0 表示无)
        - batch_size: int, 批次大小
        - seq_len: int, 序列长度
        - tp: int, 张量并行度
        - moe_tp: int, MoE 张量并行度
        - ep: int, 专家并行度
        - comm_protocol: int, 通信协议
        - is_prefill: bool, 是否为 prefill 阶段
    """
    name: str = "moe"
    layer_type: str = "MoE"
    config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """构建层内算子"""
        self.comp_ops = []
        self.comm_ops = []
        self.operator_categories = {}
        self._build_operators()

    def _build_operators(self):
        """构建 MoE 层的算子"""
        cfg = self.config
        hidden_dim = cfg.get('hidden_dim', 7168)
        inter_dim = cfg.get('inter_dim', 2048)  # 每个专家的 inter_dim
        num_experts = cfg.get('num_experts', 256)
        num_activated = cfg.get('num_activated_experts', 8)
        num_shared = cfg.get('num_shared_experts', 1)
        batch_size = cfg.get('batch_size', 1)
        seq_len = cfg.get('seq_len', 1)
        tp = cfg.get('tp', 1)
        moe_tp = cfg.get('moe_tp', 1)
        ep = cfg.get('ep', 1)
        comm_protocol = cfg.get('comm_protocol', 1)
        is_prefill = cfg.get('is_prefill', False)

        tokens = batch_size * seq_len
        dtype_bytes = 2  # bf16

        # 1. RMSNorm
        rmsnorm_op = RMSNormOperator(
            name=f"{self.name}_rmsnorm",
            parallel_params={
                'batch_size': tokens,
                'hidden_dim': hidden_dim,
                'has_scale': True,
                'has_bias': False,
            }
        )
        self.add_operator(rmsnorm_op)

        # 2. Gate Router: hidden_dim -> num_experts
        gate_op = MatMulOperator(
            name=f"{self.name}_gate",
            parallel_params={
                'G': 1,
                'M': tokens,
                'K': hidden_dim,
                'N': num_experts,
                'input_dtype': 'bf16',
                'output_dtype': 'bf16',
            }
        )
        self.add_operator(gate_op)

        # 3. Dispatch (EP > 1 时)
        if ep > 1:
            dispatch_comm_size = tokens * hidden_dim * dtype_bytes
            dispatch_op = DispatchOperator(
                name=f"{self.name}_dispatch",
                parallel_params={
                    'moe_tp': moe_tp,
                    'ep': ep,
                    'comm_size': dispatch_comm_size,
                    'batch_size': batch_size,
                    'comm_protocol': comm_protocol,
                    'is_prefill': is_prefill,
                }
            )
            self.add_operator(dispatch_op)

        # 4. Routed Experts (Gate + Up + Down)
        # 每个 EP rank 负责 num_experts / ep 个专家
        experts_per_ep = num_experts // ep
        # 激活 token 数量 = tokens * num_activated / ep (平均分配假设)
        activated_tokens = tokens * num_activated // ep

        # Routed Gate: hidden_dim -> inter_dim
        routed_gate_op = MatMulOperator(
            name=f"{self.name}_routed_gate",
            parallel_params={
                'G': experts_per_ep,
                'M': activated_tokens,
                'K': hidden_dim,
                'N': inter_dim // moe_tp,
                'input_dtype': 'bf16',
                'output_dtype': 'bf16',
            }
        )
        self.add_operator(routed_gate_op)

        # Routed Up: hidden_dim -> inter_dim
        routed_up_op = MatMulOperator(
            name=f"{self.name}_routed_up",
            parallel_params={
                'G': experts_per_ep,
                'M': activated_tokens,
                'K': hidden_dim,
                'N': inter_dim // moe_tp,
                'input_dtype': 'bf16',
                'output_dtype': 'bf16',
            }
        )
        self.add_operator(routed_up_op)

        # Routed Down: inter_dim -> hidden_dim
        routed_down_op = MatMulOperator(
            name=f"{self.name}_routed_down",
            parallel_params={
                'G': experts_per_ep,
                'M': activated_tokens,
                'K': inter_dim // moe_tp,
                'N': hidden_dim,
                'input_dtype': 'bf16',
                'output_dtype': 'bf16',
            }
        )
        self.add_operator(routed_down_op)

        # MoE TP > 1 时需要 AllReduce (routed experts)
        if moe_tp > 1:
            routed_comm_size = activated_tokens * hidden_dim * dtype_bytes
            routed_allreduce_op = AllReduceOperator(
                name=f"{self.name}_routed_allreduce",
                parallel_params={
                    'tp': moe_tp,
                    'comm_size': routed_comm_size,
                    'comm_protocol': comm_protocol,
                }
            )
            self.add_operator(routed_allreduce_op)

        # 5. Combine (EP > 1 时)
        if ep > 1:
            combine_comm_size = tokens * hidden_dim * dtype_bytes
            combine_op = CombineOperator(
                name=f"{self.name}_combine",
                parallel_params={
                    'moe_tp': moe_tp,
                    'ep': ep,
                    'comm_size': combine_comm_size,
                    'batch_size': batch_size,
                    'comm_protocol': comm_protocol,
                    'is_prefill': is_prefill,
                }
            )
            self.add_operator(combine_op)

        # 6. Shared Experts (如果有)
        if num_shared > 0:
            shared_inter_dim = inter_dim * num_shared

            # Shared Gate
            shared_gate_op = MatMulOperator(
                name=f"{self.name}_shared_gate",
                parallel_params={
                    'G': 1,
                    'M': tokens,
                    'K': hidden_dim,
                    'N': shared_inter_dim // tp,
                    'input_dtype': 'bf16',
                    'output_dtype': 'bf16',
                }
            )
            self.add_operator(shared_gate_op)

            # Shared Up
            shared_up_op = MatMulOperator(
                name=f"{self.name}_shared_up",
                parallel_params={
                    'G': 1,
                    'M': tokens,
                    'K': hidden_dim,
                    'N': shared_inter_dim // tp,
                    'input_dtype': 'bf16',
                    'output_dtype': 'bf16',
                }
            )
            self.add_operator(shared_up_op)

            # Shared Down
            shared_down_op = MatMulOperator(
                name=f"{self.name}_shared_down",
                parallel_params={
                    'G': 1,
                    'M': tokens,
                    'K': shared_inter_dim // tp,
                    'N': hidden_dim,
                    'input_dtype': 'bf16',
                    'output_dtype': 'bf16',
                }
            )
            self.add_operator(shared_down_op)

            # Shared AllReduce (TP > 1)
            if tp > 1:
                shared_comm_size = tokens * hidden_dim * dtype_bytes
                shared_allreduce_op = AllReduceOperator(
                    name=f"{self.name}_shared_allreduce",
                    parallel_params={
                        'tp': tp,
                        'comm_size': shared_comm_size,
                        'comm_protocol': comm_protocol,
                    }
                )
                self.add_operator(shared_allreduce_op)
