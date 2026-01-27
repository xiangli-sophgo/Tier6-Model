"""
MoE (Mixture of Experts) 层

包含共享专家和路由专家，支持专家并行 (EP)
"""

import math
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
from ..evaluators import get_max_expert_load_for_moe_layer


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
        - num_experts_per_tok: int, 激活的专家数 (top-k)
        - num_shared_experts: int, 共享专家数 (0 表示无)
        - batch_size: int, 全局批次大小 (对齐 DS_TPU)
        - seq_len: int, 序列长度
        - tp: int, 张量并行度
        - dp: int, 数据并行度 (用于计算 local_batch)
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
        num_activated = cfg.get('num_experts_per_tok', 8)
        num_shared = cfg.get('num_shared_experts', 1)
        batch_size = cfg.get('batch_size', 1)  # 全局 batch (对齐 DS_TPU)
        seq_len = cfg.get('seq_len', 1)
        tp = cfg.get('tp', 1)
        dp = cfg.get('dp', 1)  # 数据并行度
        moe_tp = cfg.get('moe_tp', 1)
        ep = cfg.get('ep', 1)
        comm_protocol = cfg.get('comm_protocol', 1)
        is_prefill = cfg.get('is_prefill', False)

        # 计算本地 batch (对齐 DS_TPU: local_batch = batch_size // dp)
        local_batch = batch_size // dp if dp > 0 else batch_size
        global_batch = batch_size  # batch_size 现在就是全局 batch
        tokens = local_batch * seq_len  # 本地 tokens (用于计算和通信)

        # ============================================================
        # 精度设置 (对齐 DeepSeek V3 技术报告 arXiv:2412.19437)
        # ============================================================
        #
        # DeepSeek V3 使用混合精度策略来平衡性能和精度：
        #
        # 1. GEMM 计算精度:
        #    - input_dtype (activation_dtype): 输入矩阵 A/B 的精度
        #    - output_dtype: 输出矩阵 C 的精度
        #    - W8A8 模式: input=FP8, output=BF16
        #
        # 2. MoE 通信精度 (Dispatch/Combine):
        #    - Dispatch: 使用 activation_dtype (FP8)
        #      "quantizes activation before MoE up-projections into FP8"
        #    - Combine: 始终使用 BF16
        #      "retain BF16 to preserve training precision in critical parts"
        #
        # 3. 保持高精度的算子 (不使用 FP8):
        #    - Embedding, Output Head
        #    - MoE Gating (路由网络)
        #    - Normalization (RMSNorm)
        #    - Attention 算子
        #
        weight_dtype = cfg.get('weight_dtype', 'fp8')
        activation_dtype = cfg.get('activation_dtype', 'fp8')
        output_dtype = cfg.get('output_dtype', 'bf16')

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
                'input_dtype': activation_dtype,  # 使用配置的激活精度
                'output_dtype': output_dtype,     # 使用配置的输出精度
            }
        )
        self.add_operator(gate_op)

        # ============================================================
        # 3. Dispatch 通信 (EP > 1 时)
        # ============================================================
        #
        # Dispatch: 将 token 分发到各个专家所在的芯片
        #
        # 通信精度: 使用 activation_dtype (默认 FP8)
        # 原因: "quantizes activation before MoE up-projections into FP8,
        #        which is compatible with FP8 Fprop in MoE up-projections"
        #
        # 通信量计算:
        #   dispatch_size = token_per_ep_group * hidden_dim / moe_tp * activation_dtype_bytes
        #
        # 精度字节数映射
        DTYPE_BYTES = {'fp8': 1, 'fp16': 2, 'bf16': 2, 'fp32': 4, 'int8': 1, 'int4': 0.5}
        activation_dtype_bytes = int(DTYPE_BYTES.get(activation_dtype, 1))

        if ep > 1:
            token_per_ep_group = math.ceil(global_batch * seq_len * num_activated / ep)
            dispatch_comm_size = token_per_ep_group * hidden_dim // moe_tp * activation_dtype_bytes
            dispatch_op = DispatchOperator(
                name=f"{self.name}_dispatch",
                parallel_params={
                    'moe_tp': moe_tp,
                    'ep': ep,
                    'comm_size': dispatch_comm_size,
                    'batch_size': global_batch // ep,  # 对齐 DS_TPU: bs = batch_size // ep
                    'comm_protocol': comm_protocol,
                    'is_prefill': is_prefill,
                }
            )
            self.add_operator(dispatch_op)

        # 4. Routed Experts (Gate + Up + Down)
        #
        # 关键问题：MoE 的负载不均衡
        # - Router 网络为每个 token 随机选择 top-k 个专家
        # - 专家分布到 ep 个芯片，但调用次数不均（某些专家更"热门"）
        # - 最慢的芯片决定总延迟（木桶效应）
        #
        # 解决方案：使用 MoE 负载均衡表
        # - 查表或模拟得到最忙芯片需要加载的专家数
        # - 用这个值作为 GEMM 的 G 维度
        #
        # 计算方式 (与 DS_TPU 对齐):
        # - token_per_ep_group = ceil(global_batch * seq_len * num_activated / ep)
        # - expert_per_ep_group = ceil(num_experts / ep)
        # - m_per_group = ceil(token_per_ep_group / expert_per_ep_group)
        # - G = get_max_expert(global_batch, ep)

        # 计算每个 EP 分组的 token 数和专家数
        token_per_ep_group = math.ceil(global_batch * seq_len * num_activated / ep)
        expert_per_ep_group = math.ceil(num_experts / ep)

        # 每个专家处理的 token 数 (M 维度)
        m_per_group = math.ceil(token_per_ep_group / expert_per_ep_group)
        if m_per_group == 0:
            m_per_group = 1  # 至少 1 个 token

        # 查表获取最忙芯片的专家数 (G 维度)
        # 注意: 使用全局 batch 进行查表
        max_experts_float = get_max_expert_load_for_moe_layer(
            batch_size=global_batch,  # 使用全局 batch
            ep_parallelism=ep,
            num_experts=num_experts,
            topk=num_activated
        )
        max_experts_per_chip = math.ceil(max_experts_float)
        # 限制不超过每 EP 分组的专家数
        max_experts_per_chip = min(max_experts_per_chip, expert_per_ep_group)

        # Routed Gate: hidden_dim -> inter_dim
        # G = 最忙芯片的专家数, M = 每专家处理的 token 数
        routed_gate_op = MatMulOperator(
            name=f"{self.name}_routed_gate",
            parallel_params={
                'G': max_experts_per_chip,  # 负载均衡后的专家数
                'M': m_per_group,           # 每专家处理的 tokens
                'K': hidden_dim,
                'N': inter_dim // moe_tp,
                'input_dtype': activation_dtype,  # 使用配置的激活精度
                'output_dtype': output_dtype,     # 使用配置的输出精度
            }
        )
        self.add_operator(routed_gate_op)

        # Routed Up: hidden_dim -> inter_dim
        routed_up_op = MatMulOperator(
            name=f"{self.name}_routed_up",
            parallel_params={
                'G': max_experts_per_chip,
                'M': m_per_group,
                'K': hidden_dim,
                'N': inter_dim // moe_tp,
                'input_dtype': activation_dtype,
                'output_dtype': output_dtype,
            }
        )
        self.add_operator(routed_up_op)

        # Routed Down: inter_dim -> hidden_dim
        routed_down_op = MatMulOperator(
            name=f"{self.name}_routed_down",
            parallel_params={
                'G': max_experts_per_chip,
                'M': m_per_group,
                'K': inter_dim // moe_tp,
                'N': hidden_dim,
                'input_dtype': activation_dtype,
                'output_dtype': output_dtype,
            }
        )
        self.add_operator(routed_down_op)

        # MoE TP > 1 时需要 AllReduce (routed experts)
        if moe_tp > 1:
            # 通信大小：所有 token 的输出，使用 BF16 精度
            BF16_BYTES = 2
            routed_comm_size = tokens * hidden_dim * BF16_BYTES
            routed_allreduce_op = AllReduceOperator(
                name=f"{self.name}_routed_allreduce",
                parallel_params={
                    'tp': moe_tp,
                    'comm_size': routed_comm_size,
                    'comm_protocol': comm_protocol,
                }
            )
            self.add_operator(routed_allreduce_op)

        # ============================================================
        # ============================================================
        # 5. Combine 通信 (EP > 1 时)
        # ============================================================
        #
        # Combine: 将各专家的计算结果汇聚回原始芯片
        #
        # 通信精度: 始终使用 BF16 (不受 activation_dtype 影响)
        # 原因: "retain BF16 to preserve training precision in critical parts"
        #       专家计算结果是模型输出的关键部分，需要高精度保证模型质量
        #
        # 通信量计算:
        #   combine_size = token_per_ep_group * hidden_dim / moe_tp * BF16_BYTES
        #
        if ep > 1:
            BF16_BYTES = 2  # Combine 始终使用 BF16
            combine_comm_size = token_per_ep_group * hidden_dim // moe_tp * BF16_BYTES
            combine_op = CombineOperator(
                name=f"{self.name}_combine",
                parallel_params={
                    'moe_tp': moe_tp,
                    'ep': ep,
                    'comm_size': combine_comm_size,
                    'batch_size': global_batch // ep,  # 对齐 DS_TPU: bs = batch_size // ep
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
                    'input_dtype': activation_dtype,
                    'output_dtype': output_dtype,
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
                    'input_dtype': activation_dtype,
                    'output_dtype': output_dtype,
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
                    'input_dtype': activation_dtype,
                    'output_dtype': output_dtype,
                }
            )
            self.add_operator(shared_down_op)

            # Shared AllReduce (TP > 1)
            if tp > 1:
                BF16_BYTES = 2
                shared_comm_size = tokens * hidden_dim * BF16_BYTES
                shared_allreduce_op = AllReduceOperator(
                    name=f"{self.name}_shared_allreduce",
                    parallel_params={
                        'tp': tp,
                        'comm_size': shared_comm_size,
                        'comm_protocol': comm_protocol,
                    }
                )
                self.add_operator(shared_allreduce_op)

    def _get_operator_latency(self, op_name_suffix: str) -> float:
        """
        获取算子延迟的辅助函数

        Args:
            op_name_suffix: 算子名称后缀 (如 'dispatch', 'combine', 'routed_gate')

        Returns:
            延迟 (微秒), 如果未找到或未评估则返回 0.0
        """
        # 先在计算算子中查找
        for op in self.comp_ops:
            if op.name.endswith(op_name_suffix):
                return getattr(op, 'elapse', 0.0)

        # 再在通信算子中查找
        for op in self.comm_ops:
            if op.name.endswith(op_name_suffix):
                return getattr(op, 'comm_elapse', 0.0)

        return 0.0

    def calculate_latency_with_tbo(self) -> float:
        """
        计算考虑TBO (Tensor-Bus Overlap) 重叠的MoE层延迟

        TBO原理:
        1. Dispatch通信可以与Routed Experts计算重叠
        2. Combine通信可以与Shared Experts计算重叠 (或与Routed计算重叠)
        3. 如果通信时间 < 计算时间, 通信延迟被完全隐藏

        参考DS_TPU实现:
        - combine_overlap = min_compute_elapse - combine_comm_elapse
        - dispatch_overlap = min_compute_elapse - dispatch_comm_elapse
        - 如果overlap > 0, 通信时间可以被隐藏

        Returns:
            total_latency_us: 总延迟 (微秒)
        """
        # 1. 收集各阶段延迟
        rmsnorm_lat = self._get_operator_latency('rmsnorm')
        gate_lat = self._get_operator_latency('gate')
        dispatch_lat = self._get_operator_latency('dispatch')

        # Routed experts: gate + up + down + allreduce
        routed_gate_lat = self._get_operator_latency('routed_gate')
        routed_up_lat = self._get_operator_latency('routed_up')
        routed_down_lat = self._get_operator_latency('routed_down')
        routed_allreduce_lat = self._get_operator_latency('routed_allreduce')
        routed_compute_lat = routed_gate_lat + routed_up_lat + routed_down_lat + routed_allreduce_lat

        # Shared experts (如果有)
        shared_gate_lat = self._get_operator_latency('shared_gate')
        shared_up_lat = self._get_operator_latency('shared_up')
        shared_down_lat = self._get_operator_latency('shared_down')
        shared_allreduce_lat = self._get_operator_latency('shared_allreduce')
        shared_compute_lat = shared_gate_lat + shared_up_lat + shared_down_lat + shared_allreduce_lat

        combine_lat = self._get_operator_latency('combine')

        # 2. 计算可用于重叠的计算时间
        # 参考DS_TPU: min_compute_elapse = min(mla.elapse, shared_elapse + routed_elapse)
        # 这里我们只看MoE层内部, 因此取shared + routed的总计算时间
        total_compute_lat = routed_compute_lat + shared_compute_lat

        # 3. 计算重叠
        # Dispatch可以与Routed计算重叠
        dispatch_overlap = routed_compute_lat - dispatch_lat
        dispatch_effective = 0.0 if dispatch_overlap > 0 else abs(dispatch_overlap)

        # Combine可以与Shared计算重叠 (如果没有Shared, 则与Routed重叠)
        if shared_compute_lat > 0:
            combine_overlap = shared_compute_lat - combine_lat
        else:
            combine_overlap = routed_compute_lat - combine_lat
        combine_effective = 0.0 if combine_overlap > 0 else abs(combine_overlap)

        # 4. 总延迟 = 不可重叠部分 + 重叠后的通信延迟
        total = (rmsnorm_lat + gate_lat +
                 dispatch_effective +  # 重叠后的Dispatch
                 routed_compute_lat +
                 combine_effective +   # 重叠后的Combine
                 shared_compute_lat)

        return total
