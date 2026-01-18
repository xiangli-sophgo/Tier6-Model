"""
LLM 推理模拟器核心

实现基于拓扑的 GPU/加速器侧精细模拟，包括：
- 数据搬运阶段（PCIe传输、HBM存储、权重加载）
- 推理计算阶段（细化为Attention/FFN/LayerNorm子操作）
- 结果收集阶段（HBM读取、PCIe回传）
"""
from __future__ import annotations

import time
from typing import Any
from dataclasses import dataclass, field

from .types import (
    LLMModelConfig, InferenceConfig, ParallelismStrategy,
    HardwareConfig, HierarchicalTopology,
    ChipHardwareConfig, NodeConfig, ClusterConfig,
    SimulationResult, SimulationStats, PhaseTimeStats,
    GanttTaskType, InferencePhase,
    get_bytes_per_element,
    MLAConfig, MoEConfig,
)
from .topology import TopologyParser
from .latency import (
    calc_pcie_h2d_latency, calc_pcie_d2h_latency,
    calc_weight_load_latency, calc_embedding_latency,
    calc_layernorm_latency, calc_lm_head_latency,
    calc_attention_qkv_latency, calc_attention_score_latency,
    calc_attention_softmax_latency, calc_attention_output_latency,
    calc_ffn_gate_latency, calc_ffn_up_latency, calc_ffn_down_latency,
    calc_kv_cache_read_latency, calc_kv_cache_write_latency,
    calc_tp_allreduce_latency, calc_pp_p2p_latency,
    # MLA 专用
    calc_mla_q_projection_latency, calc_mla_kv_compression_latency,
    calc_mla_attention_score_latency, calc_mla_output_latency,
    calc_mla_kv_cache_read_latency, calc_mla_kv_cache_write_latency,
    # MLA 细粒度
    calc_rmsnorm_q_lora_latency, calc_rmsnorm_kv_lora_latency,
    calc_mm_q_lora_a_latency, calc_mm_q_lora_b_latency, calc_mm_kv_lora_a_latency,
    calc_bmm_qk_latency, calc_bmm_sv_latency, calc_attn_fc_latency,
    # MoE
    calc_moe_gate_latency, calc_moe_expert_ffn_latency, calc_moe_shared_expert_latency,
    calc_ep_dispatch_latency, calc_ep_combine_latency, is_moe_layer,
    # Kernel Fusion
    calc_fused_layernorm_qkv_latency, calc_fused_ffn_gate_up_latency,
    calc_single_layer_latency_fused, OVERLAP_COEFFICIENTS,
    # SP 通信
    calc_sp_allgather_latency, calc_sp_reduce_scatter_latency, calc_sp_comm_volume_gb,
    # EP+TP 组合通信
    calc_ep_tp_combined_latency,
    # DP 梯度同步
    calc_dp_gradient_sync_latency,
)
from .gantt import GanttChartBuilder, convert_to_frontend_format


# ============================================
# 配置验证函数
# ============================================

def validate_mla_config(mla_dict: dict) -> MLAConfig:
    """
    验证并解析 MLA 配置

    Args:
        mla_dict: MLA 配置字典

    Returns:
        MLAConfig 对象

    Raises:
        ValueError: 配置无效时抛出
    """
    required_fields = ["kv_lora_rank", "q_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim"]

    # 检查必填字段
    missing = [f for f in required_fields if f not in mla_dict]
    if missing:
        raise ValueError(f"MLA 配置缺少必填字段: {missing}")

    # 检查值的有效性
    for field in required_fields:
        value = mla_dict[field]
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"MLA 配置 {field} 必须为正整数，当前值: {value}")

    return MLAConfig(
        kv_lora_rank=mla_dict["kv_lora_rank"],
        q_lora_rank=mla_dict["q_lora_rank"],
        qk_nope_head_dim=mla_dict["qk_nope_head_dim"],
        qk_rope_head_dim=mla_dict["qk_rope_head_dim"],
        v_head_dim=mla_dict["v_head_dim"],
    )


def validate_moe_config(moe_dict: dict) -> MoEConfig:
    """
    验证并解析 MoE 配置

    Args:
        moe_dict: MoE 配置字典

    Returns:
        MoEConfig 对象

    Raises:
        ValueError: 配置无效时抛出
    """
    required_fields = ["num_experts", "num_experts_per_tok"]

    # 检查必填字段
    missing = [f for f in required_fields if f not in moe_dict]
    if missing:
        raise ValueError(f"MoE 配置缺少必填字段: {missing}")

    num_experts = moe_dict["num_experts"]
    num_experts_per_tok = moe_dict["num_experts_per_tok"]

    # 检查值的有效性
    if not isinstance(num_experts, int) or num_experts <= 0:
        raise ValueError(f"MoE num_experts 必须为正整数，当前值: {num_experts}")
    if not isinstance(num_experts_per_tok, int) or num_experts_per_tok <= 0:
        raise ValueError(f"MoE num_experts_per_tok 必须为正整数，当前值: {num_experts_per_tok}")
    if num_experts_per_tok > num_experts:
        raise ValueError(f"MoE num_experts_per_tok ({num_experts_per_tok}) 不能大于 num_experts ({num_experts})")

    # 检查 expert_intermediate_size（MoE 模型必须指定）
    expert_intermediate_size = moe_dict.get("expert_intermediate_size", 0)
    if expert_intermediate_size <= 0:
        raise ValueError(f"MoE 配置必须指定 expert_intermediate_size (> 0)，当前值: {expert_intermediate_size}")

    return MoEConfig(
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        expert_capacity_factor=moe_dict.get("expert_capacity_factor", 1.0),
        num_shared_experts=moe_dict.get("num_shared_experts", 0),
        expert_intermediate_size=expert_intermediate_size,
    )


def validate_model_config(model_dict: dict) -> None:
    """
    验证模型配置的有效性

    Args:
        model_dict: 模型配置字典

    Raises:
        ValueError: 配置无效时抛出
    """
    required_fields = ["hidden_size", "num_layers", "num_attention_heads", "intermediate_size"]

    missing = [f for f in required_fields if f not in model_dict]
    if missing:
        raise ValueError(f"模型配置缺少必填字段: {missing}")

    hidden_size = model_dict["hidden_size"]
    num_heads = model_dict["num_attention_heads"]

    if hidden_size <= 0:
        raise ValueError(f"hidden_size 必须为正数，当前值: {hidden_size}")
    if num_heads <= 0:
        raise ValueError(f"num_attention_heads 必须为正数，当前值: {num_heads}")
    if hidden_size % num_heads != 0:
        raise ValueError(f"hidden_size ({hidden_size}) 必须能被 num_attention_heads ({num_heads}) 整除")


def validate_hardware_config(hardware_dict: dict) -> None:
    """
    验证硬件配置的有效性

    Args:
        hardware_dict: 硬件配置字典

    Raises:
        ValueError: 配置无效时抛出
    """
    chip_hw = hardware_dict.get("chip", {})

    # 检查关键硬件参数
    compute_tflops = chip_hw.get("compute_tflops_fp16", 989)
    memory_gb = chip_hw.get("memory_gb", 80)
    memory_bw = chip_hw.get("memory_bandwidth_gbps", 3350)

    if compute_tflops <= 0:
        raise ValueError(f"compute_tflops_fp16 必须为正数，当前值: {compute_tflops}")
    if memory_gb <= 0:
        raise ValueError(f"memory_gb 必须为正数，当前值: {memory_gb}")
    if memory_bw <= 0:
        raise ValueError(f"memory_bandwidth_gbps 必须为正数，当前值: {memory_bw}")
    if memory_bw > 10000:
        raise ValueError(f"memory_bandwidth_gbps ({memory_bw}) 超过合理范围 (最大 10000 GB/s)")


def validate_parallelism_config(parallelism_dict: dict, model_dict: dict | None = None) -> None:
    """
    验证并行策略配置的有效性

    Args:
        parallelism_dict: 并行策略配置字典
        model_dict: 模型配置字典（可选，用于交叉验证）

    Raises:
        ValueError: 配置无效时抛出
    """
    for key in ["dp", "tp", "pp", "ep"]:
        value = parallelism_dict.get(key, 1)
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"并行度 {key} 必须为正整数，当前值: {value}")

    # 交叉验证：PP 不能大于 num_layers
    if model_dict:
        pp = parallelism_dict.get("pp", 1)
        num_layers = model_dict.get("num_layers", 1)
        if pp > num_layers:
            raise ValueError(f"PP ({pp}) 不能大于模型层数 ({num_layers})")

        # 交叉验证：TP 不能大于 num_attention_heads
        tp = parallelism_dict.get("tp", 1)
        num_heads = model_dict.get("num_attention_heads", 1)
        if tp > num_heads:
            raise ValueError(f"TP ({tp}) 不能大于注意力头数 ({num_heads})")


@dataclass
class SimulationConfig:
    """模拟配置"""
    max_simulated_tokens: int = 16
    enable_data_transfer: bool = True
    enable_detailed_ops: bool = True
    enable_kv_cache: bool = True
    enable_overlap: bool = True
    # 新增: Kernel Fusion 和 MLA 优化
    enable_fusion: bool = True      # 启用 Kernel Fusion 优化
    enable_comm_overlap: bool = True  # 启用计算-通信重叠
    # 训练模式配置
    enable_training_mode: bool = False  # 启用训练模式（模拟DP梯度同步）
    enable_dp_gradient_sync: bool = False  # 启用DP梯度同步模拟
    gradient_accumulation_steps: int = 1  # 梯度累积步数


@dataclass
class ChipState:
    """芯片状态"""
    chip_id: str
    pp_stage: int
    tp_rank: int
    current_time: float = 0.0
    compute_idle_at: float = 0.0
    network_idle_at: float = 0.0


class LLMInferenceSimulator:
    """LLM 推理模拟器"""

    def __init__(
        self,
        topology_dict: dict[str, Any],
        model: LLMModelConfig,
        inference: InferenceConfig,
        parallelism: ParallelismStrategy,
        hardware: HardwareConfig,
        config: SimulationConfig | None = None,
    ):
        """
        初始化模拟器

        Args:
            topology_dict: 前端拓扑配置
            model: 模型配置
            inference: 推理配置
            parallelism: 并行策略
            hardware: 硬件配置
            config: 模拟配置
        """
        self.model = model
        self.inference = inference
        self.parallelism = parallelism
        self.hardware = hardware
        self.config = config or SimulationConfig()

        # 解析拓扑
        self.topo_parser = TopologyParser(topology_dict, hardware)
        self.interconnect = self.topo_parser.build_interconnect_graph()
        self.group_assignment = self.topo_parser.map_parallelism(parallelism)

        # 获取 TP 组的链路参数
        if self.group_assignment.tp_groups and len(self.group_assignment.tp_groups[0]) > 1:
            self.tp_bandwidth, self.tp_latency = self.topo_parser.get_link_params_for_group(
                self.group_assignment.tp_groups[0], 'allreduce'
            )
        else:
            self.tp_bandwidth = hardware.node.intra_node_bandwidth_gbps
            self.tp_latency = hardware.node.intra_node_latency_us

        # 获取 PP 组的链路参数
        if self.group_assignment.pp_groups and len(self.group_assignment.pp_groups[0]) > 1:
            self.pp_bandwidth, self.pp_latency = self.topo_parser.get_link_params_for_group(
                self.group_assignment.pp_groups[0], 'p2p'
            )
        else:
            self.pp_bandwidth = hardware.cluster.inter_node_bandwidth_gbps
            self.pp_latency = hardware.cluster.inter_node_latency_us

        # 获取 EP 组的链路参数 (MoE Expert Parallelism)
        if self.group_assignment.ep_groups and len(self.group_assignment.ep_groups[0]) > 1:
            self.ep_bandwidth, self.ep_latency = self.topo_parser.get_link_params_for_group(
                self.group_assignment.ep_groups[0], 'alltoall'
            )
        else:
            # 默认使用节点内带宽 (EP 通常在节点内)
            self.ep_bandwidth = hardware.node.intra_node_bandwidth_gbps
            self.ep_latency = hardware.node.intra_node_latency_us

        # 甘特图构建器
        self.gantt_builder = GanttChartBuilder(parallelism)

        # 芯片状态
        self.chip_states: dict[str, ChipState] = {}
        self._init_chip_states()

        # 统计
        self.prefill_stats = PhaseTimeStats()
        self.decode_stats = PhaseTimeStats()

    def _init_chip_states(self):
        """初始化芯片状态"""
        for assignment in self.group_assignment.assignments:
            self.chip_states[assignment.chip_id] = ChipState(
                chip_id=assignment.chip_id,
                pp_stage=assignment.pp_rank,
                tp_rank=assignment.tp_rank,
            )

    def simulate(self) -> SimulationResult:
        """
        运行完整模拟

        Returns:
            模拟结果
        """
        start_time = time.time()

        current_time = 0.0

        # 阶段1: 数据搬运 (H2D)
        if self.config.enable_data_transfer:
            current_time = self._simulate_data_transfer_h2d(current_time)

        # 阶段2: Prefill 推理
        prefill_end_time = self._simulate_prefill(current_time)
        phase_transition = prefill_end_time

        # 阶段3: Decode 推理
        decode_end_time = self._simulate_decode(prefill_end_time)

        # 阶段4: 数据收集 (D2H)
        if self.config.enable_data_transfer:
            final_time = self._simulate_data_transfer_d2h(decode_end_time)
        else:
            final_time = decode_end_time

        # 构建甘特图
        gantt_data = self.gantt_builder.build(phase_transition=phase_transition)

        # 计算统计信息
        stats = self._compute_stats(final_time)

        return SimulationResult(
            gantt_chart=gantt_data,
            stats=stats,
            timestamp=time.time(),
        )

    def _simulate_data_transfer_h2d(self, start_time: float) -> float:
        """模拟 Host to Device 数据传输"""
        # 计算输入数据大小
        bytes_per_elem = get_bytes_per_element(self.model.dtype)
        input_size_gb = (
            self.inference.batch_size * self.inference.input_seq_length *
            self.model.hidden_size * bytes_per_elem
        ) / (1024 ** 3)

        # PCIe 传输延迟
        pcie_latency = calc_pcie_h2d_latency(input_size_gb, self.hardware)

        # 为第一个 PP stage 的所有芯片添加传输任务
        for chip_id, state in self.chip_states.items():
            if state.pp_stage == 0:
                self.gantt_builder.add_task(
                    name="PCIe H2D",
                    start=start_time,
                    end=start_time + pcie_latency,
                    task_type=GanttTaskType.PCIE_H2D,
                    phase=InferencePhase.PREFILL,
                    chip_id=chip_id,
                    pp_stage=0,
                )
                state.compute_idle_at = start_time + pcie_latency

        return start_time + pcie_latency

    def _simulate_data_transfer_d2h(self, start_time: float) -> float:
        """模拟 Device to Host 数据传输"""
        # 计算输出数据大小 (logits)
        bytes_per_elem = get_bytes_per_element(self.model.dtype)
        output_size_gb = (
            self.inference.batch_size * self.model.vocab_size * bytes_per_elem
        ) / (1024 ** 3)

        # PCIe 传输延迟
        pcie_latency = calc_pcie_d2h_latency(output_size_gb, self.hardware)

        # 为最后一个 PP stage 的所有芯片添加传输任务
        last_stage = self.parallelism.pp - 1
        for chip_id, state in self.chip_states.items():
            if state.pp_stage == last_stage:
                self.gantt_builder.add_task(
                    name="PCIe D2H",
                    start=start_time,
                    end=start_time + pcie_latency,
                    task_type=GanttTaskType.PCIE_D2H,
                    phase=InferencePhase.DECODE,
                    chip_id=chip_id,
                    pp_stage=last_stage,
                )

        return start_time + pcie_latency

    def _simulate_prefill(self, start_time: float) -> float:
        """模拟 Prefill 阶段"""
        num_tokens = self.inference.input_seq_length
        context_length = self.inference.input_seq_length

        # 每个 PP stage 处理的层数（至少为 1，防止除零）
        layers_per_stage = max(1, self.model.num_layers // self.parallelism.pp)

        # 为每个 PP stage 模拟
        stage_times = [start_time] * self.parallelism.pp

        for layer in range(self.model.num_layers):
            pp_stage = layer // layers_per_stage
            if pp_stage >= self.parallelism.pp:
                pp_stage = self.parallelism.pp - 1

            layer_in_stage = layer % layers_per_stage

            # 获取该 stage 的第一个芯片
            chip_id = self._get_chip_for_stage(pp_stage)
            current_time = stage_times[pp_stage]

            # PP 前向传递等待上一个 stage
            if pp_stage > 0 and layer_in_stage == 0:
                prev_stage_end = stage_times[pp_stage - 1]
                if prev_stage_end > current_time:
                    # 添加气泡
                    bubble_duration = prev_stage_end - current_time
                    self.gantt_builder.add_bubble(
                        start=current_time,
                        duration=bubble_duration,
                        phase=InferencePhase.PREFILL,
                        chip_id=chip_id,
                        pp_stage=pp_stage,
                    )
                    current_time = prev_stage_end

                    # PP P2P 通信
                    pp_comm_latency = self._calc_pp_comm_latency(num_tokens)
                    self.gantt_builder.add_comm_task(
                        task_type=GanttTaskType.PP_COMM,
                        start=current_time,
                        duration=pp_comm_latency,
                        phase=InferencePhase.PREFILL,
                        chip_id=chip_id,
                        pp_stage=pp_stage,
                        layer_index=layer,
                    )
                    current_time += pp_comm_latency

            # 模拟单层
            current_time = self._simulate_single_layer(
                current_time=current_time,
                layer_index=layer,
                num_tokens=num_tokens,
                context_length=context_length,
                phase=InferencePhase.PREFILL,
                chip_id=chip_id,
                pp_stage=pp_stage,
            )

            stage_times[pp_stage] = current_time

        # Embedding (在第一层之前) 和 LM Head (在最后一层之后) 已包含在层计算中
        # 返回最后一个 stage 的结束时间
        prefill_end = max(stage_times)

        # 更新统计
        self.prefill_stats.total_time = prefill_end - start_time

        return prefill_end

    def _simulate_decode(self, start_time: float) -> float:
        """模拟 Decode 阶段"""
        current_time = start_time
        num_tokens_to_simulate = min(
            self.config.max_simulated_tokens,
            self.inference.output_seq_length
        )

        # 每个 PP stage 处理的层数（至少为 1，防止除零）
        layers_per_stage = max(1, self.model.num_layers // self.parallelism.pp)

        for token_idx in range(num_tokens_to_simulate):
            context_length = self.inference.input_seq_length + token_idx + 1
            stage_times = [current_time] * self.parallelism.pp

            for layer in range(self.model.num_layers):
                pp_stage = layer // layers_per_stage
                if pp_stage >= self.parallelism.pp:
                    pp_stage = self.parallelism.pp - 1

                layer_in_stage = layer % layers_per_stage
                chip_id = self._get_chip_for_stage(pp_stage)
                layer_start = stage_times[pp_stage]

                # PP 等待
                if pp_stage > 0 and layer_in_stage == 0:
                    prev_end = stage_times[pp_stage - 1]
                    if prev_end > layer_start:
                        bubble = prev_end - layer_start
                        self.gantt_builder.add_bubble(
                            start=layer_start,
                            duration=bubble,
                            phase=InferencePhase.DECODE,
                            chip_id=chip_id,
                            pp_stage=pp_stage,
                        )
                        layer_start = prev_end

                        pp_comm = self._calc_pp_comm_latency(1)
                        self.gantt_builder.add_comm_task(
                            task_type=GanttTaskType.PP_COMM,
                            start=layer_start,
                            duration=pp_comm,
                            phase=InferencePhase.DECODE,
                            chip_id=chip_id,
                            pp_stage=pp_stage,
                            layer_index=layer,
                            token_index=token_idx,
                        )
                        layer_start += pp_comm

                # 模拟单层 (Decode: 1 token)
                layer_end = self._simulate_single_layer(
                    current_time=layer_start,
                    layer_index=layer,
                    num_tokens=1,
                    context_length=context_length,
                    phase=InferencePhase.DECODE,
                    chip_id=chip_id,
                    pp_stage=pp_stage,
                    token_index=token_idx,
                )

                stage_times[pp_stage] = layer_end

            current_time = max(stage_times)

        # 更新统计
        self.decode_stats.total_time = current_time - start_time

        return current_time

    def _simulate_single_layer(
        self,
        current_time: float,
        layer_index: int,
        num_tokens: int,
        context_length: int,
        phase: InferencePhase,
        chip_id: str,
        pp_stage: int,
        token_index: int | None = None,
    ) -> float:
        """模拟单层 Transformer"""

        # 检查是否使用 MLA
        use_mla = self.model.attention_type == 'mla' and self.model.mla_config is not None

        if self.config.enable_detailed_ops:
            # 细粒度模拟

            # LayerNorm 1
            ln1_latency = calc_layernorm_latency(
                self.model, self.inference, self.parallelism, self.hardware, num_tokens
            )
            self.gantt_builder.add_compute_task(
                GanttTaskType.LAYERNORM, current_time, ln1_latency,
                phase, chip_id, pp_stage, layer_index, token_index
            )
            current_time += ln1_latency

            # SP AllGather (序列切分 → 张量切分)
            if self.parallelism.sp > 1:
                sp_allgather_latency = self._calc_sp_allgather_latency(num_tokens)
                self.gantt_builder.add_comm_task(
                    GanttTaskType.SP_ALLGATHER, current_time, sp_allgather_latency,
                    phase, chip_id, pp_stage, layer_index, token_index
                )
                current_time += sp_allgather_latency

            # Attention QKV (MLA 使用专用计算)
            if use_mla:
                # MLA: Q 投影 + KV 压缩
                qkv_latency = calc_mla_q_projection_latency(
                    self.model, self.inference, self.parallelism, self.hardware, num_tokens
                )
                kv_compress_latency = calc_mla_kv_compression_latency(
                    self.model, self.inference, self.parallelism, self.hardware, num_tokens
                )
                qkv_latency += kv_compress_latency
            else:
                qkv_latency = calc_attention_qkv_latency(
                    self.model, self.inference, self.parallelism, self.hardware, num_tokens
                )
            self.gantt_builder.add_compute_task(
                GanttTaskType.ATTENTION_QKV, current_time, qkv_latency,
                phase, chip_id, pp_stage, layer_index, token_index
            )
            current_time += qkv_latency

            # KV Cache 读取 (Decode 阶段)
            if self.config.enable_kv_cache and phase == InferencePhase.DECODE:
                if use_mla:
                    # MLA: 压缩后的 KV Cache (~32x 小)
                    kv_read_latency = calc_mla_kv_cache_read_latency(
                        self.model, self.inference, self.parallelism, self.hardware, context_length
                    )
                else:
                    kv_read_latency = calc_kv_cache_read_latency(
                        self.model, self.inference, self.parallelism, self.hardware, context_length
                    )
                self.gantt_builder.add_compute_task(
                    GanttTaskType.KV_CACHE_READ, current_time, kv_read_latency,
                    phase, chip_id, pp_stage, layer_index, token_index
                )
                current_time += kv_read_latency

            # Attention Score
            if use_mla:
                # MLA: 在压缩空间计算 Score
                score_latency = calc_mla_attention_score_latency(
                    self.model, self.inference, self.parallelism, self.hardware, num_tokens, context_length
                )
            else:
                score_latency = calc_attention_score_latency(
                    self.model, self.inference, self.parallelism, self.hardware, num_tokens, context_length
                )
            self.gantt_builder.add_compute_task(
                GanttTaskType.ATTENTION_SCORE, current_time, score_latency,
                phase, chip_id, pp_stage, layer_index, token_index
            )
            current_time += score_latency

            # Softmax (MLA 和标准 Attention 相同)
            softmax_latency = calc_attention_softmax_latency(
                self.model, self.inference, self.parallelism, self.hardware, num_tokens, context_length
            )
            self.gantt_builder.add_compute_task(
                GanttTaskType.ATTENTION_SOFTMAX, current_time, softmax_latency,
                phase, chip_id, pp_stage, layer_index, token_index
            )
            current_time += softmax_latency

            # Attention Output
            if use_mla:
                # MLA: V 解压缩 + Softmax@V + Output 投影
                output_latency = calc_mla_output_latency(
                    self.model, self.inference, self.parallelism, self.hardware, num_tokens, context_length
                )
            else:
                output_latency = calc_attention_output_latency(
                    self.model, self.inference, self.parallelism, self.hardware, num_tokens, context_length
                )
            self.gantt_builder.add_compute_task(
                GanttTaskType.ATTENTION_OUTPUT, current_time, output_latency,
                phase, chip_id, pp_stage, layer_index, token_index
            )
            current_time += output_latency

            # TP AllReduce (Attention)
            if self.parallelism.tp > 1:
                tp_comm_latency = self._calc_tp_allreduce_latency(num_tokens)
                self.gantt_builder.add_comm_task(
                    GanttTaskType.TP_COMM, current_time, tp_comm_latency,
                    phase, chip_id, pp_stage, layer_index, token_index
                )
                current_time += tp_comm_latency

            # SP ReduceScatter (张量切分 → 序列切分)
            if self.parallelism.sp > 1:
                sp_reduce_scatter_latency = self._calc_sp_reduce_scatter_latency(num_tokens)
                self.gantt_builder.add_comm_task(
                    GanttTaskType.SP_REDUCE_SCATTER, current_time, sp_reduce_scatter_latency,
                    phase, chip_id, pp_stage, layer_index, token_index
                )
                current_time += sp_reduce_scatter_latency

            # KV Cache 写入
            if self.config.enable_kv_cache:
                if use_mla:
                    # MLA: 压缩后的 KV Cache 写入 (~32x 小)
                    kv_write_latency = calc_mla_kv_cache_write_latency(
                        self.model, self.inference, self.parallelism, self.hardware, num_tokens
                    )
                else:
                    kv_write_latency = calc_kv_cache_write_latency(
                        self.model, self.inference, self.parallelism, self.hardware, num_tokens
                    )
                self.gantt_builder.add_compute_task(
                    GanttTaskType.KV_CACHE_WRITE, current_time, kv_write_latency,
                    phase, chip_id, pp_stage, layer_index, token_index
                )
                current_time += kv_write_latency

            # LayerNorm 2
            ln2_latency = calc_layernorm_latency(
                self.model, self.inference, self.parallelism, self.hardware, num_tokens
            )
            self.gantt_builder.add_compute_task(
                GanttTaskType.LAYERNORM, current_time, ln2_latency,
                phase, chip_id, pp_stage, layer_index, token_index
            )
            current_time += ln2_latency

            # SP AllGather (序列切分 → 张量切分) - 进入FFN TP层
            if self.parallelism.sp > 1:
                sp_allgather_latency = self._calc_sp_allgather_latency(num_tokens)
                self.gantt_builder.add_comm_task(
                    GanttTaskType.SP_ALLGATHER, current_time, sp_allgather_latency,
                    phase, chip_id, pp_stage, layer_index, token_index
                )
                current_time += sp_allgather_latency

            # 判断是否为 MoE 层
            is_moe = is_moe_layer(layer_index, self.model)

            if is_moe:
                # ========== MoE 层 ==========
                # MoE Gate (路由网络)
                gate_latency = calc_moe_gate_latency(
                    self.model, self.inference, self.parallelism, self.hardware, num_tokens
                )
                self.gantt_builder.add_compute_task(
                    GanttTaskType.MOE_GATE, current_time, gate_latency,
                    phase, chip_id, pp_stage, layer_index, token_index
                )
                current_time += gate_latency

                # EP Dispatch (Token 分发)
                if self.parallelism.ep > 1:
                    dispatch_latency = calc_ep_dispatch_latency(
                        self.model, self.inference, self.parallelism, self.hardware,
                        num_tokens, self.ep_bandwidth, self.ep_latency
                    )
                    self.gantt_builder.add_comm_task(
                        GanttTaskType.EP_DISPATCH, current_time, dispatch_latency,
                        phase, chip_id, pp_stage, layer_index, token_index
                    )
                    current_time += dispatch_latency

                # MoE Expert FFN (路由专家)
                expert_latency = calc_moe_expert_ffn_latency(
                    self.model, self.inference, self.parallelism, self.hardware, num_tokens
                )
                self.gantt_builder.add_compute_task(
                    GanttTaskType.MOE_EXPERT, current_time, expert_latency,
                    phase, chip_id, pp_stage, layer_index, token_index
                )
                current_time += expert_latency

                # MoE Shared Expert (共享专家, 并行计算)
                if self.model.moe_config and self.model.moe_config.num_shared_experts > 0:
                    shared_latency = calc_moe_shared_expert_latency(
                        self.model, self.inference, self.parallelism, self.hardware, num_tokens
                    )
                    # 共享专家与路由专家并行，取较大值
                    # 这里简化处理：假设共享专家在路由专家之后开始，但可以部分重叠
                    overlap_ratio = 0.5  # 50% 重叠
                    effective_shared_latency = shared_latency * (1 - overlap_ratio)
                    self.gantt_builder.add_compute_task(
                        GanttTaskType.MOE_SHARED_EXPERT, current_time, effective_shared_latency,
                        phase, chip_id, pp_stage, layer_index, token_index
                    )
                    current_time += effective_shared_latency

                # EP Combine (结果收集)
                if self.parallelism.ep > 1:
                    combine_latency = calc_ep_combine_latency(
                        self.model, self.inference, self.parallelism, self.hardware,
                        num_tokens, self.ep_bandwidth, self.ep_latency
                    )
                    self.gantt_builder.add_comm_task(
                        GanttTaskType.EP_COMBINE, current_time, combine_latency,
                        phase, chip_id, pp_stage, layer_index, token_index
                    )
                    current_time += combine_latency

            else:
                # ========== 普通 Dense FFN 层 ==========
                # FFN Gate
                gate_latency = calc_ffn_gate_latency(
                    self.model, self.inference, self.parallelism, self.hardware, num_tokens
                )
                self.gantt_builder.add_compute_task(
                    GanttTaskType.FFN_GATE, current_time, gate_latency,
                    phase, chip_id, pp_stage, layer_index, token_index
                )
                current_time += gate_latency

                # FFN Up
                up_latency = calc_ffn_up_latency(
                    self.model, self.inference, self.parallelism, self.hardware, num_tokens
                )
                self.gantt_builder.add_compute_task(
                    GanttTaskType.FFN_UP, current_time, up_latency,
                    phase, chip_id, pp_stage, layer_index, token_index
                )
                current_time += up_latency

                # FFN Down
                down_latency = calc_ffn_down_latency(
                    self.model, self.inference, self.parallelism, self.hardware, num_tokens
                )
                self.gantt_builder.add_compute_task(
                    GanttTaskType.FFN_DOWN, current_time, down_latency,
                    phase, chip_id, pp_stage, layer_index, token_index
                )
                current_time += down_latency

            # TP AllReduce (FFN/MoE 输出)
            if self.parallelism.tp > 1:
                tp_comm_latency = self._calc_tp_allreduce_latency(num_tokens)
                self.gantt_builder.add_comm_task(
                    GanttTaskType.TP_COMM, current_time, tp_comm_latency,
                    phase, chip_id, pp_stage, layer_index, token_index
                )
                current_time += tp_comm_latency

            # SP ReduceScatter (张量切分 → 序列切分) - 离开FFN TP层
            if self.parallelism.sp > 1:
                sp_reduce_scatter_latency = self._calc_sp_reduce_scatter_latency(num_tokens)
                self.gantt_builder.add_comm_task(
                    GanttTaskType.SP_REDUCE_SCATTER, current_time, sp_reduce_scatter_latency,
                    phase, chip_id, pp_stage, layer_index, token_index
                )
                current_time += sp_reduce_scatter_latency

        else:
            # 粗粒度模拟 - 整层计算
            layer_latency = self._calc_layer_latency_coarse(num_tokens, context_length)
            self.gantt_builder.add_compute_task(
                GanttTaskType.COMPUTE, current_time, layer_latency,
                phase, chip_id, pp_stage, layer_index, token_index
            )
            current_time += layer_latency

            # TP 通信
            if self.parallelism.tp > 1:
                tp_comm = self._calc_tp_allreduce_latency(num_tokens) * 2  # Attn + FFN
                self.gantt_builder.add_comm_task(
                    GanttTaskType.TP_COMM, current_time, tp_comm,
                    phase, chip_id, pp_stage, layer_index, token_index
                )
                current_time += tp_comm

        return current_time

    def _calc_tp_allreduce_latency(self, num_tokens: int) -> float:
        """计算 TP AllReduce 延迟"""
        bytes_per_elem = get_bytes_per_element(self.model.dtype)
        data_size_gb = (
            self.inference.batch_size * num_tokens * self.model.hidden_size * bytes_per_elem
        ) / (1024 ** 3)

        return calc_tp_allreduce_latency(
            data_size_gb, self.tp_bandwidth, self.tp_latency, self.parallelism.tp
        )

    def _calc_pp_comm_latency(self, num_tokens: int) -> float:
        """计算 PP P2P 通信延迟"""
        bytes_per_elem = get_bytes_per_element(self.model.dtype)
        data_size_gb = (
            self.inference.batch_size * num_tokens * self.model.hidden_size * bytes_per_elem
        ) / (1024 ** 3)

        return calc_pp_p2p_latency(data_size_gb, self.pp_bandwidth, self.pp_latency)

    def _calc_sp_allgather_latency(self, num_tokens: int) -> float:
        """计算 SP AllGather 延迟"""
        if self.parallelism.sp <= 1:
            return 0.0

        # 计算数据量
        data_size_gb = calc_sp_comm_volume_gb(
            self.model, self.inference, self.parallelism, num_tokens
        )

        # SP通信使用与TP相同的带宽和延迟（通常SP与TP绑定在同一组）
        return calc_sp_allgather_latency(
            data_size_gb, self.tp_bandwidth, self.tp_latency, self.parallelism.sp
        )

    def _calc_sp_reduce_scatter_latency(self, num_tokens: int) -> float:
        """计算 SP ReduceScatter 延迟"""
        if self.parallelism.sp <= 1:
            return 0.0

        # 计算数据量
        data_size_gb = calc_sp_comm_volume_gb(
            self.model, self.inference, self.parallelism, num_tokens
        )

        # SP通信使用与TP相同的带宽和延迟
        return calc_sp_reduce_scatter_latency(
            data_size_gb, self.tp_bandwidth, self.tp_latency, self.parallelism.sp
        )

    def _calc_layer_latency_coarse(self, num_tokens: int, context_length: int) -> float:
        """粗粒度计算单层延迟"""
        # 简化计算：主要是 Attention 和 FFN
        attn_latency = (
            calc_attention_qkv_latency(self.model, self.inference, self.parallelism, self.hardware, num_tokens) +
            calc_attention_score_latency(self.model, self.inference, self.parallelism, self.hardware, num_tokens, context_length) +
            calc_attention_output_latency(self.model, self.inference, self.parallelism, self.hardware, num_tokens, context_length)
        )
        ffn_latency = (
            calc_ffn_gate_latency(self.model, self.inference, self.parallelism, self.hardware, num_tokens) +
            calc_ffn_up_latency(self.model, self.inference, self.parallelism, self.hardware, num_tokens) +
            calc_ffn_down_latency(self.model, self.inference, self.parallelism, self.hardware, num_tokens)
        )
        return attn_latency + ffn_latency

    def _get_chip_for_stage(self, pp_stage: int) -> str:
        """获取指定 PP stage 的第一个芯片ID"""
        for assignment in self.group_assignment.assignments:
            if assignment.pp_rank == pp_stage:
                return assignment.chip_id
        raise ValueError(f"找不到 PP stage {pp_stage} 的芯片")

    def _compute_stats(self, total_time: float) -> SimulationStats:
        """计算统计信息"""
        # TTFT = Prefill 总时间
        ttft = self.prefill_stats.total_time

        # 平均 TPOT
        num_decode_tokens = min(self.config.max_simulated_tokens, self.inference.output_seq_length)
        avg_tpot = self.decode_stats.total_time / num_decode_tokens if num_decode_tokens > 0 else 0.0

        # 计算 MFU (简化版本)
        bytes_per_elem = get_bytes_per_element(self.model.dtype)

        # Prefill 阶段 MFU
        # MFU = 实际 FLOPs/s / 峰值 FLOPs/s
        # 注意: prefill_flops 是单个 DP 副本的 FLOPs (不需要乘 DP)
        # peak_tflops 应该是单个 DP 副本使用的芯片总算力 (tp * pp)
        prefill_flops = self._calc_total_flops(self.inference.input_seq_length)
        prefill_mfu = 0.0
        if self.prefill_stats.total_time > 0:
            # 时间单位: ms -> s
            time_s = self.prefill_stats.total_time / 1000
            achieved_tflops = (prefill_flops / 1e12) / time_s

            # 单 DP 副本的峰值算力 (tp * pp 个芯片)
            # 注意: 不乘 dp，因为每个 dp 副本独立计算相同 FLOPs
            chips_per_replica = self.parallelism.tp * self.parallelism.pp
            peak_tflops = self.hardware.chip.compute_tflops_fp16 * chips_per_replica

            prefill_mfu = achieved_tflops / peak_tflops

        # Decode 阶段 MBU (内存带宽利用率)
        # MBU = 实际带宽需求 / 峰值带宽
        # 实际带宽需求 = (模型权重 + KV Cache) / TPOT
        decode_mbu = 0.0
        if num_decode_tokens > 0 and avg_tpot > 0:
            # 模型权重大小
            model_size_gb = self._calc_model_size_gb()

            # KV Cache 大小 (平均 context 长度)
            avg_context = self.inference.input_seq_length + num_decode_tokens // 2
            kv_cache_gb = self._calc_kv_cache_size_gb(avg_context)

            # 总数据量
            data_read_gb = model_size_gb + kv_cache_gb

            # 实际带宽需求 (GB/s)
            required_bandwidth = data_read_gb / (avg_tpot / 1000)

            # 峰值带宽 (考虑 HBM 效率 85%)
            peak_bandwidth = self.hardware.chip.memory_bandwidth_gbps * 0.85
            decode_mbu = required_bandwidth / peak_bandwidth

        return SimulationStats(
            prefill=self.prefill_stats,
            decode=self.decode_stats,
            total_run_time=total_time,
            simulated_tokens=num_decode_tokens,
            ttft=ttft,
            avg_tpot=avg_tpot,
            dynamic_mfu=min(prefill_mfu, 1.0),
            dynamic_mbu=min(decode_mbu, 1.0),
            max_pp_bubble_ratio=0.0,  # TODO: 计算气泡比
            total_events=len(self.gantt_builder.tasks),
            prefill_flops=prefill_flops,
        )

    def _calc_total_flops(self, seq_length: int) -> float:
        """
        计算总 FLOPs

        标准 Transformer FLOPs 计算:
        - QKV Projection: 2 * B * S * H * (H + 2 * kv_heads * head_dim)  (考虑 GQA)
        - Attention Score: 2 * B * n_heads * S * S * head_dim
        - Attention Output: 2 * B * n_heads * S * S * head_dim + 2 * B * S * H * H
        - FFN: 3 * 2 * B * S * H * I (gate, up, down)
        - LM Head: 2 * B * S * H * V

        简化公式: 约等于 2 * num_params * seq_length
        """
        B = self.inference.batch_size
        S = seq_length
        H = self.model.hidden_size
        L = self.model.num_layers
        I = self.model.intermediate_size
        V = self.model.vocab_size
        n_heads = self.model.num_attention_heads
        kv_heads = self.model.num_kv_heads
        head_dim = H // n_heads

        # QKV Projection (考虑 GQA)
        qkv_flops = 2 * B * S * H * (H + 2 * kv_heads * head_dim) * L

        # Attention Score: Q @ K^T
        score_flops = 2 * B * n_heads * S * S * head_dim * L

        # Attention Output: Softmax @ V + Output Projection
        output_flops = (2 * B * n_heads * S * S * head_dim + 2 * B * S * H * H) * L

        # FFN: gate, up, down
        ffn_flops = 2 * B * S * H * I * 3 * L

        # LM Head
        lm_head_flops = 2 * B * S * H * V

        return qkv_flops + score_flops + output_flops + ffn_flops + lm_head_flops

    def _calc_model_size_gb(self) -> float:
        """计算模型大小 (GB)

        支持:
        - MLA (Multi-head Latent Attention) vs 标准 Attention
        - MoE (Mixture of Experts) vs Dense FFN
        """
        bytes_per_elem = get_bytes_per_element(self.model.dtype)
        H = self.model.hidden_size
        L = self.model.num_layers
        I = self.model.intermediate_size
        V = self.model.vocab_size
        num_heads = self.model.num_attention_heads
        num_kv_heads = self.model.num_kv_heads

        # === Attention 参数 ===
        if self.model.mla_config is not None:
            # MLA 参数 (DeepSeek-V3)
            mla = self.model.mla_config
            head_dim = mla.qk_nope_head_dim + mla.qk_rope_head_dim

            # Q path: W_DQ (H × q_lora_rank) + W_UQ (q_lora_rank × num_heads × head_dim)
            # + W_QR (q_lora_rank × qk_rope_head_dim × num_heads)
            q_down_params = H * mla.q_lora_rank
            q_up_params = mla.q_lora_rank * num_heads * head_dim
            q_rope_params = mla.q_lora_rank * mla.qk_rope_head_dim * num_heads

            # KV path: W_DKV (H × kv_lora_rank) + W_UK (kv_lora_rank × num_heads × head_dim)
            # + W_UV (kv_lora_rank × num_heads × v_head_dim) + W_KR (H × qk_rope_head_dim)
            kv_down_params = H * mla.kv_lora_rank
            k_up_params = mla.kv_lora_rank * num_heads * mla.qk_nope_head_dim
            v_up_params = mla.kv_lora_rank * num_heads * mla.v_head_dim
            k_rope_params = H * mla.qk_rope_head_dim

            # Output: W_O (num_heads × v_head_dim × H)
            o_params = num_heads * mla.v_head_dim * H

            attn_params_per_layer = (q_down_params + q_up_params + q_rope_params +
                                     kv_down_params + k_up_params + v_up_params +
                                     k_rope_params + o_params)
            attn_params = attn_params_per_layer * L
        else:
            # 标准 Attention: Q + K + V + O
            head_dim = H // num_heads
            q_params = H * H  # Q projection
            k_params = H * num_kv_heads * head_dim  # K projection (GQA)
            v_params = H * num_kv_heads * head_dim  # V projection (GQA)
            o_params = H * H  # Output projection
            attn_params = (q_params + k_params + v_params + o_params) * L

        # === FFN 参数 ===
        if self.model.model_type == "moe" and self.model.moe_config is not None:
            # MoE 模型
            moe = self.model.moe_config
            expert_I = moe.expert_intermediate_size if moe.expert_intermediate_size > 0 else I

            # Dense 层 (前 first_k_dense_replace 层)
            dense_layers = moe.first_k_dense_replace
            dense_ffn_params = 3 * H * I * dense_layers

            # MoE 层
            moe_layers = L - dense_layers
            # 路由专家: num_experts × (gate + up + down)
            routed_expert_params = moe.num_experts * 3 * H * expert_I * moe_layers
            # 共享专家
            shared_expert_params = moe.num_shared_experts * 3 * H * expert_I * moe_layers
            # Gate 网络: H × num_experts
            gate_params = H * moe.num_experts * moe_layers

            ffn_params = dense_ffn_params + routed_expert_params + shared_expert_params + gate_params
        else:
            # Dense FFN: (gate, up, down) per layer
            ffn_params = 3 * H * I * L

        # === Embedding (LM Head 通常与 Embedding 共享权重) ===
        embed_params = V * H

        total_params = attn_params + ffn_params + embed_params
        return (total_params * bytes_per_elem) / (1024 ** 3)

    def _calc_kv_cache_size_gb(self, context_length: int) -> float:
        """计算 KV Cache 大小 (GB)

        根据 DeepSeek-V3 论文 (arXiv:2412.19437):
        "for MLA, only c_t^KV and k_t^R need to be cached during generation"
        - c_t^KV: 压缩后的 KV 潜在向量，维度 = kv_lora_rank
        - k_t^R: RoPE 解耦 key，维度 = qk_rope_head_dim

        MLA KV Cache 维度 = kv_lora_rank + qk_rope_head_dim (如 512 + 64 = 576)
        """
        bytes_per_elem = get_bytes_per_element(self.model.dtype)
        B = self.inference.batch_size
        L = self.model.num_layers

        if self.model.mla_config is not None:
            # MLA: 只缓存 c_t^KV + k_t^R
            mla = self.model.mla_config
            kv_cache_dim = mla.kv_lora_rank + mla.qk_rope_head_dim
            kv_cache_bytes = B * context_length * kv_cache_dim * L * bytes_per_elem
        else:
            # 标准 Attention: 2 (K+V) × batch × context × kv_heads × head_dim × layers
            H = self.model.hidden_size
            num_heads = self.model.num_attention_heads
            num_kv_heads = self.model.num_kv_heads
            head_dim = H // num_heads
            kv_cache_bytes = 2 * B * context_length * num_kv_heads * head_dim * L * bytes_per_elem

        return kv_cache_bytes / (1024 ** 3)


def run_simulation(
    topology_dict: dict[str, Any],
    model_dict: dict[str, Any],
    inference_dict: dict[str, Any],
    parallelism_dict: dict[str, Any],
    hardware_dict: dict[str, Any],
    config_dict: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    运行模拟的入口函数

    Args:
        topology_dict: 拓扑配置
        model_dict: 模型配置
        inference_dict: 推理配置
        parallelism_dict: 并行策略
        hardware_dict: 硬件配置
        config_dict: 模拟配置

    Returns:
        模拟结果字典
    """
    # 验证配置
    validate_model_config(model_dict)
    validate_hardware_config(hardware_dict)
    validate_parallelism_config(parallelism_dict, model_dict)

    # 解析并验证 MLA 配置 (DeepSeek V3/R1)
    mla_config = None
    mla_dict = model_dict.get("mla_config")
    if mla_dict:
        mla_config = validate_mla_config(mla_dict)

    # 解析并验证 MoE 配置 (DeepSeek, Mixtral, Qwen-MoE)
    moe_config = None
    moe_dict = model_dict.get("moe_config")
    if moe_dict:
        moe_config = validate_moe_config(moe_dict)

    # 解析配置
    model = LLMModelConfig(
        model_name=model_dict.get("model_name", "Unknown"),
        model_type=model_dict.get("model_type", "dense"),
        hidden_size=model_dict["hidden_size"],
        num_layers=model_dict["num_layers"],
        num_attention_heads=model_dict["num_attention_heads"],
        num_kv_heads=model_dict.get("num_kv_heads", model_dict["num_attention_heads"]),
        intermediate_size=model_dict["intermediate_size"],
        vocab_size=model_dict.get("vocab_size", 32000),
        dtype=model_dict.get("dtype", "fp16"),
        max_seq_length=model_dict.get("max_seq_length", 4096),
        attention_type=model_dict.get("attention_type", "gqa"),
        mla_config=mla_config,
        moe_config=moe_config,
    )

    inference = InferenceConfig(
        batch_size=inference_dict["batch_size"],
        input_seq_length=inference_dict["input_seq_length"],
        output_seq_length=inference_dict["output_seq_length"],
        max_seq_length=inference_dict.get("max_seq_length", 4096),
    )

    parallelism = ParallelismStrategy(
        dp=parallelism_dict.get("dp", 1),
        tp=parallelism_dict.get("tp", 1),
        pp=parallelism_dict.get("pp", 1),
        ep=parallelism_dict.get("ep", 1),
        sp=parallelism_dict.get("sp", 1),
    )

    chip_hw = hardware_dict.get("chip", {})
    node_hw = hardware_dict.get("node", {})
    cluster_hw = hardware_dict.get("cluster", {})

    # 默认使用 SG2260E 芯片参数
    hardware = HardwareConfig(
        chip=ChipHardwareConfig(
            chip_type=chip_hw.get("chip_type", "SG2260E"),
            compute_tflops_fp16=chip_hw.get("compute_tflops_fp16", 64),
            memory_gb=chip_hw.get("memory_gb", 64),
            memory_bandwidth_gbps=chip_hw.get("memory_bandwidth_gbps", 273),
            compute_tops_int8=chip_hw.get("compute_tops_int8", 128),
            num_cores=chip_hw.get("num_cores", 8),
            memory_bandwidth_utilization=chip_hw.get("memory_bandwidth_utilization", 0.893),
            l2_cache_mb=chip_hw.get("l2_cache_mb", 16),
            l2_bandwidth_gbps=chip_hw.get("l2_bandwidth_gbps", 512),
            pcie_bandwidth_gbps=chip_hw.get("pcie_bandwidth_gbps", 64),
            pcie_latency_us=chip_hw.get("pcie_latency_us", 1),
            hbm_random_access_latency_ns=chip_hw.get("hbm_random_access_latency_ns", 100),
        ),
        node=NodeConfig(
            chips_per_node=node_hw.get("chips_per_node", 8),
            intra_node_bandwidth_gbps=node_hw.get("intra_node_bandwidth_gbps", 64),
            intra_node_latency_us=node_hw.get("intra_node_latency_us", 1),
            bandwidth_utilization=node_hw.get("bandwidth_utilization", 0.9),
            startup_latency_us=node_hw.get("startup_latency_us", 1),
            sync_latency_us=node_hw.get("sync_latency_us", 1),
        ),
        cluster=ClusterConfig(
            num_nodes=cluster_hw.get("num_nodes", 1),
            inter_node_bandwidth_gbps=cluster_hw.get("inter_node_bandwidth_gbps", 16),
            inter_node_latency_us=cluster_hw.get("inter_node_latency_us", 2),
        ),
    )

    config = SimulationConfig(
        max_simulated_tokens=config_dict.get("maxSimulatedTokens", 16) if config_dict else 16,
        enable_data_transfer=config_dict.get("enableDataTransferSimulation", True) if config_dict else True,
        enable_detailed_ops=config_dict.get("enableDetailedTransformerOps", True) if config_dict else True,
        enable_kv_cache=config_dict.get("enableKVCacheAccessSimulation", True) if config_dict else True,
    )

    # 运行模拟
    simulator = LLMInferenceSimulator(
        topology_dict=topology_dict,
        model=model,
        inference=inference,
        parallelism=parallelism,
        hardware=hardware,
        config=config,
    )

    result = simulator.simulate()

    # 转换为前端格式
    from .gantt import convert_to_frontend_format

    return {
        "ganttChart": convert_to_frontend_format(result.gantt_chart),
        "stats": {
            "prefill": {
                "computeTime": result.stats.prefill.compute_time,
                "commTime": result.stats.prefill.comm_time,
                "bubbleTime": result.stats.prefill.bubble_time,
                "overlapTime": result.stats.prefill.overlap_time,
                "totalTime": result.stats.prefill.total_time,
                "computeEfficiency": result.stats.prefill.compute_efficiency,
            },
            "decode": {
                "computeTime": result.stats.decode.compute_time,
                "commTime": result.stats.decode.comm_time,
                "bubbleTime": result.stats.decode.bubble_time,
                "overlapTime": result.stats.decode.overlap_time,
                "totalTime": result.stats.decode.total_time,
                "computeEfficiency": result.stats.decode.compute_efficiency,
            },
            "totalRunTime": result.stats.total_run_time,
            "simulatedTokens": result.stats.simulated_tokens,
            "ttft": result.stats.ttft,
            "avgTpot": result.stats.avg_tpot,
            "dynamicMfu": result.stats.dynamic_mfu,
            "dynamicMbu": result.stats.dynamic_mbu,
            "maxPPBubbleRatio": result.stats.max_pp_bubble_ratio,
            "totalEvents": result.stats.total_events,
        },
        "timestamp": result.timestamp,
    }
