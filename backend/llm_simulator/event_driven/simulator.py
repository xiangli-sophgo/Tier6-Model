"""
事件驱动仿真器

基于离散事件仿真（DES）的 LLM 推理模拟器主类。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional, Callable

from .event import (
    BaseEvent,
    ComputeStartEvent,
    EventType,
    ResourceType,
    reset_event_counter,
)
from .event_queue import EventQueue
from .resource import ResourceManager
from .dependency import DependencyGraph, DependencyGraphBuilder, OperatorNode

# 导入现有模块
from ..config import (
    LLMModelConfig,
    InferenceConfig,
    ParallelismStrategy,
    SimulationResult,
    SimulationStats,
    PhaseTimeStats,
    GanttTaskType,
    InferencePhase,
    get_bytes_per_element,
)
from ..core.topology import TopologyParser
from ..core.simulator import RuntimeHardwareParams
from ..core.gantt import GanttChartBuilder, convert_to_frontend_format
from ..evaluators import (
    get_arch_preset,
    AcceleratorMicroArch,
    GEMMEvaluator,
    FA2Evaluator,
    AllReduceEval,
    AllGatherEval,
    ReduceScatterEval,
    create_gemm_evaluator,
)
from ..layers import (
    MLALayer,
    MLAv32Layer,
    MLAAbsorbLayer,
    MLAAbsorbv32Layer,
    MHALayer,
    MLPLayer,
    MoELayer,
)

logger = logging.getLogger(__name__)


@dataclass
class EventDrivenSimConfig:
    """事件驱动仿真配置"""

    # 基础配置
    max_simulated_tokens: int = 16
    enable_data_transfer: bool = True
    enable_kv_cache: bool = True

    # 重叠优化
    enable_comm_overlap: bool = True
    enable_tbo: bool = True  # MoE TBO 优化

    # 评估器配置
    use_precise_evaluator: bool = True
    evaluation_granularity: str = "fine"

    # 调度策略
    pp_schedule: str = "gpipe"  # gpipe | 1f1b

    # 调试选项
    max_events: int = 1000000  # 最大事件数（防止无限循环）
    log_events: bool = False  # 是否记录事件日志
    max_simulation_time_us: float = 1e9  # 最大仿真时间


class EventDrivenSimulator:
    """事件驱动仿真器

    基于 Vidur 的设计理念，实现精确的事件驱动仿真。

    主要特点：
    - 多芯片独立推进各自时间线
    - 精确建模计算-通信重叠
    - 支持 GPipe / 1F1B 流水线策略
    - 精确计算气泡和资源利用率
    """

    def __init__(
        self,
        topology_dict: dict[str, Any],
        model: LLMModelConfig,
        inference: InferenceConfig,
        parallelism: ParallelismStrategy,
        hardware: RuntimeHardwareParams,
        config: Optional[EventDrivenSimConfig] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ):
        """初始化事件驱动仿真器

        Args:
            topology_dict: 前端拓扑配置（包含嵌入的硬件参数）
            model: 模型配置
            inference: 推理配置
            parallelism: 并行策略
            hardware: 运行时硬件参数
            config: 仿真配置
            progress_callback: 进度回调函数
        """
        self.model = model
        self.inference = inference
        self.parallelism = parallelism
        self.hardware = hardware
        self.config = config or EventDrivenSimConfig()
        self.progress_callback = progress_callback

        # 初始化拓扑解析器（硬件参数现在嵌入在拓扑配置中）
        self.topo_parser = TopologyParser(topology_dict)
        self.interconnect = self.topo_parser.build_interconnect_graph()
        is_moe = model.moe_config is not None
        self.group_assignment = self.topo_parser.map_parallelism(parallelism, is_moe=is_moe)

        # 获取芯片列表
        self.chip_ids = [
            a.chip_id for a in self.group_assignment.assignments
        ]

        # 初始化评估器
        self._init_evaluators()

        # 初始化核心组件
        self.event_queue = EventQueue()
        self.resource_manager = ResourceManager(self.chip_ids)
        self.gantt_builder = GanttChartBuilder(parallelism)
        self.dependency_graph: Optional[DependencyGraph] = None

        # 仿真状态
        self.current_time = 0.0
        self.events_processed = 0

        # 缓存
        self.eval_cache: dict[str, Any] = {}
        self._layer_cache: dict[str, Any] = {}

        # PP 阶段映射
        self._setup_pp_mapping()

    def _init_evaluators(self) -> None:
        """初始化评估器"""
        chip_type = self.hardware.chip_type
        try:
            self.arch = get_arch_preset(chip_type)
        except KeyError:
            logger.warning(f"未找到 {chip_type} 的架构预设，使用 SG2260E")
            self.arch = get_arch_preset("SG2260E")

        # GEMM 评估器
        self.gemm_evaluator = create_gemm_evaluator(
            self.arch,
            fast_mode=True,
            enable_partition_search=False,
        )

        # FA2 评估器
        self.fa2_evaluator = FA2Evaluator(self.arch)

        # 通信评估器
        self.allreduce_eval = AllReduceEval(self.arch)
        self.allgather_eval = AllGatherEval(self.arch)
        self.reduce_scatter_eval = ReduceScatterEval(self.arch)

    def _setup_pp_mapping(self) -> None:
        """设置 PP 阶段映射"""
        pp_degree = self.parallelism.pp
        num_layers = self.model.num_layers
        layers_per_stage = num_layers // pp_degree

        # 层到 PP 阶段的映射
        self.layer_to_stage: dict[int, int] = {}
        for layer_idx in range(num_layers):
            self.layer_to_stage[layer_idx] = layer_idx // layers_per_stage

        # PP 阶段到芯片的映射
        self.stage_to_chips: dict[int, list[str]] = {}
        for assignment in self.group_assignment.assignments:
            stage = assignment.pp_rank
            if stage not in self.stage_to_chips:
                self.stage_to_chips[stage] = []
            self.stage_to_chips[stage].append(assignment.chip_id)

        # 层到芯片的映射（取每个 stage 的第一个芯片作为主芯片）
        self.layer_to_chip: dict[int, str] = {}
        for layer_idx in range(num_layers):
            stage = self.layer_to_stage[layer_idx]
            chips = self.stage_to_chips.get(stage, [])
            self.layer_to_chip[layer_idx] = chips[0] if chips else "chip_0"

    def _report_progress(self, percent: float, message: str) -> None:
        """报告进度"""
        if self.progress_callback:
            self.progress_callback(percent, message)

    def simulate(self) -> SimulationResult:
        """运行仿真

        Returns:
            仿真结果
        """
        self._report_progress(0, "初始化事件驱动仿真...")

        # 重置状态
        reset_event_counter()
        self.event_queue.clear()
        self.resource_manager.reset()
        self.current_time = 0.0
        self.events_processed = 0

        # 构建层和依赖图
        self._report_progress(5, "构建依赖图...")
        self._build_dependency_graph()

        # 添加初始事件
        self._report_progress(10, "添加初始事件...")
        self._add_initial_events()

        # 运行事件循环
        self._report_progress(15, "运行事件驱动仿真...")
        context = self._create_simulation_context()
        self._run_event_loop(context)

        # 收集结果
        self._report_progress(90, "收集仿真结果...")
        result = self._collect_results(context)

        self._report_progress(100, "仿真完成")
        return result

    def _build_dependency_graph(self) -> None:
        """构建依赖图"""
        num_layers = self.model.num_layers
        num_micro_batches = self.inference.num_micro_batches

        builder = DependencyGraphBuilder()

        for micro_batch in range(num_micro_batches):
            # 为每个 micro-batch 构建层
            layers = self._build_all_layers(micro_batch)

            builder.build_from_layers(
                layers=layers,
                chip_assignments=self.layer_to_chip,
                pp_stage_map=self.layer_to_stage,
                micro_batch=micro_batch,
            )

        self.dependency_graph = builder.graph

    def _build_all_layers(self, micro_batch: int) -> list:
        """构建所有层

        Args:
            micro_batch: 微批次索引

        Returns:
            层列表
        """
        layers = []
        num_tokens = self.inference.input_seq_length  # Prefill 阶段
        context_length = num_tokens

        for layer_idx in range(self.model.num_layers):
            layer = self._build_single_layer(
                layer_idx=layer_idx,
                num_tokens=num_tokens,
                context_length=context_length,
                phase=InferencePhase.PREFILL,
            )
            layers.append(layer)

        return layers

    def _build_single_layer(
        self,
        layer_idx: int,
        num_tokens: int,
        context_length: int,
        phase: InferencePhase,
    ):
        """构建单层

        Args:
            layer_idx: 层索引
            num_tokens: token 数量
            context_length: 上下文长度
            phase: 推理阶段

        Returns:
            构建好的层
        """
        # 检查缓存
        cache_key = f"{layer_idx}_{num_tokens}_{context_length}_{phase.value}"
        if cache_key in self._layer_cache:
            return self._layer_cache[cache_key]

        model = self.model
        parallelism = self.parallelism

        # 判断是否使用 MLA
        use_mla = model.attention_type == "mla" and model.mla_config

        # 判断是否是 MoE 层
        is_moe = False
        if model.model_type == "moe" and model.moe_config:
            first_k_dense = model.moe_config.first_k_dense_replace
            is_moe = layer_idx >= first_k_dense

        # 构建 Attention 层配置
        attn_config = {
            "hidden_dim": model.hidden_size,
            "num_heads": model.num_attention_heads,
            "num_kv_heads": model.num_kv_heads,
            "head_dim": model.hidden_size // model.num_attention_heads,
            "batch_size": self.inference.batch_size,
            "seq_len": num_tokens,
            "context_len": context_length,
            "tp": parallelism.tp,
            "sp": parallelism.sp,
            "comm_protocol": 1,
            "phase": phase.value,
        }

        if use_mla and model.mla_config:
            mla_config = model.mla_config
            variant = mla_config.variant

            # 添加 MLA 特定配置
            attn_config.update({
                "kv_lora_rank": mla_config.kv_lora_rank,
                "q_lora_rank": mla_config.q_lora_rank,
                "qk_nope_head_dim": mla_config.qk_nope_head_dim,
                "qk_rope_head_dim": mla_config.qk_rope_head_dim,
                "v_head_dim": mla_config.v_head_dim,
            })

            # 选择 MLA 变体
            mla_classes = {
                "mla": MLALayer,
                "mla_v32": MLAv32Layer,
                "mla_absorb": MLAAbsorbLayer,
                "mla_absorb_v32": MLAAbsorbv32Layer,
            }
            MLAClass = mla_classes.get(variant, MLALayer)
            attention_layer = MLAClass(name=f"layer_{layer_idx}_mla", config=attn_config)
        else:
            # 标准 MHA/GQA
            attention_layer = MHALayer(name=f"layer_{layer_idx}_mha", config=attn_config)

        # 构建 FFN 层配置
        ffn_config = {
            "hidden_dim": model.hidden_size,
            "inter_dim": model.intermediate_size,
            "batch_size": self.inference.batch_size,
            "seq_len": num_tokens,
            "tp": parallelism.tp,
            "comm_protocol": 1,
        }

        if is_moe and model.moe_config:
            moe_cfg = model.moe_config
            ffn_config.update({
                "num_experts": moe_cfg.num_experts,
                "num_experts_per_tok": moe_cfg.num_experts_per_tok,
                "num_shared_experts": moe_cfg.num_shared_experts,
                "expert_inter_dim": moe_cfg.expert_intermediate_size or model.intermediate_size,
                "ep": parallelism.ep,
            })
            ffn_layer = MoELayer(name=f"layer_{layer_idx}_moe", config=ffn_config)
        else:
            ffn_layer = MLPLayer(name=f"layer_{layer_idx}_mlp", config=ffn_config)

        # 评估算子
        self._evaluate_layer(attention_layer)
        self._evaluate_layer(ffn_layer)

        # 合并为完整的 Transformer 层
        from ..layers.base import BaseLayer
        combined = BaseLayer(layer_type="TransformerLayer")
        combined.comp_ops.extend(attention_layer.comp_ops)
        combined.comm_ops.extend(attention_layer.comm_ops)
        combined.comp_ops.extend(ffn_layer.comp_ops)
        combined.comm_ops.extend(ffn_layer.comm_ops)

        # 缓存
        self._layer_cache[cache_key] = combined

        return combined

    def _evaluate_layer(self, layer) -> None:
        """评估层中的所有算子"""
        for op in layer.comp_ops:
            self._evaluate_compute_op(op)

        for op in layer.comm_ops:
            self._evaluate_comm_op(op)

    def _evaluate_compute_op(self, op) -> None:
        """评估计算算子"""
        cache_key = op.get_cache_key() if hasattr(op, 'get_cache_key') else str(op.parallel_params)

        if cache_key in self.eval_cache:
            cached = self.eval_cache[cache_key]
            op.elapse = cached.get('elapse', 0.0)
            op.comp_elapse = cached.get('comp_elapse', 0.0)
            op.dma_elapse = cached.get('dma_elapse', 0.0)
            op.dram_traffic = cached.get('dram_traffic', 0.0)
            return

        # 根据算子类型选择评估器
        if "MatMul" in op.operator_type or "GEMM" in op.operator_type.upper():
            params = op.parallel_params
            result = self.gemm_evaluator.evaluate(
                G=params.get("G", 1),
                M=params.get("M", 1),
                K=params.get("K", 1),
                N=params.get("N", 1),
                input_dtype=params.get("input_dtype", "fp16"),
            )
            op.elapse = result.latency_us
            op.comp_elapse = result.compute_time_us
            op.dma_elapse = result.memory_time_us
            op.dram_traffic = result.dram_traffic_bytes

        elif "FlashAttention" in op.operator_type or "Attention" in op.operator_type:
            params = op.parallel_params
            result = self.fa2_evaluator.evaluate(
                B=params.get("B", 1),
                QS=params.get("QS", 1),
                KS=params.get("KS", 1),
                QD=params.get("QD", 128),
                VD=params.get("VD", 128),
            )
            op.elapse = result.latency_us
            op.comp_elapse = result.compute_time_us
            op.dma_elapse = result.memory_time_us

        else:
            # 默认使用简单估算
            op.elapse = 1.0  # 1us 默认值

        # 缓存结果
        self.eval_cache[cache_key] = {
            'elapse': op.elapse,
            'comp_elapse': getattr(op, 'comp_elapse', 0.0),
            'dma_elapse': getattr(op, 'dma_elapse', 0.0),
            'dram_traffic': getattr(op, 'dram_traffic', 0.0),
        }

    def _evaluate_comm_op(self, op) -> None:
        """评估通信算子"""
        comm_kind = op.comm_kind
        comm_size = getattr(op, 'comm_size', 0)
        tp = self.parallelism.tp

        if comm_kind == "allreduce":
            result = self.allreduce_eval.evaluate(tp, comm_size)
            op.comm_elapse = result.latency_us
        elif comm_kind == "allgather":
            result = self.allgather_eval.evaluate(tp, comm_size)
            op.comm_elapse = result.latency_us
        elif comm_kind == "reduce_scatter":
            result = self.reduce_scatter_eval.evaluate(tp, comm_size)
            op.comm_elapse = result.latency_us
        else:
            # P2P 或其他通信
            bandwidth_gbps = self.hardware.node.intra_node_bandwidth_gbps
            latency_us = self.hardware.node.intra_node_latency_us
            op.comm_elapse = latency_us + (comm_size / 1e9) / bandwidth_gbps * 1e6

    def _add_initial_events(self) -> None:
        """添加初始事件"""
        if self.dependency_graph is None:
            return

        # H2D 传输时间
        h2d_time = 0.0
        if self.config.enable_data_transfer:
            h2d_time = self._calculate_h2d_time()

            # 添加 H2D 到 Gantt 图
            for chip_id in self.stage_to_chips.get(0, []):
                self.gantt_builder.add_task(
                    name="PCIe H2D",
                    start=0.0,
                    end=h2d_time,
                    task_type=GanttTaskType.PCIE_H2D,
                    phase=InferencePhase.PREFILL,
                    chip_id=chip_id,
                    pp_stage=0,
                )

        # 获取入口节点并创建初始事件
        for micro_batch in range(self.inference.num_micro_batches):
            entry_nodes = self.dependency_graph.get_entry_nodes(micro_batch)

            for node in entry_nodes:
                # 只为 stage 0 的节点创建事件
                if node.pp_stage == 0:
                    event = ComputeStartEvent(
                        timestamp=h2d_time,
                        chip_id=node.chip_id,
                        layer_index=node.layer_index,
                        token_index=-1,  # Prefill 阶段
                        micro_batch=micro_batch,
                        pp_stage=node.pp_stage,
                        operator_name=node.name,
                        operator_type=node.op_type,
                        duration_us=node.duration_us,
                        flops=node.flops,
                        dram_traffic_bytes=node.dram_traffic_bytes,
                        compute_time_us=node.compute_time_us,
                        memory_time_us=node.memory_time_us,
                        best_tile=node.best_tile,
                        best_partition=node.best_partition,
                        gemm_shape=node.gemm_shape,
                    )
                    self.event_queue.push(event)

    def _calculate_h2d_time(self) -> float:
        """计算 H2D 传输时间"""
        batch_size = self.inference.batch_size
        seq_length = self.inference.input_seq_length
        hidden_size = self.model.hidden_size
        bytes_per_elem = get_bytes_per_element(self.model.dtype)

        # 输入数据大小
        input_size_bytes = batch_size * seq_length * hidden_size * bytes_per_elem

        # 数据传输 (使用 C2C 带宽简化)
        transfer_bw = self.hardware.c2c_bandwidth_gbps
        transfer_lat = self.hardware.c2c_latency_us

        transfer_time = (input_size_bytes / 1e9) / transfer_bw * 1e6 + transfer_lat

        return transfer_time

    def _create_simulation_context(self) -> dict[str, Any]:
        """创建仿真上下文"""
        return {
            "dependency_graph": self.dependency_graph,
            "num_layers": self.model.num_layers,
            "pp_degree": self.parallelism.pp,
            "tp_degree": self.parallelism.tp,
            "stage_chips": self.stage_to_chips,
            "pp_comm_latency_us": self._calculate_pp_comm_latency(),
            "num_micro_batches": self.inference.num_micro_batches,
            "batch_end_times": {},
            "simulation_complete": False,
            "total_time": 0.0,
        }

    def _calculate_pp_comm_latency(self) -> float:
        """计算 PP P2P 通信延迟"""
        # 激活值大小
        batch_size = self.inference.batch_size
        seq_length = self.inference.input_seq_length
        hidden_size = self.model.hidden_size
        bytes_per_elem = get_bytes_per_element(self.model.dtype)

        activation_size = batch_size * seq_length * hidden_size * bytes_per_elem

        # 使用节点间带宽
        bandwidth_gbps = self.hardware.cluster.inter_node_bandwidth_gbps
        latency_us = self.hardware.cluster.inter_node_latency_us

        return latency_us + (activation_size / 1e9) / bandwidth_gbps * 1e6

    def _run_event_loop(self, context: dict[str, Any]) -> None:
        """运行事件循环"""
        max_events = self.config.max_events
        max_time = self.config.max_simulation_time_us

        while self.event_queue and self.events_processed < max_events:
            # 取出最早的事件
            event = self.event_queue.pop()

            # 更新仿真时间
            self.current_time = event.timestamp

            # 检查时间限制
            if self.current_time > max_time:
                logger.warning(f"仿真时间超过限制: {self.current_time} > {max_time}")
                break

            # 处理事件
            new_events = event.handle(
                self.resource_manager,
                self.gantt_builder,
                context,
            )

            # 添加新事件
            self.event_queue.push_many(new_events)

            self.events_processed += 1

            # 更新进度
            if self.events_processed % 1000 == 0:
                progress = min(85, 15 + self.events_processed / max_events * 70)
                self._report_progress(progress, f"处理事件 {self.events_processed}...")

            # 检查是否完成
            if context.get("simulation_complete"):
                break

            # 日志
            if self.config.log_events:
                logger.debug(f"Event {self.events_processed}: {event.to_dict()}")

    def _collect_results(self, context: dict[str, Any]) -> SimulationResult:
        """收集仿真结果"""
        total_time = context.get("total_time", self.current_time)

        # 构建 Gantt 图
        gantt_data = self.gantt_builder.build()

        # 计算统计信息
        stats = self._calculate_stats(total_time, context)

        return SimulationResult(
            gantt_chart=convert_to_frontend_format(gantt_data),
            stats=stats,
        )

    def _calculate_stats(
        self,
        total_time: float,
        context: dict[str, Any],
    ) -> SimulationStats:
        """计算统计信息"""
        # 获取气泡统计
        total_bubble = self.resource_manager.get_total_bubble_time()

        # 计算 MFU
        total_flops = self._calculate_total_flops()
        peak_tflops = self.hardware.compute_tflops_bf16 * len(self.chip_ids)
        achieved_tflops = total_flops / total_time / 1e6 if total_time > 0 else 0
        mfu = achieved_tflops / peak_tflops if peak_tflops > 0 else 0

        # Prefill 和 Decode 时间（简化：假设全部是 Prefill）
        prefill_time = total_time
        decode_time = 0.0

        # TTFT 和 TPOT
        ttft = prefill_time / 1000  # us -> ms
        tpot = decode_time / max(1, self.config.max_simulated_tokens) / 1000

        # 计算效率
        compute_efficiency = 1.0 - total_bubble / total_time if total_time > 0 else 1.0

        return SimulationStats(
            prefill=PhaseTimeStats(
                compute_time=prefill_time - total_bubble,
                comm_time=0.0,
                bubble_time=total_bubble,
                overlap_time=0.0,
                total_time=prefill_time,
                compute_efficiency=compute_efficiency,
            ),
            decode=PhaseTimeStats(
                compute_time=decode_time,
                comm_time=0.0,
                bubble_time=0.0,
                overlap_time=0.0,
                total_time=decode_time,
                compute_efficiency=1.0,
            ),
            total_run_time=total_time / 1000,  # us -> ms
            simulated_tokens=self.config.max_simulated_tokens,
            ttft=ttft,
            avg_tpot=tpot,
            dynamic_mfu=mfu,
            dynamic_mbu=0.0,
            max_pp_bubble_ratio=total_bubble / total_time if total_time > 0 else 0.0,
            total_events=self.events_processed,
            prefill_flops=total_flops,
        )

    def _calculate_total_flops(self) -> float:
        """计算总 FLOPs"""
        batch_size = self.inference.batch_size
        seq_length = self.inference.input_seq_length
        hidden_size = self.model.hidden_size
        num_layers = self.model.num_layers
        intermediate_size = self.model.intermediate_size

        # Attention FLOPs: 4 * B * S * H * S (QKV + Output)
        attn_flops = 4 * batch_size * seq_length * hidden_size * seq_length

        # FFN FLOPs: 8 * B * S * H * I (Gate + Up + Down)
        ffn_flops = 8 * batch_size * seq_length * hidden_size * intermediate_size

        # 总 FLOPs
        total_flops = (attn_flops + ffn_flops) * num_layers

        return total_flops
