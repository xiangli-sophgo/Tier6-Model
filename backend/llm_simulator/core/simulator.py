"""
LLM 推理模拟器核心

实现基于拓扑的 GPU/加速器侧精细模拟，包括：
- 数据搬运阶段（PCIe传输、HBM存储、权重加载）
- 推理计算阶段（细化为Attention/FFN/LayerNorm子操作）
- 结果收集阶段（HBM读取、PCIe回传）
"""

from __future__ import annotations

import time
from typing import Any, Optional
from dataclasses import dataclass, field

from ..config import (
    LLMModelConfig,
    InferenceConfig,
    ParallelismStrategy,
    HierarchicalTopology,
    ChipConfig,
    SimulationResult,
    SimulationStats,
    PhaseTimeStats,
    GanttTaskType,
    InferencePhase,
    get_bytes_per_element,
    MLAConfig,
    MoEConfig,
    # 验证函数
    validate_mla_config,
    validate_moe_config,
    validate_model_config,
    validate_hardware_config,
    validate_parallelism_config,
)


@dataclass
class RuntimeHardwareParams:
    """运行时硬件参数（从拓扑配置或硬件配置中提取）

    这是一个简化的数据类，用于存储模拟器运行时需要的硬件参数。
    它不代表完整的硬件配置，只是模拟器需要的参数集合。
    """
    # 芯片参数
    chip_type: str = "Unknown"
    num_cores: int = 1
    compute_tflops_fp8: float = 0.0
    compute_tflops_bf16: float = 0.0
    memory_capacity_gb: float = 0.0
    memory_bandwidth_gbps: float = 0.0
    memory_bandwidth_utilization: float = 0.85
    lmem_capacity_mb: float = 0.0
    lmem_bandwidth_gbps: float = 0.0
    # 微架构参数（可选）
    cube_m: Optional[int] = None
    cube_k: Optional[int] = None
    cube_n: Optional[int] = None
    sram_size_kb: Optional[float] = None
    sram_utilization: Optional[float] = None
    lane_num: Optional[int] = None
    align_bytes: Optional[int] = None
    compute_dma_overlap_rate: Optional[float] = None
    # 互联参数（默认值，会被拓扑配置覆盖）
    c2c_bandwidth_gbps: float = 0.0
    c2c_latency_us: float = 0.0
    b2b_bandwidth_gbps: float = 450.0  # Board-to-Board
    b2b_latency_us: float = 0.35
    r2r_bandwidth_gbps: float = 200.0  # Rack-to-Rack
    r2r_latency_us: float = 2.0
    p2p_bandwidth_gbps: float = 100.0  # Pod-to-Pod
    p2p_latency_us: float = 5.0
from .topology import TopologyParser
from .gantt import GanttChartBuilder, convert_to_frontend_format

# 新评估器系统
from ..evaluators import (
    get_arch_preset,
    AcceleratorMicroArch,
    GEMMEvaluator,
    FA2Evaluator,
    AllReduceEval,
    AllGatherEval,
    create_gemm_evaluator,
    ReduceScatterEval,
)
from .analyzer import PerformanceAnalyzer
from ..layers import (
    MLALayer,
    MLAv32Layer,
    MLAAbsorbLayer,
    MLAAbsorbv32Layer,
    MHALayer,
    MLPLayer,
    MoELayer,
)
from ..operators.base import ComputeOpType, CommOpType


@dataclass
class SimulationConfig:
    """模拟配置"""

    max_simulated_tokens: int = 16
    enable_data_transfer: bool = True
    enable_detailed_ops: bool = True
    enable_kv_cache: bool = True
    enable_overlap: bool = True
    # 新增: Kernel Fusion 和 MLA 优化
    enable_fusion: bool = True  # 启用 Kernel Fusion 优化
    enable_comm_overlap: bool = True  # 启用计算-通信重叠
    enable_tbo: bool = True  # 启用 TBO (Tensor-Bus Overlap) 重叠优化 (MoE专用) ⭐ 新增
    # 训练模式配置
    enable_training_mode: bool = False  # 启用训练模式（模拟DP梯度同步）
    enable_dp_gradient_sync: bool = False  # 启用DP梯度同步模拟
    gradient_accumulation_steps: int = 1  # 梯度累积步数
    # 新评估器系统配置
    use_precise_evaluator: bool = True  # 使用精确评估器（基于硬件建模）
    evaluation_granularity: str = "fine"  # 评估粒度: coarse（粗粒度）或 fine（细粒度）
    enable_gemm_prewarm: bool = False  # 🚀 禁用预热，改用懒加载策略（按需搜索+全局缓存）
    # 注意: mla_variant 已移至 model.mla_config.variant，从模型配置读取


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
        hardware: RuntimeHardwareParams,
        config: SimulationConfig | None = None,
        comm_latency_config: dict[str, float] | None = None,
        progress_callback: callable | None = None,
        enable_tile_search: bool = True,
        enable_partition_search: bool = False,
        max_gemm_processes: Optional[int] = None,
        moe_tp: int | None = None,
    ):
        """
        初始化模拟器

        Args:
            topology_dict: 前端拓扑配置（包含嵌入的硬件参数）
            model: 模型配置
            inference: 推理配置
            parallelism: 并行策略
            hardware: 运行时硬件参数
            config: 模拟配置
            comm_latency_config: 通信延迟配置 (前端传递的统一配置，覆盖预设值)
            progress_callback: 进度回调函数 (percent: float, message: str) -> None
        """
        self.model = model
        self.inference = inference
        self.parallelism = parallelism
        self.hardware = hardware
        self.config = config or SimulationConfig()
        self.comm_latency_config = comm_latency_config
        self.progress_callback = progress_callback
        self.moe_tp = moe_tp  # MoE 张量并行度（用于 MoE 层计算）

        # 初始化新评估器系统
        if self.config.use_precise_evaluator:
            # 根据硬件类型选择芯片架构预设
            chip_type = hardware.chip_type
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"🔧 芯片类型: {chip_type}")
            try:
                self.arch = get_arch_preset(chip_type)
                logger.info(f"✅ 使用架构预设: {self.arch.name}")
            except (KeyError, ValueError) as e:
                # 如果没有预设，使用默认 SG2260E
                logger.warning(f"未找到 {chip_type} 的架构预设 ({e})，使用 SG2260E")
                self.arch = get_arch_preset("SG2260E")

            # 使用前端传递的通信延迟配置覆盖预设值
            if comm_latency_config:
                # 覆盖芯片延迟配置
                from ..evaluators.arch_config import CommunicationLatency

                self.arch.comm_latency = CommunicationLatency(
                    chip_to_chip_us=comm_latency_config.get("chip_to_chip_us", self.arch.comm_latency.chip_to_chip_us),
                    memory_read_latency_us=comm_latency_config.get("memory_read_latency_us", self.arch.comm_latency.memory_read_latency_us),
                    memory_write_latency_us=comm_latency_config.get("memory_write_latency_us", self.arch.comm_latency.memory_write_latency_us),
                    noc_latency_us=comm_latency_config.get("noc_latency_us", self.arch.comm_latency.noc_latency_us),
                    die_to_die_latency_us=comm_latency_config.get("die_to_die_latency_us", self.arch.comm_latency.die_to_die_latency_us),
                )

            # 创建协议配置和网络基础设施配置对象 (供通信评估器使用)
            from ..config import ProtocolConfig, NetworkInfraConfig

            if comm_latency_config:
                self.protocol_cfg = ProtocolConfig(
                    rtt_tp_us=comm_latency_config.get("rtt_tp_us", 0.35),
                    rtt_ep_us=comm_latency_config.get("rtt_ep_us", 0.85),
                    bandwidth_utilization=comm_latency_config.get("bandwidth_utilization", 0.95),
                    sync_latency_us=comm_latency_config.get("sync_latency_us", 0.0),
                )
                self.network_cfg = NetworkInfraConfig(
                    switch_delay_us=comm_latency_config.get("switch_delay_us", 1.0),
                    cable_delay_us=comm_latency_config.get("cable_delay_us", 0.025),
                )
            else:
                self.protocol_cfg = ProtocolConfig()
                self.network_cfg = NetworkInfraConfig()

            # 创建 GEMM 评估器（全局单例，跨层复用）
            # fast_mode=True 时使用固定tile（关闭tile搜索），显著提升评估速度
            # enable_partition_search=False 时使用固定分区（关闭分区搜索），速度提升100倍
            fast_mode = not enable_tile_search
            import logging

            logger = logging.getLogger(__name__)
            logger.info(f"🔧 创建 GEMM 评估器: enable_tile_search={enable_tile_search}, enable_partition_search={enable_partition_search}, fast_mode={fast_mode}, max_gemm_processes={max_gemm_processes}")
            self.gemm_evaluator = create_gemm_evaluator(self.arch, fast_mode=fast_mode, enable_partition_search=enable_partition_search, max_gemm_processes=max_gemm_processes)
            evaluator_type = self.gemm_evaluator.__class__.__name__
            logger.info(f"✅ 使用 GEMM 评估器: {evaluator_type}")

            # 🚀 懒加载策略：不预热，运行时按需搜索（对齐 DS_TPU）
            # 优势：
            # - 启动时间从 17分钟 → 0秒
            # - 只搜索实际用到的形状（避免浪费）
            # - 多进程并行搜索 + 全局缓存复用
            if self.config.enable_gemm_prewarm:
                import logging

                logger = logging.getLogger(__name__)
                logger.info("🚀 GEMM 懒加载模式：预热已禁用，将按需搜索并缓存")
                # 注：如需启用预热，请在 SimulationConfig 中设置 enable_gemm_prewarm=True

            # 全局评估缓存（跨层复用）
            self.eval_cache: dict = {}
        else:
            self.arch = None
            self.gemm_evaluator = None
            self.eval_cache = None
            self.protocol_cfg = None
            self.network_cfg = None

        # 解析拓扑（硬件参数现在嵌入在拓扑配置中）
        self.topo_parser = TopologyParser(topology_dict)
        # 验证拓扑中的硬件参数是否完整
        self.topo_parser.validate_hardware_params()
        self.interconnect = self.topo_parser.build_interconnect_graph()
        is_moe = model.moe_config is not None
        self.group_assignment = self.topo_parser.map_parallelism(parallelism, is_moe=is_moe)

        # 获取 TP 组的链路参数
        if self.group_assignment.tp_groups and len(self.group_assignment.tp_groups[0]) > 1:
            self.tp_bandwidth, self.tp_latency = self.topo_parser.get_link_params_for_group(self.group_assignment.tp_groups[0], "allreduce")
        else:
            self.tp_bandwidth = hardware.b2b_bandwidth_gbps
            self.tp_latency = hardware.b2b_latency_us

        # 获取 PP 组的链路参数
        if self.group_assignment.pp_groups and len(self.group_assignment.pp_groups[0]) > 1:
            self.pp_bandwidth, self.pp_latency = self.topo_parser.get_link_params_for_group(self.group_assignment.pp_groups[0], "p2p")
        else:
            self.pp_bandwidth = hardware.r2r_bandwidth_gbps
            self.pp_latency = hardware.r2r_latency_us

        # 获取 EP 组的链路参数 (MoE Expert Parallelism)
        if self.group_assignment.ep_groups and len(self.group_assignment.ep_groups[0]) > 1:
            self.ep_bandwidth, self.ep_latency = self.topo_parser.get_link_params_for_group(self.group_assignment.ep_groups[0], "alltoall")
        else:
            # 默认使用 Board 内带宽 (EP 通常在 Board 内)
            self.ep_bandwidth = hardware.b2b_bandwidth_gbps
            self.ep_latency = hardware.b2b_latency_us

        # 甘特图构建器
        self.gantt_builder = GanttChartBuilder(parallelism)

        # 芯片状态
        self.chip_states: dict[str, ChipState] = {}
        self._init_chip_states()

        # 统计
        self.prefill_stats = PhaseTimeStats()
        self.decode_stats = PhaseTimeStats()

        # 链路流量累加器: (source_chip, target_chip) -> {traffic_mb, bandwidth_gbps, latency_us, ...}
        self._link_traffic_accumulator: dict[tuple[str, str], dict[str, Any]] = {}

    def _init_chip_states(self):
        """初始化芯片状态"""
        for assignment in self.group_assignment.assignments:
            self.chip_states[assignment.chip_id] = ChipState(
                chip_id=assignment.chip_id,
                pp_stage=assignment.pp_rank,
                tp_rank=assignment.tp_rank,
            )

    def _accumulate_link_traffic(
        self,
        source_chip: str,
        target_chip: str,
        traffic_mb: float,
        task_id: str,
        task_type: GanttTaskType,
        bandwidth_gbps: float,
        latency_us: float,
        link_type: str,
    ):
        """累加链路流量

        Args:
            source_chip: 源芯片ID
            target_chip: 目标芯片ID
            traffic_mb: 流量（MB）
            task_id: 任务ID
            task_type: 任务类型
            bandwidth_gbps: 链路带宽（Gbps）
            latency_us: 链路延迟（微秒）
            link_type: 链路类型（c2c/b2b/r2r/p2p）
        """
        # 使用有序的键（按字典序），避免重复计数
        sorted_chips = sorted([source_chip, target_chip])
        key: tuple[str, str] = (sorted_chips[0], sorted_chips[1])

        if key not in self._link_traffic_accumulator:
            self._link_traffic_accumulator[key] = {
                'source': key[0],
                'target': key[1],
                'traffic_mb': 0.0,
                'bandwidth_gbps': bandwidth_gbps,
                'latency_us': latency_us,
                'link_type': link_type,
                'contributing_tasks': [],
                'task_type_breakdown': {}
            }

        acc = self._link_traffic_accumulator[key]
        acc['traffic_mb'] += traffic_mb
        acc['contributing_tasks'].append(task_id)

        task_type_str = task_type.value if isinstance(task_type, GanttTaskType) else str(task_type)
        acc['task_type_breakdown'][task_type_str] = \
            acc['task_type_breakdown'].get(task_type_str, 0.0) + traffic_mb

    def _accumulate_pp_comm_traffic(
        self,
        from_stage: int,
        to_stage: int,
        num_tokens: int,
        task_id: str,
        task_type: GanttTaskType,
    ):
        """累加 PP 通信流量

        Args:
            from_stage: 源 PP stage
            to_stage: 目标 PP stage
            num_tokens: Token 数量
            task_id: 任务ID
            task_type: 任务类型
        """
        # 计算数据量
        bytes_per_elem = get_bytes_per_element(self.model.dtype)
        data_size_bytes = self.inference.batch_size * num_tokens * self.model.hidden_size * bytes_per_elem
        traffic_mb = data_size_bytes / (1024 ** 2)

        # 获取源和目标 stage 的芯片列表
        if from_stage >= len(self.group_assignment.pp_groups) or to_stage >= len(self.group_assignment.pp_groups):
            return

        from_chips = self.group_assignment.pp_groups[from_stage]
        to_chips = self.group_assignment.pp_groups[to_stage]

        # 累加每对芯片之间的流量
        for from_chip in from_chips:
            for to_chip in to_chips:
                self._accumulate_link_traffic(
                    source_chip=from_chip,
                    target_chip=to_chip,
                    traffic_mb=traffic_mb / (len(from_chips) * len(to_chips)),  # 平均分配
                    task_id=task_id,
                    task_type=task_type,
                    bandwidth_gbps=self.pp_bandwidth,
                    latency_us=self.pp_latency,
                    link_type='pp',
                )

    def _accumulate_tp_comm_traffic(
        self,
        chip_id: str,
        data_size_gb: float,
        task_id: str,
        task_type: GanttTaskType,
    ):
        """累加 TP 通信流量（AllReduce）

        Args:
            chip_id: 当前芯片ID（用于查找其所属的 TP 组）
            data_size_gb: 数据量（GB）
            task_id: 任务ID
            task_type: 任务类型
        """
        import logging
        logger = logging.getLogger(__name__)

        # 查找芯片所属的 TP 组
        tp_chips = None
        tp_group_idx = -1
        for idx, group in enumerate(self.group_assignment.tp_groups):
            if chip_id in group:
                tp_chips = group
                tp_group_idx = idx
                break

        if tp_chips is None or len(tp_chips) <= 1:
            logger.debug(f"芯片 {chip_id} 无 TP 通信（TP 组大小 <= 1）")
            return

        logger.debug(f"芯片 {chip_id} 属于 TP 组 {tp_group_idx}，组大小: {len(tp_chips)}")

        # Ring AllReduce: 每个芯片与相邻芯片通信
        # 简化：累加环上所有相邻芯片对的流量
        traffic_mb = data_size_gb * 1024  # GB -> MB
        tp = len(tp_chips)

        # Ring AllReduce 中，每条链路传输 (N-1)/N 的数据量（两个方向）
        per_link_traffic = traffic_mb * 2 * (tp - 1) / tp / tp

        for i in range(len(tp_chips)):
            next_i = (i + 1) % len(tp_chips)
            self._accumulate_link_traffic(
                source_chip=tp_chips[i],
                target_chip=tp_chips[next_i],
                traffic_mb=per_link_traffic,
                task_id=task_id,
                task_type=task_type,
                bandwidth_gbps=self.tp_bandwidth,
                latency_us=self.tp_latency,
                link_type='tp',
            )

    def _generate_link_traffic_stats(self) -> list:
        """生成链路流量统计

        Returns:
            LinkTrafficStats 列表
        """
        from ..config.types import LinkTrafficStats
        import logging
        logger = logging.getLogger(__name__)

        stats = []

        # 获取仿真总时长（从 gantt 任务中计算）
        if not self.gantt_builder.tasks:
            logger.warning("📊 链路流量统计: 无 Gantt 任务数据")
            return stats

        # 添加调试日志
        logger.info(f"📊 链路流量累加器: {len(self._link_traffic_accumulator)} 条链路")

        total_time_us = max(task.end for task in self.gantt_builder.tasks)
        total_time_s = total_time_us / 1_000_000

        for (source, target), acc in self._link_traffic_accumulator.items():
            # 计算利用率 = 实际流量 / (带宽 × 时间)
            # 带宽单位: Gbps -> MBps 需要乘以 1000 / 8 = 125
            bandwidth_mbps = acc['bandwidth_gbps'] * 125
            max_capacity_mb = bandwidth_mbps * total_time_s
            utilization = (acc['traffic_mb'] / max_capacity_mb) * 100 if max_capacity_mb > 0 else 0

            stats.append(LinkTrafficStats(
                source=acc['source'],
                target=acc['target'],
                traffic_mb=acc['traffic_mb'],
                bandwidth_gbps=acc['bandwidth_gbps'],
                latency_us=acc['latency_us'],
                utilization_percent=min(utilization, 100),
                link_type=acc['link_type'],
                contributing_tasks=acc['contributing_tasks'],
                task_type_breakdown=acc['task_type_breakdown']
            ))

        # 按流量大小排序
        stats.sort(key=lambda s: s.traffic_mb, reverse=True)
        return stats

    def _map_compute_op_to_task_type(self, op_type: ComputeOpType, op_name: str = "") -> GanttTaskType:
        """将计算算子类型映射到 Gantt 任务类型"""
        if op_type == ComputeOpType.MATMUL:
            # 根据算子名称细分
            if "qkv" in op_name or "q_a" in op_name or "q_b" in op_name or "kv_a" in op_name:
                return GanttTaskType.ATTENTION_QKV
            elif "o_proj" in op_name:
                return GanttTaskType.ATTENTION_OUTPUT
            elif "gate" in op_name:
                return GanttTaskType.FFN_GATE
            elif "up" in op_name:
                return GanttTaskType.FFN_UP
            elif "down" in op_name:
                return GanttTaskType.FFN_DOWN
            else:
                return GanttTaskType.COMPUTE
        elif op_type in (ComputeOpType.MHA, ComputeOpType.MQA, ComputeOpType.FA2):
            return GanttTaskType.ATTENTION_SCORE
        elif op_type == ComputeOpType.RMSNORM:
            return GanttTaskType.LAYERNORM
        elif op_type == ComputeOpType.SOFTMAX:
            return GanttTaskType.ATTENTION_SOFTMAX
        else:
            return GanttTaskType.COMPUTE

    def _map_comm_op_to_task_type(self, comm_kind: str) -> GanttTaskType:
        """将通信算子类型映射到 Gantt 任务类型"""
        if comm_kind == "allreduce":
            return GanttTaskType.TP_COMM
        elif comm_kind == "allgather":
            return GanttTaskType.SP_ALLGATHER
        elif comm_kind == "reducescatter":
            return GanttTaskType.SP_REDUCE_SCATTER
        elif comm_kind == "dispatch":
            return GanttTaskType.EP_DISPATCH
        elif comm_kind == "combine":
            return GanttTaskType.EP_COMBINE
        else:
            return GanttTaskType.TP_COMM

    def _build_layer_for_evaluation(self, layer_index: int, num_tokens: int, context_length: int, phase: InferencePhase):
        """
        为指定层构建算子并评估

        完整构建Transformer层 = Attention + FFN (对齐DS_TPU_1209)

        Args:
            layer_index: 层索引
            num_tokens: 当前处理的 token 数量
            context_length: KV cache 长度
            phase: 推理阶段

        Returns:
            评估后的层对象，包含所有算子的性能数据
        """
        from ..layers.base import BaseLayer

        # 判断层类型
        use_mla = self.model.attention_type == "mla" and self.model.mla_config is not None

        # 判断是否为 MoE 层
        is_moe = self.model.model_type == "moe" and self.model.moe_config is not None and layer_index >= self.model.moe_config.first_k_dense_replace

        # 构建层配置
        layer_config = {
            "hidden_dim": self.model.hidden_size,
            "batch_size": self.inference.batch_size,
            "seq_len": num_tokens,
            "kv_seq_len": context_length,
            "tp": self.parallelism.tp,
            "comm_protocol": 1,  # 默认协议
        }

        # ========== 1. 构建Attention层 ==========
        if use_mla and self.model.mla_config:
            # MLA 层配置
            mla = self.model.mla_config
            layer_config.update(
                {
                    "num_heads": self.model.num_attention_heads,
                    "qk_nope_dim": mla.qk_nope_head_dim,
                    "qk_rope_dim": mla.qk_rope_head_dim,
                    "v_head_dim": mla.v_head_dim,
                    "kv_lora_rank": mla.kv_lora_rank,
                    "q_lora_rank": mla.q_lora_rank,
                }
            )

            # 从模型配置读取 MLA 变体（而非模拟配置）
            mla_variant = mla.variant
            if mla_variant == "mla_v32":
                attention_layer = MLAv32Layer(name=f"layer_{layer_index}_mla", config=layer_config)
            elif mla_variant == "mla_absorb":
                attention_layer = MLAAbsorbLayer(name=f"layer_{layer_index}_mla", config=layer_config)
            elif mla_variant == "mla_absorb_v32":
                attention_layer = MLAAbsorbv32Layer(name=f"layer_{layer_index}_mla", config=layer_config)
            else:
                attention_layer = MLALayer(name=f"layer_{layer_index}_mla", config=layer_config)
        else:
            # 标准 MHA 层
            layer_config.update(
                {
                    "num_heads": self.model.num_attention_heads,
                    "num_kv_heads": self.model.num_kv_heads,
                    "head_dim": self.model.hidden_size // self.model.num_attention_heads,
                }
            )
            attention_layer = MHALayer(name=f"layer_{layer_index}_mha", config=layer_config)

        # ========== 2. 构建FFN层 ==========
        ffn_config = {
            "hidden_dim": self.model.hidden_size,
            "inter_dim": self.model.intermediate_size,
            "batch_size": self.inference.batch_size,
            "seq_len": num_tokens,
            "tp": self.parallelism.tp,
            "dp": self.parallelism.dp,
            "ep": self.parallelism.ep,
            "comm_protocol": 1,
        }

        if is_moe:
            # MoE层 - 需要额外的 moe_tp 参数
            # 从拓扑配置中获取 moe_tp，如果没有则根据 MoE 约束计算
            # MoE 约束: DP × TP = MoE_TP × EP
            moe_tp = self.moe_tp
            if moe_tp is None:
                # 根据约束计算: moe_tp = (dp * tp) / ep
                moe_tp = (self.parallelism.dp * self.parallelism.tp) // self.parallelism.ep if self.parallelism.ep > 0 else 1

            ffn_config.update(
                {
                    "num_experts": self.model.moe_config.num_experts,
                    "num_experts_per_tok": self.model.moe_config.num_experts_per_tok,
                    "expert_intermediate_size": self.model.moe_config.expert_intermediate_size,
                    "moe_tp": moe_tp,
                }
            )
            ffn_layer = MoELayer(name=f"layer_{layer_index}_moe", config=ffn_config)
        else:
            # 标准MLP层
            ffn_layer = MLPLayer(name=f"layer_{layer_index}_mlp", config=ffn_config)

        # ========== 3. 合并Attention和FFN的算子 ==========
        # 创建组合层，包含完整的Transformer层
        combined_layer = BaseLayer(name=f"layer_{layer_index}", layer_type="TransformerLayer")

        # 添加Attention的所有算子
        for op in attention_layer.comp_ops:
            combined_layer.add_operator(op)
        for op in attention_layer.comm_ops:
            combined_layer.add_operator(op)

        # 添加FFN的所有算子
        for op in ffn_layer.comp_ops:
            combined_layer.add_operator(op)
        for op in ffn_layer.comm_ops:
            combined_layer.add_operator(op)

        # ========== 4. 评估所有算子 ==========
        if self.config.use_precise_evaluator and self.arch is not None:
            self._evaluate_layer_operators(combined_layer)

        return combined_layer

    def _evaluate_layer_operators(self, layer):
        """直接评估层中的所有算子"""
        # 导入评估器（延迟导入避免循环依赖）
        from ..evaluators import (
            GEMMEvaluator,
            FA2Evaluator,
            RMSNormEvaluator,
            AllReduceEval,
            AllGatherEval,
            ReduceScatterEval,
        )

        # 🔑 使用全局评估器（复用缓存）
        gemm_eval = self.gemm_evaluator
        fa2_eval = FA2Evaluator(self.arch)
        rmsnorm_eval = RMSNormEvaluator(self.arch)
        # 通信评估器使用前端传递的配置
        allreduce_eval = AllReduceEval(self.arch, self.protocol_cfg, self.network_cfg)
        allgather_eval = AllGatherEval(self.arch, self.protocol_cfg, self.network_cfg)
        reducescatter_eval = ReduceScatterEval(self.arch, self.protocol_cfg, self.network_cfg)

        # 评估所有计算算子
        import logging

        logger = logging.getLogger(__name__)

        total_ops = len(layer.comp_ops)
        cached_ops = 0
        evaluated_ops = 0

        for op_idx, op in enumerate(layer.comp_ops):
            cache_key = op.get_cache_key()

            # 检查缓存
            if cache_key in self.eval_cache:
                op.apply_result(self.eval_cache[cache_key])
                cached_ops += 1
                continue

            # 报告详细进度（每10个算子或最后一个）
            # if (op_idx + 1) % 10 == 0 or (op_idx + 1) == total_ops:
            # logger.info(f"      评估算子 {op_idx + 1}/{total_ops} (缓存命中: {cached_ops}, 已评估: {evaluated_ops})")

            # 评估算子
            if op.operator_type == "MatMulOperator":
                result = gemm_eval.evaluate(
                    G=op.parallel_params.get("G", 1),
                    M=op.parallel_params.get("M", 1),
                    K=op.parallel_params.get("K", 1),
                    N=op.parallel_params.get("N", 1),
                    input_dtype=op.parallel_params.get("input_dtype", "bf16"),
                    output_dtype=op.parallel_params.get("output_dtype", "bf16"),
                    use_multiprocess=True,  # 🚀 运行时启用多进程搜索
                )
                op.elapse = result.latency_us
                op.comp_elapse = result.compute_time_us
                op.dma_elapse = result.memory_time_us
                op.dram_traffic = result.dram_traffic_bytes
                op.urate = result.effective_utilization

            elif op.operator_type == "FA2Operator":
                result = fa2_eval.evaluate(
                    B=op.parallel_params.get("B", 1),
                    QS=op.parallel_params.get("QS", 1),
                    KS=op.parallel_params.get("KS", 1),
                    QD=op.parallel_params.get("QD", 1),
                    VD=op.parallel_params.get("VD", 1),
                )
                op.elapse = result.latency_us
                op.comp_elapse = result.compute_time_us
                op.dma_elapse = result.memory_time_us
                op.dram_traffic = result.dram_traffic_bytes
                op.urate = result.effective_utilization

            elif op.operator_type == "MHAOperator":
                # MHA 使用 FA2 评估器，等效 B = B * H
                B = op.parallel_params.get("B", 1)
                H = op.parallel_params.get("H", 1)
                result = fa2_eval.evaluate(
                    B=B * H,
                    QS=op.parallel_params.get("QS", 1),
                    KS=op.parallel_params.get("KS", 1),
                    QD=op.parallel_params.get("QD", 1),
                    VD=op.parallel_params.get("VD", 1),
                )
                op.elapse = result.latency_us
                op.comp_elapse = result.compute_time_us
                op.dma_elapse = result.memory_time_us
                op.dram_traffic = result.dram_traffic_bytes
                op.urate = result.effective_utilization

            elif op.operator_type == "MQAOperator":
                # MQA 也使用 FA2 评估器
                result = fa2_eval.evaluate(
                    B=op.parallel_params.get("B", 1),
                    QS=op.parallel_params.get("QS", 1),
                    KS=op.parallel_params.get("KS", 1),
                    QD=op.parallel_params.get("QD", 1),
                    VD=op.parallel_params.get("VD", 1),
                )
                op.elapse = result.latency_us
                op.comp_elapse = result.compute_time_us
                op.dma_elapse = result.memory_time_us
                op.dram_traffic = result.dram_traffic_bytes
                op.urate = result.effective_utilization

            elif op.operator_type == "RMSNormOperator":
                result = rmsnorm_eval.evaluate(
                    batch_size=op.parallel_params.get("batch_size", 1),
                    hidden_dim=op.parallel_params.get("hidden_dim", 1),
                    has_scale=op.parallel_params.get("has_scale", True),
                    has_bias=op.parallel_params.get("has_bias", False),
                )
                # RMSNorm 主要受带宽限制
                data_bytes = op.parallel_params.get("batch_size", 1) * op.parallel_params.get("hidden_dim", 1) * 2 * 2
                op.elapse = (data_bytes / self.arch.dram_bandwidth_bytes) * 1e6
                op.comp_elapse = op.elapse * 0.1
                op.dma_elapse = op.elapse * 0.9
                op.dram_traffic = data_bytes
                op.urate = result.utilization

            # 缓存结果
            self.eval_cache[cache_key] = {
                "elapse": op.elapse,
                "comp_elapse": op.comp_elapse,
                "dma_elapse": op.dma_elapse,
                "dram_traffic": op.dram_traffic,
                "urate": op.urate,
            }

        # 评估所有通信算子
        for op in layer.comm_ops:
            cache_key = op.get_cache_key()

            if cache_key in self.eval_cache:
                op.apply_result(self.eval_cache[cache_key])
                continue

            # 评估通信算子
            tp = op.parallel_params.get("tp", 1)
            comm_size = op.parallel_params.get("comm_size", 0)
            comm_protocol = op.parallel_params.get("comm_protocol", 1)

            if op.comm_kind == "allreduce":
                result = allreduce_eval.evaluate(tp, comm_size, comm_protocol)
                op.comm_elapse = result.latency_us
            elif op.comm_kind == "allgather":
                result = allgather_eval.evaluate(tp, comm_size, comm_protocol)
                op.comm_elapse = result.latency_us
            elif op.comm_kind == "reducescatter":
                result = reducescatter_eval.evaluate(tp, comm_size, comm_protocol)
                op.comm_elapse = result.latency_us
            else:
                # 默认使用简单的带宽模型
                op.comm_elapse = (comm_size / self.tp_bandwidth) * 1e6

            # 缓存结果
            self.eval_cache[cache_key] = {"comm_elapse": op.comm_elapse}

    def _report_progress(self, percent: float, message: str):
        """报告进度"""
        import sys

        print(f"[DEBUG SIMULATOR] _report_progress: percent={percent}, message={message}", flush=True)
        sys.stdout.flush()
        if self.progress_callback:
            try:
                self.progress_callback(percent, message)
            except Exception as e:
                print(f"[DEBUG SIMULATOR] callback error: {e}", flush=True)
                pass  # 忽略回调错误

    def simulate(self) -> SimulationResult:
        """
        运行完整模拟

        Returns:
            模拟结果
        """
        import logging

        logger = logging.getLogger(__name__)

        wall_start = time.time()
        current_time = 0.0

        # 进度划分:
        # 0-10%: H2D 数据传输
        # 10-50%: Prefill 推理 (按层细分)
        # 50-90%: Decode 推理 (按 token 细分)
        # 90-100%: D2H + Gantt + 统计

        # 阶段1: 数据搬运 (H2D)
        self._report_progress(0, "H2D 数据传输...")
        phase_start = time.time()
        if self.config.enable_data_transfer:
            current_time = self._simulate_data_transfer_h2d(current_time)
        h2d_wall_time = (time.time() - phase_start) * 1000
        self._report_progress(10, "H2D 完成")

        # 阶段2: Prefill 推理 (10-50%)
        phase_start = time.time()
        prefill_end_time = self._simulate_prefill(current_time, report_progress=True)
        phase_transition = prefill_end_time
        prefill_wall_time = (time.time() - phase_start) * 1000

        # 阶段3: Decode 推理 (50-90%)
        phase_start = time.time()
        decode_end_time = self._simulate_decode(prefill_end_time, report_progress=True)
        decode_wall_time = (time.time() - phase_start) * 1000
        num_tokens = min(self.config.max_simulated_tokens, self.inference.output_seq_length)

        # 阶段4: 数据收集 (D2H)
        self._report_progress(90, "D2H 数据传输...")
        phase_start = time.time()
        if self.config.enable_data_transfer:
            final_time = self._simulate_data_transfer_d2h(decode_end_time)
        else:
            final_time = decode_end_time
        d2h_wall_time = (time.time() - phase_start) * 1000

        # 构建甘特图
        self._report_progress(93, "构建 Gantt 图...")
        phase_start = time.time()
        gantt_data = self.gantt_builder.build(phase_transition=phase_transition)
        gantt_wall_time = (time.time() - phase_start) * 1000

        # 计算统计信息
        self._report_progress(96, "计算统计信息...")
        phase_start = time.time()
        stats = self._compute_stats(final_time)
        stats_wall_time = (time.time() - phase_start) * 1000

        total_wall_time = (time.time() - wall_start) * 1000

        # 📊 打印 GEMM 缓存统计（如果使用了精确评估器）
        if self.config.use_precise_evaluator and hasattr(self, "gemm_evaluator"):
            logger.info("")  # 空行分隔
            self.gemm_evaluator.print_cache_stats()

        # 📊 打印性能摘要

        # 计算各阶段时间占比
        stages = [
            ("H2D数据传输", h2d_wall_time),
            ("Prefill推理", prefill_wall_time),
            ("Decode推理", decode_wall_time),
            ("D2H数据传输", d2h_wall_time),
            ("Gantt图构建", gantt_wall_time),
            ("统计计算", stats_wall_time),
        ]

        for stage_name, stage_time in stages:
            percent = (stage_time / total_wall_time * 100) if total_wall_time > 0 else 0
            logger.info(f"   {stage_name:12s}: {stage_time:7.2f}ms ({percent:5.1f}%)")

        logger.info(f"   {'─' * 35}")
        logger.info(f"   {'总计':12s}: {total_wall_time:7.2f}ms")

        # 识别瓶颈
        max_stage = max(stages, key=lambda x: x[1])
        if max_stage[1] > 0:
            logger.info(f"   🎯 最慢阶段: {max_stage[0]} ({max_stage[1]:.2f}ms)")

        logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        # 生成链路流量统计
        link_traffic_stats = self._generate_link_traffic_stats()
        if link_traffic_stats:
            logger.info(f"📊 链路流量统计: {len(link_traffic_stats)} 条链路")

        return SimulationResult(
            gantt_chart=gantt_data,
            stats=stats,
            link_traffic_stats=link_traffic_stats,
            timestamp=time.time(),
        )

    def _simulate_data_transfer_h2d(self, start_time: float) -> float:
        """模拟 Host to Device 数据传输"""
        # 计算输入数据大小
        bytes_per_elem = get_bytes_per_element(self.model.dtype)
        input_size_gb = (self.inference.batch_size * self.inference.input_seq_length * self.model.hidden_size * bytes_per_elem) / (1024**3)

        # 数据传输延迟 (使用 C2C 带宽，简化 Host-Device 传输)
        # 实际 PCIe 带宽约 32-64 GB/s，但对 LLM 推理影响很小，使用 C2C 带宽简化
        transfer_bw_gbps = self.hardware.c2c_bandwidth_gbps
        transfer_latency_us = self.hardware.c2c_latency_us
        transfer_latency = (input_size_gb / transfer_bw_gbps) * 1000 + transfer_latency_us / 1000  # 转换为 ms

        # 为第一个 PP stage 的所有芯片添加传输任务
        for chip_id, state in self.chip_states.items():
            if state.pp_stage == 0:
                self.gantt_builder.add_task(
                    name="H2D Transfer",
                    start=start_time,
                    end=start_time + transfer_latency,
                    task_type=GanttTaskType.PCIE_H2D,
                    phase=InferencePhase.PREFILL,
                    chip_id=chip_id,
                    pp_stage=0,
                )
                state.compute_idle_at = start_time + transfer_latency

        return start_time + transfer_latency

    def _simulate_data_transfer_d2h(self, start_time: float) -> float:
        """模拟 Device to Host 数据传输"""
        # 计算输出数据大小 (logits)
        bytes_per_elem = get_bytes_per_element(self.model.dtype)
        output_size_gb = (self.inference.batch_size * self.model.vocab_size * bytes_per_elem) / (1024**3)

        # 数据传输延迟 (使用 C2C 带宽，简化 Device-Host 传输)
        transfer_bw_gbps = self.hardware.c2c_bandwidth_gbps
        transfer_latency_us = self.hardware.c2c_latency_us
        transfer_latency = (output_size_gb / transfer_bw_gbps) * 1000 + transfer_latency_us / 1000  # 转换为 ms

        # 为最后一个 PP stage 的所有芯片添加传输任务
        last_stage = self.parallelism.pp - 1
        for chip_id, state in self.chip_states.items():
            if state.pp_stage == last_stage:
                self.gantt_builder.add_task(
                    name="D2H Transfer",
                    start=start_time,
                    end=start_time + transfer_latency,
                    task_type=GanttTaskType.PCIE_D2H,
                    phase=InferencePhase.DECODE,
                    chip_id=chip_id,
                    pp_stage=last_stage,
                )

        return start_time + transfer_latency

    def _simulate_prefill(self, start_time: float, report_progress: bool = False) -> float:
        """模拟 Prefill 阶段

        Args:
            start_time: 开始时间
            report_progress: 是否报告进度（默认 False）

        Returns:
            Prefill 结束时间
        """
        import logging
        logger = logging.getLogger(__name__)

        num_tokens = self.inference.input_seq_length
        context_length = self.inference.input_seq_length
        num_layers = self.model.num_layers

        # 每个 PP stage 处理的层数（至少为 1，防止除零）
        layers_per_stage = max(1, num_layers // self.parallelism.pp)

        # 为每个 PP stage 模拟
        stage_times = [start_time] * self.parallelism.pp

        if report_progress:
            logger.info(f"━━ 开始 Prefill 阶段：共 {num_layers} 层 ━━")

        for layer in range(num_layers):
            layer_wall_start = time.time() if report_progress else None

            # 报告进度: 10% + (layer / num_layers) * 40%
            if report_progress:
                progress = 10 + (layer / num_layers) * 40
                layer_progress_msg = f"Prefill Layer {layer + 1}/{num_layers}"
                self._report_progress(progress, layer_progress_msg)
                logger.info(f"")
                logger.info(f"  🔹 开始评估 Layer {layer + 1}/{num_layers} (进度: {progress:.1f}%)")

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

                    # 累加 PP 通信流量
                    task_id = f"pp_comm_prefill_layer{layer}_stage{pp_stage}"
                    self._accumulate_pp_comm_traffic(
                        from_stage=pp_stage - 1,
                        to_stage=pp_stage,
                        num_tokens=num_tokens,
                        task_id=task_id,
                        task_type=GanttTaskType.PP_COMM,
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

            # 打印层评估墙上时间
            if report_progress and layer_wall_start is not None:
                layer_wall_time = (time.time() - layer_wall_start) * 1000
                logger.info(f"  ✅ Layer {layer + 1}/{num_layers} 完成，墙上时间: {layer_wall_time:.2f}ms")

        # 返回最后一个 stage 的结束时间
        prefill_end = max(stage_times)

        # 更新统计
        self.prefill_stats.total_time = prefill_end - start_time

        if report_progress:
            self._report_progress(50, "Prefill 完成")

        return prefill_end

    def _simulate_decode(self, start_time: float, report_progress: bool = False) -> float:
        """模拟 Decode 阶段

        Args:
            start_time: 开始时间
            report_progress: 是否报告进度（默认 False）

        Returns:
            Decode 结束时间
        """
        import logging
        logger = logging.getLogger(__name__)

        current_time = start_time
        num_tokens_to_simulate = min(self.config.max_simulated_tokens, self.inference.output_seq_length)

        # 每个 PP stage 处理的层数（至少为 1，防止除零）
        layers_per_stage = max(1, self.model.num_layers // self.parallelism.pp)

        for token_idx in range(num_tokens_to_simulate):
            # 报告进度: 50% + (token_idx / num_tokens) * 40%
            if report_progress:
                progress = 50 + (token_idx / num_tokens_to_simulate) * 40
                self._report_progress(progress, f"Decode Token {token_idx + 1}/{num_tokens_to_simulate}")

            token_wall_start = time.time()
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

                        # 累加 PP 通信流量
                        task_id = f"pp_comm_decode_token{token_idx}_layer{layer}_stage{pp_stage}"
                        self._accumulate_pp_comm_traffic(
                            from_stage=pp_stage - 1,
                            to_stage=pp_stage,
                            num_tokens=1,
                            task_id=task_id,
                            task_type=GanttTaskType.PP_COMM,
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

            # 📊 每个token的性能日志
            token_wall_time = (time.time() - token_wall_start) * 1000
            logger.info(f"    🔹 Token {token_idx}/{num_tokens_to_simulate}: 墙上时间 {token_wall_time:.2f}ms, 遍历了 {self.model.num_layers} 层")

        # 更新统计
        self.decode_stats.total_time = current_time - start_time

        if report_progress:
            self._report_progress(90, "Decode 完成")

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

        # 使用新的精确评估器
        if self.config.use_precise_evaluator and self.arch is not None:
            return self._simulate_single_layer_precise(current_time, layer_index, num_tokens, context_length, phase, chip_id, pp_stage, token_index)

        # 回退到简化模拟（粗粒度）
        return self._simulate_single_layer_coarse(current_time, layer_index, num_tokens, context_length, phase, chip_id, pp_stage, token_index)

    def _simulate_single_layer_precise(
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
        """使用精确评估器模拟单层（基于算子）"""

        # 构建并评估层
        layer_wall_start = time.time()
        layer = self._build_layer_for_evaluation(layer_index, num_tokens, context_length, phase)
        build_time = (time.time() - layer_wall_start) * 1000

        # 根据评估粒度决定是否展开所有算子
        gantt_wall_start = time.time()
        if self.config.evaluation_granularity == "fine":
            # 检查是否为 MoE 层且启用了 TBO 优化
            from ..layers import MoELayer

            if self.config.enable_tbo and isinstance(layer, MoELayer):
                # TBO 模式: 标记被重叠隐藏的通信算子
                dispatch_lat = layer._get_operator_latency("dispatch")
                combine_lat = layer._get_operator_latency("combine")

                routed_gate_lat = layer._get_operator_latency("routed_gate")
                routed_up_lat = layer._get_operator_latency("routed_up")
                routed_down_lat = layer._get_operator_latency("routed_down")
                routed_allreduce_lat = layer._get_operator_latency("routed_allreduce")
                routed_compute_lat = routed_gate_lat + routed_up_lat + routed_down_lat + routed_allreduce_lat

                shared_gate_lat = layer._get_operator_latency("shared_gate")
                shared_up_lat = layer._get_operator_latency("shared_up")
                shared_down_lat = layer._get_operator_latency("shared_down")
                shared_allreduce_lat = layer._get_operator_latency("shared_allreduce")
                shared_compute_lat = shared_gate_lat + shared_up_lat + shared_down_lat + shared_allreduce_lat

                # 计算被隐藏的延迟
                dispatch_hidden = min(dispatch_lat, routed_compute_lat)
                if shared_compute_lat > 0:
                    combine_hidden = min(combine_lat, shared_compute_lat)
                else:
                    combine_hidden = min(combine_lat, routed_compute_lat)

                # 遍历所有计算算子 (正常添加)
                for op in layer.comp_ops:
                    task_type = self._map_compute_op_to_task_type(op.op_type, op.name)
                    latency_ms = op.elapse / 1000

                    # 构造详细信息字典
                    extra_fields = {
                        "flops": op.flops,
                        "params_bytes": op.param,
                        "dram_occupy_bytes": op.dram_occupy,
                        "dram_traffic_bytes": op.dram_traffic,
                        "compute_time_us": op.comp_elapse,
                        "memory_time_us": op.dma_elapse,
                        "arch_utilization": op.urate,
                        "parallel_config": {
                            "tp": self.parallelism.tp,
                            "dp": self.parallelism.dp,
                            "pp": self.parallelism.pp,
                            "ep": self.parallelism.ep,
                            "sp": self.parallelism.sp,
                        },
                    }

                    # 添加 GEMM 优化结果
                    if op.best_tile is not None:
                        extra_fields["best_tile"] = op.best_tile
                    if op.best_partition is not None:
                        extra_fields["best_partition"] = op.best_partition
                    if hasattr(op, "parallel_params") and op.parallel_params:
                        extra_fields["gemm_shape"] = {
                            "G": op.parallel_params.get("G"),
                            "M": op.parallel_params.get("M"),
                            "K": op.parallel_params.get("K"),
                            "N": op.parallel_params.get("N"),
                        }

                    self.gantt_builder.add_compute_task(task_type, current_time, latency_ms, phase, chip_id, pp_stage, layer_index, token_index, **extra_fields)
                    current_time += latency_ms

                # 遍历通信算子 (应用 TBO 重叠)
                for op in layer.comm_ops:
                    task_type = self._map_comm_op_to_task_type(op.comm_kind)
                    latency_ms = op.comm_elapse / 1000

                    # 如果是 dispatch 或 combine，减去被隐藏的部分
                    if op.name.endswith("dispatch") and dispatch_hidden > 0:
                        effective_latency_ms = max(0, latency_ms - dispatch_hidden / 1000)
                    elif op.name.endswith("combine") and combine_hidden > 0:
                        effective_latency_ms = max(0, latency_ms - combine_hidden / 1000)
                    else:
                        effective_latency_ms = latency_ms

                    if effective_latency_ms > 0:
                        # 推断通信组大小
                        comm_group_size = 1
                        if "tp" in op.comm_kind or "allreduce" in op.comm_kind.lower():
                            comm_group_size = self.parallelism.tp
                        elif "dp" in op.comm_kind:
                            comm_group_size = self.parallelism.dp
                        elif "ep" in op.comm_kind or "dispatch" in op.comm_kind or "combine" in op.comm_kind:
                            comm_group_size = self.parallelism.ep
                        elif "sp" in op.comm_kind:
                            comm_group_size = self.parallelism.sp

                        # 构造通信详细信息
                        comm_extra = {
                            "comm_size_bytes": op.comm_size,
                            "comm_time_us": op.comm_elapse,
                            "comm_algorithm": op.parallel_params.get("algorithm", "unknown"),
                            "comm_group_size": comm_group_size,
                            "parallel_config": {
                                "tp": self.parallelism.tp,
                                "dp": self.parallelism.dp,
                                "pp": self.parallelism.pp,
                                "ep": self.parallelism.ep,
                                "sp": self.parallelism.sp,
                            },
                        }

                        self.gantt_builder.add_comm_task(task_type, current_time, effective_latency_ms, phase, chip_id, pp_stage, layer_index, token_index, **comm_extra)
                        current_time += effective_latency_ms
            else:
                # 标准模式: 细粒度遍历所有算子
                for op in layer.comp_ops:
                    task_type = self._map_compute_op_to_task_type(op.op_type, op.name)
                    latency_ms = op.elapse / 1000

                    # 构造详细信息字典
                    extra_fields = {
                        "flops": op.flops,
                        "params_bytes": op.param,
                        "dram_occupy_bytes": op.dram_occupy,
                        "dram_traffic_bytes": op.dram_traffic,
                        "compute_time_us": op.comp_elapse,
                        "memory_time_us": op.dma_elapse,
                        "arch_utilization": op.urate,
                        "parallel_config": {
                            "tp": self.parallelism.tp,
                            "dp": self.parallelism.dp,
                            "pp": self.parallelism.pp,
                            "ep": self.parallelism.ep,
                            "sp": self.parallelism.sp,
                        },
                    }

                    # 添加 GEMM 优化结果
                    if op.best_tile is not None:
                        extra_fields["best_tile"] = op.best_tile
                    if op.best_partition is not None:
                        extra_fields["best_partition"] = op.best_partition
                    if hasattr(op, "parallel_params") and op.parallel_params:
                        extra_fields["gemm_shape"] = {
                            "G": op.parallel_params.get("G"),
                            "M": op.parallel_params.get("M"),
                            "K": op.parallel_params.get("K"),
                            "N": op.parallel_params.get("N"),
                        }

                    self.gantt_builder.add_compute_task(task_type, current_time, latency_ms, phase, chip_id, pp_stage, layer_index, token_index, **extra_fields)
                    current_time += latency_ms

                # 遍历所有通信算子
                for op in layer.comm_ops:
                    task_type = self._map_comm_op_to_task_type(op.comm_kind)
                    latency_ms = op.comm_elapse / 1000

                    # 推断通信组大小
                    comm_group_size = 1
                    if "tp" in op.comm_kind or "allreduce" in op.comm_kind.lower():
                        comm_group_size = self.parallelism.tp
                    elif "dp" in op.comm_kind:
                        comm_group_size = self.parallelism.dp
                    elif "ep" in op.comm_kind or "dispatch" in op.comm_kind or "combine" in op.comm_kind:
                        comm_group_size = self.parallelism.ep
                    elif "sp" in op.comm_kind:
                        comm_group_size = self.parallelism.sp

                    # 构造通信详细信息
                    comm_extra = {
                        "comm_size_bytes": op.comm_size,
                        "comm_time_us": op.comm_elapse,
                        "comm_algorithm": op.parallel_params.get("algorithm", "unknown"),
                        "comm_group_size": comm_group_size,
                        "parallel_config": {
                            "tp": self.parallelism.tp,
                            "dp": self.parallelism.dp,
                            "pp": self.parallelism.pp,
                            "ep": self.parallelism.ep,
                            "sp": self.parallelism.sp,
                        },
                    }

                    self.gantt_builder.add_comm_task(task_type, current_time, latency_ms, phase, chip_id, pp_stage, layer_index, token_index, **comm_extra)

                    # 累加 TP 通信流量（AllReduce）
                    if "tp" in op.comm_kind or "allreduce" in op.comm_kind.lower():
                        data_size_gb = op.comm_size / (1024 ** 3)
                        task_id_comm = f"tp_comm_{phase.value}_layer{layer_index}_token{token_index}_{chip_id}"
                        import logging
                        logger = logging.getLogger(__name__)
                        logger.debug(f"累加 TP 流量: {data_size_gb:.4f} GB, chip={chip_id}")
                        self._accumulate_tp_comm_traffic(
                            chip_id=chip_id,
                            data_size_gb=data_size_gb,
                            task_id=task_id_comm,
                            task_type=task_type,
                        )

                    current_time += latency_ms
        else:
            # 粗粒度：聚合整层
            # 检查是否为 MoE 层且启用了 TBO 优化
            from ..layers import MoELayer

            if self.config.enable_tbo and isinstance(layer, MoELayer):
                # 使用 TBO 优化计算延迟
                total_layer_time = layer.calculate_latency_with_tbo() / 1000  # us -> ms

                # 添加聚合任务到甘特图
                if total_layer_time > 0:
                    self.gantt_builder.add_compute_task(GanttTaskType.MOE_EXPERT, current_time, total_layer_time, phase, chip_id, pp_stage, layer_index, token_index)
                    current_time += total_layer_time
            else:
                # 标准模式：简单求和
                total_compute_time = sum(op.elapse for op in layer.comp_ops) / 1000
                total_comm_time = sum(op.comm_elapse for op in layer.comm_ops) / 1000

                if total_compute_time > 0:
                    self.gantt_builder.add_compute_task(GanttTaskType.COMPUTE, current_time, total_compute_time, phase, chip_id, pp_stage, layer_index, token_index)
                    current_time += total_compute_time

                if total_comm_time > 0:
                    self.gantt_builder.add_comm_task(GanttTaskType.TP_COMM, current_time, total_comm_time, phase, chip_id, pp_stage, layer_index, token_index)
                    current_time += total_comm_time

        gantt_time = (time.time() - gantt_wall_start) * 1000

        # 📊 性能日志（打印前3层的详细timing，或decode第一个token的所有层）
        import logging

        logger = logging.getLogger(__name__)

        # 条件1: Prefill阶段的前3层
        # 条件2: Decode第一个token的前3层
        # 条件3: 如果环境变量设置了详细日志，打印所有层
        import os

        verbose_logging = os.environ.get("GEMM_VERBOSE_LOGGING", "0") == "1"

        should_log = False
        if phase == InferencePhase.PREFILL and layer_index < 3:
            should_log = True
        elif phase == InferencePhase.DECODE and token_index == 0 and layer_index < 3:
            should_log = True
        elif verbose_logging:
            should_log = True

        if should_log:
            logger.info(f"      🔸 [{phase.value}] 层{layer_index}: build={build_time:.2f}ms, gantt={gantt_time:.2f}ms, ops={len(layer.comp_ops)}+{len(layer.comm_ops)}")

        return current_time

    def _simulate_single_layer_coarse(
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
        """粗粒度模拟单层（简化公式，用于 fallback）"""

        # 简化计算：使用固定公式估算
        # 注意：这是一个非常粗略的估算，仅作为 fallback

        bytes_per_elem = get_bytes_per_element(self.model.dtype)
        hidden_size = self.model.hidden_size

        # Attention 部分延迟估算（简化）
        # QKV投影 + Score计算 + Output投影
        qkv_size = hidden_size * hidden_size * 3
        qkv_flops = 2 * num_tokens * qkv_size
        attn_score_flops = 2 * num_tokens * context_length * hidden_size
        compute_tflops = self.hardware.compute_tflops_bf16 * 1e12
        attn_latency_ms = (qkv_flops + attn_score_flops) / compute_tflops * 1000

        # FFN 部分延迟估算
        intermediate_size = self.model.intermediate_size
        ffn_flops = 2 * num_tokens * hidden_size * intermediate_size * 3  # gate, up, down
        ffn_latency_ms = ffn_flops / compute_tflops * 1000

        total_compute_ms = attn_latency_ms + ffn_latency_ms

        self.gantt_builder.add_compute_task(GanttTaskType.COMPUTE, current_time, total_compute_ms, phase, chip_id, pp_stage, layer_index, token_index)
        current_time += total_compute_ms

        # TP 通信
        if self.parallelism.tp > 1:
            tp_comm_latency = self._calc_tp_allreduce_latency(num_tokens)
            self.gantt_builder.add_comm_task(GanttTaskType.TP_COMM, current_time, tp_comm_latency, phase, chip_id, pp_stage, layer_index, token_index)

            # 累加 TP 通信流量
            bytes_per_elem = get_bytes_per_element(self.model.dtype)
            data_size_bytes = self.inference.batch_size * num_tokens * self.model.hidden_size * bytes_per_elem
            data_size_gb = data_size_bytes / (1024 ** 3)
            task_id_tp = f"tp_comm_coarse_{phase.value}_layer{layer_index}_token{token_index}_{chip_id}"
            self._accumulate_tp_comm_traffic(
                chip_id=chip_id,
                data_size_gb=data_size_gb,
                task_id=task_id_tp,
                task_type=GanttTaskType.TP_COMM,
            )

            current_time += tp_comm_latency

        return current_time

    def _calc_tp_allreduce_latency(self, num_tokens: int) -> float:
        """计算 TP AllReduce 延迟（Ring AllReduce 算法）"""
        bytes_per_elem = get_bytes_per_element(self.model.dtype)
        data_size_gb = (self.inference.batch_size * num_tokens * self.model.hidden_size * bytes_per_elem) / (1024**3)

        # Ring AllReduce: 2 * (N-1) / N * data_size / bandwidth + latency
        tp = self.parallelism.tp
        if tp <= 1:
            return 0.0

        transfer_time = 2 * (tp - 1) / tp * data_size_gb / self.tp_bandwidth * 1000  # ms
        latency_overhead = self.tp_latency / 1000  # us -> ms
        return transfer_time + latency_overhead

    def _calc_pp_comm_latency(self, num_tokens: int) -> float:
        """计算 PP P2P 通信延迟"""
        bytes_per_elem = get_bytes_per_element(self.model.dtype)
        data_size_gb = (self.inference.batch_size * num_tokens * self.model.hidden_size * bytes_per_elem) / (1024**3)

        # P2P: data_size / bandwidth + latency
        transfer_time = data_size_gb / self.pp_bandwidth * 1000  # ms
        latency_overhead = self.pp_latency / 1000  # us -> ms
        return transfer_time + latency_overhead

    def _calc_sp_allgather_latency(self, num_tokens: int) -> float:
        """计算 SP AllGather 延迟"""
        if self.parallelism.sp <= 1:
            return 0.0

        # 计算数据量
        bytes_per_elem = get_bytes_per_element(self.model.dtype)
        data_size_gb = (self.inference.batch_size * num_tokens * self.model.hidden_size * bytes_per_elem) / (1024**3)

        # AllGather: (N-1) / N * data_size / bandwidth + latency
        sp = self.parallelism.sp
        transfer_time = (sp - 1) / sp * data_size_gb / self.tp_bandwidth * 1000
        latency_overhead = self.tp_latency / 1000
        return transfer_time + latency_overhead

    def _calc_sp_reduce_scatter_latency(self, num_tokens: int) -> float:
        """计算 SP ReduceScatter 延迟"""
        if self.parallelism.sp <= 1:
            return 0.0

        # 计算数据量
        bytes_per_elem = get_bytes_per_element(self.model.dtype)
        data_size_gb = (self.inference.batch_size * num_tokens * self.model.hidden_size * bytes_per_elem) / (1024**3)

        # ReduceScatter: (N-1) / N * data_size / bandwidth + latency
        sp = self.parallelism.sp
        transfer_time = (sp - 1) / sp * data_size_gb / self.tp_bandwidth * 1000
        latency_overhead = self.tp_latency / 1000
        return transfer_time + latency_overhead

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
            peak_tflops = self.hardware.compute_tflops_bf16 * chips_per_replica

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
            peak_bandwidth = self.hardware.memory_bandwidth_gbps * self.hardware.memory_bandwidth_utilization
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

            attn_params_per_layer = q_down_params + q_up_params + q_rope_params + kv_down_params + k_up_params + v_up_params + k_rope_params + o_params
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
        return (total_params * bytes_per_elem) / (1024**3)

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

        return kv_cache_bytes / (1024**3)


def run_simulation(
    topology_dict: dict[str, Any],
    model_dict: dict[str, Any],
    inference_dict: dict[str, Any],
    parallelism_dict: dict[str, Any],
    hardware_dict: dict[str, Any],
    config_dict: dict[str, Any] | None = None,
    progress_callback: callable | None = None,
    enable_tile_search: bool = True,
    enable_partition_search: bool = False,
    max_simulated_tokens: int = 4,
    max_gemm_processes: Optional[int] = None,
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
        progress_callback: 进度回调函数 (percent: float, message: str) -> None

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

    # 获取 MoE 相关的 moe_tp 参数（从 parallelism_dict 中获取）
    moe_tp = parallelism_dict.get("moe_tp")

    # 从 hardware_dict 获取芯片参数（顶层 chips 字典）
    chips_dict = hardware_dict.get("chips", {})
    if not chips_dict:
        raise ValueError("硬件配置缺少 'chips' 字段，请确保使用新格式配置")
    first_chip_name = next(iter(chips_dict))
    chip_hw = chips_dict[first_chip_name]

    # ========== 互联参数获取（支持两种格式） ==========
    # 格式1: topology_dict.interconnect.links (YAML 配置文件格式)
    # 格式2: hardware_dict 中的 chip/board/rack/pod (前端传入格式)
    interconnect = topology_dict.get("interconnect", {}).get("links", {})
    c2c_config = interconnect.get("c2c", {})
    b2b_config = interconnect.get("b2b", {})
    r2r_config = interconnect.get("r2r", {})
    p2p_config = interconnect.get("p2p", {})

    # 前端传入的硬件配置（备用来源）
    board_hw = hardware_dict.get("board", {})
    rack_hw = hardware_dict.get("rack", {})
    pod_hw = hardware_dict.get("pod", {})

    # ========== 严格参数验证（不使用默认值） ==========
    def _require_field(config: dict, field: str, config_name: str) -> Any:
        """要求字段必须存在，否则抛出错误"""
        if field not in config:
            raise ValueError(f"{config_name} 缺少必需字段: {field}")
        return config[field]

    def _require_positive(value: float, field_name: str) -> float:
        """要求值必须为正数"""
        if value <= 0:
            raise ValueError(f"{field_name} 必须为正数，当前值: {value}")
        return value

    def _get_interconnect_param(
        yaml_config: dict, yaml_field: str,
        frontend_config: dict, frontend_field: str,
        param_name: str
    ) -> float:
        """
        从两种格式中获取互联参数（优先 YAML 格式，备用前端格式）

        Args:
            yaml_config: YAML 格式的配置（如 c2c_config）
            yaml_field: YAML 格式的字段名（如 "bandwidth_gbps"）
            frontend_config: 前端格式的配置（如 chip_hw 或 board_hw）
            frontend_field: 前端格式的字段名（如 "c2c_bandwidth_gbps"）
            param_name: 参数名称（用于错误信息）

        Returns:
            参数值
        """
        # 优先从 YAML 格式获取
        if yaml_field in yaml_config:
            return yaml_config[yaml_field]
        # 备用：从前端格式获取
        if frontend_field in frontend_config:
            return frontend_config[frontend_field]
        # 都没有则报错
        raise ValueError(f"互联配置缺少必需字段: {param_name}（支持格式：topology.interconnect.links.*.{yaml_field} 或 hardware.*.{frontend_field}）")

    # 验证芯片必需参数
    chip_type = _require_field(chip_hw, "name", "芯片配置")
    num_cores = _require_positive(_require_field(chip_hw, "num_cores", "芯片配置"), "num_cores")
    compute_tflops_bf16 = _require_positive(_require_field(chip_hw, "compute_tflops_bf16", "芯片配置"), "compute_tflops_bf16")
    memory_capacity_gb = _require_positive(_require_field(chip_hw, "memory_capacity_gb", "芯片配置"), "memory_capacity_gb")
    memory_bandwidth_gbps = _require_positive(_require_field(chip_hw, "memory_bandwidth_gbps", "芯片配置"), "memory_bandwidth_gbps")

    # 验证互联必需参数（支持两种格式）
    c2c_bandwidth_gbps = _require_positive(
        _get_interconnect_param(c2c_config, "bandwidth_gbps", chip_hw, "c2c_bandwidth_gbps", "c2c_bandwidth"),
        "c2c_bandwidth_gbps"
    )
    c2c_latency_us = _get_interconnect_param(c2c_config, "latency_us", chip_hw, "c2c_latency_us", "c2c_latency")
    b2b_bandwidth_gbps = _require_positive(
        _get_interconnect_param(b2b_config, "bandwidth_gbps", board_hw, "b2b_bandwidth_gbps", "b2b_bandwidth"),
        "b2b_bandwidth_gbps"
    )
    b2b_latency_us = _get_interconnect_param(b2b_config, "latency_us", board_hw, "b2b_latency_us", "b2b_latency")
    r2r_bandwidth_gbps = _require_positive(
        _get_interconnect_param(r2r_config, "bandwidth_gbps", rack_hw, "r2r_bandwidth_gbps", "r2r_bandwidth"),
        "r2r_bandwidth_gbps"
    )
    r2r_latency_us = _get_interconnect_param(r2r_config, "latency_us", rack_hw, "r2r_latency_us", "r2r_latency")
    p2p_bandwidth_gbps = _require_positive(
        _get_interconnect_param(p2p_config, "bandwidth_gbps", pod_hw, "p2p_bandwidth_gbps", "p2p_bandwidth"),
        "p2p_bandwidth_gbps"
    )
    p2p_latency_us = _get_interconnect_param(p2p_config, "latency_us", pod_hw, "p2p_latency_us", "p2p_latency")

    # 构建运行时硬件参数（所有必需参数已验证）
    hardware = RuntimeHardwareParams(
        # 芯片参数（必需）
        chip_type=chip_type,
        num_cores=num_cores,
        compute_tflops_fp8=chip_hw.get("compute_tflops_fp8", compute_tflops_bf16 * 2),  # FP8 默认为 BF16 的 2 倍
        compute_tflops_bf16=compute_tflops_bf16,
        memory_capacity_gb=memory_capacity_gb,
        memory_bandwidth_gbps=memory_bandwidth_gbps,
        memory_bandwidth_utilization=chip_hw.get("memory_bandwidth_utilization", 0.85),
        lmem_capacity_mb=chip_hw.get("lmem_capacity_mb", 0.0),
        lmem_bandwidth_gbps=chip_hw.get("lmem_bandwidth_gbps", 0.0),
        c2c_bandwidth_gbps=c2c_bandwidth_gbps,
        c2c_latency_us=c2c_latency_us,
        # 微架构参数（可选）
        cube_m=chip_hw.get("cube_m"),
        cube_k=chip_hw.get("cube_k"),
        cube_n=chip_hw.get("cube_n"),
        sram_size_kb=chip_hw.get("sram_size_kb"),
        sram_utilization=chip_hw.get("sram_utilization"),
        lane_num=chip_hw.get("lane_num"),
        align_bytes=chip_hw.get("align_bytes"),
        compute_dma_overlap_rate=chip_hw.get("compute_dma_overlap_rate"),
        # 互联参数（必需）
        b2b_bandwidth_gbps=b2b_bandwidth_gbps,
        b2b_latency_us=b2b_latency_us,
        r2r_bandwidth_gbps=r2r_bandwidth_gbps,
        r2r_latency_us=r2r_latency_us,
        p2p_bandwidth_gbps=p2p_bandwidth_gbps,
        p2p_latency_us=p2p_latency_us,
    )

    config = SimulationConfig(
        max_simulated_tokens=max_simulated_tokens,  # 使用传入的参数
        enable_data_transfer=config_dict.get("enableDataTransferSimulation", True) if config_dict else True,
        enable_detailed_ops=config_dict.get("enableDetailedTransformerOps", True) if config_dict else True,
        enable_kv_cache=config_dict.get("enableKVCacheAccessSimulation", True) if config_dict else True,
    )

    # 从拓扑配置中提取通信延迟配置 (interconnect.comm_params)
    comm_latency_config = topology_dict.get("interconnect", {}).get("comm_params")

    # 运行模拟
    simulator = LLMInferenceSimulator(
        topology_dict=topology_dict,
        model=model,
        inference=inference,
        parallelism=parallelism,
        hardware=hardware,
        config=config,
        comm_latency_config=comm_latency_config,
        progress_callback=progress_callback,
        enable_tile_search=enable_tile_search,
        enable_partition_search=enable_partition_search,
        max_gemm_processes=max_gemm_processes,
        moe_tp=moe_tp,
    )

    result = simulator.simulate()

    # 转换为前端格式
    from .gantt import convert_to_frontend_format

    # 计算吞吐量指标（使用统一的芯片计算函数）
    from ..tasks.deployment import calculate_required_chips
    total_chips = calculate_required_chips(parallelism_dict, model_dict)

    # TPOT 转换：微秒 -> 毫秒
    tpot_ms = result.stats.avg_tpot / 1000.0 if result.stats.avg_tpot > 0 else 0.0

    # TPS per Batch: 单个请求每秒生成的token数 (用户体验指标)
    # 公式: 1000ms/s / TPOT(ms/token) = tokens/s per request
    tps_per_batch = 1000.0 / tpot_ms if tpot_ms > 0 else 0.0

    # TPS per Chip: 单芯片（单DP rank）每秒处理的总token数 (成本效益指标)
    # 公式: TPS_batch × batch_size = tokens/s per chip
    tps_per_chip = tps_per_batch * inference.batch_size

    # Total TPS: 集群总吞吐量 (tokens/s)
    # 公式: TPS_chip × DP = total tokens/s (DP线性扩展吞吐)
    tokens_per_second = tps_per_chip * parallelism.dp

    # 理论峰值吞吐量（基于硬件算力，仅作参考）
    theoretical_max_tps = tokens_per_second / max(result.stats.dynamic_mfu, 0.01) if result.stats.dynamic_mfu > 0 else 0.0

    # Requests per second: 每秒处理的请求数
    # 在持续decode场景下，每个请求占用一个batch slot
    requests_per_second = tokens_per_second / inference.output_seq_length if inference.output_seq_length > 0 else 0.0

    # 转换链路流量统计为前端格式（将 snake_case 转换为 camelCase）
    from dataclasses import asdict
    link_traffic_stats_dict = []
    for stat in result.link_traffic_stats:
        link_traffic_stats_dict.append({
            "source": stat.source,
            "target": stat.target,
            "trafficMb": stat.traffic_mb,
            "bandwidthGbps": stat.bandwidth_gbps,
            "latencyUs": stat.latency_us,
            "utilizationPercent": stat.utilization_percent,
            "linkType": stat.link_type,
            "contributingTasks": stat.contributing_tasks,
            "taskTypeBreakdown": stat.task_type_breakdown,
        })

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
            "totalChips": total_chips,
            "linkTrafficStats": link_traffic_stats_dict,  # 新增：链路流量统计
        },
        # 吞吐量指标（独立对象，与前端 ThroughputAnalysis 对应）
        "throughput": {
            "tokens_per_second": tokens_per_second,           # 集群总吞吐 (tokens/s)
            "tps_per_batch": tps_per_batch,                   # 单请求TPS (tokens/s per request) - 用户体验指标
            "tps_per_chip": tps_per_chip,                     # 单芯片TPS (tokens/s per chip) - 成本效益指标
            "requests_per_second": requests_per_second,       # 请求吞吐 (requests/s)
            "model_flops_utilization": result.stats.dynamic_mfu,  # MFU (0-1)
            "memory_bandwidth_utilization": result.stats.dynamic_mbu,  # MBU (0-1)
            "theoretical_max_throughput": theoretical_max_tps,  # 理论峰值吞吐 (tokens/s)
        },
        "timestamp": result.timestamp,
    }
