"""评估引擎.

统一评估入口，负责输入校验、评估器选择、结果聚合与 EngineResult 输出。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from math_model.L4_evaluation.calibration import Calibration, CalibrationConfig
from math_model.L4_evaluation.cost_models import ChipCostModel, CoreCostModel
from math_model.L4_evaluation.evaluators import (
    CommEvaluator,
    ComputeEvaluator,
    FallbackEvaluator,
)
from math_model.L4_evaluation.metrics import (
    Aggregates,
    CommProtocolSpec,
    EngineResult,
    Granularity,
    HardwareSpec,
    StepMetrics,
    TopologySpec,
    merge_specs,
)
from math_model.L4_evaluation.registry import CostModelRegistry, OpTypeRouter
from math_model.L3_mapping.plan.distributed_model import NodeRole

if TYPE_CHECKING:
    from math_model.L3_mapping.plan.distributed_model import DistributedModel
    from math_model.L3_mapping.plan.exec_plan import ExecPlan


class EvaluationEngine:
    """评估引擎

    统一评估入口，支持:
        - 基于 ExecPlan + HardwareSpec 做统一口径的性能评估
        - 按 granularity (Chip/Core/Lane) 切换评估精度
        - 按 OpType 路由到子评估器
        - Step 级别时延分解与瓶颈归因

    计算流程:
        - 口径校验 → 模型选择 → 类型路由 → Step 估时 → 精细评估 → 聚合输出
    """

    # 计算类 Op 类型
    COMPUTE_OP_TYPES = {
        "matmul",
        "linear",
        "gemm",
        "conv",
        "softmax",
        "layernorm",
        "rmsnorm",
        "relu",
        "gelu",
        "silu",
        "add",
        "mul",
        "concat",
        "split",
        "embedding",
        "lmhead",
        "attention",
        "mha",
        "mla",
    }

    # 通信类 Op 类型
    COMM_OP_TYPES = {
        "allreduce",
        "allgather",
        "all2all",
        "p2p",
        "send",
        "recv",
        "reduce_scatter",
        "broadcast",
    }

    def __init__(self) -> None:
        """初始化评估引擎

        关键步骤:
            - 创建 CostModelRegistry 并注册 Chip/Core 级代价模型
            - 创建 OpTypeRouter 并注册评估器
        """
        # 初始化代价模型注册表
        self.cost_model_registry = CostModelRegistry()
        self._register_default_cost_models()

        # 初始化 OpType 路由器
        self.op_type_router = OpTypeRouter()
        self._register_default_evaluators()

    def _register_default_cost_models(self) -> None:
        """注册默认代价模型"""
        self.cost_model_registry.register(Granularity.CHIP, ChipCostModel())
        self.cost_model_registry.register(Granularity.CORE, CoreCostModel())
        # Lane 级暂不实现，回退到 Core 级

    def _register_default_evaluators(self) -> None:
        """注册默认评估器"""
        compute_eval = ComputeEvaluator()
        comm_eval = CommEvaluator()
        fallback_eval = FallbackEvaluator()

        # 注册计算类 Op
        for op_type in self.COMPUTE_OP_TYPES:
            self.op_type_router.register(op_type, Granularity.CHIP, compute_eval)
            self.op_type_router.register(op_type, Granularity.CORE, compute_eval)

        # 注册通信类 Op
        for op_type in self.COMM_OP_TYPES:
            self.op_type_router.register(op_type, Granularity.CHIP, comm_eval)
            self.op_type_router.register(op_type, Granularity.CORE, comm_eval)

        # 注册回退评估器
        self.op_type_router.register_fallback(fallback_eval)

    def evaluate(
        self,
        exec_plan: ExecPlan,
        distributed_model: DistributedModel,
        hardware: dict[str, float],
        granularity: Granularity = Granularity.CHIP,
        calibration: CalibrationConfig | None = None,
        output_tokens: int = 1,
        prefill_ops: set[str] | None = None,
    ) -> EngineResult:
        """执行评估

        输入:
            - exec_plan: 执行计划（包含 timeline/binding/precedence/buffer_plan）
            - distributed_model: 分布式模型（包含 Op 定义和 shape）
            - hardware: 硬件参数（如 compute_tflops, memory_bandwidth_gbps）
            - granularity: 评估精度（Chip/Core/Lane）
            - calibration: 校准参数（可选）
            - output_tokens: 输出 token 数（用于计算 TPOT/TPS）
            - prefill_ops: Prefill 阶段的 op_id 集合（用于计算 TTFT）
        输出:
            - EngineResult（StepMetrics + Aggregates + trace_meta）
        关键步骤:
            - 口径校验 → 模型选择 → 类型路由 → Step 估时 → 校准 → 聚合
        """
        # 1. 口径校验
        self._validate_inputs(exec_plan, hardware, granularity)

        # 2. 获取默认代价模型（作为 fallback）
        default_cost_model = self.cost_model_registry.get(granularity)
        if default_cost_model is None:
            # 回退到 Chip 级
            default_cost_model = self.cost_model_registry.get(Granularity.CHIP)
            if default_cost_model is None:
                raise ValueError(f"No cost model available for {granularity}")

        # 3. 遍历 timeline 进行评估
        step_metrics_list: list[StepMetrics] = []
        for event in exec_plan.timeline:
            op_id = event.get("op_id", "")
            op = distributed_model.get_op(op_id)
            if op is None:
                continue

            # 获取 Op 类型
            op_type = op.op_type.lower()

            # 4. 类型路由：选择评估器
            evaluator = self.op_type_router.resolve(op_type, granularity)
            if evaluator is None:
                evaluator = FallbackEvaluator()

            # 5. 模型选择：根据 op_type 选择代价模型，回退到默认模型
            cost_model = self.cost_model_registry.get(granularity, op_type)
            if cost_model is None:
                cost_model = default_cost_model

            # 6. 构建评估参数
            kernel_config = exec_plan.kernel_config.get(op_id, {}) if exec_plan else {}
            attrs = self._build_attrs(op, kernel_config)
            local_shape = op.local_shape.copy()

            # 7. Step 估时
            step_metrics = evaluator.evaluate(
                op_id=op_id,
                op_type=op_type,
                local_shape=local_shape,
                attrs=attrs,
                hardware=hardware,
                cost_model=cost_model,
            )

            # 8. 添加等待时间（从 timeline 获取）
            wait_time = event.get("wait_time", 0.0)
            step_metrics.t_wait = wait_time
            step_metrics.t_total = (
                step_metrics.t_compute + step_metrics.t_comm + step_metrics.t_wait
            )

            step_metrics_list.append(step_metrics)

        # 9. 应用校准（如果提供）
        if calibration is not None:
            calibrator = Calibration(calibration)
            step_metrics_list = calibrator.apply_batch(step_metrics_list)

        # 10. 聚合指标
        aggregates = self._aggregate_metrics(
            step_metrics_list,
            hardware,
            output_tokens,
            prefill_ops or set(),
        )

        # 11. 构建结果
        result = EngineResult(
            step_metrics=step_metrics_list,
            aggregates=aggregates,
            granularity=granularity,
            trace_meta={
                "num_ops": len(step_metrics_list),
                "hardware": hardware,
                "calibration": calibration.__dict__ if calibration else None,
            },
        )

        return result

    def _validate_inputs(
        self,
        exec_plan: ExecPlan,
        hardware: dict[str, float],
        granularity: Granularity,
    ) -> None:
        """校验输入

        输入:
            - exec_plan, hardware, granularity
        输出:
            - 无（校验失败抛出异常）
        关键步骤:
            - 检查 timeline 是否存在
            - 检查硬件参数是否完整
        """
        if not exec_plan.timeline:
            raise ValueError("ExecPlan.timeline is empty")

        # 获取所需字段
        required_fields = self.cost_model_registry.required_fields(granularity)

        # 检查硬件参数
        missing_fields = required_fields - set(hardware.keys())
        if missing_fields:
            raise ValueError(f"Missing hardware fields: {missing_fields}")

    def _build_attrs(self, op: Any, kernel_config: dict[str, Any] | None = None) -> dict[str, str]:
        """构建 Op 属性字典

        输入:
            - op: DistributedOp
            - kernel_config: Tile/Kernel 级元信息（可选）
        输出:
            - attrs 字典（用于评估器）
        关键步骤:
            - 通信 Op 需提取 comm_bytes/path_key/participants
            - 计算 Op 可注入 tile 级 traffic/lmem 信息
        """
        attrs = dict(op.attrs) if op.attrs else {}
        kernel_config = kernel_config or {}

        # 通信 Op 特殊处理
        if op.role == NodeRole.COMM:
            attrs["comm_bytes"] = str(op.comm_bytes)
            attrs["path_key"] = op.topology_path_key or "inter_board"
            attrs["participants"] = str(len(op.participants) if op.participants else 2)
            if op.comm_type:
                attrs["comm_type"] = op.comm_type.name.lower()
            if op.reason:
                attrs["reason"] = op.reason
        else:
            traffic = kernel_config.get("traffic")
            if traffic is not None:
                attrs["tile_traffic_bytes"] = str(traffic)
            lmem_bytes = kernel_config.get("lmem_bytes")
            if lmem_bytes is not None:
                attrs["tile_lmem_bytes"] = str(lmem_bytes)

        return attrs

    def _aggregate_metrics(
        self,
        step_metrics_list: list[StepMetrics],
        hardware: dict[str, float],
        output_tokens: int,
        prefill_ops: set[str],
    ) -> Aggregates:
        """聚合指标

        输入:
            - step_metrics_list: 所有 StepMetrics
            - hardware: 硬件参数
            - output_tokens: 输出 token 数
            - prefill_ops: Prefill 阶段 op_id 集合
        输出:
            - Aggregates
        关键步骤:
            - 累加 t_compute/t_comm/t_wait/flops/bytes
            - 计算 TTFT/TPOT/TPS/MFU/MBU
        """
        total_compute_time = 0.0
        total_comm_time = 0.0
        total_wait_time = 0.0
        total_flops = 0
        total_bytes = 0
        prefill_time = 0.0
        bottleneck_summary: dict[str, int] = {}
        seen_weight_ops: set[str] = set()
        total_weight_bytes = 0

        for step in step_metrics_list:
            total_compute_time += step.t_compute
            total_comm_time += step.t_comm
            total_wait_time += step.t_wait
            total_flops += step.flops
            total_bytes += step.bytes_read + step.bytes_write
            local_weight_bytes = step.meta.get("local_weight_bytes")
            if local_weight_bytes is not None and step.op_id not in seen_weight_ops:
                try:
                    total_weight_bytes += int(local_weight_bytes)
                    seen_weight_ops.add(step.op_id)
                except (TypeError, ValueError):
                    pass

            # 统计瓶颈类型
            tag = step.bottleneck_tag.name
            bottleneck_summary[tag] = bottleneck_summary.get(tag, 0) + 1

            # 累加 Prefill 时间
            if step.op_id in prefill_ops:
                prefill_time += step.t_total

        total_time = total_compute_time + total_comm_time + total_wait_time

        # 计算 TTFT（首 Token 延迟）
        ttft = prefill_time if prefill_time > 0 else total_time

        # 计算 TPOT（每 Token 时间）
        decode_time = total_time - prefill_time
        tpot = decode_time / output_tokens if output_tokens > 0 else 0

        # 计算 TPS（每秒 Token 数）
        tps = output_tokens / (decode_time / 1000) if decode_time > 0 else 0

        # 计算 MFU（Model FLOPS Utilization）
        compute_tflops = hardware.get("compute_tflops", 125.0)
        peak_flops = compute_tflops * 1e12  # FLOPS
        achieved_flops = (
            total_flops / (total_time / 1000) if total_time > 0 else 0
        )
        mfu = achieved_flops / peak_flops if peak_flops > 0 else 0

        # 计算 MBU（Memory Bandwidth Utilization）
        memory_bw_gbps = hardware.get("memory_bandwidth_gbps", 2000.0)
        peak_bw = memory_bw_gbps * 1e9  # Bytes/s
        achieved_bw = total_bytes / (total_time / 1000) if total_time > 0 else 0
        mbu = achieved_bw / peak_bw if peak_bw > 0 else 0

        # 内存占用口径：优先使用本地权重占用（近似 DS_TPU dram_occupy 语义）。
        memory_peak = total_weight_bytes if total_weight_bytes > 0 else total_bytes

        return Aggregates(
            ttft=ttft,
            tpot=tpot,
            tps=tps,
            mfu=mfu,
            mbu=mbu,
            memory_peak=memory_peak,
            total_time=total_time,
            total_compute_time=total_compute_time,
            total_comm_time=total_comm_time,
            total_wait_time=total_wait_time,
            total_flops=total_flops,
            total_bytes=total_bytes,
            num_steps=len(step_metrics_list),
            bottleneck_summary=bottleneck_summary,
        )


def create_default_hardware_spec(
    # HardwareSpec: 芯片级参数
    compute_tflops: float = 125.0,
    memory_bandwidth_gbps: float = 2000.0,
    num_cores: int = 64,
    sram_per_core_kb: float = 512.0,
    noc_bandwidth_gbps: float = 1000.0,
    # TopologySpec: chip 间通信参数
    intra_board_bw_gbps: float = 400.0,
    inter_board_bw_gbps: float = 200.0,
    inter_node_bw_gbps: float = 100.0,
    # TopologySpec: 通信硬件延迟参数（L2/L4 共享口径）
    c2c_lat_us: float = 0.15,
    ddr_r_lat_us: float = 0.15,
    ddr_w_lat_us: float = 0.01,
    noc_lat_us: float = 0.05,
    d2d_lat_us: float = 0.04,
    link_delay_us: float = 0.0,
    switch_delay_us: float = 0.25,
    cable_delay_us: float = 0.025,
    # CommProtocolSpec: 评估协议参数（L4 口径）
    rtt_tp_us: float = 0.35,
    rtt_ep_us: float = 0.85,
    sync_lat_us: float = 0.0,
    bw_utilization: float = 0.95,
    cpu_fetch_delay_us: float = 0.0,
    moe_topk: float = 8.0,
    prefill_topk_factor: float = 8 / 128,
) -> dict[str, float]:
    """创建默认硬件参数（HardwareSpec + TopologySpec）

    HardwareSpec（芯片级参数）:
        compute_tflops: 峰值算力（TFLOPS）
        memory_bandwidth_gbps: 显存带宽（GB/s）
        num_cores: Core 数量（Core 级需要）
        sram_per_core_kb: 每 Core SRAM 容量（KB，Core 级需要）
        noc_bandwidth_gbps: 片内 NoC 带宽（GB/s，Core 级需要）

    TopologySpec（chip 间通信参数）:
        intra_board_bw_gbps: 板内互联带宽（GB/s）
        inter_board_bw_gbps: 板间互联带宽（GB/s）
        inter_node_bw_gbps: 节点间互联带宽（GB/s）

    Returns:
        合并后的硬件参数字典
    """
    hw = HardwareSpec(
        compute_tflops=compute_tflops,
        memory_bandwidth_gbps=memory_bandwidth_gbps,
        num_cores=num_cores,
        sram_per_core_kb=sram_per_core_kb,
        noc_bandwidth_gbps=noc_bandwidth_gbps,
    )
    topo = TopologySpec(
        intra_board_bw_gbps=intra_board_bw_gbps,
        inter_board_bw_gbps=inter_board_bw_gbps,
        inter_node_bw_gbps=inter_node_bw_gbps,
        c2c_lat_us=c2c_lat_us,
        ddr_r_lat_us=ddr_r_lat_us,
        ddr_w_lat_us=ddr_w_lat_us,
        noc_lat_us=noc_lat_us,
        d2d_lat_us=d2d_lat_us,
        link_delay_us=link_delay_us,
        switch_delay_us=switch_delay_us,
        cable_delay_us=cable_delay_us,
    )
    comm = CommProtocolSpec(
        rtt_tp_us=rtt_tp_us,
        rtt_ep_us=rtt_ep_us,
        sync_lat_us=sync_lat_us,
        bw_utilization=bw_utilization,
        cpu_fetch_delay_us=cpu_fetch_delay_us,
        moe_topk=moe_topk,
        prefill_topk_factor=prefill_topk_factor,
    )
    return merge_specs(hw, topo, comm)
