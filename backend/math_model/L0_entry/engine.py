"""评估引擎模块

集成 L1-L5 层，执行完整的评估流程（对齐 CHIPMathica）。
支持从前端 EvaluationRequest 格式配置运行评估。
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from math_model.L0_entry.config_loader import (
    load_chip,
    load_model,
)
from math_model.L0_entry.eval_config import (
    EvalConfig,
    build_eval_config,
    _extract_first_chip_config,
    _require,
)

logger = logging.getLogger(__name__)


def run_evaluation(
    eval_config: EvalConfig,
    progress_callback: Callable[[float], None] | None = None,
) -> dict[str, Any]:
    """执行评估

    完整流程（对齐 CHIPMathica）:
    1. L1: 构建 WorkloadIR (DeepSeekV3Model/LlamaModel)
    2. L2: 加载 ChipSpec (SG2262Chip 等)
    3. L3: ParallelismPlanner -> DistributedModel
    4. L3: TilingPlanner -> TilePlan (使用 PreciseTileEvaluator)
    5. L3: Scheduler -> ExecPlan
    6. L4: EvaluationEngine -> StepMetrics[] + Aggregates
    7. L5: ReportingEngine -> 报告 (Gantt/成本/内存/Roofline/流量)

    Args:
        eval_config: 类型化评估配置
        progress_callback: 进度回调函数 (0.0 ~ 1.0)

    Returns:
        dict: 评估结果
    """
    def _report(p: float) -> None:
        if progress_callback:
            progress_callback(p)

    _report(0.02)

    # ==================== L1: 构建 WorkloadIR ====================
    from math_model.L1_workload.models.llm.deepseek import DeepSeekV3Model

    model = DeepSeekV3Model.from_model_config(eval_config.model)
    ir = model.to_ir()

    print(f"[L1] WorkloadIR created: {len(ir.get_layers())} layers")
    _report(0.05)

    # ==================== L2: 加载 ChipSpec ====================
    from math_model.L2_arch.chip import ChipSpecImpl

    chip_config = eval_config.chip_config
    chip_name = _require(chip_config, "name", "chip config")
    chip = ChipSpecImpl.from_config(chip_name, chip_config)
    print(f"[L2] ChipSpec loaded: {chip.name}")
    _report(0.08)

    # ==================== L3: Parallelism Planning ====================
    from math_model.L3_mapping.parallelism.planner import DeploymentSpec, ParallelismPlanner, BoardSpec

    dep = eval_config.deployment
    deployment = DeploymentSpec(
        tp=dep.tp,
        pp=dep.pp,
        ep=dep.ep,
        dp=dep.dp,
        moe_tp=dep.moe_tp,
        seq_len=dep.seq_len,
        batch_size=dep.batch_size,
        enable_tp_sp=dep.enable_tp_sp,
        enable_ring_attention=dep.enable_ring_attention,
        embed_tp=dep.embed_tp,
        lmhead_tp=dep.lmhead_tp,
        comm_protocol=dep.comm_protocol,
        kv_cache_rate=dep.kv_cache_rate,
        is_prefill=dep.is_prefill,
    )

    brd = eval_config.board
    board = BoardSpec(
        num_chips=brd.num_chips,
        chip_memory_gb=brd.chip_memory_gb,
        inter_chip_bw_gbps=brd.inter_chip_bw_gbps,
    )

    print("[L3] Running ParallelismPlanner...")
    planner = ParallelismPlanner(deployment, board)
    dist_model = planner.plan(ir)
    _report(0.12)

    # ==================== L3: Tiling Planning ====================
    from math_model.L3_mapping.tiling.planner import TilingPlanner
    from math_model.L4_evaluation.evaluators.precise import PreciseTileEvaluator

    print("[L3] Running TilingPlanner...")
    compute_tflops = chip.get_peak_flops("BF16", "cube") / 1e12
    memory_bw_gbps = chip.get_gmem_bandwidth()

    precise_evaluator = PreciseTileEvaluator(
        compute_tflops=compute_tflops,
        memory_bandwidth_gbps=memory_bw_gbps,
        is_prefill=dep.is_prefill,
        enable_zigzag=dep.enable_zigzag,
    )

    # TilingPlanner 进度映射到 12% -> 75%
    def _tiling_progress(current: int, total: int) -> None:
        if total > 0:
            _report(0.12 + (current / total) * 0.63)

    tile_plan = TilingPlanner(chip, l4_evaluator=precise_evaluator).plan(
        dist_model, progress_callback=_tiling_progress,
    )
    _report(0.75)

    # ==================== L3: Scheduling ====================
    from math_model.L3_mapping.scheduling.scheduler import Scheduler

    print("[L3] Running Scheduler...")
    exec_plan = Scheduler().plan(dist_model, tile_plan)
    _report(0.78)

    # ==================== L4: Evaluation ====================
    from math_model.L4_evaluation.engine import EvaluationEngine
    from math_model.L4_evaluation.metrics import Granularity

    print("[L4] Running EvaluationEngine...")

    # 构建硬件规格 -- 使用 eval_config 中的拓扑/通信参数（修复数据丢失 bug）
    hardware = _build_hardware_spec(chip, eval_config)

    # deployment_config dict 用于传递给 engine (ring attention 等)
    deployment_dict = {
        "enable_ring_attention": dep.enable_ring_attention,
        "tp": dep.tp,
    }

    engine = EvaluationEngine()
    engine_result = engine.evaluate(
        exec_plan=exec_plan,
        distributed_model=dist_model,
        hardware=hardware,
        granularity=Granularity.CHIP,
        output_tokens=dep.batch_size,
        deployment_config=deployment_dict,
    )
    _report(0.85)

    # ==================== L5: Reporting ====================
    print("[L5] Generating reports...")

    mc = eval_config.model
    run_config = {
        "model": {
            "name": mc.name,
            "batch": mc.batch,
            "q_seq_len": mc.q_seq_len,
            "kv_seq_len": mc.kv_seq_len,
        },
        "deployment": {
            "tp": dep.tp,
            "dp": dep.dp,
            "moe_tp": dep.moe_tp,
            "ep": dep.ep,
            "comm_protocol": dep.comm_protocol,
            "batch_size": dep.batch_size,
        },
        "board": {
            "num_chips": brd.num_chips,
            "inter_chip_bw_gbps": brd.inter_chip_bw_gbps,
        },
    }

    from dataclasses import asdict
    from math_model.L5_reporting.engine import ReportingEngine

    reporting_engine = ReportingEngine()
    report = reporting_engine.run(engine_result=engine_result, config=run_config)
    _report(0.95)

    result = {
        "aggregates": asdict(report.performance) if hasattr(report.performance, '__dataclass_fields__') else report.performance,
        "step_metrics": [asdict(s) if hasattr(s, '__dataclass_fields__') else s for s in report.step_metrics],
        "config": run_config,
        "schema_version": report.schema_version,
        "granularity": report.granularity,
    }

    perf = report.performance
    print(f"\n[Result Summary]")
    print(f"  Total time: {perf.total_time_ms:.2f} ms")
    print(f"  Compute time: {perf.compute_time_ms:.2f} ms")
    print(f"  Comm time: {perf.comm_time_ms:.2f} ms")
    print(f"  MFU: {perf.mfu:.4f}")
    print(f"  TPS: {perf.tps:.2f}")
    print(f"  TPS per chip: {perf.tps / brd.num_chips:.2f}")

    return result


def _build_hardware_spec(chip: Any, eval_config: EvalConfig) -> dict[str, Any]:
    """构建硬件规格字典（对齐 CHIPMathica）

    核心 Bug 修复: 使用 eval_config.topology 中的用户配置，
    不再使用 TopologySpec() 的默认值。
    """
    from math_model.L4_evaluation.metrics import CommProtocolSpec, HardwareSpec, merge_specs
    from math_model.L2_arch.topology import TopologySpec

    # 硬件规格（从 chip 对象获取）
    compute_tflops = chip.get_peak_flops("BF16", "cube") / 1e12
    memory_bw_gbps = chip.get_gmem_bandwidth()
    sram_per_core_kb = chip.get_total_sram() / max(1, chip.core_count) / 1024

    hardware_spec = HardwareSpec(
        compute_tflops=compute_tflops,
        memory_bandwidth_gbps=memory_bw_gbps,
        num_cores=chip.core_count,
        sram_per_core_kb=sram_per_core_kb,
        noc_bandwidth_gbps=chip.interconnect.noc_bandwidth_gbps,
    )

    # FIX: 使用用户配置的拓扑参数，不再用默认值！
    topo = eval_config.topology
    topology_spec = TopologySpec(
        c2c_bandwidth_gbps=topo.c2c_bandwidth_gbps,
        c2c_latency_us=topo.c2c_latency_us,
        b2b_bandwidth_gbps=topo.b2b_bandwidth_gbps,
        b2b_latency_us=topo.b2b_latency_us,
        r2r_bandwidth_gbps=topo.r2r_bandwidth_gbps,
        r2r_latency_us=topo.r2r_latency_us,
        p2p_bandwidth_gbps=topo.p2p_bandwidth_gbps,
        p2p_latency_us=topo.p2p_latency_us,
        switch_latency_us=topo.switch_latency_us,
        cable_latency_us=topo.cable_latency_us,
        memory_read_latency_us=topo.memory_read_latency_us,
        memory_write_latency_us=topo.memory_write_latency_us,
        noc_latency_us=topo.noc_latency_us,
        die_to_die_latency_us=topo.die_to_die_latency_us,
    )

    comm = eval_config.comm
    comm_spec = CommProtocolSpec(
        bw_utilization=comm.bw_utilization,
        sync_lat_us=comm.sync_lat_us,
    )

    hardware = merge_specs(hardware_spec, topology_spec, comm_spec)

    # compute_efficiency 从芯片配置读取
    chip_config = eval_config.chip_config
    if "compute_efficiency" not in chip_config:
        raise ValueError(
            f"Missing 'compute_efficiency' in chip config: {chip_config.get('name', 'unknown')}"
        )
    hardware["compute_efficiency"] = float(chip_config["compute_efficiency"])

    # compute_dma_overlap_rate 从芯片配置读取
    if "compute_dma_overlap_rate" not in chip_config:
        raise ValueError(
            f"Missing 'compute_dma_overlap_rate' in chip config: {chip_config.get('name', 'unknown')}"
        )
    hardware["compute_dma_overlap_rate"] = float(chip_config["compute_dma_overlap_rate"])

    return hardware


# ============================================
# 前端 EvaluationRequest 格式支持
# ============================================


def _calculate_memory_breakdown(
    eval_config: EvalConfig,
    aggregates: dict[str, Any],
    step_metrics: list[dict[str, Any]],
) -> dict[str, Any]:
    """计算内存分解

    混合方法：
    - 权重：使用运行时 local_weight_bytes（反映实际分布式模型）
    - KV Cache：理论计算（支持 MLA 压缩）
    - 激活值：理论计算（推理时占比小）
    - 开销：估算（15%）

    Args:
        eval_config: 类型化配置
        aggregates: 性能聚合（包含 memory_peak）
        step_metrics: 步骤指标（包含 local_weight_bytes）

    Returns:
        dict: 内存分解（匹配前端 MemoryAnalysis 接口）
    """
    import logging
    from math_model.L5_reporting.memory_analysis import MemoryAnalyzer

    logger = logging.getLogger(__name__)

    # 提取配置
    mc = eval_config.model
    dep = eval_config.deployment
    chip_memory_gb = eval_config.board.chip_memory_gb

    # 数据类型字节数映射
    dtype_map = {"fp32": 4, "fp16": 2, "bf16": 2, "fp8": 1, "int8": 1, "int4": 0.5}
    weight_dtype_bytes = dtype_map.get(mc.weight_dtype.lower(), 2)
    activation_dtype_bytes = dtype_map.get(mc.activation_dtype.lower(), 2)

    # 1. 权重：使用运行时数据（L4 已累加 local_weight_bytes）
    weight_bytes = aggregates.get("memory_peak", 0)

    # 回退：如果运行时数据缺失，使用理论估算
    if weight_bytes == 0:
        logger.warning("[WARN] No runtime weight data (memory_peak=0), using theoretical estimate")

        # 简化估算（不支持 MoE/MLA 细节，但优于 0）
        total_params = (
            mc.num_layers * (4 * mc.hidden_size ** 2 + 3 * mc.hidden_size * mc.intermediate_size)
            + mc.vocab_size * mc.hidden_size * 2
        )
        shard_factor = dep.tp * dep.pp
        weight_bytes = int((total_params / shard_factor) * weight_dtype_bytes) if shard_factor > 0 else 0

    # 2. KV Cache：理论计算
    analyzer = MemoryAnalyzer(dtype_bytes=weight_dtype_bytes)

    # 计算 head_dim
    head_dim = mc.hidden_size // mc.num_attention_heads if mc.num_attention_heads > 0 else mc.hidden_size

    # 检测 MLA 启用
    mla_enabled = mc.mla.kv_lora_rank > 0

    # num_kv_heads 处理（对于非 GQA 模型可能等于 num_attention_heads）
    # 目前 ModelConfig 中没有单独的 num_kv_heads 字段，使用 num_attention_heads
    num_kv_heads = mc.num_attention_heads
    if mla_enabled:
        logger.debug(f"[OK] MLA enabled: kv_lora_rank={mc.mla.kv_lora_rank}, qk_rope_dim={mc.mla.qk_rope_head_dim}")

    kv_cache_bytes = analyzer.calculate_kv_cache_memory(
        batch_size=dep.batch_size,
        seq_len=dep.kv_seq_len,
        num_layers=mc.num_layers,
        hidden_size=mc.hidden_size,
        num_kv_heads=num_kv_heads,
        num_heads=mc.num_attention_heads,
        tp_degree=dep.tp,
        pp_degree=dep.pp,
        mla_enabled=mla_enabled,
        kv_lora_rank=mc.mla.kv_lora_rank if mla_enabled else 0,
        qk_rope_dim=mc.mla.qk_rope_head_dim if mla_enabled else 0,
    )

    # 3. 激活值：理论计算
    activation_bytes = analyzer.calculate_activation_memory(
        batch_size=dep.batch_size,
        seq_len=max(dep.q_seq_len, 1),  # 使用 q_seq_len（decode 时为 1）
        hidden_size=mc.hidden_size,
        intermediate_size=mc.intermediate_size,
        num_layers=mc.num_layers,
        tp_degree=dep.tp,
        pp_degree=dep.pp,
    )

    # 4. 开销：估算
    overhead_bytes = analyzer.calculate_overhead(weight_bytes, kv_cache_bytes)

    # 5. 总计和利用率
    total_bytes = weight_bytes + kv_cache_bytes + activation_bytes + overhead_bytes
    total_gb = total_bytes / (1024 ** 3)

    is_sufficient = total_gb <= chip_memory_gb
    utilization = total_gb / chip_memory_gb if chip_memory_gb > 0 else 0.0

    # 返回格式：完全匹配前端 MemoryAnalysis 接口
    return {
        "model_memory_gb": weight_bytes / (1024 ** 3),
        "kv_cache_memory_gb": kv_cache_bytes / (1024 ** 3),
        "activation_memory_gb": activation_bytes / (1024 ** 3),
        "overhead_gb": overhead_bytes / (1024 ** 3),
        "total_per_chip_gb": total_gb,
        "is_memory_sufficient": is_sufficient,
        "memory_utilization": utilization,
    }


def run_evaluation_from_request(
    config: dict[str, Any],
    progress_callback: Callable[[float], None] | None = None,
) -> dict[str, Any]:
    """从前端 EvaluationRequest 格式运行评估

    Args:
        config: 前端格式的配置，包含:
            - experiment_name: 实验名称
            - benchmark_name: Benchmark 名称
            - topology_config_name: 拓扑配置名称
            - benchmark_config: { model: {...}, inference: {...} }
            - topology_config: 完整拓扑配置 (含 interconnect.comm_params)
            - manual_parallelism: 手动并行配置
            - search_mode: 搜索模式
        progress_callback: 进度回调函数 (0.0 ~ 1.0)

    Returns:
        dict: 评估结果
    """
    # 提取配置
    benchmark_config = _require(config, "benchmark_config", "evaluation request")
    topology_config = _require(config, "topology_config", "evaluation request")
    manual_parallelism = _require(config, "manual_parallelism", "evaluation request")

    model_config = _require(benchmark_config, "model", "benchmark_config")
    inference_config = _require(benchmark_config, "inference", "benchmark_config")

    # 从 topology_config.chips 提取芯片配置
    chips_dict = topology_config.get("chips")
    if not chips_dict:
        raise ValueError("Missing 'chips' in topology_config")
    chip_config = _extract_first_chip_config(chips_dict)

    # 单一转换点: dict -> EvalConfig
    eval_config = build_eval_config(
        chip_config=chip_config,
        model_config=model_config,
        topology_config=topology_config,
        manual_parallelism=manual_parallelism,
        inference_config=inference_config,
    )

    dep = eval_config.deployment
    logger.info(f"Running evaluation: {config.get('experiment_name', 'N/A')}")
    logger.info(f"  Model: {eval_config.model.name}")
    logger.info(f"  Chips: {eval_config.board.num_chips}")
    logger.info(f"  Parallelism: TP={dep.tp}, PP={dep.pp}, DP={dep.dp}, EP={dep.ep}")

    # 执行评估
    try:
        result = run_evaluation(eval_config, progress_callback=progress_callback)

        aggregates = _require(result, "aggregates", "evaluation result")
        step_metrics = _require(result, "step_metrics", "evaluation result")

        # 计算内存分解
        memory_breakdown = _calculate_memory_breakdown(eval_config, aggregates, step_metrics)

        # 计算成本分析
        cost_result = _calculate_deployment_cost(eval_config, aggregates)

        # 生成与前端兼容的 gantt_chart 和 stats
        from math_model.L0_entry.compat import convert_to_gantt_chart, convert_to_stats

        # 构建 deployment_config dict (前端兼容层需要)
        deployment_dict = {
            "tp": dep.tp, "pp": dep.pp, "dp": dep.dp, "ep": dep.ep,
            "moe_tp": dep.moe_tp, "seq_len": dep.seq_len,
            "batch_size": dep.batch_size, "enable_tp_sp": dep.enable_tp_sp,
            "embed_tp": dep.embed_tp, "lmhead_tp": dep.lmhead_tp,
            "comm_protocol": dep.comm_protocol, "kv_cache_rate": dep.kv_cache_rate,
            "is_prefill": dep.is_prefill, "enable_zigzag": dep.enable_zigzag,
            "enable_ring_attention": dep.enable_ring_attention,
            "q_seq_len": dep.q_seq_len, "kv_seq_len": dep.kv_seq_len,
        }

        gantt_chart = convert_to_gantt_chart(
            step_metrics=step_metrics,
            parallelism=deployment_dict,
            aggregates=aggregates,
            topology_config=topology_config,
        )
        stats = convert_to_stats(
            aggregates=aggregates,
            step_metrics=step_metrics,
            inference_config=inference_config,
            parallelism=deployment_dict,
            topology_config=topology_config,
        )

        # 构建与前端兼容的返回格式
        tps = _require(aggregates, "tps", "aggregates")
        tpot = aggregates.get("tpot_ms")
        if tpot is None:
            total_time_ms = _require(aggregates, "total_time_ms", "aggregates")
            output_seq_length = _require(inference_config, "output_seq_length", "inference config")
            tpot = total_time_ms / max(output_seq_length, 1)

        chips = dep.tp * dep.pp * dep.dp * dep.ep
        batch_size = dep.batch_size
        tps_per_chip = tps / chips if chips > 0 else 0
        tps_per_batch = tps / batch_size if batch_size > 0 else 0

        # 显存占用
        if "memory_peak_mb" in aggregates:
            memory_peak_mb = aggregates["memory_peak_mb"]
        elif "memory_peak" in aggregates:
            memory_peak_mb = aggregates["memory_peak"] / (1024 * 1024)
        else:
            raise ValueError("Missing 'memory_peak_mb' or 'memory_peak' in aggregates")
        dram_occupy = memory_peak_mb * 1024 * 1024  # MB -> bytes

        ttft = aggregates.get("ttft_ms")
        if ttft is None:
            ttft = _require(aggregates, "total_time_ms", "aggregates")

        mbu = aggregates.get("mbu")
        if mbu is None:
            mbu = aggregates.get("memory_utilization")
            if mbu is None:
                raise ValueError("Missing 'mbu' or 'memory_utilization' in aggregates")

        plan = {
            "parallelism": deployment_dict,
            "tps": tps,
            "ttft": ttft,
            "tpot": tpot,
            "mfu": _require(aggregates, "mfu", "aggregates"),
            "mbu": mbu,
            "score": tps,
            "is_feasible": True,
            "chips": chips,
            "tps_per_chip": tps_per_chip,
            "tps_per_batch": tps_per_batch,
            "dram_occupy": dram_occupy,
            "memory": memory_breakdown,
            "aggregates": aggregates,
            "step_metrics": step_metrics,
            "config": _require(result, "config", "evaluation result"),
            "cost": cost_result,
            "gantt_chart": gantt_chart,
            "stats": stats,
        }

        if progress_callback:
            progress_callback(0.98)
        return {
            "top_k_plans": [plan],
            "infeasible_plans": [],
            "search_stats": {
                "total_plans": 1,
                "feasible_plans": 1,
                "infeasible_plans": 0,
            },
            "config_snapshot": {
                "experiment_name": config.get("experiment_name"),
                "benchmark_name": config.get("benchmark_name"),
                "topology_config_name": config.get("topology_config_name"),
                "parallelism": deployment_dict,
                "inference": inference_config,
            },
        }

    except Exception as e:
        logger.exception("Evaluation failed")
        _locals = locals()
        parallelism_info = _locals.get('deployment_dict', manual_parallelism)
        inference_info = _locals.get('inference_config', {})
        return {
            "top_k_plans": [],
            "infeasible_plans": [{
                "parallelism": parallelism_info,
                "infeasible_reason": str(e),
                "is_feasible": False,
            }],
            "search_stats": {
                "total_plans": 1,
                "feasible_plans": 0,
                "infeasible_plans": 1,
            },
            "config_snapshot": {
                "experiment_name": config.get("experiment_name"),
                "benchmark_name": config.get("benchmark_name"),
                "topology_config_name": config.get("topology_config_name"),
                "parallelism": parallelism_info,
                "inference": inference_info,
            },
        }


def _calculate_deployment_cost(
    eval_config: EvalConfig,
    aggregates: dict[str, Any],
) -> dict[str, Any]:
    """计算部署成本

    Args:
        eval_config: 类型化配置
        aggregates: 性能聚合指标

    Returns:
        dict: 成本分析结果
    """
    from math_model.L5_reporting.cost_analysis import CostAnalyzer

    dep = eval_config.deployment
    mc = eval_config.model
    chips = dep.tp * dep.pp * dep.dp * dep.ep

    # 获取芯片类型
    chip_type = _require(eval_config.chip_config, "name", "chip config")

    # 估算模型大小
    hidden_size = mc.hidden_size
    num_layers = mc.num_layers
    vocab_size = mc.vocab_size
    num_experts = mc.moe.num_routed_experts

    embed_params = hidden_size * vocab_size
    attn_params = 4 * hidden_size * hidden_size * num_layers

    intermediate_size = mc.intermediate_size
    if num_experts > 0:
        num_dense_layers = mc.num_dense_layers
        num_moe_layers = mc.num_moe_layers
        moe_intermediate_size = mc.moe.intermediate_size
        ffn_params = (
            3 * hidden_size * intermediate_size * num_dense_layers
            + num_experts * 3 * hidden_size * moe_intermediate_size * num_moe_layers
        )
    else:
        ffn_params = 3 * hidden_size * intermediate_size * num_layers

    total_params = embed_params + attn_params + ffn_params
    bytes_per_param = 2  # BF16
    model_size_gb = total_params * bytes_per_param / 1e9

    tps = _require(aggregates, "tps", "aggregates")
    c2c_bandwidth_gbps = eval_config.topology.c2c_bandwidth_gbps

    analyzer = CostAnalyzer()
    cost_breakdown = analyzer.analyze(
        chip_type=chip_type,
        chip_count=chips,
        tps=tps,
        c2c_bandwidth_gbps=c2c_bandwidth_gbps,
    )

    return {
        "server_cost": cost_breakdown.server_cost,
        "rdma_cost": cost_breakdown.rdma_cost,
        "per_chip_cost": cost_breakdown.per_chip_cost,
        "interconnect_cost": cost_breakdown.interconnect_cost,
        "total_cost": cost_breakdown.total_cost,
        "cost_per_chip": cost_breakdown.cost_per_chip,
        "cost_per_million_tokens": cost_breakdown.cost_per_million_tokens,
        "dfop": cost_breakdown.dfop,
        "model_size_gb": model_size_gb,
        "chips": chips,
    }
