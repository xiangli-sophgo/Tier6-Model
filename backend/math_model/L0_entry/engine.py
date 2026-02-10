"""评估引擎模块

集成 L1-L5 层，执行完整的评估流程（对齐 CHIPMathica）。
支持从前端 EvaluationRequest 格式配置运行评估。
"""

from __future__ import annotations

import logging
from typing import Any

from math_model.L0_entry.config_loader import (
    load_chip,
    load_model,
)

logger = logging.getLogger(__name__)


def run_evaluation(config: dict[str, Any]) -> dict[str, Any]:
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
        config: 评估配置，支持两种格式:
            - 方式1: scenario 引用（推荐）
                {"scenario": "deepseek_v3_sg2262"}
            - 方式2: 显式配置
                {
                    "chip": "sg2262" | chip_config_dict,
                    "model": "deepseek_v3" | model_config_dict,
                    "deployment": {...},
                    "board": {...},
                    "inference": {...}
                }

    Returns:
        dict: 评估结果
    """
    # 解析配置
    chip_config = _load_chip_config(config)
    model_config = _load_model_config(config)
    deployment_config = _get_required(config, "deployment", "evaluation config")
    board_config = _get_required(config, "board", "evaluation config")
    inference_config = _get_required(config, "inference", "evaluation config")

    # ==================== L1: 构建 WorkloadIR ====================
    from math_model.L1_workload.models.llm.deepseek import DeepSeekV3Model

    # 映射模型配置为 DeepSeekV3Model 需要的格式
    model_params = _map_model_config(model_config, deployment_config, inference_config)
    model = DeepSeekV3Model(model_params)
    ir = model.to_ir()

    print(f"[L1] WorkloadIR created: {len(ir.get_layers())} layers")

    # ==================== L2: 加载 ChipSpec ====================
    from math_model.L2_arch.chip import ChipSpecImpl

    chip = ChipSpecImpl.from_config(_get_required(chip_config, "name", "chip config"), chip_config)
    print(f"[L2] ChipSpec loaded: {chip.name}")

    # ==================== L3: Parallelism Planning ====================
    from math_model.L3_mapping.parallelism.planner import DeploymentSpec, ParallelismPlanner, BoardSpec

    deployment = DeploymentSpec(
        tp=int(_get_required(deployment_config, "tp", "deployment config")),
        pp=int(_get_required(deployment_config, "pp", "deployment config")),
        ep=int(_get_required(deployment_config, "ep", "deployment config")),
        dp=int(_get_required(deployment_config, "dp", "deployment config")),
        moe_tp=int(_get_required(deployment_config, "moe_tp", "deployment config")),
        seq_len=int(_get_required(deployment_config, "seq_len", "deployment config")),
        batch_size=int(_get_required(deployment_config, "batch_size", "deployment config")),
        enable_tp_sp=bool(_get_required(deployment_config, "enable_tp_sp", "deployment config")),
        enable_ring_attention=bool(_get_required(deployment_config, "enable_ring_attention", "deployment config")),
        embed_tp=int(_get_required(deployment_config, "embed_tp", "deployment config")),
        lmhead_tp=int(_get_required(deployment_config, "lmhead_tp", "deployment config")),
        comm_protocol=int(_get_required(deployment_config, "comm_protocol", "deployment config")),
        kv_cache_rate=float(_get_required(deployment_config, "kv_cache_rate", "deployment config")),
        is_prefill=bool(_get_required(deployment_config, "is_prefill", "deployment config")),
    )

    board = BoardSpec(
        num_chips=int(_get_required(board_config, "num_chips", "board config")),
        chip_memory_gb=int(_get_required(board_config, "chip_memory_gb", "board config")),
        inter_chip_bw_gbps=float(_get_required(board_config, "inter_chip_bw_gbps", "board config")),
    )

    print("[L3] Running ParallelismPlanner...")
    planner = ParallelismPlanner(deployment, board)
    dist_model = planner.plan(ir)

    # ==================== L3: Tiling Planning ====================
    from math_model.L3_mapping.tiling.planner import TilingPlanner
    from math_model.L4_evaluation.evaluators.precise import PreciseTileEvaluator

    print("[L3] Running TilingPlanner...")
    compute_tflops = chip.get_peak_flops("BF16", "cube") / 1e12
    memory_bw_gbps = chip.get_gmem_bandwidth()

    precise_evaluator = PreciseTileEvaluator(
        compute_tflops=compute_tflops,
        memory_bandwidth_gbps=memory_bw_gbps,
        is_prefill=deployment.is_prefill,
        enable_zigzag=bool(deployment_config.get("enable_zigzag", False)),
    )

    tile_plan = TilingPlanner(chip, l4_evaluator=precise_evaluator).plan(dist_model)

    # ==================== L3: Scheduling ====================
    from math_model.L3_mapping.scheduling.scheduler import Scheduler

    print("[L3] Running Scheduler...")
    exec_plan = Scheduler().plan(dist_model, tile_plan)

    # ==================== L4: Evaluation ====================
    from math_model.L4_evaluation.engine import EvaluationEngine
    from math_model.L4_evaluation.metrics import Granularity

    print("[L4] Running EvaluationEngine...")

    # 构建硬件规格（对齐 CHIPMathica）
    hardware = _build_hardware_spec(chip, chip_config)

    engine = EvaluationEngine()
    engine_result = engine.evaluate(
        exec_plan=exec_plan,
        distributed_model=dist_model,
        hardware=hardware,
        granularity=Granularity.CHIP,
        output_tokens=deployment.batch_size,
    )

    # ==================== L5: Reporting ====================
    print("[L5] Generating reports...")

    # 构建运行配置（用于报告）
    run_config = {
        "model": {
            "name": _get_required(model_config, "name", "model config"),
            "batch": model_params.get("batch"),
            "q_seq_len": model_params.get("q_seq_len"),
            "kv_seq_len": model_params.get("kv_seq_len"),
        },
        "deployment": {
            "tp": deployment.tp,
            "dp": deployment.dp,
            "moe_tp": deployment.moe_tp,
            "ep": deployment.ep,
            "comm_protocol": deployment.comm_protocol,
            "batch_size": deployment.batch_size,
        },
        "board": {
            "num_chips": board.num_chips,
            "inter_chip_bw_gbps": board.inter_chip_bw_gbps,
        },
    }

    # 生成 L5 报告
    from dataclasses import asdict
    from math_model.L5_reporting.engine import ReportingEngine

    reporting_engine = ReportingEngine()
    report = reporting_engine.run(engine_result=engine_result, config=run_config)

    # 转换为字典 (使用 asdict 处理 dataclass)
    result = {
        "aggregates": asdict(report.performance) if hasattr(report.performance, '__dataclass_fields__') else report.performance,
        "step_metrics": [asdict(s) if hasattr(s, '__dataclass_fields__') else s for s in report.step_metrics],
        "config": run_config,
        "schema_version": report.schema_version,
        "granularity": report.granularity,
    }

    # 打印摘要
    perf = report.performance
    print(f"\n[Result Summary]")
    print(f"  Total time: {perf.total_time_ms:.2f} ms")
    print(f"  Compute time: {perf.compute_time_ms:.2f} ms")
    print(f"  Comm time: {perf.comm_time_ms:.2f} ms")
    print(f"  MFU: {perf.mfu:.4f}")
    print(f"  TPS: {perf.tps:.2f}")
    print(f"  TPS per chip: {perf.tps / board.num_chips:.2f}")

    return result


def _load_chip_config(config: dict[str, Any]) -> dict[str, Any]:
    """加载芯片配置"""
    if "chip" in config:
        chip_ref = config["chip"]
        if isinstance(chip_ref, str):
            # 引用芯片预设
            chip_data = load_chip(chip_ref)
            return chip_data.get("chip", chip_data)
        else:
            # 内联配置
            return chip_ref
    elif "chip_config" in config:
        return config["chip_config"]
    elif "chip_preset" in config:
        return load_chip(config["chip_preset"])
    else:
        raise ValueError("Missing chip configuration")


def _load_model_config(config: dict[str, Any]) -> dict[str, Any]:
    """加载模型配置"""
    if "model" in config:
        model_ref = config["model"]
        if isinstance(model_ref, str):
            # 引用模型预设
            return load_model(model_ref)
        else:
            # 内联配置
            return model_ref
    elif "model_config" in config:
        return config["model_config"]
    elif "model_preset" in config:
        return load_model(config["model_preset"])
    else:
        raise ValueError("Missing model configuration")


def _get_required(cfg: dict[str, Any], key: str, source: str) -> Any:
    """从配置中获取必需字段，缺失时报错"""
    if key not in cfg:
        raise ValueError(f"Missing required field '{key}' in {source}")
    return cfg[key]


def _map_model_config(
    model_cfg: dict[str, Any],
    deployment_cfg: dict[str, Any],
    inference_cfg: dict[str, Any],
) -> dict[str, Any]:
    """映射模型配置为 DeepSeekV3Model 需要的扁平结构

    所有模型必需字段缺失时报错，不使用默认值。
    """
    # 必需的模型字段（对齐 configs/models/DeepSeek-v3.yaml 字段名）
    hidden_size = int(_get_required(model_cfg, "hidden_size", "model config"))
    num_layers = int(_get_required(model_cfg, "num_layers", "model config"))
    num_heads = int(_get_required(model_cfg, "num_attention_heads", "model config"))
    vocab_size = int(_get_required(model_cfg, "vocab_size", "model config"))

    # dense/moe 层数
    num_dense_layers = int(_get_required(model_cfg, "num_dense_layers", "model config"))
    num_moe_layers = int(_get_required(model_cfg, "num_moe_layers", "model config"))

    # FFN 中间层大小（顶层字段，不是嵌套在 ffn 下）
    intermediate_size = int(_get_required(model_cfg, "intermediate_size", "model config"))

    # MLA 配置 (必需)
    mla = model_cfg.get("MLA")
    if mla is None:
        raise ValueError("Missing 'MLA' section in model config")
    q_lora_rank = int(_get_required(mla, "q_lora_rank", "model.MLA"))
    kv_lora_rank = int(_get_required(mla, "kv_lora_rank", "model.MLA"))
    qk_nope_head_dim = int(_get_required(mla, "qk_nope_head_dim", "model.MLA"))
    qk_rope_head_dim = int(_get_required(mla, "qk_rope_head_dim", "model.MLA"))
    v_head_dim = int(_get_required(mla, "v_head_dim", "model.MLA"))

    # MoE 配置 (必需)
    moe = model_cfg.get("MoE")
    if moe is None:
        raise ValueError("Missing 'MoE' section in model config")
    n_routed_experts = int(_get_required(moe, "num_routed_experts", "model.MoE"))
    n_shared_experts = int(_get_required(moe, "num_shared_experts", "model.MoE"))
    n_activated_experts = int(_get_required(moe, "num_activated_experts", "model.MoE"))
    moe_intermediate_size = int(_get_required(moe, "intermediate_size", "model.MoE"))

    # 精度配置 - 优先从 inference_config 获取，其次从 deployment_config
    if "weight_dtype" in inference_cfg:
        weight_dtype = inference_cfg["weight_dtype"]
    elif "weight_dtype" in deployment_cfg:
        weight_dtype = deployment_cfg["weight_dtype"]
    else:
        raise ValueError("Missing 'weight_dtype' in inference config or deployment config")

    if "activation_dtype" in inference_cfg:
        activation_dtype = inference_cfg["activation_dtype"]
    elif "activation_dtype" in deployment_cfg:
        activation_dtype = deployment_cfg["activation_dtype"]
    else:
        raise ValueError("Missing 'activation_dtype' in inference config or deployment config")

    # 运行时参数 - 优先从 deployment，其次从 inference
    seq_len = int(_get_required(deployment_cfg, "seq_len", "deployment config"))
    kv_seq_len = int(_get_required(deployment_cfg, "kv_seq_len", "deployment config"))
    q_seq_len = int(_get_required(deployment_cfg, "q_seq_len", "deployment config"))
    batch = int(_get_required(deployment_cfg, "batch_size", "deployment config"))
    is_prefill = bool(_get_required(deployment_cfg, "is_prefill", "deployment config"))

    return {
        "weight_dtype": weight_dtype,
        "activation_dtype": activation_dtype,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "num_dense_layers": num_dense_layers,
        "num_moe_layers": num_moe_layers,
        "num_heads": num_heads,
        "vocab_size": vocab_size,
        "q_lora_rank": q_lora_rank,
        "kv_lora_rank": kv_lora_rank,
        "qk_nope_head_dim": qk_nope_head_dim,
        "qk_rope_head_dim": qk_rope_head_dim,
        "v_head_dim": v_head_dim,
        "n_routed_experts": n_routed_experts,
        "n_shared_experts": n_shared_experts,
        "n_activated_experts": n_activated_experts,
        "moe_intermediate_size": moe_intermediate_size,
        "intermediate_size": intermediate_size,
        "seq_len": seq_len,
        "kv_seq_len": kv_seq_len,
        "q_seq_len": q_seq_len,
        "batch": batch,
        "is_prefill": is_prefill,
    }


def _build_hardware_spec(chip: Any, chip_config: dict[str, Any]) -> dict[str, Any]:
    """构建硬件规格字典（对齐 CHIPMathica）"""
    from math_model.L4_evaluation.metrics import CommProtocolSpec, HardwareSpec
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

    # 拓扑规格（使用默认值初始化，后续可从拓扑配置覆盖）
    topology_spec = TopologySpec()

    # 通信协议规格
    comm_spec = CommProtocolSpec()

    # 合并规格
    from math_model.L4_evaluation import merge_specs

    hardware = merge_specs(hardware_spec, topology_spec, comm_spec)

    # compute_efficiency 从芯片配置读取
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


def run_evaluation_from_request(config: dict[str, Any]) -> dict[str, Any]:
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

    Returns:
        dict: 评估结果
    """
    # WebSocket 广播由 TaskManager 处理，此处不需要直接调用

    # 提取配置
    benchmark_config = _get_required(config, "benchmark_config", "evaluation request")
    topology_config = _get_required(config, "topology_config", "evaluation request")
    manual_parallelism = _get_required(config, "manual_parallelism", "evaluation request")

    # 从 benchmark_config 提取模型和推理配置
    model_config = _get_required(benchmark_config, "model", "benchmark_config")
    inference_config = _get_required(benchmark_config, "inference", "benchmark_config")

    # 从 topology_config.chips 提取芯片配置
    chips_dict = topology_config.get("chips")
    if not chips_dict:
        raise ValueError("Missing 'chips' in topology_config")
    chip_config = _extract_first_chip_config(chips_dict)

    # 提取通信延迟配置 (interconnect.comm_params)
    ic = topology_config.get("interconnect", {})
    comm_latency_config = ic.get("comm_params")
    if not comm_latency_config:
        raise ValueError("Missing 'interconnect.comm_params' in topology_config")

    # 从 topology_config 提取 BoardSpec
    board_spec = _extract_board_spec(topology_config)

    # 映射通信延迟配置到 tier6 格式
    topology_overrides, comm_overrides = _map_comm_latency_config(comm_latency_config)

    # 构建部署配置
    deployment_config = _build_deployment_config(manual_parallelism, inference_config)

    # 构建内部评估配置
    internal_config = {
        "chip_config": chip_config,
        "model_config": model_config,
        "deployment": deployment_config,
        "board": {
            "num_chips": board_spec["num_chips"],
            "chip_memory_gb": board_spec["chip_memory_gb"],
            "inter_chip_bw_gbps": board_spec["inter_chip_bw_gbps"],
        },
        "inference": inference_config,
        "topology_overrides": topology_overrides,
        "comm_overrides": comm_overrides,
    }

    logger.info(f"Running evaluation: {config.get('experiment_name', 'N/A')}")
    logger.info(f"  Model: {model_config.get('name', 'N/A')}")
    logger.info(f"  Chips: {board_spec['num_chips']}")
    logger.info(f"  Parallelism: TP={deployment_config['tp']}, PP={deployment_config['pp']}, "
                f"DP={deployment_config['dp']}, EP={deployment_config['ep']}")

    # 执行评估
    try:
        result = run_evaluation(internal_config)

        # 提取性能指标
        aggregates = _get_required(result, "aggregates", "evaluation result")
        step_metrics = _get_required(result, "step_metrics", "evaluation result")

        # 计算成本分析
        cost_result = _calculate_deployment_cost(
            deployment_config=deployment_config,
            model_config=model_config,
            aggregates=aggregates,
        )

        # 生成与前端兼容的 gantt_chart 和 stats
        from math_model.L0_entry.compat import convert_to_gantt_chart, convert_to_stats

        gantt_chart = convert_to_gantt_chart(
            step_metrics=step_metrics,
            parallelism=deployment_config,
            aggregates=aggregates,
            topology_config=topology_config,
        )
        stats = convert_to_stats(
            aggregates=aggregates,
            step_metrics=step_metrics,
            inference_config=inference_config,
            parallelism=deployment_config,
            topology_config=topology_config,
        )

        # 构建与前端兼容的返回格式
        tps = _get_required(aggregates, "tps", "aggregates")
        tpot = aggregates.get("tpot_ms")
        if tpot is None:
            total_time_ms = _get_required(aggregates, "total_time_ms", "aggregates")
            output_seq_length = _get_required(inference_config, "output_seq_length", "inference config")
            tpot = total_time_ms / max(output_seq_length, 1)

        # 计算芯片数和每芯片/每批次 TPS
        tp = deployment_config["tp"]
        pp = deployment_config["pp"]
        dp = deployment_config["dp"]
        ep = deployment_config["ep"]
        chips = tp * pp * dp * ep
        batch_size = deployment_config["batch_size"]
        tps_per_chip = tps / chips if chips > 0 else 0
        tps_per_batch = tps / batch_size if batch_size > 0 else 0

        # 显存占用（从 aggregates 获取，单位转换为 bytes）
        if "memory_peak_mb" in aggregates:
            memory_peak_mb = aggregates["memory_peak_mb"]
        elif "memory_peak" in aggregates:
            memory_peak_mb = aggregates["memory_peak"] / (1024 * 1024)
        else:
            raise ValueError("Missing 'memory_peak_mb' or 'memory_peak' in aggregates")
        dram_occupy = memory_peak_mb * 1024 * 1024  # MB -> bytes

        # ttft: 优先 ttft_ms，否则用 total_time_ms
        ttft = aggregates.get("ttft_ms")
        if ttft is None:
            ttft = _get_required(aggregates, "total_time_ms", "aggregates")

        # mbu: 优先 mbu，否则 memory_utilization
        mbu = aggregates.get("mbu")
        if mbu is None:
            mbu = aggregates.get("memory_utilization")
            if mbu is None:
                raise ValueError("Missing 'mbu' or 'memory_utilization' in aggregates")

        plan = {
            "parallelism": deployment_config,
            "tps": tps,
            "ttft": ttft,
            "tpot": tpot,
            "mfu": _get_required(aggregates, "mfu", "aggregates"),
            "mbu": mbu,
            "score": tps,  # 使用 TPS 作为 score
            "is_feasible": True,
            # 前端期望的额外字段
            "chips": chips,
            "tps_per_chip": tps_per_chip,
            "tps_per_batch": tps_per_batch,
            "dram_occupy": dram_occupy,
            # 保留原始结果
            "aggregates": aggregates,
            "step_metrics": step_metrics,
            "config": _get_required(result, "config", "evaluation result"),
            # 成本分析
            "cost": cost_result,
            # 前端可视化兼容数据
            "gantt_chart": gantt_chart,
            "stats": stats,
        }

        # 返回与 llm_simulator 兼容的格式
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
                "parallelism": deployment_config,
                "inference": inference_config,
            },
        }

    except Exception as e:
        logger.exception("Evaluation failed")
        # 返回失败结果
        return {
            "top_k_plans": [],
            "infeasible_plans": [{
                "parallelism": deployment_config,
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
                "parallelism": deployment_config,
                "inference": inference_config,
            },
        }


def _extract_first_chip_config(chips_dict: dict[str, Any]) -> dict[str, Any]:
    """从芯片字典中提取第一个芯片配置

    Args:
        chips_dict: 芯片配置字典 { chip_name: chip_config, ... }

    Returns:
        dict: 芯片配置
    """
    if not chips_dict:
        raise ValueError("No chip configuration found in topology_config.chips")

    # 获取第一个芯片
    first_chip_name = next(iter(chips_dict))
    first_chip = chips_dict[first_chip_name]

    # 确保 name 字段存在
    if "name" not in first_chip:
        first_chip["name"] = first_chip_name

    return first_chip


def _extract_board_spec(topology_config: dict[str, Any]) -> dict[str, Any]:
    """从拓扑配置提取板卡规格

    Args:
        topology_config: 完整拓扑配置 (grouped_pods 格式, 含 chips + interconnect)

    Returns:
        dict: 板卡规格 { num_chips, chip_memory_gb, inter_chip_bw_gbps }
    """
    from .topology_format import count_chips

    # 计算芯片总数 - 从 pods 结构
    num_chips = count_chips(topology_config)

    # 获取互联带宽 - 从 interconnect.links
    ic = topology_config.get("interconnect", {})
    links = ic.get("links", {})
    if not links:
        raise ValueError("Missing 'interconnect.links' in topology_config")
    c2c = _get_required(links, "c2c", "topology config interconnect.links")
    inter_chip_bw = _get_required(c2c, "bandwidth_gbps", "topology config interconnect.links.c2c")

    # 获取芯片内存 - 从 chips
    chips = topology_config.get("chips")
    if not chips:
        raise ValueError("Missing 'chips' in topology_config")

    first_chip = next(iter(chips.values()))
    memory = _get_required(first_chip, "memory", "chip config")
    gmem = _get_required(memory, "gmem", "chip config memory")
    chip_memory = _get_required(gmem, "capacity_gb", "chip config memory.gmem")

    return {
        "num_chips": num_chips,
        "chip_memory_gb": chip_memory,
        "inter_chip_bw_gbps": inter_chip_bw,
    }


def _count_topology_chips(topology: dict[str, Any]) -> int:
    """计算拓扑中的芯片总数

    Args:
        topology: 拓扑配置 (grouped_pods 格式)

    Returns:
        int: 芯片总数
    """
    from .topology_format import count_chips
    return count_chips(topology)


def _map_comm_latency_config(frontend_config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """将前端 comm_latency_config 映射到 tier6 TopologySpec 和 CommProtocolSpec

    Args:
        frontend_config: 前端通信延迟配置

    Returns:
        tuple: (topology_overrides, comm_overrides)
    """
    topology_overrides: dict[str, Any] = {}
    comm_overrides: dict[str, Any] = {}

    # TopologySpec 字段映射
    if "switch_delay_us" in frontend_config:
        topology_overrides["switch_delay_us"] = frontend_config["switch_delay_us"]
    if "cable_delay_us" in frontend_config:
        topology_overrides["cable_delay_us"] = frontend_config["cable_delay_us"]
    if "memory_read_latency_us" in frontend_config:
        topology_overrides["ddr_r_lat_us"] = frontend_config["memory_read_latency_us"]
    if "memory_write_latency_us" in frontend_config:
        topology_overrides["ddr_w_lat_us"] = frontend_config["memory_write_latency_us"]
    if "noc_latency_us" in frontend_config:
        topology_overrides["noc_lat_us"] = frontend_config["noc_latency_us"]
    if "die_to_die_latency_us" in frontend_config:
        topology_overrides["d2d_lat_us"] = frontend_config["die_to_die_latency_us"]
    if "c2c_latency_us" in frontend_config:
        topology_overrides["c2c_lat_us"] = frontend_config["c2c_latency_us"]

    # 带宽映射
    if "intra_board_bw_gbps" in frontend_config:
        topology_overrides["intra_board_bw_gbps"] = frontend_config["intra_board_bw_gbps"]
    if "inter_board_bw_gbps" in frontend_config:
        topology_overrides["inter_board_bw_gbps"] = frontend_config["inter_board_bw_gbps"]
    if "inter_node_bw_gbps" in frontend_config:
        topology_overrides["inter_node_bw_gbps"] = frontend_config["inter_node_bw_gbps"]

    # CommProtocolSpec 字段映射
    if "rtt_tp_us" in frontend_config:
        comm_overrides["rtt_tp_us"] = frontend_config["rtt_tp_us"]
    if "rtt_ep_us" in frontend_config:
        comm_overrides["rtt_ep_us"] = frontend_config["rtt_ep_us"]
    if "bandwidth_utilization" in frontend_config:
        comm_overrides["bw_utilization"] = frontend_config["bandwidth_utilization"]
    if "sync_latency_us" in frontend_config:
        comm_overrides["sync_lat_us"] = frontend_config["sync_latency_us"]

    return topology_overrides, comm_overrides


def _calculate_deployment_cost(
    deployment_config: dict[str, Any],
    model_config: dict[str, Any],
    aggregates: dict[str, Any],
) -> dict[str, Any]:
    """计算部署成本

    Args:
        deployment_config: 部署配置（含并行策略）
        model_config: 模型配置
        aggregates: 性能聚合指标

    Returns:
        dict: 成本分析结果
    """
    from math_model.L5_reporting.cost_analysis import CostAnalyzer

    # 计算芯片数
    tp = _get_required(deployment_config, "tp", "deployment config")
    pp = _get_required(deployment_config, "pp", "deployment config")
    dp = _get_required(deployment_config, "dp", "deployment config")
    ep = _get_required(deployment_config, "ep", "deployment config")
    chips = tp * pp * dp * ep

    # 获取芯片类型（从 chip_config 或 deployment_config，而非 model_config）
    chip_config = config.get("chip", {})
    if isinstance(chip_config, str):
        chip_type = chip_config  # 引用预设名称
    elif "name" in chip_config:
        chip_type = chip_config["name"]
    elif "chip_type" in deployment_config:
        chip_type = deployment_config["chip_type"]
    else:
        raise ValueError("Cannot determine chip_type from config (check chip or deployment sections)")

    # 估算模型大小（粗略）
    hidden_size = _get_required(model_config, "hidden_size", "model config")
    num_layers = _get_required(model_config, "num_layers", "model config")
    vocab_size = _get_required(model_config, "vocab_size", "model config")
    moe = model_config.get("MoE", model_config.get("moe"))
    if moe is None:
        raise ValueError("Missing 'MoE' section in model config")
    num_experts = _get_required(moe, "num_routed_experts", "model config MoE")

    # 简化的模型参数估算
    # Embedding + Attention + FFN/MoE + LMHead
    embed_params = hidden_size * vocab_size
    attn_params = 4 * hidden_size * hidden_size * num_layers

    # FFN/MoE 参数（使用配置而非硬编码）
    intermediate_size = model_config.get("intermediate_size", hidden_size * 4)
    if num_experts > 0:
        # MoE: 从配置获取 dense/moe 层数拆分
        num_dense_layers = model_config.get("num_dense_layers", 0)
        num_moe_layers = model_config.get("num_moe_layers", num_layers)
        moe_intermediate_size = moe.get("intermediate_size", intermediate_size // 4)
        ffn_params = (
            3 * hidden_size * intermediate_size * num_dense_layers
            + num_experts * 3 * hidden_size * moe_intermediate_size * num_moe_layers
        )
    else:
        ffn_params = 3 * hidden_size * intermediate_size * num_layers

    total_params = embed_params + attn_params + ffn_params
    bytes_per_param = 2  # BF16
    model_size_gb = total_params * bytes_per_param / 1e9

    # 获取性能指标
    tps = _get_required(aggregates, "tps", "aggregates")
    tpot = aggregates.get("tpot_ms")
    if tpot is None:
        tpot = _get_required(aggregates, "total_time_ms", "aggregates")

    # 从模型参数估算互联带宽需求 -> lanes_per_chip
    effective_tpot = tpot if tpot > 0 else 1.0
    bandwidth_gbps = 2 * model_size_gb * tp / effective_tpot * 1000
    lanes_per_chip = max(1, int(bandwidth_gbps / 112 * 8))  # 8 modules/server

    # 使用 L5 CostAnalyzer 计算成本
    analyzer = CostAnalyzer()
    cost_breakdown = analyzer.analyze(
        chip_type=chip_type,
        chip_count=chips,
        tps=tps,
        lanes_per_chip=lanes_per_chip,
    )

    return {
        "server_cost": cost_breakdown.server_cost,
        "interconnect_cost": cost_breakdown.interconnect_cost,
        "total_cost": cost_breakdown.total_cost,
        "cost_per_chip": cost_breakdown.cost_per_chip,
        "cost_per_million_tokens": cost_breakdown.cost_per_million_tokens,
        "model_size_gb": model_size_gb,
        "chips": chips,
    }


def _build_deployment_config(
    manual_parallelism: dict[str, Any],
    inference_config: dict[str, Any],
) -> dict[str, Any]:
    """构建部署配置

    Args:
        manual_parallelism: 手动并行配置
        inference_config: 推理配置

    Returns:
        dict: 部署配置
    """
    # 从 manual_parallelism 获取并行参数
    tp = _get_required(manual_parallelism, "tp", "manual_parallelism")
    pp = _get_required(manual_parallelism, "pp", "manual_parallelism")
    dp = _get_required(manual_parallelism, "dp", "manual_parallelism")
    ep = _get_required(manual_parallelism, "ep", "manual_parallelism")
    moe_tp = _get_required(manual_parallelism, "moe_tp", "manual_parallelism")

    # 从 inference_config 获取推理参数
    batch_size = _get_required(inference_config, "batch_size", "inference config")
    input_seq_length = _get_required(inference_config, "input_seq_length", "inference config")
    output_seq_length = _get_required(inference_config, "output_seq_length", "inference config")

    return {
        "tp": tp,
        "pp": pp,
        "dp": dp,
        "ep": ep,
        "moe_tp": moe_tp,
        "seq_len": _get_required(manual_parallelism, "seq_len", "manual_parallelism"),
        "batch_size": batch_size,
        "enable_tp_sp": _get_required(manual_parallelism, "enable_tp_sp", "manual_parallelism"),
        "embed_tp": _get_required(manual_parallelism, "embed_tp", "manual_parallelism"),
        "lmhead_tp": _get_required(manual_parallelism, "lmhead_tp", "manual_parallelism"),
        "comm_protocol": _get_required(manual_parallelism, "comm_protocol", "manual_parallelism"),
        "kv_cache_rate": _get_required(manual_parallelism, "kv_cache_rate", "manual_parallelism"),
        "is_prefill": _get_required(manual_parallelism, "is_prefill", "manual_parallelism"),
        "enable_zigzag": _get_required(manual_parallelism, "enable_zigzag", "manual_parallelism"),
        "enable_ring_attention": _get_required(manual_parallelism, "enable_ring_attention", "manual_parallelism"),
        # 推理参数
        "q_seq_len": input_seq_length,
        "kv_seq_len": input_seq_length + output_seq_length,
    }
