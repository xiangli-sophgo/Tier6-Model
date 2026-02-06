"""评估引擎模块

集成 L1-L5 层，执行完整的评估流程（对齐 CHIPMathica）。
支持从前端 EvaluationRequest 格式配置运行评估。
"""

from __future__ import annotations

import logging
from typing import Any

from math_model.L0_entry.config_loader import (
    load_chip,
    load_evaluation_config,
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
    deployment_config = config.get("deployment", {})
    board_config = config.get("board", {})
    inference_config = config.get("inference", {})

    # ==================== L1: 构建 WorkloadIR ====================
    from math_model.L1_workload.models.llm.deepseek import DeepSeekV3Model

    # 映射模型配置为 DeepSeekV3Model 需要的格式
    model_params = _map_model_config(model_config, deployment_config, inference_config)
    model = DeepSeekV3Model(model_params)
    ir = model.to_ir()

    print(f"[L1] WorkloadIR created: {len(ir.get_layers())} layers")

    # ==================== L2: 加载 ChipSpec ====================
    from math_model.L2_arch.chip import ChipSpecImpl

    chip = ChipSpecImpl.from_config(chip_config.get("name", "unknown"), chip_config)
    print(f"[L2] ChipSpec loaded: {chip.name}")

    # ==================== L3: Parallelism Planning ====================
    from math_model.L3_mapping.parallelism.planner import DeploymentSpec, ParallelismPlanner, BoardSpec

    deployment = DeploymentSpec(
        tp=int(deployment_config.get("tp", 2)),
        pp=int(deployment_config.get("pp", 1)),
        ep=int(deployment_config.get("ep", 16)),
        dp=int(deployment_config.get("dp", 16)),
        moe_tp=int(deployment_config.get("moe_tp", 2)),
        seq_len=int(deployment_config.get("seq_len", 1)),
        batch_size=int(deployment_config.get("batch_size", 2048)),
        enable_tp_sp=bool(deployment_config.get("enable_tp_sp", True)),
        embed_tp=int(deployment_config.get("embed_tp", 1)),
        lmhead_tp=int(deployment_config.get("lmhead_tp", 2)),
        comm_protocol=int(deployment_config.get("comm_protocol", 1)),
        kv_cache_rate=float(deployment_config.get("kv_cache_rate", 0.0)),
        is_prefill=bool(deployment_config.get("is_prefill", False)),
    )

    board = BoardSpec(
        num_chips=int(board_config.get("num_chips", 32)),
        chip_memory_gb=int(board_config.get("chip_memory_gb", 64)),
        inter_chip_bw_gbps=float(board_config.get("inter_chip_bw_gbps", 400.0)),
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
            "name": model_config.get("name", "DeepSeek-V3"),
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
    # 必需的模型字段
    dtype = _get_required(model_cfg, "dtype", "model config")
    hidden_size = int(_get_required(model_cfg, "hidden_size", "model config"))
    num_layers = int(_get_required(model_cfg, "num_layers", "model config"))
    num_heads = int(_get_required(model_cfg, "num_heads", "model config"))
    vocab_size = int(_get_required(model_cfg, "vocab_size", "model config"))

    # dense/moe 层数
    num_dense_layers = int(_get_required(model_cfg, "num_dense_layers", "model config"))
    num_moe_layers = int(_get_required(model_cfg, "num_moe_layers", "model config"))

    # MLA 配置 (必需)
    mla = model_cfg.get("MLA")
    if mla is None:
        raise ValueError("Missing 'mla' section in model config")
    q_lora_rank = int(_get_required(mla, "q_lora_rank", "model.mla"))
    kv_lora_rank = int(_get_required(mla, "kv_lora_rank", "model.mla"))
    qk_nope_head_dim = int(_get_required(mla, "qk_nope_head_dim", "model.mla"))
    qk_rope_head_dim = int(_get_required(mla, "qk_rope_head_dim", "model.mla"))
    v_head_dim = int(_get_required(mla, "v_head_dim", "model.mla"))

    # MoE 配置 (必需)
    moe = model_cfg.get("MoE")
    if moe is None:
        raise ValueError("Missing 'moe' section in model config")
    n_routed_experts = int(_get_required(moe, "num_routed_experts", "model.moe"))
    n_shared_experts = int(_get_required(moe, "num_shared_experts", "model.moe"))
    n_activated_experts = int(_get_required(moe, "num_activated_experts", "model.moe"))
    moe_intermediate_size = int(_get_required(moe, "intermediate_size", "model.moe"))

    # FFN 配置 (必需)
    ffn = model_cfg.get("ffn")
    if ffn is None:
        raise ValueError("Missing 'ffn' section in model config")
    intermediate_size = int(_get_required(ffn, "intermediate_size", "model.ffn"))

    # 运行时参数 - 从 deployment 或 inference 获取
    seq_len = int(deployment_cfg.get("seq_len", inference_cfg.get("seq_len", 1)))
    kv_seq_len = int(deployment_cfg.get("kv_seq_len", inference_cfg.get("kv_seq_len", 4096)))
    q_seq_len = int(deployment_cfg.get("q_seq_len", inference_cfg.get("q_seq_len", 4096)))
    batch = int(deployment_cfg.get("batch_size", inference_cfg.get("batch_size", 1)))
    is_prefill = bool(deployment_cfg.get("is_prefill", inference_cfg.get("is_prefill", False)))

    return {
        "dtype": dtype,
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

    # 加载评估配置
    eval_cfg = load_evaluation_config("default")
    hardware_cfg = eval_cfg.get("evaluation", {}).get("hardware", {})
    topology_cfg = eval_cfg.get("evaluation", {}).get("topology", {})
    comm_cfg = eval_cfg.get("evaluation", {}).get("comm_protocol", {})

    # 硬件规格
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

    # 拓扑规格
    topology_spec = TopologySpec(
        intra_board_bw_gbps=float(topology_cfg.get("intra_board_bw_gbps", 448.0)),
        inter_board_bw_gbps=float(topology_cfg.get("inter_board_bw_gbps", 448.0)),
        inter_node_bw_gbps=float(topology_cfg.get("inter_node_bw_gbps", 448.0)),
        c2c_lat_us=float(topology_cfg.get("c2c_lat_us", 0.15)),
        ddr_r_lat_us=float(topology_cfg.get("ddr_r_lat_us", 0.15)),
        ddr_w_lat_us=float(topology_cfg.get("ddr_w_lat_us", 0.01)),
        noc_lat_us=float(topology_cfg.get("noc_lat_us", 0.05)),
        d2d_lat_us=float(topology_cfg.get("d2d_lat_us", 0.04)),
        link_delay_us=float(topology_cfg.get("link_delay_us", 0.0)),
        switch_delay_us=float(topology_cfg.get("switch_delay_us", 0.25)),
        cable_delay_us=float(topology_cfg.get("cable_delay_us", 0.025)),
    )

    # 通信协议规格
    comm_spec = CommProtocolSpec(
        rtt_tp_us=float(comm_cfg.get("rtt_tp_us", 0.35)),
        rtt_ep_us=float(comm_cfg.get("rtt_ep_us", 0.85)),
        sync_lat_us=float(comm_cfg.get("sync_lat_us", 0.0)),
        bw_utilization=float(comm_cfg.get("bw_utilization", 0.95)),
        cpu_fetch_delay_us=float(comm_cfg.get("cpu_fetch_delay_us", 0.0)),
        moe_topk=float(comm_cfg.get("moe_topk", 8.0)),
        prefill_topk_factor=float(comm_cfg.get("prefill_topk_factor", 0.0625)),
    )

    # 合并规格
    from math_model.L4_evaluation import merge_specs

    hardware = merge_specs(hardware_spec, topology_spec, comm_spec)

    # 加入计算效率因子
    hardware["compute_efficiency"] = float(hardware_cfg.get("compute_efficiency", 0.9))

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
            - topology_config: 完整拓扑配置 (含 comm_latency_config)
            - manual_parallelism: 手动并行配置
            - search_mode: 搜索模式

    Returns:
        dict: 评估结果
    """
    # WebSocket 广播由 TaskManager 处理，此处不需要直接调用

    # 提取配置
    benchmark_config = config.get("benchmark_config", {})
    topology_config = config.get("topology_config", {})
    manual_parallelism = config.get("manual_parallelism", {})

    # 从 benchmark_config 提取模型和推理配置
    model_config = benchmark_config.get("model", {})
    inference_config = benchmark_config.get("inference", {})

    # 从 topology_config.hardware_params.chips 提取芯片配置
    hw_params = topology_config.get("hardware_params", {})
    chips_dict = hw_params.get("chips", topology_config.get("chips", {}))
    chip_config = _extract_first_chip_config(chips_dict)

    # 提取通信延迟配置
    comm_latency_config = topology_config.get("comm_latency_config", {})

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

    logger.info(f"Running evaluation: {config.get('experiment_name', 'unnamed')}")
    logger.info(f"  Model: {model_config.get('name', 'unknown')}")
    logger.info(f"  Chips: {board_spec['num_chips']}")
    logger.info(f"  Parallelism: TP={deployment_config.get('tp')}, PP={deployment_config.get('pp')}, "
                f"DP={deployment_config.get('dp')}, EP={deployment_config.get('ep')}")

    # 执行评估
    try:
        result = run_evaluation(internal_config)

        # 提取性能指标
        aggregates = result.get("aggregates", {})
        step_metrics = result.get("step_metrics", [])

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
        tps = aggregates.get("tps", 0)
        tpot = aggregates.get("tpot_ms", aggregates.get("total_time_ms", 0) / max(inference_config.get("output_seq_length", 1), 1))

        # 计算芯片数和每芯片/每批次 TPS
        tp = deployment_config.get("tp", 1)
        pp = deployment_config.get("pp", 1)
        dp = deployment_config.get("dp", 1)
        ep = deployment_config.get("ep", 1)
        chips = tp * pp * dp * ep
        batch_size = deployment_config.get("batch_size", 1)
        tps_per_chip = tps / chips if chips > 0 else 0
        tps_per_batch = tps / batch_size if batch_size > 0 else 0

        # 显存占用（从 aggregates 获取，单位转换为 bytes）
        memory_peak_mb = aggregates.get("memory_peak_mb", aggregates.get("memory_peak", 0) / (1024 * 1024) if aggregates.get("memory_peak") else 0)
        dram_occupy = memory_peak_mb * 1024 * 1024  # MB -> bytes

        plan = {
            "parallelism": deployment_config,
            "tps": tps,
            "ttft": aggregates.get("ttft_ms", aggregates.get("total_time_ms", 0)),
            "tpot": tpot,
            "mfu": aggregates.get("mfu", 0),
            "mbu": aggregates.get("mbu", aggregates.get("memory_utilization", 0)),
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
            "config": result.get("config", {}),
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
                "experiment_name": config.get("experiment_name", ""),
                "benchmark_name": config.get("benchmark_name", ""),
                "topology_config_name": config.get("topology_config_name", ""),
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
                "experiment_name": config.get("experiment_name", ""),
                "benchmark_name": config.get("benchmark_name", ""),
                "topology_config_name": config.get("topology_config_name", ""),
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
        topology_config: 完整拓扑配置 (含 hardware_params)

    Returns:
        dict: 板卡规格 { num_chips, chip_memory_gb, inter_chip_bw_gbps }
    """
    # 计算芯片总数 - 从 rack_config 计算
    hw = topology_config.get("hardware_params", {})
    rack_config = topology_config.get("rack_config", {})
    boards = rack_config.get("boards", [])

    num_chips = 0
    pod_count = topology_config.get("pod_count", 1)
    racks_per_pod = topology_config.get("racks_per_pod", 1)
    for board in boards:
        board_count = board.get("count", 1)
        for chip_group in board.get("chips", []):
            num_chips += chip_group.get("count", 1) * board_count
    num_chips = num_chips * pod_count * racks_per_pod
    if num_chips == 0:
        num_chips = _count_topology_chips(topology_config.get("topology", {}))

    # 获取互联带宽 - 从 hardware_params.interconnect
    interconnect = hw.get("interconnect", topology_config.get("interconnect", {}))
    c2c = interconnect.get("c2c", {})
    if "bandwidth_gbps" not in c2c:
        raise ValueError("Missing 'hardware_params.interconnect.c2c.bandwidth_gbps' in topology config")
    inter_chip_bw = c2c["bandwidth_gbps"]

    # 获取芯片内存 - 从 hardware_params.chips
    chips = hw.get("chips", topology_config.get("chips", {}))
    if not chips:
        raise ValueError("No chip configuration found in topology config (hardware_params.chips)")

    first_chip = next(iter(chips.values()))
    memory = first_chip.get("memory", {})
    gmem = memory.get("gmem", {})
    if "capacity_gb" not in gmem:
        raise ValueError(f"Missing 'memory.gmem.capacity_gb' in chip config")
    chip_memory = gmem["capacity_gb"]

    return {
        "num_chips": num_chips,
        "chip_memory_gb": chip_memory,
        "inter_chip_bw_gbps": inter_chip_bw,
    }


def _count_topology_chips(topology: dict[str, Any]) -> int:
    """计算拓扑中的芯片总数

    Args:
        topology: 拓扑结构配置

    Returns:
        int: 芯片总数
    """
    # 尝试从 pods 结构计算
    pods = topology.get("pods", [])
    if pods:
        count = 0
        for pod in pods:
            for rack in pod.get("racks", []):
                for board in rack.get("boards", []):
                    for chip_group in board.get("chips", []):
                        count += chip_group.get("count", 1)
        return max(count, 1)

    # 尝试从简单字段获取
    pod_count = topology.get("pod_count", 1)
    racks_per_pod = topology.get("racks_per_pod", 1)
    boards_per_rack = topology.get("boards_per_rack", 1)
    chips_per_board = topology.get("chips_per_board", 8)

    return pod_count * racks_per_pod * boards_per_rack * chips_per_board


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
    from math_model.evaluators.cost_evaluator import CostEvaluator

    # 计算芯片数
    tp = deployment_config.get("tp", 1)
    pp = deployment_config.get("pp", 1)
    dp = deployment_config.get("dp", 1)
    ep = deployment_config.get("ep", 1)
    chips = tp * pp * dp * ep

    # 获取芯片类型
    chip_type = model_config.get("chip_type", "SG2262")

    # 估算模型大小（粗略）
    hidden_size = model_config.get("hidden_size", 7168)
    num_layers = model_config.get("num_layers", 61)
    vocab_size = model_config.get("vocab_size", 129280)
    moe = model_config.get("moe", {})
    num_experts = moe.get("num_routed_experts", 256)

    # 简化的模型参数估算
    # Embedding + Attention + FFN/MoE + LMHead
    embed_params = hidden_size * vocab_size
    attn_params = 4 * hidden_size * hidden_size * num_layers
    ffn_params = 3 * hidden_size * hidden_size * 4 * num_layers  # 简化
    if num_experts > 0:
        ffn_params = num_experts * 3 * hidden_size * 2048 * (num_layers - 3)  # MoE 层

    total_params = embed_params + attn_params + ffn_params
    bytes_per_param = 2  # BF16
    model_size_gb = total_params * bytes_per_param / 1e9

    # 获取性能指标
    tps = aggregates.get("tps", 0)
    tpot = aggregates.get("tpot_ms", aggregates.get("total_time_ms", 0))

    # 计算成本
    evaluator = CostEvaluator()
    cost_result = evaluator.calculate_total_cost(
        cp_num=chips,
        chip_type=chip_type,
        model_size_gb=model_size_gb,
        tp=tp,
        tpot_ms=tpot if tpot > 0 else 1.0,
    )

    # 计算每百万 tokens 成本
    cost_per_m_tokens = evaluator.calculate_cost_per_million_tokens(
        total_cost=cost_result["total_cost"],
        tps=tps,
    ) if tps > 0 else 0

    return {
        **cost_result,
        "model_size_gb": model_size_gb,
        "cost_per_million_tokens": cost_per_m_tokens,
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
    tp = manual_parallelism.get("tp", 1)
    pp = manual_parallelism.get("pp", 1)
    dp = manual_parallelism.get("dp", 1)
    ep = manual_parallelism.get("ep", 1)
    moe_tp = manual_parallelism.get("moe_tp", 1)

    # 从 inference_config 获取推理参数
    batch_size = inference_config.get("batch_size", manual_parallelism.get("batch_size", 1))
    input_seq_length = inference_config.get("input_seq_length", 4096)
    output_seq_length = inference_config.get("output_seq_length", 128)

    return {
        "tp": tp,
        "pp": pp,
        "dp": dp,
        "ep": ep,
        "moe_tp": moe_tp,
        "seq_len": manual_parallelism.get("seq_len", 1),
        "batch_size": batch_size,
        "enable_tp_sp": manual_parallelism.get("enable_tp_sp", False),
        "embed_tp": manual_parallelism.get("embed_tp", 1),
        "lmhead_tp": manual_parallelism.get("lmhead_tp", tp),
        "comm_protocol": manual_parallelism.get("comm_protocol", 1),
        "kv_cache_rate": manual_parallelism.get("kv_cache_rate", 0.0),
        "is_prefill": manual_parallelism.get("is_prefill", False),
        # 推理参数
        "q_seq_len": input_seq_length,
        "kv_seq_len": input_seq_length + output_seq_length,
    }
