"""评估引擎模块

集成 L1-L5 层，执行完整的评估流程（对齐 CHIPMathica）。
"""

from __future__ import annotations

from typing import Any

from tier6.L0_entry.config_loader import load_chip, load_evaluation_config, load_model, load_scenario


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
    if "scenario" in config:
        # 方式1: 从 scenario 加载
        scenario_name = config["scenario"]
        scenario = load_scenario(scenario_name, resolve=True)
        chip_config = scenario["chip_config"]
        model_config = scenario["model_config"]
        deployment_config = scenario.get("deployment", {})
        board_config = scenario.get("board", {})
        inference_config = config.get("inference", {})
    else:
        # 方式2: 显式配置
        chip_config = _load_chip_config(config)
        model_config = _load_model_config(config)
        deployment_config = config.get("deployment", {})
        board_config = config.get("board", {})
        inference_config = config.get("inference", {})

    # ==================== L1: 构建 WorkloadIR ====================
    from tier6.L1_workload.models.llm.deepseek import DeepSeekV3Model

    # 映射模型配置为 DeepSeekV3Model 需要的格式
    model_params = _map_model_config(model_config, deployment_config, inference_config)
    model = DeepSeekV3Model(model_params)
    ir = model.to_ir()

    print(f"[L1] WorkloadIR created: {len(ir.get_layers())} layers")

    # ==================== L2: 加载 ChipSpec ====================
    from tier6.L2_arch.chip import ChipSpecImpl

    chip = ChipSpecImpl.from_config(chip_config.get("name", "unknown"), chip_config)
    print(f"[L2] ChipSpec loaded: {chip.name}")

    # ==================== L3: Parallelism Planning ====================
    from tier6.L3_mapping.parallelism.planner import DeploymentSpec, ParallelismPlanner
    from tier6.L3_mapping.parallelism.board import BoardSpec

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
    from tier6.L3_mapping.tiling.planner import TilingPlanner
    from tier6.L4_evaluation.evaluators.precise import PreciseTileEvaluator

    print("[L3] Running TilingPlanner...")
    compute_tflops = chip.get_peak_flops("BF16", "cube") / 1e12
    memory_bw_gbps = chip.get_gmem_bandwidth()

    precise_evaluator = PreciseTileEvaluator(
        compute_tflops=compute_tflops,
        memory_bandwidth_gbps=memory_bw_gbps,
    )

    tile_plan = TilingPlanner(chip, l4_evaluator=precise_evaluator).plan(dist_model)

    # ==================== L3: Scheduling ====================
    from tier6.L3_mapping.scheduling.scheduler import Scheduler

    print("[L3] Running Scheduler...")
    exec_plan = Scheduler().plan(dist_model, tile_plan)

    # ==================== L4: Evaluation ====================
    from tier6.L4_evaluation.engine import EvaluationEngine
    from tier6.L4_evaluation.metrics import Granularity

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
    from tier6.L5_reporting.engine import ReportingEngine

    reporting_engine = ReportingEngine()
    report = reporting_engine.run(engine_result=engine_result, config=run_config)

    # 转换为字典
    result = {
        "aggregates": report.performance.to_dict(),
        "step_metrics": [s.to_dict() for s in report.step_metrics],
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
        from tier6.L0_entry.config_loader import load_chip_preset

        return load_chip_preset(config["chip_preset"])
    else:
        raise ValueError("Missing chip configuration")


def _load_model_config(config: dict[str, Any]) -> dict[str, Any]:
    """加载模型配置"""
    if "model" in config:
        model_ref = config["model"]
        if isinstance(model_ref, str):
            # 引用模型预设
            model_data = load_model(model_ref)
            return model_data.get("model", model_data)
        else:
            # 内联配置
            return model_ref
    elif "model_config" in config:
        return config["model_config"]
    elif "model_preset" in config:
        from tier6.L0_entry.config_loader import load_model_preset

        return load_model_preset(config["model_preset"])
    else:
        raise ValueError("Missing model configuration")


def _map_model_config(
    model_cfg: dict[str, Any],
    deployment_cfg: dict[str, Any],
    inference_cfg: dict[str, Any],
) -> dict[str, Any]:
    """映射模型配置为 DeepSeekV3Model 需要的扁平结构"""
    mla = model_cfg.get("mla", {})
    moe = model_cfg.get("moe", {})
    ffn = model_cfg.get("ffn", {})

    return {
        "dtype": model_cfg.get("dtype", "bf16"),
        "hidden_size": int(model_cfg.get("hidden_size", 7168)),
        "num_layers": int(model_cfg.get("num_layers", 61)),
        "num_dense_layers": int(model_cfg.get("num_dense_layers", 3)),
        "num_moe_layers": int(model_cfg.get("num_moe_layers", 58)),
        "num_heads": int(model_cfg.get("num_heads", 128)),
        "vocab_size": int(model_cfg.get("vocab_size", 129280)),
        # MLA
        "q_lora_rank": int(mla.get("q_lora_rank", 1536)),
        "kv_lora_rank": int(mla.get("kv_lora_rank", 512)),
        "qk_nope_head_dim": int(mla.get("qk_nope_head_dim", 128)),
        "qk_rope_head_dim": int(mla.get("qk_rope_head_dim", 64)),
        "v_head_dim": int(mla.get("v_head_dim", 128)),
        # MoE
        "n_routed_experts": int(moe.get("num_routed_experts", 256)),
        "n_shared_experts": int(moe.get("num_shared_experts", 1)),
        "n_activated_experts": int(moe.get("num_activated_experts", 8)),
        "moe_intermediate_size": int(moe.get("intermediate_size", 2048)),
        # FFN
        "intermediate_size": int(ffn.get("intermediate_size", 18432)),
        # 运行时参数
        "seq_len": int(deployment_cfg.get("seq_len", inference_cfg.get("seq_len", 1))),
        "kv_seq_len": int(deployment_cfg.get("kv_seq_len", inference_cfg.get("kv_seq_len", 4096))),
        "q_seq_len": int(deployment_cfg.get("q_seq_len", inference_cfg.get("q_seq_len", 4096))),
        "batch": int(deployment_cfg.get("batch_size", inference_cfg.get("batch_size", 2048))),
        "is_prefill": bool(deployment_cfg.get("is_prefill", inference_cfg.get("is_prefill", False))),
    }


def _build_hardware_spec(chip: Any, chip_config: dict[str, Any]) -> dict[str, Any]:
    """构建硬件规格字典（对齐 CHIPMathica）"""
    from tier6.L4_evaluation.metrics import CommProtocolSpec, HardwareSpec
    from tier6.L2_arch.topology import TopologySpec

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
    from tier6.L4_evaluation import merge_specs

    hardware = merge_specs(hardware_spec, topology_spec, comm_spec)

    # 加入计算效率因子
    hardware["compute_efficiency"] = float(hardware_cfg.get("compute_efficiency", 0.9))

    return hardware
