"""
部署方案评估适配器

将现有的模拟器逻辑包装成任务管理器需要的接口
"""

import logging
from typing import Dict, Any, Optional, Callable, List
from ..core.simulator import run_simulation
from ..core.analyzer import PerformanceAnalyzer
from ..evaluators import get_arch_preset
from ..models import create_deepseek_v3, create_deepseek_v3_absorb, create_deepseek_v32

logger = logging.getLogger(__name__)


def evaluate_deployment(
    topology: dict,
    model_config: dict,
    inference_config: dict,
    search_mode: str,
    manual_parallelism: Optional[dict] = None,
    search_constraints: Optional[dict] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    enable_tile_search: bool = True,
    enable_partition_search: bool = False,
    max_simulated_tokens: int = 4,
) -> dict:
    """
    评估部署方案

    Args:
        topology: 完整拓扑配置（包含 pods/racks/boards/chips/connections + protocol_config + network_config + chip_latency_config）
        model_config: 模型配置
        inference_config: 推理配置
        search_mode: 'manual' 或 'auto'
        manual_parallelism: 手动模式的并行策略
        search_constraints: 自动模式的搜索约束
        progress_callback: 进度回调函数 (current, total, message)

    Returns:
        评估结果字典，包含:
        - top_k_plans: 可行方案列表
        - infeasible_plans: 不可行方案列表
        - search_stats: 搜索统计
    """

    # 计算总芯片数
    total_chips = 0
    for pod in topology.get("pods", []):
        for rack in pod.get("racks", []):
            for board in rack.get("boards", []):
                total_chips += len(board.get("chips", []))

    logger.info(f"开始评估: mode={search_mode}, chips={total_chips}")

    if search_mode == 'manual':
        return _evaluate_manual_mode(
            topology,
            model_config,
            inference_config,
            manual_parallelism,
            total_chips,
            progress_callback,
            enable_tile_search,
            enable_partition_search,
            max_simulated_tokens
        )
    else:
        return _evaluate_auto_mode(
            topology,
            model_config,
            inference_config,
            search_constraints,
            total_chips,
            progress_callback,
            enable_tile_search,
            enable_partition_search,
            max_simulated_tokens
        )


def _evaluate_manual_mode(
    topology: dict,
    model_config: dict,
    inference_config: dict,
    manual_parallelism: dict,
    total_chips: int,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    enable_tile_search: bool = True,
    enable_partition_search: bool = False,
    max_simulated_tokens: int = 4,
) -> dict:
    """手动模式评估（带细粒度进度）"""

    if progress_callback:
        progress_callback(0, 100, "开始手动模式评估...")

    # 提取并行策略（严格模式：不使用默认值）
    required_parallelism_fields = ["dp", "tp", "pp", "ep"]
    missing_fields = [f for f in required_parallelism_fields if f not in manual_parallelism]
    if missing_fields:
        raise ValueError(f"手动并行策略缺少必需字段: {', '.join(missing_fields)}")

    dp = manual_parallelism["dp"]
    tp = manual_parallelism["tp"]
    pp = manual_parallelism["pp"]
    ep = manual_parallelism["ep"]
    sp = manual_parallelism.get("sp", 1)  # sp 是可选的，默认为 1

    required_chips = dp * tp * pp * ep

    if progress_callback:
        progress_callback(5, 100, "检查芯片数量...")

    # 检查芯片数量是否足够
    if required_chips > total_chips:
        result = _create_infeasible_result(
            manual_parallelism,
            required_chips,
            f"需要 {required_chips} 个芯片，但拓扑中只有 {total_chips} 个"
        )

        if progress_callback:
            progress_callback(100, 100, "评估完成（不可行）")

        return {
            "top_k_plans": [],
            "infeasible_plans": [result],
            "search_stats": {
                "total_plans": 1,
                "feasible_plans": 0,
                "infeasible_plans": 1,
            }
        }

    # 从拓扑配置中提取硬件配置
    if progress_callback:
        progress_callback(8, 100, "提取硬件配置...")
    hardware_config = _extract_hardware_config(topology)

    # 调用真实的模拟器（带细粒度进度回调）
    try:
        # 创建内部进度回调，将模拟器的 0-100% 映射到外部的 10-95%
        def sim_progress_callback(percent: float, message: str):
            print(f"[DEBUG SIM] sim_progress_callback called: percent={percent}, message={message}", flush=True)
            if progress_callback:
                # 映射: 模拟器的 0-100% -> 外部的 10-95%
                external_progress = 10 + percent * 0.85
                print(f"[DEBUG SIM] calling progress_callback: {int(external_progress)}/100", flush=True)
                progress_callback(int(external_progress), 100, message)

        if progress_callback:
            progress_callback(10, 100, "运行模拟器...")

        sim_result = run_simulation(
            topology_dict=topology,
            model_dict=model_config,
            inference_dict=inference_config,
            parallelism_dict=manual_parallelism,
            hardware_dict=hardware_config,
            progress_callback=sim_progress_callback,
            enable_tile_search=enable_tile_search,
            enable_partition_search=enable_partition_search,
            max_simulated_tokens=max_simulated_tokens,
        )

        if progress_callback:
            progress_callback(95, 100, "转换结果格式...")

        # 转换为 DS_TPU 格式
        result = _transform_to_ds_tpu_format(
            sim_result=sim_result,
            parallelism=manual_parallelism,
            chips=required_chips,
            model_config=model_config,
            inference_config=inference_config,
            topology=topology,
        )

        if progress_callback:
            progress_callback(100, 100, "评估完成")

        return {
            "top_k_plans": [result],
            "infeasible_plans": [],
            "search_stats": {
                "total_plans": 1,
                "feasible_plans": 1,
                "infeasible_plans": 0,
            }
        }

    except Exception as e:
        logger.error(f"模拟器运行失败: {e}")
        result = _create_infeasible_result(
            manual_parallelism,
            required_chips,
            f"模拟器运行失败: {str(e)}"
        )
        return {
            "top_k_plans": [],
            "infeasible_plans": [result],
            "search_stats": {
                "total_plans": 1,
                "feasible_plans": 0,
                "infeasible_plans": 1,
            }
        }


def _evaluate_auto_mode(
    topology: dict,
    model_config: dict,
    inference_config: dict,
    search_constraints: dict,
    total_chips: int,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    enable_tile_search: bool = True,
    enable_partition_search: bool = False,
    max_simulated_tokens: int = 4,
) -> dict:
    """自动模式评估（搜索最优方案，带累加进度）

    进度计算方式：
    - 总进度 = (已完成方案数 / 总方案数) * 100% + (当前方案内部进度 / 总方案数)
    - 例如：10 个方案，第 3 个方案执行到 50%
      总进度 = (2/10)*100% + (50%/10) = 20% + 5% = 25%
    """

    max_chips = search_constraints.get("max_chips", total_chips)
    top_k = search_constraints.get("top_k", 10)

    if progress_callback:
        progress_callback(0, 100, "准备评估...")

    # 从拓扑配置中提取硬件配置
    hardware_config = _extract_hardware_config(topology)

    # 生成所有可能的并行组合
    candidates = _generate_parallelism_candidates(
        max_chips,
        model_config,
        hardware_config
    )

    total_candidates = len(candidates)
    logger.info(f"自动模式: 生成 {total_candidates} 个候选方案")

    if progress_callback:
        progress_callback(2, 100, f"开始评估 {total_candidates} 个方案...")

    # 评估所有候选方案
    feasible_plans = []
    infeasible_plans = []

    # 预留 2% 给准备工作，98% 给方案评估
    progress_per_plan = 98.0 / total_candidates if total_candidates > 0 else 98.0

    for i, candidate in enumerate(candidates):
        # 计算当前方案开始时的基础进度
        base_progress = 2 + i * progress_per_plan

        # 创建内部进度回调，将方案的 0-100% 映射到该方案的进度份额
        def make_inner_callback(plan_index: int, base: float, per_plan: float):
            def inner_callback(inner_percent: float, message: str):
                if progress_callback:
                    # 累加进度: 基础进度 + 内部进度 * 该方案的份额
                    total_progress = base + (inner_percent / 100.0) * per_plan
                    progress_callback(int(total_progress), 100, f"[{plan_index + 1}/{total_candidates}] {message}")
            return inner_callback

        inner_progress = make_inner_callback(i, base_progress, progress_per_plan)

        # 报告方案开始
        if progress_callback:
            progress_callback(int(base_progress), 100, f"评估方案 {i + 1}/{total_candidates}...")

        result = _evaluate_single_plan_with_progress(
            parallelism=candidate,
            total_chips=total_chips,
            model_config=model_config,
            inference_config=inference_config,
            topology=topology,
            hardware_config=hardware_config,
            progress_callback=inner_progress,
            enable_tile_search=enable_tile_search,
            enable_partition_search=enable_partition_search,
            max_simulated_tokens=max_simulated_tokens,
        )

        if result.get("is_feasible", False):
            feasible_plans.append(result)
        else:
            infeasible_plans.append(result)

    # 按得分排序，取 Top-K
    feasible_plans.sort(key=lambda x: x.get("score", 0), reverse=True)
    top_k_plans = feasible_plans[:top_k]

    if progress_callback:
        progress_callback(100, 100, f"评估完成，找到 {len(feasible_plans)} 个可行方案")

    return {
        "top_k_plans": top_k_plans,
        "infeasible_plans": infeasible_plans,
        "search_stats": {
            "total_plans": total_candidates,
            "feasible_plans": len(feasible_plans),
            "infeasible_plans": len(infeasible_plans),
        }
    }


def _generate_parallelism_candidates(
    max_chips: int,
    model_config: dict,
    hardware_config: dict
) -> List[dict]:
    """生成并行策略候选"""

    candidates = []

    # TODO: 实现智能的候选生成算法
    # 应该根据以下因素生成合法的因子分解：
    # 1. 模型大小和层数（决定 PP 的可能性）
    # 2. 注意力头数（决定 TP 的约束）
    # 3. 专家数量（决定 EP 的需求，如果是 MoE 模型）
    # 4. 可用芯片数（max_chips）
    # 5. 拓扑结构（同板/同机/跨机的带宽差异）

    # 示例：简单的因子分解（需要改进）
    for tp in [1, 2, 4, 8, 16]:
        if tp > max_chips:
            continue
        for pp in [1, 2, 4, 8]:
            if tp * pp > max_chips:
                continue
            for ep in [1, 2, 4, 8]:
                if tp * pp * ep > max_chips:
                    continue
                dp = max_chips // (tp * pp * ep)
                if dp > 0:
                    candidates.append({
                        "dp": dp,
                        "tp": tp,
                        "pp": pp,
                        "ep": ep,
                        "sp": 1  # Sequence Parallelism 通常与 TP 绑定
                    })

    return candidates


def _evaluate_single_plan(
    parallelism: dict,
    total_chips: int,
    model_config: dict,
    inference_config: dict,
    topology: dict,
) -> dict:
    """评估单个方案（不使用默认值，确保数据完整性）"""

    # 直接访问字段，缺失时会立即 KeyError
    dp = parallelism["dp"]
    tp = parallelism["tp"]
    pp = parallelism["pp"]
    ep = parallelism["ep"]

    required_chips = dp * tp * pp * ep

    if required_chips > total_chips:
        return _create_infeasible_result(
            parallelism,
            required_chips,
            f"需要 {required_chips} 个芯片，超出 {total_chips}"
        )

    # 从拓扑配置中提取硬件配置
    hardware_config = _extract_hardware_config(topology)

    # 调用真实的模拟器
    try:
        sim_result = run_simulation(
            topology_dict=topology,
            model_dict=model_config,
            inference_dict=inference_config,
            parallelism_dict=parallelism,
            hardware_dict=hardware_config,
        )

        # 转换为 DS_TPU 格式
        return _transform_to_ds_tpu_format(
            sim_result=sim_result,
            parallelism=parallelism,
            chips=required_chips,
            model_config=model_config,
            inference_config=inference_config,
            topology=topology,
        )

    except Exception as e:
        logger.error(f"评估方案失败 {parallelism}: {e}")
        return _create_infeasible_result(
            parallelism,
            required_chips,
            f"模拟失败: {str(e)}"
        )


def _evaluate_single_plan_with_progress(
    parallelism: dict,
    total_chips: int,
    model_config: dict,
    inference_config: dict,
    topology: dict,
    hardware_config: dict,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    enable_tile_search: bool = True,
    enable_partition_search: bool = False,
    max_simulated_tokens: int = 4,
) -> dict:
    """评估单个方案（带细粒度进度回调）

    Args:
        parallelism: 并行策略配置
        total_chips: 可用芯片总数
        model_config: 模型配置
        inference_config: 推理配置
        topology: 拓扑配置
        hardware_config: 硬件配置（已提取）
        progress_callback: 进度回调函数 (percent: float, message: str) -> None
            percent: 0-100 的进度百分比
            message: 进度描述信息
    """

    # 直接访问字段，缺失时会立即 KeyError
    dp = parallelism["dp"]
    tp = parallelism["tp"]
    pp = parallelism["pp"]
    ep = parallelism["ep"]

    required_chips = dp * tp * pp * ep

    # 报告检查进度
    if progress_callback:
        progress_callback(5, "检查芯片数量")

    if required_chips > total_chips:
        if progress_callback:
            progress_callback(100, "芯片数量不足")
        return _create_infeasible_result(
            parallelism,
            required_chips,
            f"需要 {required_chips} 个芯片，超出 {total_chips}"
        )

    # 调用真实的模拟器（带进度回调）
    try:
        if progress_callback:
            progress_callback(10, "启动模拟器")

        sim_result = run_simulation(
            topology_dict=topology,
            model_dict=model_config,
            inference_dict=inference_config,
            parallelism_dict=parallelism,
            hardware_dict=hardware_config,
            progress_callback=progress_callback,  # 传递进度回调
            enable_tile_search=enable_tile_search,
            max_simulated_tokens=max_simulated_tokens,
        )

        if progress_callback:
            progress_callback(95, "转换结果格式")

        # 转换为 DS_TPU 格式
        result = _transform_to_ds_tpu_format(
            sim_result=sim_result,
            parallelism=parallelism,
            chips=required_chips,
            model_config=model_config,
            inference_config=inference_config,
            topology=topology,
        )

        if progress_callback:
            progress_callback(100, "评估完成")

        return result

    except Exception as e:
        logger.error(f"评估方案失败 {parallelism}: {e}")
        if progress_callback:
            progress_callback(100, f"模拟失败: {str(e)}")
        return _create_infeasible_result(
            parallelism,
            required_chips,
            f"模拟失败: {str(e)}"
        )


def _create_infeasible_result(parallelism: dict, chips: int, reason: str) -> dict:
    """创建不可行方案结果"""
    return {
        "parallelism": parallelism,
        "chips": chips,
        "is_feasible": False,
        "infeasible_reason": reason,
        "total_elapse_us": 0,
        "total_elapse_ms": 0,
        "comm_elapse_us": 0,
        "tps": 0,
        "tps_per_batch": 0,
        "tps_per_chip": 0,
        "mfu": 0,
        "flops": 0,
        "dram_occupy": 0,
        "score": 0,
    }


def _create_model_instance(model_config: dict, inference_config: dict, parallelism: dict):
    """根据配置创建模型实例（不使用默认值）"""
    # 准备模型配置（合并 model_config, inference_config, parallelism）
    # 直接访问字段，缺失时会立即 KeyError
    full_config = {
        **model_config,
        "batch_size": inference_config["batch_size"],
        "seq_len": inference_config["input_seq_length"],
        "kv_seq_len": inference_config["input_seq_length"],
        "is_prefill": False,  # Decode 阶段
        "tp": parallelism["tp"],
        "dp": parallelism["dp"],
        "moe_tp": parallelism.get("moe_tp", 1),  # moe_tp 是可选的，非 MoE 模型没有
        "ep": parallelism.get("ep", 1),  # ep 也是可选的
        "comm_protocol": 1,
    }

    # 根据模型类型创建实例
    model_name = model_config.get("model_name", "").lower()
    mla_config = model_config.get("mla_config")

    if "deepseek" in model_name or mla_config:
        # 判断 MLA 变体
        if mla_config:
            # 有 MLA 配置，选择对应的工厂函数
            # 简化：默认使用 absorb 变体（Decode 阶段）
            return create_deepseek_v3_absorb(full_config)
        else:
            # 标准 DeepSeek V3
            return create_deepseek_v3(full_config)
    else:
        # 其他模型类型，使用基础工厂
        logger.warning(f"Unknown model type: {model_name}, using deepseek_v3 as fallback")
        return create_deepseek_v3(full_config)


def _transform_to_ds_tpu_format(
    sim_result: dict,
    parallelism: dict,
    chips: int,
    model_config: dict,
    inference_config: dict,
    topology: dict,
) -> dict:
    """将 Tier6-Model 模拟结果转换为 DS_TPU 格式（包含详细的 layers 信息）"""

    stats = sim_result.get("stats", {})

    # 提取基础性能指标
    avg_tpot = stats.get("avgTpot", 0)  # 微秒/token
    ttft = stats.get("ttft", 0)  # 微秒
    mfu = stats.get("dynamicMfu", 0)

    # Decode 阶段通信时间
    decode_stats = stats.get("decode", {})
    comm_time_us = decode_stats.get("commTime", 0)

    # 计算吞吐量指标
    total_elapse_us = avg_tpot if avg_tpot > 0 else 1.0
    total_elapse_ms = total_elapse_us / 1000.0

    # tokens/s = 1,000,000 us/s / (us/token)
    tps = 1_000_000.0 / total_elapse_us if total_elapse_us > 0 else 0

    batch_size = inference_config.get("batch_size", 1)
    tps_per_batch = tps / batch_size if batch_size > 0 else 0

    # 修正 tps_per_chip 计算（对齐 DS_TPU）
    tp = parallelism.get("tp", 1)
    dp = parallelism.get("dp", 1)
    tps_per_chip = tps / (tp * dp) if (tp * dp) > 0 else 0

    # 使用 PerformanceAnalyzer 获取详细的 layers 信息
    layers_info = {}
    flops = 0
    dram_occupy = 0

    try:
        # 创建模型实例
        model = _create_model_instance(model_config, inference_config, parallelism)

        # 从拓扑配置中提取硬件配置
        hardware_config = _extract_hardware_config(topology)
        chip_hw = hardware_config.get("chip", {})
        chip_type = chip_hw.get("chip_type", "SG2260E")
        arch = get_arch_preset(chip_type)

        # 创建 PerformanceAnalyzer
        analyzer = PerformanceAnalyzer(model, arch, global_cache={})

        # 获取性能分析结果
        analysis = model.analyze_performance(
            batch_size=batch_size,
            seq_len=inference_config.get("input_seq_length", 1),
            tpu_flops=getattr(arch, 'flops', 0)
        )

        # 提取 layers 信息和指标
        layers_info = analysis.get("layers", {})
        flops = analysis.get("total_flops", 0)
        dram_occupy = analysis.get("dram_occupy", 0)

        logger.info(f"PerformanceAnalyzer 成功生成 {len(layers_info)} 层的详细信息")

    except Exception as e:
        logger.warning(f"PerformanceAnalyzer 运行失败，使用简化估算: {e}")

        # 回退到简化估算
        hidden_size = model_config.get("hidden_size", 0)
        num_layers = model_config.get("num_layers", 0)
        intermediate_size = model_config.get("intermediate_size", 0)

        flops_per_token = num_layers * (
            4 * hidden_size * hidden_size +
            2 * hidden_size * intermediate_size * 2
        )
        flops = flops_per_token * batch_size

        bytes_per_param = 2
        model_params = num_layers * (
            4 * hidden_size * hidden_size +
            2 * hidden_size * intermediate_size
        )
        seq_len = inference_config.get("input_seq_length", 0) + inference_config.get("output_seq_length", 0)
        kv_cache_params = 2 * num_layers * hidden_size * seq_len * batch_size
        dram_occupy = (model_params + kv_cache_params) * bytes_per_param

    # 计算综合得分：tps_per_chip * mfu
    score = tps_per_chip * mfu

    return {
        "parallelism": parallelism,
        "chips": chips,
        "is_feasible": True,
        "total_elapse_us": total_elapse_us,
        "total_elapse_ms": total_elapse_ms,
        "comm_elapse_us": comm_time_us,
        "tps": tps,
        "tps_per_batch": tps_per_batch,
        "tps_per_chip": tps_per_chip,
        "mfu": mfu,
        "flops": flops,
        "dram_occupy": dram_occupy,
        "score": score,
        "layers": layers_info,  # 添加详细的 layers 信息
    }


def _extract_hardware_config(topology: dict) -> dict:
    """从拓扑配置中提取硬件配置

    Args:
        topology: 完整拓扑配置

    Returns:
        硬件配置字典，包含 chip, node, cluster 配置

    Raises:
        ValueError: 如果无法从拓扑中提取芯片配置
    """
    # 如果拓扑配置中直接包含硬件配置，直接返回
    if "hardware_config" in topology:
        return topology["hardware_config"]

    # 否则从拓扑结构中提取第一个芯片的配置
    # 假设所有芯片配置相同（同构集群）
    hardware_config = {
        "chip": {},
        "node": {},
        "cluster": {}
    }

    # 从拓扑中提取第一个芯片的配置
    pods = topology.get("pods", [])
    if pods:
        for pod in pods:
            racks = pod.get("racks", [])
            if racks:
                for rack in racks:
                    boards = rack.get("boards", [])
                    if boards:
                        for board in boards:
                            chips = board.get("chips", [])
                            if chips:
                                chip = chips[0]
                                # 提取芯片配置
                                hardware_config["chip"] = {
                                    "chip_type": chip.get("name", "SG2260E"),
                                    "compute_tflops_fp16": chip.get("compute_tflops_fp16", 2000),
                                    "memory_gb": chip.get("memory_gb", 80),
                                    "memory_bandwidth_gbps": chip.get("memory_bandwidth_gbps", 3000),
                                    "memory_bandwidth_utilization": chip.get("memory_bandwidth_utilization", 0.9),
                                }
                                return hardware_config

    # 如果找不到芯片配置，抛出错误
    raise ValueError("无法从拓扑配置中提取芯片硬件配置，请确保拓扑中包含有效的芯片数据")
