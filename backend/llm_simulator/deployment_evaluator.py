"""
部署方案评估适配器

将现有的模拟器逻辑包装成任务管理器需要的接口
"""

import logging
from typing import Dict, Any, Optional, Callable, List

logger = logging.getLogger(__name__)


def evaluate_deployment(
    topology: dict,
    model_config: dict,
    hardware_config: dict,
    inference_config: dict,
    search_mode: str,
    manual_parallelism: Optional[dict] = None,
    search_constraints: Optional[dict] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> dict:
    """
    评估部署方案

    Args:
        topology: 拓扑配置
        model_config: 模型配置
        hardware_config: 硬件配置
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

    # 模拟评估过程（这里是示例实现，你需要替换为实际的评估逻辑）
    if search_mode == 'manual':
        return _evaluate_manual_mode(
            topology,
            model_config,
            hardware_config,
            inference_config,
            manual_parallelism,
            total_chips,
            progress_callback
        )
    else:
        return _evaluate_auto_mode(
            topology,
            model_config,
            hardware_config,
            inference_config,
            search_constraints,
            total_chips,
            progress_callback
        )


def _evaluate_manual_mode(
    topology: dict,
    model_config: dict,
    hardware_config: dict,
    inference_config: dict,
    manual_parallelism: dict,
    total_chips: int,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> dict:
    """手动模式评估"""

    if progress_callback:
        progress_callback(0, 1, "开始手动模式评估...")

    # 提取并行策略
    dp = manual_parallelism.get("dp", 1)
    tp = manual_parallelism.get("tp", 1)
    pp = manual_parallelism.get("pp", 1)
    ep = manual_parallelism.get("ep", 1)

    required_chips = dp * tp * pp * ep

    # 检查芯片数量是否足够
    if required_chips > total_chips:
        result = {
            "parallelism": manual_parallelism,
            "chips": required_chips,
            "is_feasible": False,
            "infeasible_reason": f"需要 {required_chips} 个芯片，但拓扑中只有 {total_chips} 个",
            "throughput": 0,
            "tps_per_chip": 0,
            "ttft": 0,
            "tpot": 0,
            "mfu": 0,
            "mbu": 0,
            "score": 0,
        }

        if progress_callback:
            progress_callback(1, 1, "评估完成（不可行）")

        return {
            "top_k_plans": [],
            "infeasible_plans": [result],
            "search_stats": {
                "total_plans": 1,
                "feasible_plans": 0,
                "infeasible_plans": 1,
            }
        }

    # 这里应该调用实际的模拟器来计算性能指标
    # 目前使用模拟数据作为示例

    if progress_callback:
        progress_callback(1, 2, "计算性能指标...")

    # 模拟性能指标（需要替换为真实计算）
    throughput = 1000.0 * total_chips / required_chips  # 示例值
    tps_per_chip = throughput / required_chips
    ttft = 50.0  # ms
    tpot = 10.0  # ms
    mfu = 0.45
    mbu = 0.60
    score = tps_per_chip * 100

    result = {
        "parallelism": manual_parallelism,
        "chips": required_chips,
        "is_feasible": True,
        "throughput": throughput,
        "tps_per_chip": tps_per_chip,
        "ttft": ttft,
        "tpot": tpot,
        "mfu": mfu,
        "mbu": mbu,
        "score": score,
    }

    if progress_callback:
        progress_callback(2, 2, "评估完成")

    return {
        "top_k_plans": [result],
        "infeasible_plans": [],
        "search_stats": {
            "total_plans": 1,
            "feasible_plans": 1,
            "infeasible_plans": 0,
        }
    }


def _evaluate_auto_mode(
    topology: dict,
    model_config: dict,
    hardware_config: dict,
    inference_config: dict,
    search_constraints: dict,
    total_chips: int,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> dict:
    """自动模式评估（搜索最优方案）"""

    max_chips = search_constraints.get("max_chips", total_chips)

    # 生成所有可能的并行组合
    candidates = _generate_parallelism_candidates(
        max_chips,
        model_config,
        hardware_config
    )

    total_candidates = len(candidates)
    logger.info(f"自动模式: 生成 {total_candidates} 个候选方案")

    if progress_callback:
        progress_callback(0, total_candidates, f"开始评估 {total_candidates} 个方案...")

    top_k_plans = []
    infeasible_plans = []

    for i, parallelism in enumerate(candidates):
        # 模拟评估每个方案
        result = _evaluate_single_plan(
            parallelism,
            total_chips,
            model_config,
            hardware_config,
            inference_config
        )

        if result["is_feasible"]:
            top_k_plans.append(result)
        else:
            infeasible_plans.append(result)

        if progress_callback:
            progress_callback(
                i + 1,
                total_candidates,
                f"已评估 {i + 1}/{total_candidates} 个方案"
            )

    # 按得分排序
    top_k_plans.sort(key=lambda x: x["score"], reverse=True)
    top_k_plans = top_k_plans[:10]  # 只保留前 10 个

    return {
        "top_k_plans": top_k_plans,
        "infeasible_plans": infeasible_plans,
        "search_stats": {
            "total_plans": total_candidates,
            "feasible_plans": len(top_k_plans),
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

    # 简化版本：只生成几个示例方案
    # 实际应该根据模型配置生成所有合法的因子分解

    for tp in [1, 2, 4, 8]:
        if tp > max_chips:
            continue
        dp = max_chips // tp
        if dp > 0:
            candidates.append({"dp": dp, "tp": tp, "pp": 1, "ep": 1})

    return candidates


def _evaluate_single_plan(
    parallelism: dict,
    total_chips: int,
    model_config: dict,
    hardware_config: dict,
    inference_config: dict
) -> dict:
    """评估单个方案"""

    dp = parallelism.get("dp", 1)
    tp = parallelism.get("tp", 1)
    pp = parallelism.get("pp", 1)
    ep = parallelism.get("ep", 1)

    required_chips = dp * tp * pp * ep

    if required_chips > total_chips:
        return {
            "parallelism": parallelism,
            "chips": required_chips,
            "is_feasible": False,
            "infeasible_reason": f"需要 {required_chips} 个芯片，超出 {total_chips}",
            "throughput": 0,
            "tps_per_chip": 0,
            "ttft": 0,
            "tpot": 0,
            "mfu": 0,
            "mbu": 0,
            "score": 0,
        }

    # 模拟性能指标
    throughput = 800.0 * total_chips / required_chips + tp * 50
    tps_per_chip = throughput / required_chips
    ttft = 50.0 / tp
    tpot = 10.0 / tp
    mfu = 0.40 + tp * 0.02
    mbu = 0.55
    score = tps_per_chip * 100 + mfu * 50

    return {
        "parallelism": parallelism,
        "chips": required_chips,
        "is_feasible": True,
        "throughput": throughput,
        "tps_per_chip": tps_per_chip,
        "ttft": ttft,
        "tpot": tpot,
        "mfu": mfu,
        "mbu": mbu,
        "score": score,
    }
