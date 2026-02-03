"""
部署方案评估适配器

将现有的模拟器逻辑包装成任务管理器需要的接口
"""

import logging
import threading
from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, Optional, Callable, List
from ..core.simulator import run_simulation
from ..core.analyzer import PerformanceAnalyzer
from ..evaluators import get_arch_preset
from ..evaluators.cost_evaluator import evaluate_deployment_cost
from ..models import create_deepseek_v3, create_deepseek_v3_absorb, create_deepseek_v32

logger = logging.getLogger(__name__)


class TaskCancelledException(Exception):
    """任务取消异常"""
    pass


# ============================================
# 芯片数量计算公共函数
# ============================================

def count_topology_chips(topology: dict) -> int:
    """统计拓扑中的芯片总数

    支持两种拓扑格式：
    1. 展开格式：pods[].racks[].boards[].chips[]
    2. 配置文件格式：pod_count, racks_per_pod, rack_config.boards[].count/chips[].count

    Args:
        topology: 拓扑配置字典

    Returns:
        芯片总数
    """
    # 格式1：展开后的 pods/racks/boards/chips 结构
    if "pods" in topology:
        total = 0
        for pod in topology.get("pods", []):
            for rack in pod.get("racks", []):
                for board in rack.get("boards", []):
                    total += len(board.get("chips", []))
        return total

    # 格式2：配置文件格式 (pod_count, racks_per_pod, rack_config)
    if "rack_config" in topology:
        pod_count = topology.get("pod_count", 1)
        racks_per_pod = topology.get("racks_per_pod", 1)
        rack_config = topology.get("rack_config", {})

        chips_per_rack = 0
        for board in rack_config.get("boards", []):
            board_count = board.get("count", 1)
            for chip in board.get("chips", []):
                chip_count = chip.get("count", 1)
                chips_per_rack += board_count * chip_count

        return pod_count * racks_per_pod * chips_per_rack

    return 0


def calculate_required_chips(parallelism: dict, model_config: dict) -> int:
    """计算并行策略所需的芯片数

    Args:
        parallelism: 并行策略配置 (dp, tp, ep, ...)
        model_config: 模型配置

    Returns:
        所需芯片数

    Note:
        - MoE 模型：DP × TP（因为 MoE 约束 DP×TP = MoE_TP×EP）
        - 非 MoE 模型：DP × TP × EP
        - PP 暂未启用，不参与计算
    """
    dp = parallelism.get("dp", 1)
    tp = parallelism.get("tp", 1)
    ep = parallelism.get("ep", 1)

    is_moe = model_config.get("moe_config") is not None
    if is_moe:
        return dp * tp
    else:
        return dp * tp * ep


def evaluate_deployment(
    topology: dict,
    model_config: dict,
    inference_config: dict,
    search_mode: str,
    manual_parallelism: Optional[dict] = None,
    search_constraints: Optional[dict] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    result_callback: Optional[Callable[[dict], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    enable_tile_search: bool = True,
    enable_partition_search: bool = False,
    max_simulated_tokens: int = 4,
    max_workers: int = 1,
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
        result_callback: 结果回调函数 (自动模式下边评估边保存)
        cancel_check: 取消检查函数，返回 True 表示任务被取消

    Returns:
        评估结果字典，包含:
        - top_k_plans: 可行方案列表
        - infeasible_plans: 不可行方案列表
        - search_stats: 搜索统计

    Raises:
        TaskCancelledException: 任务被取消时抛出
    """

    # 计算拓扑中的总芯片数
    total_chips = count_topology_chips(topology)

    logger.info(f"开始评估: mode={search_mode}, chips={total_chips}")

    if search_mode == 'manual':
        return _evaluate_manual_mode(
            topology,
            model_config,
            inference_config,
            manual_parallelism,
            total_chips,
            progress_callback,
            cancel_check,
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
            result_callback,
            cancel_check,
            enable_tile_search,
            enable_partition_search,
            max_simulated_tokens,
            max_workers
        )


def _evaluate_manual_mode(
    topology: dict,
    model_config: dict,
    inference_config: dict,
    manual_parallelism: dict,
    total_chips: int,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    enable_tile_search: bool = True,
    enable_partition_search: bool = False,
    max_simulated_tokens: int = 4,
) -> dict:
    """手动模式评估（带细粒度进度）"""

    # 检查取消
    if cancel_check and cancel_check():
        logger.info("任务在开始前被取消")
        raise TaskCancelledException("任务已取消")

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

    # 计算实际使用的芯片数
    required_chips = calculate_required_chips(manual_parallelism, model_config)

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
            if progress_callback:
                # 映射: 模拟器的 0-100% -> 外部的 10-95%
                external_progress = 10 + percent * 0.85
                progress_callback(int(external_progress), 100, message)

        if progress_callback:
            progress_callback(10, 100, "运行模拟器...")

        # 手动模式使用保守的进程数，避免多任务串行执行时过度竞争
        # 公式: max(2, cpu_count() // 4)
        # 例如: 16核CPU -> 4进程, 8核CPU -> 2进程
        manual_gemm_processes = max(2, cpu_count() // 4)
        logger.info(f"手动模式 GEMM 进程数: {manual_gemm_processes} (CPU核数: {cpu_count()})")

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
            max_gemm_processes=manual_gemm_processes,
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
    result_callback: Optional[Callable[[dict], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    enable_tile_search: bool = True,
    enable_partition_search: bool = False,
    max_simulated_tokens: int = 4,
    max_workers: int = 1,
) -> dict:
    """自动模式评估（搜索最优方案，支持并行评估）

    进度计算方式：
    - 总进度 = (已完成方案数 / 总方案数) * 100% + (当前方案内部进度 / 总方案数)
    - 例如：10 个方案，第 3 个方案执行到 50%
      总进度 = (2/10)*100% + (50%/10) = 20% + 5% = 25%

    Args:
        max_workers: 并行评估的最大 worker 数量（1 表示串行）
    """

    # 检查取消
    if cancel_check and cancel_check():
        logger.info("任务在开始前被取消")
        raise TaskCancelledException("任务已取消")

    max_chips = search_constraints.get("max_chips", total_chips)
    # 注意：已移除 top_k 限制，现在返回所有可行方案

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

    # 初始化子任务进度跟踪（所有方案初始为 pending）
    sub_tasks = []
    for i, candidate in enumerate(candidates):
        sub_tasks.append({
            "candidate_index": i,
            "parallelism": candidate,
            "status": "pending",
            "progress": 0,
            "chips": calculate_required_chips(candidate, model_config),
        })

    # 定义一个辅助函数，用于推送子任务状态
    def update_sub_task(index: int, status: str, progress: int):
        sub_tasks[index]["status"] = status
        sub_tasks[index]["progress"] = progress

    if progress_callback:
        progress_callback(2, 100, f"开始评估 {total_candidates} 个方案（并行度={max_workers}）...")

    # 计算每个方案的 GEMM 进程数（动态调整避免系统过载）
    # 公式：max(1, cpu_count() // (2 * max_workers))
    # 例如：16核CPU，max_workers=4，则每个方案用 16//(2*4)=2 个进程
    max_gemm_processes = max(1, cpu_count() // (2 * max_workers))
    logger.info(f"每个评估任务使用 {max_gemm_processes} 个 GEMM 进程（总 worker={max_workers}，CPU核={cpu_count()}）")

    # 评估所有候选方案
    feasible_plans = []
    infeasible_plans = []

    # 预留 2% 给准备工作，98% 给方案评估
    progress_per_plan = 98.0 / total_candidates if total_candidates > 0 else 98.0

    # 线程安全的进度更新锁
    progress_lock = threading.Lock()

    # 跟踪已完成方案数
    completed_count = [0]  # 使用列表以便在闭包中修改

    # 创建内部进度回调，将方案的 0-100% 映射到该方案的进度份额
    def make_inner_callback(plan_index: int):
        def inner_callback(inner_percent: float, message: str):
            if progress_callback:
                with progress_lock:
                    # 更新子任务进度
                    update_sub_task(plan_index, "running", int(inner_percent))
                    # 计算总进度：已完成方案进度 + 当前方案内部进度
                    base_progress = 2 + completed_count[0] * progress_per_plan
                    current_plan_progress = (inner_percent / 100.0) * progress_per_plan
                    total_progress = base_progress + current_plan_progress
                    # 尝试传递 sub_tasks 信息给前端（使用 try-except 处理不同签名）
                    try:
                        progress_callback(int(total_progress), 100, f"[{completed_count[0] + 1}/{total_candidates}] {message}", sub_tasks=sub_tasks)
                    except TypeError:
                        # 如果 progress_callback 不支持 sub_tasks 参数，只传递基本参数
                        progress_callback(int(total_progress), 100, f"[{completed_count[0] + 1}/{total_candidates}] {message}")
        return inner_callback

    # 评估单个方案的worker函数
    def evaluate_plan_worker(plan_index: int, candidate: dict):
        """Worker函数：评估单个方案"""
        try:
            # 检查取消
            if cancel_check and cancel_check():
                logger.info(f"任务在评估方案 {plan_index + 1} 前被取消")
                raise TaskCancelledException("任务已取消")

            # 标记当前方案为 running
            with progress_lock:
                update_sub_task(plan_index, "running", 0)
                # 报告任务开始状态
                if progress_callback:
                    base_progress = 2 + completed_count[0] * progress_per_plan
                    try:
                        progress_callback(int(base_progress), 100, f"评估方案 {plan_index + 1}/{total_candidates}...", sub_tasks=sub_tasks)
                    except TypeError:
                        progress_callback(int(base_progress), 100, f"评估方案 {plan_index + 1}/{total_candidates}...")

            inner_progress = make_inner_callback(plan_index)

            result = _evaluate_single_plan(
                parallelism=candidate,
                total_chips=total_chips,
                model_config=model_config,
                inference_config=inference_config,
                topology=topology,
                hardware_config=hardware_config,
                progress_callback=inner_progress,
                cancel_check=cancel_check,
                enable_tile_search=enable_tile_search,
                enable_partition_search=enable_partition_search,
                max_simulated_tokens=max_simulated_tokens,
                max_gemm_processes=max_gemm_processes,  # 传递动态调整后的 GEMM 进程数
            )

            return (plan_index, result, None)

        except Exception as e:
            logger.error(f"评估方案 {plan_index + 1} 失败: {e}")
            return (plan_index, None, str(e))

    # 根据 max_workers 选择执行模式
    if max_workers == 1:
        # 串行模式（保持原有行为）
        for i, candidate in enumerate(candidates):
            plan_index, result, error = evaluate_plan_worker(i, candidate)

            if error:
                with progress_lock:
                    update_sub_task(plan_index, "failed", 0)
                completed_count[0] += 1
                continue

            if result and result.get("is_feasible", False):
                with progress_lock:
                    update_sub_task(plan_index, "completed", 100)
                    feasible_plans.append(result)
                # 每完成一个可行方案就立即保存
                if result_callback:
                    try:
                        result_callback(result)
                    except Exception as e:
                        logger.error(f"保存结果失败: {e}")
            else:
                with progress_lock:
                    update_sub_task(plan_index, "failed", 100)
                    if result:
                        infeasible_plans.append(result)

            completed_count[0] += 1

    else:
        # 并行模式
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_index = {
                executor.submit(evaluate_plan_worker, i, candidate): i
                for i, candidate in enumerate(candidates)
            }

            # 按完成顺序处理结果
            for future in as_completed(future_to_index):
                # 检查取消
                if cancel_check and cancel_check():
                    logger.info("任务被取消，停止评估")
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise TaskCancelledException("任务已取消")

                plan_index, result, error = future.result()

                if error:
                    with progress_lock:
                        update_sub_task(plan_index, "failed", 0)
                    completed_count[0] += 1
                    continue

                if result and result.get("is_feasible", False):
                    with progress_lock:
                        update_sub_task(plan_index, "completed", 100)
                        feasible_plans.append(result)
                    # 每完成一个可行方案就立即保存
                    if result_callback:
                        try:
                            result_callback(result)
                        except Exception as e:
                            logger.error(f"保存结果失败: {e}")
                else:
                    with progress_lock:
                        update_sub_task(plan_index, "failed", 100)
                        if result:
                            infeasible_plans.append(result)

                completed_count[0] += 1

    # 按得分排序（返回所有可行方案，不再截断为 top_k）
    feasible_plans.sort(key=lambda x: x.get("score", 0), reverse=True)

    if progress_callback:
        progress_callback(100, 100, f"评估完成，找到 {len(feasible_plans)} 个可行方案")

    return {
        "top_k_plans": feasible_plans,  # 返回所有可行方案，按分数排序
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
    """生成并行策略候选

    芯片数计算规则：
    - MoE 模型：chips = DP × TP（因为 DP×TP = MoE_TP×EP 约束）
    - 非 MoE 模型：chips = DP × TP × EP
    - PP 暂不启用，默认为1，不影响芯片数计算
    """

    candidates = []
    is_moe = model_config.get("moe_config") is not None

    # TODO: 实现智能的候选生成算法
    # 应该根据以下因素生成合法的因子分解：
    # 1. 模型大小和层数（决定 PP 的可能性）
    # 2. 注意力头数（决定 TP 的约束）
    # 3. 专家数量（决定 EP 的需求，如果是 MoE 模型）
    # 4. 可用芯片数（max_chips）
    # 5. 拓扑结构（同板/同机/跨机的带宽差异）

    # 候选生成策略
    if is_moe:
        # MoE 模型：实际芯片数 = DP × TP
        # EP 由 MoE 约束自动确定，这里也枚举不同的 EP
        for tp in [1, 2, 4, 8, 16, 32]:
            if tp > max_chips:
                continue
            for ep in [1, 2, 4, 8, 16, 32]:
                # 计算 DP，使得 DP * TP <= max_chips
                dp = max_chips // tp
                if dp > 0:
                    # MoE 约束：DP * TP = MoE_TP * EP
                    # 设置 moe_tp = (DP * TP) / EP
                    if (dp * tp) % ep == 0:
                        moe_tp = (dp * tp) // ep
                        candidate = {
                            "dp": dp,
                            "tp": tp,
                            "pp": 1,  # PP 暂不启用
                            "ep": ep,
                            "sp": 1,
                            "moe_tp": moe_tp,
                        }
                        candidates.append(candidate)
                        logger.debug(f"生成 MoE 候选: DP={dp}, TP={tp}, EP={ep}, MoE_TP={moe_tp}, 芯片数={dp*tp}")
    else:
        # 非 MoE 模型：实际芯片数 = DP × TP × EP
        for tp in [1, 2, 4, 8, 16, 32]:
            if tp > max_chips:
                continue
            for ep in [1, 2, 4, 8, 16, 32]:
                if tp * ep > max_chips:
                    continue
                # 计算 DP，使得 DP * TP * EP <= max_chips
                dp = max_chips // (tp * ep)
                if dp > 0:
                    candidates.append({
                        "dp": dp,
                        "tp": tp,
                        "pp": 1,  # PP 暂不启用
                        "ep": ep,
                        "sp": 1,
                    })

    logger.info(f"生成 {len(candidates)} 个候选方案 (is_moe={is_moe}, max_chips={max_chips})")

    # 打印前5个候选方案用于调试
    if len(candidates) > 0:
        logger.info(f"前5个候选方案示例:")
        for i, c in enumerate(candidates[:5]):
            logger.info(f"  方案{i+1}: DP={c['dp']}, TP={c['tp']}, EP={c['ep']}, MoE_TP={c.get('moe_tp', 'N/A')}, 芯片数={calculate_required_chips(c, model_config)}")

    # 打印候选方案的统计信息
    if len(candidates) > 0:
        ep_values = set(c['ep'] for c in candidates)
        moe_tp_values = set(c.get('moe_tp') for c in candidates if 'moe_tp' in c)
        logger.info(f"候选方案统计: 总数={len(candidates)}, EP取值={sorted(ep_values)}, MoE_TP取值={sorted(moe_tp_values) if moe_tp_values else 'N/A'}")

    return candidates


def _evaluate_single_plan(
    parallelism: dict,
    total_chips: int,
    model_config: dict,
    inference_config: dict,
    topology: dict,
    hardware_config: Optional[dict] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    enable_tile_search: bool = True,
    enable_partition_search: bool = False,
    max_simulated_tokens: int = 4,
    max_gemm_processes: Optional[int] = None,
) -> dict:
    """评估单个方案

    Args:
        parallelism: 并行策略配置
        total_chips: 可用芯片总数
        model_config: 模型配置
        inference_config: 推理配置
        topology: 拓扑配置
        hardware_config: 硬件配置（可选，未提供时自动从 topology 提取）
        progress_callback: 进度回调函数 (percent: float, message: str) -> None
            percent: 0-100 的进度百分比
            message: 进度描述信息
        cancel_check: 取消检查函数
        enable_tile_search: 是否启用 tile 搜索
        enable_partition_search: 是否启用分区搜索
        max_simulated_tokens: 最大模拟 token 数
        max_gemm_processes: GEMM 进程数限制
    """
    # 检查取消
    if cancel_check and cancel_check():
        logger.info("单个方案评估被取消")
        raise TaskCancelledException("任务已取消")

    # 直接访问字段，缺失时会立即 KeyError
    dp = parallelism["dp"]
    tp = parallelism["tp"]
    pp = parallelism["pp"]
    ep = parallelism["ep"]

    # 计算实际使用的芯片数
    required_chips = calculate_required_chips(parallelism, model_config)

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

    # 如果未提供硬件配置，从拓扑中提取
    if hardware_config is None:
        hardware_config = _extract_hardware_config(topology)

    # 调用真实的模拟器
    try:
        if progress_callback:
            progress_callback(10, "启动模拟器")

        sim_result = run_simulation(
            topology_dict=topology,
            model_dict=model_config,
            inference_dict=inference_config,
            parallelism_dict=parallelism,
            hardware_dict=hardware_config,
            progress_callback=progress_callback,
            enable_tile_search=enable_tile_search,
            max_simulated_tokens=max_simulated_tokens,
            max_gemm_processes=max_gemm_processes,
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
            return create_deepseek_v3_absorb(**full_config)
        else:
            # 标准 DeepSeek V3
            return create_deepseek_v3(**full_config)
    else:
        # 其他模型类型，使用基础工厂
        logger.warning(f"Unknown model type: {model_name}, using deepseek_v3 as fallback")
        return create_deepseek_v3(**full_config)


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
    mbu = stats.get("dynamicMbu", 0)

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

    # 修正 tps_per_chip 计算：除以实际使用的芯片数
    actual_chips = calculate_required_chips(parallelism, model_config)
    tps_per_chip = tps / actual_chips if actual_chips > 0 else 0

    # 从拓扑配置中提取硬件配置（只调用一次，后续复用）
    hardware_config = _extract_hardware_config(topology)
    # 新格式 v2.1.0+: hardware_params.chips 字典
    chips_dict = hardware_config.get("hardware_params", {}).get("chips", {})
    if chips_dict:
        first_chip_name = next(iter(chips_dict))
        chip_hw = chips_dict[first_chip_name]
        chip_type = chip_hw.get("name", "SG2260E")
    else:
        chip_type = "SG2260E"

    # 使用 PerformanceAnalyzer 获取详细的 layers 信息
    layers_info = {}
    flops = 0
    dram_occupy = 0

    try:
        # 创建模型实例
        model = _create_model_instance(model_config, inference_config, parallelism)

        # 获取架构预设
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

    # TODO: 综合得分计算公式需要重新设计，暂时设置为 0
    # 原公式：score = tps_per_chip * mfu（不太合理，后续讨论修改）
    score = 0

    # 成本评估（复用前面提取的 hardware_config 和 chip_type）
    cost_result = {}
    try:

        # 获取模型参数量
        num_parameters = model_config.get("num_parameters", 0)
        if num_parameters == 0:
            # 如果没有直接提供 num_parameters，尝试从其他参数估算
            hidden_size = model_config.get("hidden_size", 0)
            num_layers = model_config.get("num_layers", 0)
            intermediate_size = model_config.get("intermediate_size", 0)
            vocab_size = model_config.get("vocab_size", 0)
            if all([hidden_size, num_layers, intermediate_size]):
                # 简化估算：Attention + FFN 参数
                num_parameters = num_layers * (
                    4 * hidden_size * hidden_size +  # QKV + O
                    3 * hidden_size * intermediate_size  # Gate + Up + Down
                ) + vocab_size * hidden_size  # Embedding + LM Head

        # 计算成本（仅当有足够信息时）
        if num_parameters > 0 and avg_tpot > 0 and tps > 0:
            cost_result = evaluate_deployment_cost(
                chips=chips,
                chip_type=chip_type,
                num_parameters=num_parameters,
                tp=parallelism.get("tp", 1),
                tpot_ms=avg_tpot / 1000.0,  # 微秒转毫秒
                tps=tps,
                bytes_per_param=2  # FP16/BF16
            )
            logger.info(f"成本评估完成: 总成本=${cost_result.get('total_cost', 0):,.2f}, 单位成本=${cost_result.get('cost_per_million_tokens', 0):.4f}/M tokens")
        else:
            logger.warning(f"跳过成本计算: num_parameters={num_parameters}, avg_tpot={avg_tpot}, tps={tps}")
    except Exception as e:
        logger.warning(f"成本评估失败，跳过: {e}")
        cost_result = {}

    # 记录日志以便调试
    logger.info(f"保存结果: DP={parallelism['dp']}, TP={parallelism['tp']}, PP={parallelism.get('pp', 1)}, EP={parallelism.get('ep', 1)}, MoE_TP={parallelism.get('moe_tp', 'N/A')}, 芯片数={chips}")

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
        "ttft": ttft / 1000.0,  # 转换为毫秒
        "tpot": avg_tpot / 1000.0,  # 转换为毫秒
        "mfu": mfu,
        "mbu": mbu,
        "flops": flops,
        "dram_occupy": dram_occupy,
        "score": score,
        "layers": layers_info,  # 添加详细的 layers 信息
        "cost": cost_result,  # 成本评估结果
        "stats": stats,  # 完整的统计数据（用于前端展示）
        "gantt_chart": sim_result.get("ganttChart"),  # 甘特图数据（用于前端可视化）
    }


def _extract_hardware_config(topology: dict) -> dict:
    """从拓扑配置中提取硬件配置

    支持多种配置格式：
    1. hardware_config: 直接嵌入的硬件配置
    2. hardware_params.chip: 配置文件格式的芯片参数
    3. pods/racks/boards/chips: 展开后的拓扑结构

    Args:
        topology: 完整拓扑配置

    Returns:
        硬件配置字典，包含 chip, node, cluster 配置

    Raises:
        ValueError: 如果无法从拓扑中提取芯片配置
    """
    # 格式1：直接包含 hardware_config
    if "hardware_config" in topology:
        return topology["hardware_config"]

    # 格式2：配置文件格式 hardware_params.chips (新格式)
    if "hardware_params" in topology:
        hardware_params = topology["hardware_params"]
        chips_dict = hardware_params.get("chips", {})

        logger.warning(f"[DEBUG] _extract_hardware_config: hardware_params keys = {hardware_params.keys()}")
        logger.warning(f"[DEBUG] _extract_hardware_config: chips_dict keys = {chips_dict.keys() if chips_dict else 'EMPTY'}")
        if chips_dict:
            first_chip_name = next(iter(chips_dict))
            logger.warning(f"[DEBUG] _extract_hardware_config: first chip '{first_chip_name}' = {chips_dict[first_chip_name]}")

        if chips_dict:
            result = {
                "hardware_params": {
                    "chips": chips_dict,
                    "interconnect": hardware_params.get("interconnect", {}),
                    "comm_latency_config": hardware_params.get("comm_latency_config", {}),
                }
            }
            logger.warning(f"[DEBUG] _extract_hardware_config: returning result with keys = {result.keys()}")
            return result

    # 格式3：展开后的 pods/racks/boards/chips 结构 -> 转换为新格式
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
                                chip_name = chip.get("name", "SG2260E")
                                # 构建新格式的 hardware_params.chips 结构
                                return {
                                    "hardware_params": {
                                        "chips": {
                                            chip_name: {
                                                "name": chip_name,
                                                "num_cores": chip.get("num_cores", 64),
                                                "compute_tflops_fp8": chip.get("compute_tflops_fp8", 0),
                                                "compute_tflops_bf16": chip.get("compute_tflops_bf16", 0),
                                                "memory_capacity_gb": chip.get("memory_capacity_gb", chip.get("memory_gb", 80)),
                                                "memory_bandwidth_gbps": chip.get("memory_bandwidth_gbps", 3000),
                                                "memory_bandwidth_utilization": chip.get("memory_bandwidth_utilization", 0.85),
                                                "lmem_capacity_mb": chip.get("lmem_capacity_mb", 0),
                                                "lmem_bandwidth_gbps": chip.get("lmem_bandwidth_gbps", 0),
                                                "cube_m": chip.get("cube_m"),
                                                "cube_k": chip.get("cube_k"),
                                                "cube_n": chip.get("cube_n"),
                                                "sram_size_kb": chip.get("sram_size_kb"),
                                                "sram_utilization": chip.get("sram_utilization"),
                                                "lane_num": chip.get("lane_num"),
                                                "align_bytes": chip.get("align_bytes"),
                                                "compute_dma_overlap_rate": chip.get("compute_dma_overlap_rate"),
                                            }
                                        },
                                        "interconnect": {},
                                        "comm_latency_config": {},
                                    }
                                }

    # 如果找不到芯片配置，抛出错误
    raise ValueError("无法从拓扑配置中提取芯片硬件配置，请确保拓扑中包含有效的芯片数据或 hardware_params.chips 配置")
