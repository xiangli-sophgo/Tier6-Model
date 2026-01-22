"""
任务队列管理器

使用 ThreadPoolExecutor 运行后台评估任务，支持并发执行。
"""

import uuid
import traceback
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Dict, Optional, Callable, Any
from sqlalchemy.orm import Session

from .db_models import Experiment, EvaluationTask, EvaluationResult, TaskStatus
from .database import SessionLocal
from .deployment_evaluator import evaluate_deployment

logger = logging.getLogger(__name__)

# 全局任务执行器（最多 4 个并发任务）
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="eval_worker")

# 正在运行的任务 Future 映射
_running_tasks: Dict[str, Future] = {}

# WebSocket 广播回调（由 websocket_manager 设置）
_ws_broadcast_callback: Optional[Callable[[str, dict], None]] = None


def set_ws_broadcast_callback(callback: Callable[[str, dict], None]):
    """设置 WebSocket 广播回调"""
    global _ws_broadcast_callback
    _ws_broadcast_callback = callback


def _broadcast_task_update(task_id: str, data: dict):
    """广播任务更新（通过 WebSocket）"""
    if _ws_broadcast_callback:
        try:
            _ws_broadcast_callback(task_id, data)
        except Exception as e:
            logger.error(f"WebSocket broadcast failed: {e}")


def _update_task_status(
    task_id: str,
    status: TaskStatus,
    progress: float = 0.0,
    message: Optional[str] = None,
    error: Optional[str] = None,
    search_stats: Optional[dict] = None,
    top_plan: Optional[dict] = None,
):
    """更新任务状态到数据库并广播"""
    db = SessionLocal()
    try:
        task = db.query(EvaluationTask).filter(EvaluationTask.task_id == task_id).first()
        if not task:
            logger.error(f"Task {task_id} not found")
            return

        task.status = status
        task.progress = progress
        if message:
            task.message = message
        if error:
            task.error = error
        if search_stats:
            task.search_stats = search_stats
        if top_plan:
            # 将最优方案保存到search_stats中
            if not task.search_stats:
                task.search_stats = {}
            task.search_stats['top_plan'] = top_plan

        if status == TaskStatus.RUNNING and not task.started_at:
            task.started_at = datetime.utcnow()
        elif status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
            task.completed_at = datetime.utcnow()

        db.commit()
        db.refresh(task)

        # 广播更新
        _broadcast_task_update(task_id, {
            "task_id": task_id,
            "status": status.value,
            "progress": progress,
            "message": message or "",
            "error": error,
            "search_stats": search_stats,
        })

    except Exception as e:
        logger.error(f"Failed to update task status: {e}")
        db.rollback()
    finally:
        db.close()


def _save_results(task_id: str, results: list, db: Session):
    """保存评估结果到数据库（对齐 DS_TPU 数据格式）"""
    task = db.query(EvaluationTask).filter(EvaluationTask.task_id == task_id).first()
    if not task:
        raise ValueError(f"Task {task_id} not found")

    for result_data in results:
        result = EvaluationResult(
            task_id=task.id,
            dp=result_data["parallelism"]["dp"],
            tp=result_data["parallelism"]["tp"],
            pp=result_data["parallelism"].get("pp", 1),
            ep=result_data["parallelism"].get("ep", 1),
            sp=result_data["parallelism"].get("sp", 1),
            moe_tp=result_data["parallelism"].get("moe_tp"),
            chips=result_data["chips"],
            # DS_TPU 字段
            total_elapse_us=result_data["total_elapse_us"],
            total_elapse_ms=result_data["total_elapse_ms"],
            comm_elapse_us=result_data["comm_elapse_us"],
            tps=result_data["tps"],
            tps_per_batch=result_data["tps_per_batch"],
            tps_per_chip=result_data["tps_per_chip"],
            mfu=result_data["mfu"],
            flops=result_data["flops"],
            dram_occupy=result_data["dram_occupy"],
            score=result_data["score"],
            is_feasible=1 if result_data.get("is_feasible", True) else 0,
            infeasible_reason=result_data.get("infeasible_reason"),
            full_result=result_data,
        )
        db.add(result)

    # 更新实验统计
    task.experiment.completed_tasks += 1
    db.commit()


def _execute_evaluation(
    task_id: str,
    topology: dict,
    model_config: dict,
    inference_config: dict,
    search_mode: str,
    manual_parallelism: Optional[dict],
    search_constraints: Optional[dict],
):
    """执行评估任务（在工作线程中运行）"""
    logger.info(f"[Task {task_id}] Starting evaluation")
    _update_task_status(task_id, TaskStatus.RUNNING, 0.0, "开始评估...")

    db = SessionLocal()
    try:
        # 调用评估函数
        def progress_callback(current: int, total: int, message: str = ""):
            progress = (current / total * 100) if total > 0 else 0
            _update_task_status(task_id, TaskStatus.RUNNING, progress, message)

        result = evaluate_deployment(
            topology=topology,
            model_config=model_config,
            inference_config=inference_config,
            search_mode=search_mode,
            manual_parallelism=manual_parallelism,
            search_constraints=search_constraints,
            progress_callback=progress_callback,
        )

        # 保存结果
        if result["top_k_plans"]:
            _save_results(task_id, result["top_k_plans"], db)

        # 保存搜索统计
        search_stats = result.get("search_stats", {})

        # 获取最优方案信息
        top_plan = None
        if result["top_k_plans"]:
            best = result["top_k_plans"][0]
            top_plan = {
                "parallelism": {
                    "dp": best.get("dp", best.get("parallelism", {}).get("dp", 1)),
                    "tp": best.get("tp", best.get("parallelism", {}).get("tp", 1)),
                    "pp": best.get("pp", best.get("parallelism", {}).get("pp", 1)),
                    "ep": best.get("ep", best.get("parallelism", {}).get("ep", 1)),
                    "sp": best.get("sp", best.get("parallelism", {}).get("sp", 1)),
                },
                "throughput": float(best.get("throughput", 0)),
                "tps_per_chip": float(best.get("tps_per_chip", 0)),
                "ttft": float(best.get("ttft", 0)),
                "tpot": float(best.get("tpot", 0)),
                "mfu": float(best.get("mfu", 0)),
                "mbu": float(best.get("mbu", 0)),
                "score": float(best.get("score", 0)),
            }

        _update_task_status(
            task_id,
            TaskStatus.COMPLETED,
            100.0,
            f"评估完成，找到 {len(result['top_k_plans'])} 个方案",
            search_stats=search_stats,
            top_plan=top_plan,
        )
        logger.info(f"[Task {task_id}] Completed successfully")

    except Exception as e:
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        logger.error(f"[Task {task_id}] Failed: {error_msg}")
        _update_task_status(task_id, TaskStatus.FAILED, 0.0, "评估失败", error_msg)
    finally:
        db.close()
        # 清理运行中任务映射
        _running_tasks.pop(task_id, None)


def create_and_submit_task(
    experiment_name: str,
    description: str,
    topology: dict,
    model_config: dict,
    inference_config: dict,
    search_mode: str,
    benchmark_name: Optional[str] = None,
    topology_config_name: Optional[str] = None,
    manual_parallelism: Optional[dict] = None,
    search_constraints: Optional[dict] = None,
) -> str:
    """
    创建评估任务并提交到后台执行

    Args:
        experiment_name: 实验名称
        description: 实验描述
        topology: 完整拓扑数据（包含物理拓扑 + protocol_config + network_config + chip_latency_config）
        model_config: 模型配置
        inference_config: 推理配置
        search_mode: 搜索模式 ('manual' or 'auto')
        benchmark_name: Benchmark 配置文件名称（可选）
        topology_config_name: 拓扑配置文件名称（可选）
        manual_parallelism: 手动并行策略（可选）
        search_constraints: 搜索约束（可选）

    Returns:
        task_id: 任务 UUID
    """
    db = SessionLocal()
    try:
        # 查找或创建实验
        experiment = db.query(Experiment).filter(Experiment.name == experiment_name).first()
        if not experiment:
            experiment = Experiment(
                name=experiment_name,
                description=description,
            )
            db.add(experiment)
            db.flush()

        # 构建完整配置快照（topology 已包含所有硬件和延迟配置）
        config_snapshot = {
            "model": model_config,
            "inference": inference_config,
            "topology": topology,  # 包含 pods/racks/boards/chips/connections + protocol_config + network_config + chip_latency_config
        }

        # 创建任务记录
        task_id = str(uuid.uuid4())
        task = EvaluationTask(
            task_id=task_id,
            experiment_id=experiment.id,
            status=TaskStatus.PENDING,
            config_snapshot=config_snapshot,
            benchmark_name=benchmark_name,
            topology_config_name=topology_config_name,
            search_mode=search_mode,
            manual_parallelism=manual_parallelism,
            search_constraints=search_constraints,
        )
        db.add(task)
        experiment.total_tasks += 1
        db.commit()

        # 提交到线程池
        future = _executor.submit(
            _execute_evaluation,
            task_id,
            topology,
            model_config,
            inference_config,
            search_mode,
            manual_parallelism,
            search_constraints,
        )
        _running_tasks[task_id] = future

        logger.info(f"Task {task_id} submitted to executor")
        return task_id

    except Exception as e:
        db.rollback()
        logger.error(f"Failed to create task: {e}")
        raise
    finally:
        db.close()


def get_task_status(task_id: str) -> Optional[dict]:
    """获取任务状态"""
    db = SessionLocal()
    try:
        task = db.query(EvaluationTask).filter(EvaluationTask.task_id == task_id).first()
        if not task:
            return None

        return {
            "task_id": task.task_id,
            "status": task.status.value,
            "progress": task.progress,
            "message": task.message,
            "error": task.error,
            "search_stats": task.search_stats,
            "created_at": task.created_at.isoformat() if task.created_at else None,
            "started_at": task.started_at.isoformat() if task.started_at else None,
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
            "experiment_name": task.experiment.name,
        }
    finally:
        db.close()


def cancel_task(task_id: str) -> bool:
    """取消任务（尽力而为，无法中断正在运行的评估）"""
    future = _running_tasks.get(task_id)
    if future and not future.done():
        future.cancel()

    _update_task_status(task_id, TaskStatus.CANCELLED, 0.0, "任务已取消")
    return True


def get_running_tasks() -> list:
    """获取所有运行中的任务"""
    db = SessionLocal()
    try:
        tasks = db.query(EvaluationTask).filter(
            EvaluationTask.status.in_([TaskStatus.PENDING, TaskStatus.RUNNING])
        ).all()

        return [
            {
                "task_id": task.task_id,
                "status": task.status.value,
                "progress": task.progress,
                "created_at": task.created_at.isoformat() if task.created_at else None,
                "experiment_name": task.experiment.name,
            }
            for task in tasks
        ]
    finally:
        db.close()


def get_task_results(task_id: str) -> Optional[dict]:
    """获取任务的完整结果"""
    db = SessionLocal()
    try:
        task = db.query(EvaluationTask).filter(EvaluationTask.task_id == task_id).first()
        if not task:
            return None

        results = db.query(EvaluationResult).filter(
            EvaluationResult.task_id == task.id,
            EvaluationResult.is_feasible == 1
        ).order_by(EvaluationResult.score.desc()).all()

        infeasible_results = db.query(EvaluationResult).filter(
            EvaluationResult.task_id == task.id,
            EvaluationResult.is_feasible == 0
        ).all()

        return {
            "task_id": task.task_id,
            "status": task.status.value,
            "experiment_name": task.experiment.name,
            "top_k_plans": [r.full_result for r in results],
            "infeasible_plans": [r.full_result for r in infeasible_results],
            "search_stats": task.search_stats,
        }
    finally:
        db.close()


def delete_task(task_id: str) -> bool:
    """删除任务（及其结果）"""
    db = SessionLocal()
    try:
        task = db.query(EvaluationTask).filter(EvaluationTask.task_id == task_id).first()
        if not task:
            return False

        # 删除任务会级联删除结果
        db.delete(task)
        db.commit()
        return True
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to delete task: {e}")
        return False
    finally:
        db.close()
