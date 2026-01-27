"""
任务队列管理器

使用 ThreadPoolExecutor 运行后台评估任务，支持并发执行。
集成 WebSocket 订阅功能，支持实时进度推送。
"""

import uuid
import asyncio
import traceback
import logging
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Dict, List, Optional, Callable, Any
from sqlalchemy.orm import Session

from ..core.database import Experiment, EvaluationTask, EvaluationResult, TaskStatus, get_db_session
from .deployment import evaluate_deployment
from ..config import get_max_global_workers

logger = logging.getLogger(__name__)


class GlobalWorkerPool:
    """
    全局 worker 资源池管理器（单例模式）

    管理所有任务共享的 worker 资源分配和释放。
    集成 WebSocket 订阅功能。
    """

    _instance = None
    _lock = threading.RLock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        with self._lock:
            if not self._initialized:
                self._max_global_workers = get_max_global_workers()
                logger.info(f"初始化全局资源池，最大 worker 数量: {self._max_global_workers}")
                self._allocated_workers: Dict[str, int] = {}
                self._task_executors: Dict[str, ThreadPoolExecutor] = {}
                # WebSocket 订阅者管理
                self._global_subscribers: List[asyncio.Queue] = []
                self._main_loop: Optional[asyncio.AbstractEventLoop] = None
                self._initialized = True

    def set_main_loop(self, loop: asyncio.AbstractEventLoop):
        """设置主事件循环（在应用启动时调用）"""
        with self._lock:
            self._main_loop = loop
            logger.info("主事件循环已注册")

    def subscribe_global(self) -> asyncio.Queue:
        """订阅所有任务状态更新（全局）"""
        queue = asyncio.Queue()
        with self._lock:
            self._global_subscribers.append(queue)
            logger.info(f"全局订阅者已添加，当前订阅者数量: {len(self._global_subscribers)}")
        return queue

    def unsubscribe_global(self, queue: asyncio.Queue):
        """取消全局订阅"""
        with self._lock:
            if queue in self._global_subscribers:
                self._global_subscribers.remove(queue)
                logger.info(f"全局订阅者已移除，当前订阅者数量: {len(self._global_subscribers)}")

    def notify_subscribers_sync(self, task_id: str, data: dict):
        """
        线程安全的同步通知方法
        从普通线程中调用，自动调度到事件循环
        """
        with self._lock:
            subscribers = list(self._global_subscribers)
            main_loop = self._main_loop

        if not subscribers:
            return

        print(f"[DEBUG] notify_subscribers_sync: task_id={task_id}, progress={data.get('progress')}, subscribers={len(subscribers)}")

        message = {"type": "task_update", **data}

        try:
            # 尝试获取当前运行的事件循环（在异步上下文中）
            loop = asyncio.get_running_loop()
            loop.create_task(self._push_to_queues(subscribers, message))
        except RuntimeError:
            # 没有运行中的事件循环（在子线程中），使用保存的主事件循环
            if main_loop and main_loop.is_running():
                asyncio.run_coroutine_threadsafe(self._push_to_queues(subscribers, message), main_loop)
            else:
                logger.warning(f"无法通知订阅者（任务 {task_id}）: 主事件循环未设置或未运行")

    async def _push_to_queues(self, subscribers: List[asyncio.Queue], message: dict):
        """向所有订阅者队列推送消息"""
        print(f"[DEBUG] _push_to_queues: pushing to {len(subscribers)} queues")
        for queue in subscribers:
            try:
                await queue.put(message)
            except Exception as e:
                logger.error(f"推送消息到队列失败: {e}")

    def allocate_workers(self, task_id: str, requested: int) -> int:
        """为任务分配 worker 资源"""
        with self._lock:
            total_allocated = sum(self._allocated_workers.values())
            available = self._max_global_workers - total_allocated
            actual = max(1, min(requested, available))
            self._allocated_workers[task_id] = actual
            logger.info(f"[Task {task_id}] 分配 {actual} workers (请求 {requested}, 可用 {available})")
            return actual

    def release_workers(self, task_id: str):
        """释放任务的 worker 资源"""
        with self._lock:
            if task_id in self._allocated_workers:
                released = self._allocated_workers.pop(task_id)
                logger.info(f"[Task {task_id}] 释放 {released} workers")
            if task_id in self._task_executors:
                del self._task_executors[task_id]

    def register_executor(self, task_id: str, executor: ThreadPoolExecutor):
        """注册任务的 executor"""
        with self._lock:
            self._task_executors[task_id] = executor

    def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        with self._lock:
            executor = self._task_executors.get(task_id)
            if executor:
                executor.shutdown(wait=False, cancel_futures=True)
                return True
            return False


# 全局单例
_worker_pool = GlobalWorkerPool()


def set_main_loop(loop: asyncio.AbstractEventLoop):
    """设置主事件循环"""
    _worker_pool.set_main_loop(loop)


def subscribe_global() -> asyncio.Queue:
    """订阅全局任务更新"""
    return _worker_pool.subscribe_global()


def unsubscribe_global(queue: asyncio.Queue):
    """取消全局订阅"""
    _worker_pool.unsubscribe_global(queue)


def _broadcast_task_update(task_id: str, data: dict):
    """广播任务更新到所有订阅者"""
    _worker_pool.notify_subscribers_sync(task_id, data)


def _update_task_status(
    task_id: str,
    status: TaskStatus,
    progress: float,
    message: Optional[str] = None,
    error: Optional[str] = None,
    search_stats: Optional[dict] = None,
    top_plan: Optional[dict] = None,
):
    """更新任务状态到数据库并广播"""
    print(f"[DEBUG] _update_task_status: task_id={task_id}, status={status.value}, progress={progress}")

    with get_db_session() as db:
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


def _save_results(task_id: str, results: list, db: Session):
    """保存评估结果到数据库"""
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

    task.experiment.completed_tasks += 1
    db.commit()


def _execute_evaluation(
    task_id: str,
    max_workers: int,
    topology: dict,
    model_config: dict,
    inference_config: dict,
    search_mode: str,
    manual_parallelism: Optional[dict],
    search_constraints: Optional[dict],
    enable_tile_search: bool = True,
    enable_partition_search: bool = False,
    max_simulated_tokens: int = 4,
):
    """执行评估任务（在独立线程中运行）"""
    logger.info(f"[Task {task_id}] Starting evaluation with max_workers={max_workers}")

    # 分配 worker 资源
    actual_workers = _worker_pool.allocate_workers(task_id, max_workers)
    _update_task_status(task_id, TaskStatus.RUNNING, 0.0, f"开始评估（使用 {actual_workers} workers）...")

    # 创建任务专属的 ThreadPoolExecutor
    executor = ThreadPoolExecutor(max_workers=actual_workers, thread_name_prefix=f"task_{task_id[:8]}")
    _worker_pool.register_executor(task_id, executor)

    try:
        # 调用评估函数
        def progress_callback(current: int, total: int, message: str = ""):
            progress = (current / total * 100) if total > 0 else 0
            print(f"[DEBUG MANAGER] progress_callback: current={current}, total={total}, progress={progress}, message={message}")
            _update_task_status(task_id, TaskStatus.RUNNING, progress, message)

        result = evaluate_deployment(
            topology=topology,
            model_config=model_config,
            inference_config=inference_config,
            search_mode=search_mode,
            manual_parallelism=manual_parallelism,
            search_constraints=search_constraints,
            progress_callback=progress_callback,
            enable_tile_search=enable_tile_search,
            enable_partition_search=enable_partition_search,
            max_simulated_tokens=max_simulated_tokens,
        )

        # 保存结果
        with get_db_session() as db:
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
        # 清理资源
        executor.shutdown(wait=True)
        _worker_pool.release_workers(task_id)


def create_and_submit_task(
    experiment_name: str,
    description: str,
    topology: dict,
    model_config: dict,
    inference_config: dict,
    search_mode: str,
    max_workers: int = 4,
    benchmark_name: Optional[str] = None,
    topology_config_name: Optional[str] = None,
    manual_parallelism: Optional[dict] = None,
    search_constraints: Optional[dict] = None,
    enable_tile_search: bool = True,
    enable_partition_search: bool = False,
    max_simulated_tokens: int = 4,
) -> str:
    """
    创建并提交评估任务

    Returns:
        task_id: 任务ID
    """
    task_id = str(uuid.uuid4())

    with get_db_session() as db:
        # 查找或创建实验
        experiment = db.query(Experiment).filter(
            Experiment.name == experiment_name
        ).first()

        if not experiment:
            experiment = Experiment(
                name=experiment_name,
                description=description,
            )
            db.add(experiment)
            db.commit()
            db.refresh(experiment)

        # 创建评估任务
        task = EvaluationTask(
            task_id=task_id,
            experiment_id=experiment.id,
            status=TaskStatus.PENDING,
            progress=0.0,
            message="等待执行",
            benchmark_name=benchmark_name,
            topology_config_name=topology_config_name,
            search_mode=search_mode,
            manual_parallelism=manual_parallelism,
            search_constraints=search_constraints,
            config_snapshot={
                "topology": topology,
                "model": model_config,
                "inference": inference_config,
                "max_workers": max_workers,
                "enable_tile_search": enable_tile_search,
                "enable_partition_search": enable_partition_search,
                "max_simulated_tokens": max_simulated_tokens,
            },
        )
        db.add(task)
        db.commit()

    # 在后台线程中执行任务
    thread = threading.Thread(
        target=_execute_evaluation,
        args=(
            task_id,
            max_workers,
            topology,
            model_config,
            inference_config,
            search_mode,
            manual_parallelism,
            search_constraints,
            enable_tile_search,
            enable_partition_search,
            max_simulated_tokens,
        ),
        daemon=True,
    )
    thread.start()

    logger.info(f"[Task {task_id}] Submitted for experiment '{experiment_name}'")
    return task_id


def cancel_task(task_id: str) -> bool:
    """取消任务"""
    with get_db_session() as db:
        task = db.query(EvaluationTask).filter(EvaluationTask.task_id == task_id).first()
        if task and task.status == TaskStatus.RUNNING:
            _worker_pool.cancel_task(task_id)
            _update_task_status(task_id, TaskStatus.CANCELLED, 0.0, "任务已取消")
            return True
    return False


def get_task_status(task_id: str) -> Optional[dict]:
    """获取任务状态"""
    with get_db_session() as db:
        task = db.query(EvaluationTask).filter(EvaluationTask.task_id == task_id).first()
        if task:
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
            }
    return None


def get_task_results(task_id: str) -> Optional[dict]:
    """获取任务结果"""
    with get_db_session() as db:
        task = db.query(EvaluationTask).filter(EvaluationTask.task_id == task_id).first()
        if not task:
            return None

        results = db.query(EvaluationResult).filter(EvaluationResult.task_id == task.id).all()
        return {
            "task_id": task.task_id,
            "experiment_name": task.experiment.name,
            "status": task.status.value,
            "top_k_plans": [r.full_result for r in results],
            "search_stats": task.search_stats,
        }


def delete_task(task_id: str) -> bool:
    """删除任务及其结果"""
    with get_db_session() as db:
        task = db.query(EvaluationTask).filter(EvaluationTask.task_id == task_id).first()
        if not task:
            return False

        # 先删除关联的结果
        db.query(EvaluationResult).filter(EvaluationResult.task_id == task.id).delete()
        # 再删除任务
        db.delete(task)
        db.commit()
        logger.info(f"[Task {task_id}] Deleted")
        return True


def get_running_tasks() -> List[dict]:
    """获取所有运行中的任务"""
    with get_db_session() as db:
        tasks = db.query(EvaluationTask).filter(
            EvaluationTask.status == TaskStatus.RUNNING
        ).all()
        return [
            {
                "task_id": task.task_id,
                "experiment_name": task.experiment.name,
                "status": task.status.value,
                "progress": task.progress,
                "message": task.message,
                "started_at": task.started_at.isoformat() if task.started_at else None,
            }
            for task in tasks
        ]


def get_executor_info() -> dict:
    """获取全局资源池信息"""
    with _worker_pool._lock:
        allocated = sum(_worker_pool._allocated_workers.values())
        active_count = len(_worker_pool._allocated_workers)
    return {
        "max_workers": _worker_pool._max_global_workers,
        "running_tasks": allocated,
        "active_tasks": active_count,
    }
