"""任务管理模块

提供异步任务队列和执行管理，集成 WebSocket 广播。
"""

from __future__ import annotations

import logging
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    """任务状态"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskInfo:
    """任务信息

    Attributes:
        task_id: 任务 ID
        name: 任务名称
        status: 任务状态
        progress: 进度 (0-1)
        created_at: 创建时间
        started_at: 开始时间
        completed_at: 完成时间
        result: 执行结果
        error: 错误信息
        config_snapshot: 配置快照
    """

    task_id: str
    name: str
    status: TaskStatus = TaskStatus.PENDING
    progress: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    result: Any = None
    error: str | None = None
    config_snapshot: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "taskId": self.task_id,
            "name": self.name,
            "status": self.status.value,
            "progress": self.progress,
            "createdAt": self.created_at.isoformat() if self.created_at else None,
            "startedAt": self.started_at.isoformat() if self.started_at else None,
            "completedAt": self.completed_at.isoformat() if self.completed_at else None,
            "error": self.error,
        }


class TaskManager:
    """任务管理器

    管理异步任务的提交和执行，集成 WebSocket 广播。
    """

    def __init__(
        self,
        max_workers: int = 4,
        max_queued: int = 100,
        enable_ws_broadcast: bool = True,
    ) -> None:
        """初始化

        Args:
            max_workers: 最大工作线程数
            max_queued: 最大排队任务数
            enable_ws_broadcast: 是否启用 WebSocket 广播
        """
        self.max_workers = max_workers
        self.max_queued = max_queued
        self._enable_ws_broadcast = enable_ws_broadcast
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._tasks: dict[str, TaskInfo] = {}
        self._futures: dict[str, Future] = {}
        self._callbacks: dict[str, list[Callable[[TaskInfo], None]]] = {}
        self._global_callbacks: list[Callable[[TaskInfo], None]] = []

    def submit(
        self,
        name: str,
        func: Callable[..., Any],
        *args: Any,
        config_snapshot: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> str:
        """提交任务

        Args:
            name: 任务名称
            func: 任务函数
            *args: 位置参数
            config_snapshot: 配置快照
            **kwargs: 关键字参数

        Returns:
            str: 任务 ID
        """
        # 检查队列是否已满
        pending_count = sum(
            1 for t in self._tasks.values() if t.status == TaskStatus.PENDING
        )
        if pending_count >= self.max_queued:
            raise RuntimeError(f"Task queue is full (max: {self.max_queued})")

        # 创建任务
        task_id = str(uuid.uuid4())
        task = TaskInfo(
            task_id=task_id,
            name=name,
            config_snapshot=config_snapshot or {},
        )
        self._tasks[task_id] = task
        self._callbacks[task_id] = []

        # 提交到线程池
        future = self._executor.submit(self._run_task, task_id, func, *args, **kwargs)
        self._futures[task_id] = future

        return task_id

    def _run_task(
        self,
        task_id: str,
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """执行任务

        Args:
            task_id: 任务 ID
            func: 任务函数
            *args: 位置参数
            **kwargs: 关键字参数

        Returns:
            Any: 执行结果
        """
        task = self._tasks.get(task_id)
        if not task:
            return None

        # 更新状态为运行中
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now()
        self._notify_callbacks(task_id)

        # 注入 progress_callback，让任务函数可以上报进度
        def progress_callback(progress: float) -> None:
            self.update_progress(task_id, progress)

        kwargs["progress_callback"] = progress_callback

        try:
            # 执行任务
            result = func(*args, **kwargs)

            # 更新状态为完成
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            task.result = result
            task.progress = 1.0
            self._notify_callbacks(task_id)

            return result

        except Exception as e:
            # 更新状态为失败
            task.status = TaskStatus.FAILED
            task.completed_at = datetime.now()
            task.error = str(e)
            self._notify_callbacks(task_id)

            raise

    def get_task(self, task_id: str) -> TaskInfo | None:
        """获取任务信息

        Args:
            task_id: 任务 ID

        Returns:
            TaskInfo | None: 任务信息
        """
        return self._tasks.get(task_id)

    def get_result(self, task_id: str, timeout: float | None = None) -> Any:
        """获取任务结果 (阻塞)

        Args:
            task_id: 任务 ID
            timeout: 超时时间 (秒)

        Returns:
            Any: 任务结果
        """
        future = self._futures.get(task_id)
        if not future:
            raise KeyError(f"Task not found: {task_id}")

        return future.result(timeout=timeout)

    def cancel(self, task_id: str) -> bool:
        """取消任务

        Args:
            task_id: 任务 ID

        Returns:
            bool: 是否成功取消
        """
        future = self._futures.get(task_id)
        task = self._tasks.get(task_id)

        if not future or not task:
            return False

        # 尝试取消
        if future.cancel():
            task.status = TaskStatus.CANCELLED
            task.completed_at = datetime.now()
            self._notify_callbacks(task_id)
            return True

        return False

    def update_progress(self, task_id: str, progress: float) -> None:
        """更新任务进度

        Args:
            task_id: 任务 ID
            progress: 进度 (0-1)
        """
        task = self._tasks.get(task_id)
        if task:
            task.progress = max(task.progress, min(1.0, progress))
            self._notify_callbacks(task_id)

    def add_callback(
        self,
        task_id: str,
        callback: Callable[[TaskInfo], None],
    ) -> None:
        """添加回调函数

        Args:
            task_id: 任务 ID
            callback: 回调函数
        """
        if task_id in self._callbacks:
            self._callbacks[task_id].append(callback)

    def _notify_callbacks(self, task_id: str) -> None:
        """通知回调函数

        Args:
            task_id: 任务 ID
        """
        task = self._tasks.get(task_id)
        if not task:
            return

        # 通知任务特定回调
        for callback in self._callbacks.get(task_id, []):
            try:
                callback(task)
            except Exception:
                pass  # 忽略回调错误

        # 通知全局回调
        for callback in self._global_callbacks:
            try:
                callback(task)
            except Exception:
                pass

        # WebSocket 广播
        if self._enable_ws_broadcast:
            self._broadcast_task_update(task)

    def _broadcast_task_update(self, task: TaskInfo) -> None:
        """通过 WebSocket 广播任务更新

        Args:
            task: 任务信息
        """
        try:
            from perf_model.L0_entry.websocket import get_ws_manager

            ws_manager = get_ws_manager()
            ws_manager.broadcast_task_update(
                task_id=task.task_id,
                status=task.status.value,
                progress=task.progress,
                error=task.error,
                result=task.result if task.status == TaskStatus.COMPLETED else None,
            )
        except Exception as e:
            logger.debug(f"Failed to broadcast task update: {e}")

    def add_global_callback(self, callback: Callable[[TaskInfo], None]) -> None:
        """添加全局回调函数（所有任务状态变更时触发）

        Args:
            callback: 回调函数
        """
        self._global_callbacks.append(callback)

    def remove_global_callback(self, callback: Callable[[TaskInfo], None]) -> None:
        """移除全局回调函数

        Args:
            callback: 回调函数
        """
        try:
            self._global_callbacks.remove(callback)
        except ValueError:
            pass

    def list_tasks(
        self,
        status: TaskStatus | None = None,
        limit: int = 100,
    ) -> list[TaskInfo]:
        """列出任务

        Args:
            status: 过滤状态
            limit: 返回数量限制

        Returns:
            list[TaskInfo]: 任务列表
        """
        tasks = list(self._tasks.values())

        if status:
            tasks = [t for t in tasks if t.status == status]

        # 按创建时间倒序
        tasks.sort(key=lambda t: t.created_at, reverse=True)

        return tasks[:limit]

    def cleanup(self, max_age_hours: int = 24) -> int:
        """清理旧任务

        Args:
            max_age_hours: 最大保留时间 (小时)

        Returns:
            int: 清理的任务数
        """
        now = datetime.now()
        to_remove = []

        for task_id, task in self._tasks.items():
            if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
                age = (now - task.created_at).total_seconds() / 3600
                if age > max_age_hours:
                    to_remove.append(task_id)

        for task_id in to_remove:
            del self._tasks[task_id]
            self._futures.pop(task_id, None)
            self._callbacks.pop(task_id, None)

        return len(to_remove)

    def shutdown(self, wait: bool = True) -> None:
        """关闭任务管理器

        Args:
            wait: 是否等待所有任务完成
        """
        self._executor.shutdown(wait=wait)


# 全局任务管理器实例
_task_manager: TaskManager | None = None


def get_task_manager() -> TaskManager:
    """获取全局任务管理器

    Returns:
        TaskManager: 任务管理器实例
    """
    global _task_manager
    if _task_manager is None:
        _task_manager = TaskManager()
    return _task_manager
