"""WebSocket 管理模块

提供跨线程的 WebSocket 消息广播能力。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)


class WebSocketManager:
    """WebSocket 连接管理器

    支持:
    - 多客户端订阅
    - 从任意线程广播消息（通过 asyncio.run_coroutine_threadsafe）
    - 消息队列机制

    使用示例:
        # 在 FastAPI 启动时设置事件循环
        @app.on_event("startup")
        async def startup():
            ws_manager.set_event_loop(asyncio.get_running_loop())

        # WebSocket 端点
        @app.websocket("/ws/tasks")
        async def websocket_tasks(websocket: WebSocket):
            await websocket.accept()
            queue = ws_manager.subscribe()
            try:
                while True:
                    message = await queue.get()
                    await websocket.send_json(message)
            except WebSocketDisconnect:
                ws_manager.unsubscribe(queue)

        # 从线程池任务中广播
        def background_task():
            ws_manager.broadcast({"type": "task_update", "data": {...}})
    """

    def __init__(self) -> None:
        """初始化 WebSocket 管理器"""
        self._subscribers: list[asyncio.Queue[dict[str, Any]]] = []
        self._main_loop: asyncio.AbstractEventLoop | None = None
        self._lock: asyncio.Lock | None = asyncio.Lock() if asyncio else None

    def set_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """设置主事件循环

        Args:
            loop: asyncio 事件循环实例
        """
        self._main_loop = loop
        # 重新创建锁，绑定到正确的事件循环
        self._lock = asyncio.Lock()
        logger.info("WebSocket manager event loop set")

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        """订阅消息

        Returns:
            asyncio.Queue: 消息队列，用于接收广播消息
        """
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._subscribers.append(queue)
        logger.debug(f"New subscriber added, total: {len(self._subscribers)}")
        return queue

    def unsubscribe(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        """取消订阅

        Args:
            queue: 之前订阅时返回的队列
        """
        try:
            self._subscribers.remove(queue)
            logger.debug(f"Subscriber removed, total: {len(self._subscribers)}")
        except ValueError:
            pass  # 队列不存在，忽略

    def broadcast(self, message: dict[str, Any]) -> None:
        """从任意线程广播消息

        Args:
            message: 要广播的消息字典
        """
        if not self._subscribers:
            logger.warning(f"[WS] No subscribers, message dropped: {message.get('type')}")
            return

        if self._main_loop is None:
            logger.warning("[WS] Event loop not set, message dropped")
            return

        try:
            # 从线程池安全地调度到主事件循环
            logger.debug(f"[WS] Scheduling broadcast to {len(self._subscribers)} subscribers")
            asyncio.run_coroutine_threadsafe(
                self._broadcast_async(message),
                self._main_loop
            )
        except Exception as e:
            logger.error(f"[WS] Failed to broadcast message: {e}")

    async def _broadcast_async(self, message: dict[str, Any]) -> None:
        """异步广播消息（内部方法）

        Args:
            message: 要广播的消息字典
        """
        # 复制订阅者列表，避免迭代时修改
        subscribers = list(self._subscribers)

        for queue in subscribers:
            try:
                await queue.put(message)
            except Exception as e:
                logger.error(f"Failed to put message to queue: {e}")

    def broadcast_sync(self, message: dict[str, Any]) -> None:
        """同步广播消息（仅在事件循环中使用）

        Args:
            message: 要广播的消息字典
        """
        if not self._subscribers:
            return

        for queue in list(self._subscribers):
            try:
                queue.put_nowait(message)
            except asyncio.QueueFull:
                logger.warning("Queue full, message dropped")
            except Exception as e:
                logger.error(f"Failed to broadcast: {e}")

    @property
    def subscriber_count(self) -> int:
        """获取当前订阅者数量"""
        return len(self._subscribers)

    def broadcast_task_update(
        self,
        task_id: str,
        status: str,
        progress: float = 0.0,
        error: str | None = None,
        result: dict[str, Any] | None = None,
    ) -> None:
        """广播任务状态更新

        Args:
            task_id: 任务 ID
            status: 任务状态
            progress: 进度 (0-1)
            error: 错误信息
            result: 任务结果（仅在完成时发送）
        """
        # 使用前端期望的格式（与 llm_simulator 兼容）
        message: dict[str, Any] = {
            "type": "task_update",
            "task_id": task_id,
            "status": status,
            "progress": progress,
            "message": f"Task {status}",
        }

        if error:
            message["error"] = error

        if result and status == "completed":
            # 添加搜索统计信息
            search_stats = result.get("search_stats", {})
            message["search_stats"] = {
                "total_plans": search_stats.get("total_plans", 1),
                "feasible_plans": search_stats.get("feasible_plans", 1),
                "infeasible_plans": search_stats.get("infeasible_plans", 0),
            }

        logger.debug(f"[WS] Broadcasting task update: task_id={task_id}, status={status}, subscribers={self.subscriber_count}")
        self.broadcast(message)

    def broadcast_experiment_update(
        self,
        experiment_id: str,
        event: str,
        data: dict[str, Any] | None = None,
    ) -> None:
        """广播实验更新

        Args:
            experiment_id: 实验 ID
            event: 事件类型 (created, updated, deleted, task_added, etc.)
            data: 附加数据
        """
        message: dict[str, Any] = {
            "type": "experiment_update",
            "data": {
                "experimentId": experiment_id,
                "event": event,
            }
        }

        if data:
            message["data"].update(data)

        self.broadcast(message)


# 全局 WebSocket 管理器实例
ws_manager = WebSocketManager()


def get_ws_manager() -> WebSocketManager:
    """获取全局 WebSocket 管理器

    Returns:
        WebSocketManager: WebSocket 管理器实例
    """
    return ws_manager
