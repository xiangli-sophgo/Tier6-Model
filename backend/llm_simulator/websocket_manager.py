"""
WebSocket 连接管理器

管理所有活跃的 WebSocket 连接并广播任务更新。
"""

import logging
from typing import Set
from fastapi import WebSocket

logger = logging.getLogger(__name__)


class WebSocketManager:
    """WebSocket 连接管理器"""

    def __init__(self):
        # 存储所有活跃连接
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        """接受新的 WebSocket 连接"""
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        """断开 WebSocket 连接"""
        self.active_connections.discard(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        """广播消息到所有连接的客户端"""
        if not self.active_connections:
            return

        dead_connections = set()
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Failed to send message to WebSocket: {e}")
                dead_connections.add(connection)

        # 清理失效连接
        for connection in dead_connections:
            self.active_connections.discard(connection)

    def broadcast_task_update(self, task_id: str, data: dict):
        """
        同步方法：广播任务更新（从工作线程调用）

        注意：这个方法会在工作线程中被调用，不能直接使用 async/await。
        我们将更新放入队列，由主事件循环处理。
        """
        import asyncio

        # 获取当前事件循环（如果有）
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 将协程调度到事件循环
                asyncio.run_coroutine_threadsafe(
                    self.broadcast({
                        "type": "task_update",
                        "task_id": task_id,
                        **data
                    }),
                    loop
                )
        except RuntimeError:
            # 没有运行中的事件循环，忽略
            logger.warning("No running event loop, skipping WebSocket broadcast")


# 全局单例
ws_manager = WebSocketManager()
