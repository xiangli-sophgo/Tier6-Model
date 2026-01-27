"""
WebSocket 连接管理器

使用队列订阅模式管理所有活跃的 WebSocket 连接并广播任务更新。
"""

import asyncio
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class WebSocketManager:
    """WebSocket 连接管理器（队列订阅模式）"""

    def __init__(self):
        # 全局订阅者队列列表
        self._global_subscribers: List[asyncio.Queue] = []
        # 保存主事件循环的引用（在应用启动时设置）
        self._main_loop: Optional[asyncio.AbstractEventLoop] = None

    def set_event_loop(self, loop: asyncio.AbstractEventLoop):
        """设置主事件循环引用（在应用启动时调用）"""
        self._main_loop = loop
        logger.info("WebSocket manager: event loop registered")

    def subscribe_global(self) -> asyncio.Queue:
        """订阅所有任务状态更新（全局）"""
        queue = asyncio.Queue()
        self._global_subscribers.append(queue)
        logger.info(f"Global subscriber added. Total subscribers: {len(self._global_subscribers)}")
        return queue

    def unsubscribe_global(self, queue: asyncio.Queue):
        """取消全局订阅"""
        if queue in self._global_subscribers:
            self._global_subscribers.remove(queue)
            logger.info(f"Global subscriber removed. Total subscribers: {len(self._global_subscribers)}")

    def broadcast_task_update(self, task_id: str, data: dict):
        """
        同步方法：广播任务更新（从工作线程调用）

        注意：这个方法会在工作线程中被调用，不能直接使用 async/await。
        使用预先保存的事件循环引用来调度协程。
        """
        print(f"[DEBUG WS] broadcast_task_update called: task_id={task_id}, progress={data.get('progress', 'N/A')}, subscribers={len(self._global_subscribers)}")

        if not self._global_subscribers:
            print(f"[DEBUG WS] No subscribers, skipping broadcast")
            return

        logger.info(f"[WS Broadcast] task_id={task_id}, progress={data.get('progress', 'N/A')}, subscribers={len(self._global_subscribers)}")

        message = {
            "type": "task_update",
            "task_id": task_id,
            **data
        }

        try:
            # 尝试获取当前运行的事件循环（在异步上下文中）
            loop = asyncio.get_running_loop()
            loop.create_task(self._broadcast_to_queues(message))
        except RuntimeError:
            # 没有运行中的事件循环（在子线程中），使用保存的主事件循环
            if self._main_loop and self._main_loop.is_running():
                asyncio.run_coroutine_threadsafe(self._broadcast_to_queues(message), self._main_loop)
            else:
                logger.warning(f"Cannot broadcast (task {task_id}): main event loop not set or not running")

    async def _broadcast_to_queues(self, message: dict):
        """向所有订阅者队列推送消息"""
        print(f"[DEBUG WS] _broadcast_to_queues: sending to {len(self._global_subscribers)} queues")
        for queue in self._global_subscribers:
            try:
                await queue.put(message)
                print(f"[DEBUG WS] Message put to queue successfully")
            except Exception as e:
                print(f"[DEBUG WS] Failed to put message to queue: {e}")
                logger.error(f"Failed to put message to queue: {e}")


# 全局单例
ws_manager = WebSocketManager()
