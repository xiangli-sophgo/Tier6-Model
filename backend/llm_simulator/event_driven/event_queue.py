"""
事件队列模块

基于 heapq 实现的优先队列，按照事件的 (timestamp, event_type, event_id) 排序。
"""

from __future__ import annotations

import heapq
from typing import Optional, Iterator

from .event import BaseEvent


class EventQueue:
    """事件优先队列

    使用 Python heapq 实现，按照事件的自然顺序排序：
    1. 首先按 timestamp 升序
    2. 相同时间按 event_type 升序（END 事件优先于 START 事件）
    3. 仍相同按 event_id 升序（保证确定性）
    """

    def __init__(self):
        self._heap: list[BaseEvent] = []
        self._counter = 0  # 用于统计

    def push(self, event: BaseEvent) -> None:
        """添加事件到队列

        时间复杂度: O(log n)
        """
        heapq.heappush(self._heap, event)
        self._counter += 1

    def pop(self) -> BaseEvent:
        """取出并返回最早的事件

        时间复杂度: O(log n)

        Raises:
            IndexError: 队列为空时抛出
        """
        return heapq.heappop(self._heap)

    def peek(self) -> Optional[BaseEvent]:
        """查看最早的事件但不移除

        时间复杂度: O(1)

        Returns:
            最早的事件，队列为空时返回 None
        """
        if self._heap:
            return self._heap[0]
        return None

    def push_many(self, events: list[BaseEvent]) -> None:
        """批量添加事件

        Args:
            events: 事件列表
        """
        for event in events:
            self.push(event)

    def is_empty(self) -> bool:
        """检查队列是否为空"""
        return len(self._heap) == 0

    def __len__(self) -> int:
        """返回队列中的事件数量"""
        return len(self._heap)

    def __bool__(self) -> bool:
        """队列非空时返回 True"""
        return len(self._heap) > 0

    def __iter__(self) -> Iterator[BaseEvent]:
        """迭代所有事件（按顺序取出）

        注意：这会清空队列！
        """
        while self._heap:
            yield self.pop()

    @property
    def total_events_processed(self) -> int:
        """返回累计处理的事件数量"""
        return self._counter

    def clear(self) -> None:
        """清空队列"""
        self._heap.clear()

    def get_events_at_time(self, timestamp: float, tolerance: float = 1e-6) -> list[BaseEvent]:
        """获取特定时间点的所有事件（不移除）

        Args:
            timestamp: 目标时间
            tolerance: 时间容差

        Returns:
            该时间点的所有事件列表
        """
        return [
            e for e in self._heap
            if abs(e.timestamp - timestamp) < tolerance
        ]

    def remove_events_for_chip(self, chip_id: str) -> int:
        """移除特定芯片的所有事件（用于错误处理）

        Args:
            chip_id: 芯片ID

        Returns:
            移除的事件数量
        """
        original_len = len(self._heap)
        self._heap = [e for e in self._heap if e.chip_id != chip_id]
        heapq.heapify(self._heap)
        return original_len - len(self._heap)

    def stats(self) -> dict:
        """返回队列统计信息"""
        if not self._heap:
            return {
                "size": 0,
                "total_processed": self._counter,
                "earliest_time": None,
                "latest_time": None,
            }

        times = [e.timestamp for e in self._heap]
        return {
            "size": len(self._heap),
            "total_processed": self._counter,
            "earliest_time": min(times),
            "latest_time": max(times),
            "event_types": self._count_event_types(),
        }

    def _count_event_types(self) -> dict[str, int]:
        """统计各类型事件数量"""
        counts: dict[str, int] = {}
        for event in self._heap:
            type_name = event.event_type.name
            counts[type_name] = counts.get(type_name, 0) + 1
        return counts
