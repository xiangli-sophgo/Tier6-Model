"""全局事件驱动仿真内核

对标 SystemC sc_main / gem5 EventManager:
- 全局事件队列 (heapq)
- 时间单位 ns, 支持多时钟域 cycle 转换
- 事件按 (time_ns, seq_id) 排序, 保证同时间 FIFO

参考设计: docs/plans/2026-02-19-g5-full-architecture-design.md Section 5.1
"""

from __future__ import annotations

import heapq
from typing import Callable

from perf_model.L4_evaluation.g5.kernel.stats import StatGroup


class SimKernel:
    """轻量级事件驱动仿真内核"""

    def __init__(self) -> None:
        self._current_time: float = 0.0
        self._event_queue: list[tuple[float, int, Callable]] = []
        self._seq_counter: int = 0
        self._clocks: dict[str, float] = {}  # name -> frequency_ghz
        self._event_count: int = 0

        # 统计框架: 顶层 StatGroup
        self.stats = StatGroup("kernel")
        self._stat_total_events = self.stats.scalar(
            "total_events", "事件队列处理的事件总数"
        )
        self._stat_sim_time = self.stats.scalar(
            "total_sim_time_ns", "仿真总时长 (ns)"
        )

    def now(self) -> float:
        """当前仿真时间 (ns)"""
        return self._current_time

    @property
    def event_count(self) -> int:
        """已执行的事件总数"""
        return self._event_count

    def schedule(self, delay_ns: float, callback: Callable) -> None:
        """延迟调度事件"""
        if delay_ns < 0:
            raise ValueError(f"delay_ns must be >= 0, got {delay_ns}")
        time = self._current_time + delay_ns
        seq = self._seq_counter
        self._seq_counter += 1
        heapq.heappush(self._event_queue, (time, seq, callback))

    def schedule_at(self, time_ns: float, callback: Callable) -> None:
        """绝对时间调度事件"""
        if time_ns < self._current_time:
            raise ValueError(
                f"Cannot schedule in the past: time_ns={time_ns}, now={self._current_time}"
            )
        seq = self._seq_counter
        self._seq_counter += 1
        heapq.heappush(self._event_queue, (time_ns, seq, callback))

    def run(self) -> None:
        """主事件循环: 弹出事件 -> 推进时间 -> 执行回调"""
        while self._event_queue:
            time_ns, _seq, callback = heapq.heappop(self._event_queue)
            self._current_time = time_ns
            self._event_count += 1
            callback()
        # 仿真结束, 更新统计
        self._stat_total_events.value = float(self._event_count)
        self._stat_sim_time.value = self._current_time

    def add_clock(self, name: str, frequency_ghz: float) -> None:
        """注册时钟域"""
        if frequency_ghz <= 0:
            raise ValueError(f"frequency_ghz must be > 0, got {frequency_ghz}")
        self._clocks[name] = frequency_ghz

    def cycle_to_ns(self, cycles: int, clock_name: str) -> float:
        """cycle -> ns"""
        freq = self._clocks.get(clock_name)
        if freq is None:
            raise KeyError(f"Clock '{clock_name}' not registered")
        return cycles / freq

    def ns_to_cycle(self, ns: float, clock_name: str) -> int:
        """ns -> cycle (向下取整)"""
        freq = self._clocks.get(clock_name)
        if freq is None:
            raise KeyError(f"Clock '{clock_name}' not registered")
        return int(ns * freq)
