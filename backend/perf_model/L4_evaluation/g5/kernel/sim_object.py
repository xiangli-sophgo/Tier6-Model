"""SimObject 硬件模块基类

所有 G5 仿真模块的基类, 提供:
- kernel 引用 (全局事件队列)
- 时钟域绑定 (cycle <-> ns 转换)
- schedule / schedule_cycles 便捷方法

参考设计: docs/plans/2026-02-19-g5-full-architecture-design.md Section 5.2
"""

from __future__ import annotations

from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from perf_model.L4_evaluation.g5.kernel.sim_kernel import SimKernel


class SimObject:
    """硬件模块基类"""

    def __init__(self, kernel: SimKernel, name: str, clock_name: str) -> None:
        self.kernel = kernel
        self.name = name
        self.clock_name = clock_name

    def now(self) -> float:
        """当前仿真时间 (ns)"""
        return self.kernel.now()

    def cycle_now(self) -> int:
        """当前仿真 cycle (基于绑定时钟域)"""
        return self.kernel.ns_to_cycle(self.kernel.now(), self.clock_name)

    def schedule(self, delay_ns: float, callback: Callable) -> None:
        """延迟调度事件 (ns)"""
        self.kernel.schedule(delay_ns, callback)

    def schedule_cycles(self, cycles: int, callback: Callable) -> None:
        """延迟调度事件 (cycles, 自动转 ns)"""
        delay_ns = self.kernel.cycle_to_ns(cycles, self.clock_name)
        self.kernel.schedule(delay_ns, callback)

    def schedule_at(self, time_ns: float, callback: Callable) -> None:
        """绝对时间调度"""
        self.kernel.schedule_at(time_ns, callback)
