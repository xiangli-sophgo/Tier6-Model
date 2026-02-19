"""事件驱动仿真调度器 (G5 指令级仿真模式)

Phase 1 重构: 内部委托 SingleChipSim (SimKernel 架构)。
外部 API (G5SimEngine.simulate) 保持不变。

支持多核仿真: 每个 CoreInstructions 在独立 CoreSubsys 中并行执行,
共享 SimKernel 全局事件队列。

同步机制 (4 引擎):
    tiu_sync_id: TIU 完成时写入 cmd_id
    tdma_sync_id: DMA 完成时写入 cmd_id
    sdma_sync_id: SDMA 完成时写入 cmd_id
    hau_sync_id: HAU 完成时写入 cmd_id

依赖解析:
    TIU:  cmd.cmd_id_dep <= tdma_sync_id (固定依赖 DMA)
    DMA:  cmd.cmd_id_dep <= tiu_sync_id (固定依赖 TIU)
    SDMA: cmd.dep_engine 指定依赖引擎 ("tiu"/"hau"/"sdma"/"tdma")
    HAU:  cmd.dep_engine 指定依赖引擎 ("tiu"/"tdma")

参考设计: docs/plans/2026-02-19-g5-full-architecture-design.md
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Any

from perf_model.L2_arch.chip import ChipSpecImpl
from perf_model.L3_mapping.g5.program import CoreProgram

# SimRecord 的权威定义在 kernel/sim_record.py, 这里重新导出保持兼容
from perf_model.L4_evaluation.g5.kernel.sim_record import SimRecord  # noqa: F401


class EventType(Enum):
    """仿真事件类型 (保留导出兼容)"""
    TIU_FINISH = auto()
    DMA_FINISH = auto()
    SDMA_FINISH = auto()
    HAU_FINISH = auto()


class G5SimEngine:
    """G5 事件驱动仿真引擎

    支持单核和多核仿真。内部委托 SingleChipSim (SimKernel 架构)。
    """

    def __init__(self, chip: ChipSpecImpl) -> None:
        self._chip = chip
        self._last_stats: dict[str, Any] = {}

    def simulate(self, program: CoreProgram) -> list[SimRecord]:
        """仿真 CoreProgram

        Args:
            program: 多核程序

        Returns:
            SimRecord 列表
        """
        self._last_stats = {}

        if not program.cores:
            return []

        # 延迟导入避免循环引用
        from perf_model.L4_evaluation.g5.top.single_chip import SingleChipSim

        sim = SingleChipSim(self._chip)
        records = sim.simulate(program)
        self._last_stats = sim.get_stats()
        return records

    def get_stats(self) -> dict[str, Any]:
        """获取最近一次仿真的统计数据"""
        return dict(self._last_stats)
