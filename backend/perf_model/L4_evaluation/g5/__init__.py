"""G5 指令级仿真 - L4 评估层

事件驱动仿真引擎: TIU + GDMA + SDMA + HAU 协同仿真。
当前支持: 单核 MatMul/MoE 仿真, SimRecord -> EngineResult 适配。

参考设计: docs/plans/G5-instruction-level-simulator.md
"""

from perf_model.L4_evaluation.g5.sim_engine import G5SimEngine
from perf_model.L4_evaluation.g5.adapter import G5ResultAdapter

__all__ = [
    "G5SimEngine",
    "G5ResultAdapter",
]
