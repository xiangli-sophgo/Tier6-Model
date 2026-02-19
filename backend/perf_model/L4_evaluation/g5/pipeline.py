"""G5 指令级仿真管线封装

DistributedModel -> G5InstructionEmitter -> G5SimEngine -> G5ResultAdapter -> EngineResult

将 L3.g5 和 L4.g5 组件串联为单一调用入口，供 engine.py 的 G5 路由使用。
"""

from __future__ import annotations

from typing import Any, Callable

from perf_model.L2_arch.chip import ChipSpecImpl
from perf_model.L3_mapping.common.plan.distributed_model import DistributedModel
from perf_model.L3_mapping.g5.instruction_emitter import G5InstructionEmitter
from perf_model.L4_evaluation.common.metrics import EngineResult
from perf_model.L4_evaluation.g5.sim_engine import G5SimEngine
from perf_model.L4_evaluation.g5.adapter import G5ResultAdapter


def run_g5_pipeline(
    dist_model: DistributedModel,
    chip: ChipSpecImpl,
    progress_callback: Callable[[float], None] | None = None,
) -> EngineResult:
    """G5 指令级仿真管线

    Args:
        dist_model: 分布式模型 (ParallelismPlanner 产出)
        chip: 芯片规格
        progress_callback: 进度回调 (0.0 ~ 1.0)

    Returns:
        EngineResult (与 Math 模式的 EvaluationEngine 产出格式一致)
    """
    # L3.g5: DistributedOp -> CoreProgram
    emitter = G5InstructionEmitter(chip)
    program = emitter.emit(dist_model.ops)

    if progress_callback:
        progress_callback(0.5)

    # L4.g5: CoreProgram -> SimRecord[]
    engine = G5SimEngine(chip)
    records = engine.simulate(program)
    stats = engine.get_stats()

    if progress_callback:
        progress_callback(0.8)

    # L4.g5: SimRecord[] -> EngineResult (含统计数据)
    adapter = G5ResultAdapter(chip)
    result = adapter.convert(records, stats=stats)

    if progress_callback:
        progress_callback(1.0)

    return result
