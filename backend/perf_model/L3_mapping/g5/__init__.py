"""G5 指令级仿真 - L3 映射层

DistributedModel -> CoreProgram 指令生成。
支持: MatMul (MM2_NN) 单核 double buffering, MoE dispatch/combine, AllReduce。

参考设计: docs/plans/G5-instruction-level-simulator.md
"""

from perf_model.L3_mapping.g5.program import (
    CoreInstructions,
    CoreProgram,
    DMACommand,
    DMADirection,
    HAUCommand,
    HAUMsgAction,
    HAUOpType,
    SDMACommand,
    SDMACommandType,
    TIUCommand,
    TIUOpType,
)
from perf_model.L3_mapping.g5.instruction_emitter import G5InstructionEmitter

__all__ = [
    "CoreInstructions",
    "CoreProgram",
    "DMACommand",
    "DMADirection",
    "HAUCommand",
    "HAUMsgAction",
    "HAUOpType",
    "SDMACommand",
    "SDMACommandType",
    "TIUCommand",
    "TIUOpType",
    "G5InstructionEmitter",
]
