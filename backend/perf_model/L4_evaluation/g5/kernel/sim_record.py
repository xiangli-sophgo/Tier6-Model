"""SimRecord: 仿真记录数据结构

独立模块, 避免循环导入 (sim_engine <-> core_subsys)。
"""

from __future__ import annotations

from dataclasses import dataclass

from perf_model.L3_mapping.g5.program import DMADirection


@dataclass
class SimRecord:
    """仿真记录 (单条指令的执行记录)

    Attributes:
        engine: 执行引擎 ("TIU" / "DMA" / "SDMA" / "HAU")
        cmd_id: 指令 ID
        start_ns: 开始时间 (ns)
        end_ns: 结束时间 (ns)
        flops: 浮点运算量 (仅 TIU)
        data_bytes: 搬运数据量 (仅 DMA/SDMA)
        direction: DMA 方向 (仅 DMA)
        source_op_id: 关联的 DistributedOp ID
    """
    engine: str
    cmd_id: int
    start_ns: float
    end_ns: float
    flops: int = 0
    data_bytes: int = 0
    direction: DMADirection | None = None
    source_op_id: str = ""
