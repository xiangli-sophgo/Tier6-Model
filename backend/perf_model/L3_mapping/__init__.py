"""L3: 映射与调度层

实现并行策略映射、切片规划、调度器等。
支持 math 和 g5 两种模式。
"""

from perf_model.L3_mapping.common.plan.distributed_model import (
    CommType,
    DistributedModel,
    DistributedOp,
    NodeRole,
)
from perf_model.L3_mapping.math.plan.exec_plan import ExecPlan

__all__ = [
    # 执行计划
    "DistributedModel",
    "ExecPlan",
    "CommType",
    "DistributedOp",
    "NodeRole",
]
