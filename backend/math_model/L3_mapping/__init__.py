"""L3: 映射与调度层

实现并行策略映射、切片规划、调度器等。
"""

from math_model.L3_mapping.plan import DistributedModel, ExecPlan
from math_model.L3_mapping.plan.distributed_model import CommType, DistributedOp, NodeRole

__all__ = [
    # 执行计划
    "DistributedModel",
    "ExecPlan",
    "CommType",
    "DistributedOp",
    "NodeRole",
]
