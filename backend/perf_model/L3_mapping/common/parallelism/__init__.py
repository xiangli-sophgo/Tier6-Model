"""第一层：chip 间切分"""

from perf_model.L3_mapping.common.parallelism.parallel_spec import (
    ParallelSpec,
    ParallelType,
)
from perf_model.L3_mapping.common.parallelism.planner import ParallelismPlanner

__all__ = [
    "ParallelSpec",
    "ParallelType",
    "ParallelismPlanner",
]
