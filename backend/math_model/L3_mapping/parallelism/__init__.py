"""第一层：chip 间切分"""

from math_model.L3_mapping.parallelism.parallel_spec import (
    ParallelSpec,
    ParallelType,
)
from math_model.L3_mapping.parallelism.planner import ParallelismPlanner

__all__ = [
    "ParallelSpec",
    "ParallelType",
    "ParallelismPlanner",
]
