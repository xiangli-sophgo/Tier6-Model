"""第一层：chip 间切分"""

from tier6.L3_mapping.parallelism.parallel_spec import (
    ParallelSpec,
    ParallelType,
)
from tier6.L3_mapping.parallelism.planner import ParallelismPlanner

__all__ = [
    "ParallelSpec",
    "ParallelType",
    "ParallelismPlanner",
]
