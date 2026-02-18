"""算子定义模块

提供算子基类和注册表。
"""

from perf_model.L0_entry.registry import Registry
from perf_model.L1_workload.operators.base import OpBase, OpRole

# 全局算子注册表
op_registry: Registry[OpBase] = Registry("op")

__all__ = ["OpBase", "OpRole", "op_registry"]
