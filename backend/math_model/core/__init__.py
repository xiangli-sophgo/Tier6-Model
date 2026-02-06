"""Tier6 核心模块

包含类型定义、协议接口、工具函数等基础组件。
"""

from math_model.core.types import (
    DataType,
    ParallelMode,
    BottleneckType,
    EngineType,
    Latency,
    MemorySize,
    Bandwidth,
    FLOPs,
    TFLOPS,
    LatencyBreakdown,
    UtilizationMetrics,
    ThroughputMetrics,
    ParallelismConfig,
    BatchConfig,
)
from math_model.core.protocols import (
    Configurable,
    WorkloadIR,
    OpsBreakdown,
    MemoryFootprint,
    CommPattern,
    ExecPlan,
    EngineResult,
    Exportable,
)
from math_model.core.registry import InstanceRegistry
from math_model.core.exceptions import (
    Tier6Error,
    ConfigError,
    ValidationError,
    EvaluationError,
)

__all__ = [
    # 类型
    "DataType",
    "ParallelMode",
    "BottleneckType",
    "EngineType",
    "Latency",
    "MemorySize",
    "Bandwidth",
    "FLOPs",
    "TFLOPS",
    "LatencyBreakdown",
    "UtilizationMetrics",
    "ThroughputMetrics",
    "ParallelismConfig",
    "BatchConfig",
    # 协议
    "Configurable",
    "WorkloadIR",
    "OpsBreakdown",
    "MemoryFootprint",
    "CommPattern",
    "ExecPlan",
    "EngineResult",
    "Exportable",
    # 工具
    "InstanceRegistry",
    # 异常
    "Tier6Error",
    "ConfigError",
    "ValidationError",
    "EvaluationError",
]
