"""L4: 成本模型子模块

提供算子/层级成本估算的各种模型实现。
"""

from perf_model.L4_evaluation.common.cost_models.base import BaseCostModel
from perf_model.L4_evaluation.common.cost_models.chip import (
    AttentionCostModel,
    ChipCostModel,
    MatMulCostModel,
    RooflineResult,
)
from perf_model.L4_evaluation.common.cost_models.core import CoreCostModel
from perf_model.L4_evaluation.common.cost_models.comm_protocol import (
    CommArchSpec,
    CommProtocolParams,
    CommProtocolCostModel,
)

__all__ = [
    # 基类
    "BaseCostModel",
    # 芯片级模型
    "ChipCostModel",
    "RooflineResult",
    # Core 级模型
    "CoreCostModel",
    # 专用模型
    "MatMulCostModel",
    "AttentionCostModel",
    # 通信协议模型
    "CommArchSpec",
    "CommProtocolParams",
    "CommProtocolCostModel",
]
