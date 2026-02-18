"""L4 Evaluation Engine 模块.

基于 ExecPlan + HardwareSpec + TopologySpec 做统一口径的性能评估。
支持 math 和 g5 两种模式。

主要组件:
    - EvaluationEngine: 统一评估入口
    - Granularity: 评估精度层级（Chip/Core/Lane）
    - HardwareSpec: 芯片级硬件参数
    - TopologySpec: chip 间通信拓扑参数
    - StepMetrics: Step 级时延分解
    - Aggregates: 聚合指标
    - EngineResult: 评估结果
    - CostModelRegistry: 代价模型注册表
    - OpTypeRouter: OpType 路由器
    - Calibration: 校准模块
"""

from perf_model.L4_evaluation.math.calibration import Calibration, CalibrationConfig
from perf_model.L4_evaluation.math.engine import EvaluationEngine, create_default_hardware_spec
from perf_model.L2_arch.topology import TopologySpec
from perf_model.L4_evaluation.common.metrics import (
    Aggregates,
    BottleneckTag,
    CommProtocolSpec,
    EngineResult,
    Granularity,
    HardwareSpec,
    StepMetrics,
    merge_specs,
)
from perf_model.L4_evaluation.math.evaluators.precise import (
    AttentionPreciseEvaluator,
    ConvPreciseEvaluator,
    ElementwisePreciseEvaluator,
    FallbackPreciseEvaluator,
    MatMulPreciseEvaluator,
    PreciseEvaluatorRegistry,
    PreciseMetrics,
    PreciseTileEvaluator,
    create_precise_evaluator,
)
from perf_model.L4_evaluation.math.registry import CostModel, CostModelRegistry, OpTypeRouter

__all__ = [
    # Engine
    "EvaluationEngine",
    "create_default_hardware_spec",
    # Specs
    "HardwareSpec",
    "TopologySpec",
    "CommProtocolSpec",
    "merge_specs",
    # Metrics
    "Granularity",
    "BottleneckTag",
    "StepMetrics",
    "Aggregates",
    "EngineResult",
    # Precise Evaluator (for L3 TilingPlanner)
    "PreciseTileEvaluator",
    "PreciseMetrics",
    "PreciseEvaluatorRegistry",
    "create_precise_evaluator",
    # Op-specific Precise Evaluators
    "MatMulPreciseEvaluator",
    "AttentionPreciseEvaluator",
    "ConvPreciseEvaluator",
    "ElementwisePreciseEvaluator",
    "FallbackPreciseEvaluator",
    # Registry
    "CostModel",
    "CostModelRegistry",
    "OpTypeRouter",
    # Calibration
    "Calibration",
    "CalibrationConfig",
]
