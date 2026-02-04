"""事件驱动仿真引擎 (预留接口)

定义 SimulationEngine 接口，后续实现。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tier6.L2_arch.protocols import ClusterSpec
    from tier6.L3_mapping.plan import exec_plan import ExecPlanImpl
    from tier6.L4_evaluation.evaluators.base import BaseEvaluator


@dataclass
class SimulationResult:
    """仿真结果

    Attributes:
        total_time_ns: 总时延 (ns)
        events: 事件列表
        resource_utilization: 资源利用率
        bottleneck_analysis: 瓶颈分析
    """

    total_time_ns: float = 0.0
    events: list[dict[str, Any]] = field(default_factory=list)
    resource_utilization: dict[str, float] = field(default_factory=dict)
    bottleneck_analysis: dict[str, Any] = field(default_factory=dict)


class SimulationEngine:
    """事件驱动仿真引擎

    精确模拟资源竞争和并发执行。

    与 Analytical (L3+L4) 的区别:
        - Analytical: 累加各 Op 时间，简化资源竞争模型
        - Simulation: 事件驱动，精确模拟并发和队列等待

    Example:
        >>> engine = SimulationEngine()
        >>> result = engine.simulate(exec_plan, cluster_spec)
        >>> print(result.total_time_ns)
    """

    def __init__(self) -> None:
        """初始化仿真引擎"""
        pass

    def simulate(
        self,
        exec_plan: "ExecPlanImpl",
        cluster_spec: "ClusterSpec",
        evaluator: "BaseEvaluator | None" = None,
    ) -> SimulationResult:
        """执行事件驱动仿真

        Args:
            exec_plan: 来自 L3 的执行计划
            cluster_spec: 来自 L2 的集群规格
            evaluator: 来自 L4 的评估器（可选）

        Returns:
            仿真结果

        Raises:
            NotImplementedError: 此功能尚未实现
        """
        raise NotImplementedError(
            "SimulationEngine.simulate() is not implemented yet. "
            "Use L3+L4 analytical evaluation for now."
        )
