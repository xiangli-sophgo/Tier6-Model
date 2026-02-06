"""评估器基类."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from math_model.L4_evaluation.metrics import BottleneckTag, StepMetrics

if TYPE_CHECKING:
    from math_model.L4_evaluation.registry import CostModel


class BaseEvaluator(ABC):
    """评估器基类

    定义 Op 评估器接口，子类需实现具体的评估逻辑。
    """

    @abstractmethod
    def evaluate(
        self,
        op_id: str,
        op_type: str,
        local_shape: dict[str, int],
        attrs: dict[str, str],
        hardware: dict[str, float],
        cost_model: "CostModel",
    ) -> StepMetrics:
        """评估单个 Op

        Args:
            op_id: Op 标识
            op_type: Op 类型
            local_shape: 切分后的 shape
            attrs: Op 属性
            hardware: 硬件参数
            cost_model: 代价模型

        Returns:
            StepMetrics 评估结果
        """
        ...


class FallbackEvaluator(BaseEvaluator):
    """回退评估器

    在 OpType 未注册或 granularity 不可用时提供降级评估。
    降级评估仍需输出完整 StepMetrics 字段。
    """

    def evaluate(
        self,
        op_id: str,
        op_type: str,
        local_shape: dict[str, int],
        attrs: dict[str, str],
        hardware: dict[str, float],
        cost_model: "CostModel",
    ) -> StepMetrics:
        """降级估时

        输入:
            - op_id, op_type, local_shape, attrs, hardware, cost_model
        输出:
            - StepMetrics（使用 Chip 粗粒度估时）
        关键步骤:
            - 调用 cost_model.estimate_compute 获取基础估时
            - 标记为 UNKNOWN 瓶颈类型
        """
        t_compute = cost_model.estimate_compute(op_type, local_shape, hardware)

        # 估算 FLOPs 和访存量
        flops = cost_model.estimate_flops(op_type, local_shape)
        bytes_read, bytes_write = cost_model.estimate_bytes(op_type, local_shape)

        return StepMetrics(
            op_id=op_id,
            t_compute=t_compute,
            t_comm=0.0,
            t_wait=0.0,
            t_total=t_compute,
            bottleneck_tag=BottleneckTag.UNKNOWN,
            flops=flops,
            bytes_read=bytes_read,
            bytes_write=bytes_write,
            meta={"evaluator": "fallback", "op_type": op_type},
        )
