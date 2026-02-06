"""代价模型注册表与 OpType 路由器.

- CostModelRegistry: 以 granularity 为主轴选择代价模型
- OpTypeRouter: 按 OpType 将节点路由到子评估器
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from math_model.L4_evaluation.metrics import Granularity, StepMetrics

if TYPE_CHECKING:
    from math_model.L3_mapping.plan.exec_plan import ExecPlan


class CostModel(Protocol):
    """代价模型协议"""

    def required_fields(self) -> set[str]:
        """返回该模型所需的最小字段集合"""
        raise NotImplementedError

    def estimate_compute(
        self,
        op_type: str,
        local_shape: dict[str, int],
        hardware: dict[str, float],
    ) -> float:
        """估算计算时间（ms）"""
        raise NotImplementedError

    def estimate_comm(
        self,
        comm_bytes: int,
        path_key: str,
        participants: int,
        hardware: dict[str, float],
    ) -> float:
        """估算通信时间（ms）"""
        raise NotImplementedError


class OpEvaluator(Protocol):
    """Op 评估器协议"""

    def evaluate(
        self,
        op_id: str,
        op_type: str,
        local_shape: dict[str, int],
        attrs: dict[str, str],
        hardware: dict[str, float],
        cost_model: CostModel,
    ) -> StepMetrics:
        """评估单个 Op，返回 StepMetrics"""
        raise NotImplementedError


class CostModelRegistry:
    """代价模型注册表

    支持两种注册方式:
    1. 按 granularity 注册默认代价模型（所有 op_type 共用）
    2. 按 (op_type, granularity) 注册专属代价模型（特定 Op 使用）

    查找顺序: op_type 专属模型 -> granularity 默认模型 -> 精度回退
    """

    def __init__(self) -> None:
        # 默认模型: granularity -> CostModel
        self._default_models: dict[Granularity, CostModel] = {}
        # 专属模型: (op_type, granularity) -> CostModel
        self._op_models: dict[tuple[str, Granularity], CostModel] = {}

    def register(
        self,
        granularity: Granularity,
        model: CostModel,
        op_type: str | None = None,
    ) -> None:
        """注册代价模型

        Args:
            granularity: 评估精度
            model: 代价模型实例
            op_type: Op 类型（可选，不指定则作为该精度的默认模型）
        """
        if op_type is not None:
            self._op_models[(op_type, granularity)] = model
        else:
            self._default_models[granularity] = model

    def get(
        self,
        granularity: Granularity,
        op_type: str | None = None,
    ) -> CostModel | None:
        """获取代价模型

        查找顺序:
        1. 精确匹配 (op_type, granularity)
        2. 默认模型 granularity
        3. 精度回退（Core -> Chip）

        Args:
            granularity: 评估精度
            op_type: Op 类型（可选）

        Returns:
            对应的代价模型，如果未找到则返回 None
        """
        # 1. 精确匹配专属模型
        if op_type is not None:
            model = self._op_models.get((op_type, granularity))
            if model is not None:
                return model

        # 2. 默认模型
        model = self._default_models.get(granularity)
        if model is not None:
            return model

        # 3. 精度回退
        if granularity == Granularity.CORE:
            return self._default_models.get(Granularity.CHIP)
        if granularity == Granularity.LANE:
            model = self._default_models.get(Granularity.CORE)
            if model is not None:
                return model
            return self._default_models.get(Granularity.CHIP)

        return None

    def required_fields(self, granularity: Granularity) -> set[str]:
        """获取指定精度所需的最小字段集合

        Args:
            granularity: 评估精度

        Returns:
            字段名称集合
        """
        model = self.get(granularity)
        if model is None:
            return set()
        return model.required_fields()

    def available_granularities(self) -> list[Granularity]:
        """返回所有已注册的精度（默认模型）"""
        return list(self._default_models.keys())

    def available_op_models(self) -> list[tuple[str, Granularity]]:
        """返回所有已注册的 (op_type, granularity) 专属模型"""
        return list(self._op_models.keys())


class OpTypeRouter:
    """OpType 路由器

    按 OpType 将节点路由到子评估器，支持 Compute/Comm/Memory/Control 分类。
    不允许改变 ExecPlan 的依赖/绑定，仅做评估实现选择。
    """

    def __init__(self) -> None:
        # registry[(op_type, granularity)] = evaluator
        self._registry: dict[tuple[str, Granularity], OpEvaluator] = {}
        self._fallback: OpEvaluator | None = None

    def register(
        self, op_type: str, granularity: Granularity, evaluator: OpEvaluator
    ) -> None:
        """注册评估器

        Args:
            op_type: Op 类型（如 "matmul", "allreduce"）
            granularity: 评估精度
            evaluator: 评估器实例
        """
        self._registry[(op_type, granularity)] = evaluator

    def register_fallback(self, evaluator: OpEvaluator) -> None:
        """注册回退评估器

        Args:
            evaluator: 回退评估器实例
        """
        self._fallback = evaluator

    def resolve(self, op_type: str, granularity: Granularity) -> OpEvaluator | None:
        """解析评估器

        Args:
            op_type: Op 类型
            granularity: 评估精度

        Returns:
            评估器实例，如果未找到且有回退则返回回退评估器，否则返回 None
        """
        # 精确匹配
        evaluator = self._registry.get((op_type, granularity))
        if evaluator is not None:
            return evaluator

        # 尝试降级匹配（Core -> Chip）
        if granularity == Granularity.CORE:
            evaluator = self._registry.get((op_type, Granularity.CHIP))
            if evaluator is not None:
                return evaluator

        # 尝试 Lane -> Core -> Chip
        if granularity == Granularity.LANE:
            evaluator = self._registry.get((op_type, Granularity.CORE))
            if evaluator is not None:
                return evaluator
            evaluator = self._registry.get((op_type, Granularity.CHIP))
            if evaluator is not None:
                return evaluator

        # 回退
        return self._fallback

    def available_evaluators(self) -> list[tuple[str, Granularity]]:
        """返回所有已注册的 (op_type, granularity) 组合"""
        return list(self._registry.keys())
