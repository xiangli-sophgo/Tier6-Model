"""Tier6 - AI 加速芯片性能建模平台

基于 CHIPMathica 六层架构的 LLM 推理部署分析系统。

架构层级:
    L0: Entry & Orchestration - 入口与编排
    L1: Workload Representation - 工作负载表示
    L2: Architecture Spec - 硬件规格
    L3: Mapping & Scheduling - 映射调度
    L4: Evaluation Engines - 评估引擎
    L5: Reporting & Visualization - 报告与可视化
    Simulation: 事件驱动仿真 (预留)

Example:
    >>> from math_model import run_evaluation
    >>> result = run_evaluation({
    ...     "chip_preset": "SG2262",
    ...     "model_preset": "deepseek-v3",
    ...     "parallelism": {"tp": 4, "pp": 1, "dp": 1},
    ...     "inference": {"batch_size": 32, "input_seq_length": 1024}
    ... })
    >>> print(result["aggregates"]["tps"])
"""

__version__ = "3.0.0"
__author__ = "Tier6 Team"


# 延迟导入，避免循环依赖
def __getattr__(name: str):
    """延迟导入公开 API"""
    if name == "run_evaluation":
        from math_model.L0_entry.engine import run_evaluation
        return run_evaluation
    elif name == "get_task_manager":
        from math_model.L0_entry.tasks import get_task_manager
        return get_task_manager
    elif name == "get_config_loader":
        from math_model.L0_entry.config_loader import get_config_loader
        return get_config_loader
    elif name == "DataType":
        from math_model.L0_entry.types import DataType
        return DataType
    elif name == "router":
        from math_model.L0_entry.api import router
        return router
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "__version__",
    # L0 Entry
    "run_evaluation",
    "get_task_manager",
    "get_config_loader",
    "router",
    # Core
    "DataType",
]
