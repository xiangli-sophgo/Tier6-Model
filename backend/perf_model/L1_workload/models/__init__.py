"""模型定义模块

提供模型基类和注册表。
"""

from perf_model.L0_entry.registry import Registry
from perf_model.L1_workload.models.base import ModelBase

# 全局模型注册表
model_registry: Registry[ModelBase] = Registry("model")

# 导入所有模型实现以触发注册
from perf_model.L1_workload.models.llm import DeepSeekModel, DeepSeekV3Model, Llama2Model

__all__ = [
    "ModelBase",
    "model_registry",
    # 模型实现
    "DeepSeekModel",
    "DeepSeekV3Model",
    "Llama2Model",
]
