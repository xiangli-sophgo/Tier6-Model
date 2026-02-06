"""LLM 模型模块

提供大语言模型实现。
"""

from math_model.L1_workload.models.llm.deepseek import DeepSeekModel, DeepSeekV3Model
from math_model.L1_workload.models.llm.llama import Llama2Model

__all__ = ["Llama2Model", "DeepSeekV3Model", "DeepSeekModel"]
