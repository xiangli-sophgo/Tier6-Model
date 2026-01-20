"""
模型模块 - DS_TPU 风格的模型定义

Model 是最顶层抽象，包含:
- 多个 Layer 实例
- operator_map: 按算子类型分组
- 性能指标聚合
"""

from .base import BaseModel
from .deepseek import (
    DeepSeekModel,
    create_deepseek_v3,
    create_deepseek_v3_absorb,
    create_deepseek_v32,
    MLA_VARIANTS,
)
from .llama import LlamaModel

__all__ = [
    'BaseModel',
    'DeepSeekModel',
    'create_deepseek_v3',
    'create_deepseek_v3_absorb',
    'create_deepseek_v32',
    'MLA_VARIANTS',
    'LlamaModel',
]
