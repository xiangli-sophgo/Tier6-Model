"""
层模块 - DS_TPU 风格的层定义

每个 Layer 包含多个 Operator，提供:
- 算子注册和管理
- 性能指标聚合
- 信息导出
"""

from .base import BaseLayer
from .embedding import EmbeddingLayer
from .attention import MLALayer, MHALayer
from .ffn import MLPLayer
from .moe import MoELayer
from .lmhead import LMHeadLayer

__all__ = [
    'BaseLayer',
    'EmbeddingLayer',
    'MLALayer',
    'MHALayer',
    'MLPLayer',
    'MoELayer',
    'LMHeadLayer',
]
