"""
层模块 - DS_TPU 风格的层定义

每个 Layer 包含多个 Operator，提供:
- 算子注册和管理
- 性能指标聚合
- 信息导出
"""

from .base import BaseLayer
from .embedding import EmbeddingLayer
from .attention import (
    MLALayer,
    MLAv32Layer,
    MLAAbsorbLayer,
    MLAAbsorbv32Layer,
    MHALayer,
)
from .dsa import DSALayer
from .ffn import MLPLayer
from .moe import MoELayer
from .lmhead import LMHeadLayer

__all__ = [
    'BaseLayer',
    'EmbeddingLayer',
    # MLA 变体 (对齐 DS_TPU)
    'MLALayer',
    'MLAv32Layer',
    'MLAAbsorbLayer',
    'MLAAbsorbv32Layer',
    # 标准 MHA
    'MHALayer',
    # DSA (DeepSeek V3.2 稀疏注意力)
    'DSALayer',
    'MLPLayer',
    'MoELayer',
    'LMHeadLayer',
]
