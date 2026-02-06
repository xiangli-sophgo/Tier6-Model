"""层定义模块

提供层基类和注册表。
"""

from math_model.core.registry import Registry
from math_model.L1_workload.layers.base import LayerBase, LayerRole

# 全局层注册表
layer_registry: Registry[LayerBase] = Registry("layer")

# 导入所有层实现以触发注册
from math_model.L1_workload.layers.attention import AttentionLayer
from math_model.L1_workload.layers.dsa import DSALayer
from math_model.L1_workload.layers.embedding import EmbeddingLayer
from math_model.L1_workload.layers.ffn import FFNLayer
from math_model.L1_workload.layers.lmhead import LMHeadLayer
from math_model.L1_workload.layers.mla import MLALayer
from math_model.L1_workload.layers.mla_absorb import MLAAbsorbLayer
from math_model.L1_workload.layers.mla_absorb_v3_2 import MLAAbsorbv32Layer
from math_model.L1_workload.layers.mla_v3_2 import MLAv32Layer
from math_model.L1_workload.layers.mlp import MLPLayer
from math_model.L1_workload.layers.moe import MoELayer

__all__ = [
    "LayerBase",
    "LayerRole",
    "layer_registry",
    # 层实现
    "AttentionLayer",
    "DSALayer",
    "EmbeddingLayer",
    "FFNLayer",
    "LMHeadLayer",
    "MLALayer",
    "MLAAbsorbLayer",
    "MLAAbsorbv32Layer",
    "MLAv32Layer",
    "MLPLayer",
    "MoELayer",
]
