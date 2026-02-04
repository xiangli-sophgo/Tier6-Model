"""层定义模块

提供层基类和注册表。
"""

from tier6.core.registry import Registry
from tier6.L1_workload.layers.base import LayerBase, LayerRole

# 全局层注册表
layer_registry: Registry[LayerBase] = Registry("layer")

# 导入所有层实现以触发注册
from tier6.L1_workload.layers.attention import AttentionLayer
from tier6.L1_workload.layers.dsa import DSALayer
from tier6.L1_workload.layers.embedding import EmbeddingLayer
from tier6.L1_workload.layers.ffn import FFNLayer
from tier6.L1_workload.layers.lmhead import LMHeadLayer
from tier6.L1_workload.layers.mla import MLALayer
from tier6.L1_workload.layers.mla_absorb import MLAAbsorbLayer
from tier6.L1_workload.layers.mla_absorb_v3_2 import MLAAbsorbv32Layer
from tier6.L1_workload.layers.mla_v3_2 import MLAv32Layer
from tier6.L1_workload.layers.mlp import MLPLayer
from tier6.L1_workload.layers.moe import MoELayer

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
