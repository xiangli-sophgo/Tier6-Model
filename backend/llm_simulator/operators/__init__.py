"""
算子模块 - DS_TPU 风格的算子定义

三层抽象体系:
- Operator: 最小计算/通信单元
- Layer: 由多个 Operator 组成
- Model: 由多个 Layer 组成
"""

from .base import ComputeOperator, CommunicationOperator
from .compute import (
    MatMulOperator,
    FA2Operator,
    MHAOperator,
    MQAOperator,
    SoftmaxOperator,
    RMSNormOperator,
)
from .communication import (
    AllReduceOperator,
    AllGatherOperator,
    ReduceScatterOperator,
    DispatchOperator,
    CombineOperator,
)

__all__ = [
    # 基类
    'ComputeOperator',
    'CommunicationOperator',
    # 计算算子
    'MatMulOperator',
    'FA2Operator',
    'MHAOperator',
    'MQAOperator',
    'SoftmaxOperator',
    'RMSNormOperator',
    # 通信算子
    'AllReduceOperator',
    'AllGatherOperator',
    'ReduceScatterOperator',
    'DispatchOperator',
    'CombineOperator',
]
