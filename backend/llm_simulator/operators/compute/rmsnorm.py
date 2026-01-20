"""
RMSNorm 算子

RMSNorm(x) = x / sqrt(mean(x^2) + eps) * gamma
"""

from dataclasses import dataclass, field
from typing import Dict, Any

from ..base import ComputeOperator, ComputeOpType


@dataclass
class RMSNormOperator(ComputeOperator):
    """
    RMSNorm 算子

    parallel_params 必须包含:
        - batch_size: int, 批次大小
        - hidden_dim: int, 隐藏维度
        - has_scale: bool, 是否有 gamma 参数
        - has_bias: bool, 是否有 beta 参数
    """
    name: str = ""
    op_type: ComputeOpType = ComputeOpType.RMSNORM
    parallel_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """初始化后计算基础指标"""
        params = self.parallel_params
        batch_size = params.get('batch_size', 1)
        hidden_dim = params.get('hidden_dim', 1)
        has_scale = params.get('has_scale', True)
        has_bias = params.get('has_bias', False)

        # gamma 参数
        dtype_bytes = 2  # bf16
        self.param = hidden_dim * dtype_bytes if has_scale else 0
        if has_bias:
            self.param += hidden_dim * dtype_bytes
        self.dram_occupy = self.param

        # FLOPs 估算 (square + reduce_sum + rsqrt + mul)
        self.flops = batch_size * hidden_dim * 4

    @property
    def batch_size(self) -> int:
        return self.parallel_params.get('batch_size', 1)

    @property
    def hidden_dim(self) -> int:
        return self.parallel_params.get('hidden_dim', 1)
