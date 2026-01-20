"""
MatMul (GEMM) 算子

矩阵乘法: (G, M, K) × (G, K, N) = (G, M, N)
"""

from dataclasses import dataclass, field
from typing import Dict, Any

from ..base import ComputeOperator, ComputeOpType


@dataclass
class MatMulOperator(ComputeOperator):
    """
    矩阵乘法算子

    parallel_params 必须包含:
        - G: int, 分组数/批次维度
        - M: int, 输出行数 (通常是 batch * seq_len)
        - K: int, 累加维度 (输入特征维度)
        - N: int, 输出列数 (输出特征维度)
        - input_dtype: str, 输入数据类型 ('bf16', 'fp16', 'fp8')
        - output_dtype: str, 输出数据类型 ('bf16', 'fp16', 'fp32')
    """
    name: str = ""
    op_type: ComputeOpType = ComputeOpType.MATMUL
    parallel_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """初始化后计算基础指标"""
        params = self.parallel_params
        G = params.get('G', 1)
        M = params.get('M', 1)
        K = params.get('K', 1)
        N = params.get('N', 1)

        # 计算浮点操作数: 2 * G * M * K * N (乘加各算一次)
        self.flops = 2 * G * M * K * N

        # 权重参数量 (假设权重 shape 为 (G, K, N))
        input_dtype = params.get('input_dtype', 'bf16')
        dtype_bytes = {'fp8': 1, 'bf16': 2, 'fp16': 2, 'fp32': 4}.get(input_dtype, 2)
        self.param = G * K * N * dtype_bytes
        self.dram_occupy = self.param

    @property
    def G(self) -> int:
        return self.parallel_params.get('G', 1)

    @property
    def M(self) -> int:
        return self.parallel_params.get('M', 1)

    @property
    def K(self) -> int:
        return self.parallel_params.get('K', 1)

    @property
    def N(self) -> int:
        return self.parallel_params.get('N', 1)

    @property
    def input_dtype(self) -> str:
        return self.parallel_params.get('input_dtype', 'bf16')

    @property
    def output_dtype(self) -> str:
        return self.parallel_params.get('output_dtype', 'bf16')
