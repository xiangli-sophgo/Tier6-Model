"""
RMSNorm 向量操作评估器 (Numba JIT 加速)

RMSNorm 计算步骤:
1. square: x^2
2. reduce_sum: sum(x^2)
3. div_constant: sum / dim (31 cycles)
4. add_constant: + epsilon
5. rsqrt: 1 / sqrt(...) (30 cycles)
6. mul: x * rsqrt_result
7. mul_scale: * gamma (可选)
8. add_bias: + beta (可选, LayerNorm)
9. data_convert: 类型转换

迁移自 llm_simulator/evaluators/rmsnorm_eval.py
"""

import math
import numba
import numpy as np
from typing import Tuple


# RMSNorm 操作步骤（NumPy 数组，Numba 兼容）
# (shape_type, op_count, skip_if_no_scale, skip_if_no_bias)
RMSNORM_STEPS_ARRAY = np.array([
    [0, 1, 0, 0],   # square
    [1, 1, 0, 0],   # reduce_sum
    [1, 31, 0, 0],  # div_constant
    [1, 1, 0, 0],   # add_constant
    [1, 30, 0, 0],  # rsqrt
    [0, 1, 0, 0],   # mul
    [0, 1, 1, 0],   # mul_scale (skip if no scale)
    [0, 1, 0, 1],   # add_bias (skip if no bias)
    [0, 1, 0, 0],   # data_convert
], dtype=np.int32)


@numba.jit(nopython=True, cache=True)
def _align_up(value: int, alignment: int) -> int:
    """Round up to nearest multiple of alignment."""
    if alignment <= 1:
        return value
    return int(math.ceil(value / alignment)) * alignment


@numba.jit(nopython=True, cache=True)
def _rmsnorm_core(
    batch_size: int, hidden_dim: int, lane_num: int, eu_num: int,
    dtype_bytes: int, has_scale: int, has_bias: int,
) -> Tuple[int, int]:
    """RMSNorm 核心计算（Numba JIT）"""
    QS = batch_size
    KS = hidden_dim

    vector_theo = 0
    vector_real = 0

    aligned_qs = _align_up(QS, lane_num)

    if lane_num > 0 and dtype_bytes > 0:
        eu_block = eu_num // lane_num // dtype_bytes
        eu_block = max(1, eu_block)
    else:
        eu_block = 1

    aligned_ks = _align_up(KS, eu_block)

    steps = RMSNORM_STEPS_ARRAY
    for i in range(9):
        shape_type = steps[i, 0]
        op_count = steps[i, 1]
        skip_no_scale = steps[i, 2]
        skip_no_bias = steps[i, 3]

        if skip_no_scale and has_scale == 0:
            continue
        if skip_no_bias and has_bias == 0:
            continue

        if shape_type == 0:
            theo = aligned_qs * aligned_ks * op_count
            real = QS * KS * op_count
        else:
            theo = aligned_qs * op_count
            real = QS * op_count

        vector_theo += theo
        vector_real += real

    return vector_theo, vector_real


def rmsnorm_theoretical_and_real(
    batch_size: int,
    hidden_dim: int,
    lane_num: int,
    eu_num: int,
    dtype_bytes: int = 2,
    has_scale: bool = True,
    has_bias: bool = False,
) -> Tuple[int, int]:
    """
    计算 RMSNorm 的理论和实际向量操作数

    Args:
        batch_size: 批次大小 (行维)
        hidden_dim: 隐藏维度 (列维)
        lane_num: SIMD lane 数量 (行对齐基数)
        eu_num: 执行单元数量 (列对齐基数)
        dtype_bytes: 数据类型字节数 (BF16=2)
        has_scale: 是否包含 scale (gamma) 操作
        has_bias: 是否包含 bias (beta) 操作 (LayerNorm 有, RMSNorm 无)

    Returns:
        (vector_theo, vector_real): 理论操作数, 实际操作数
    """
    return _rmsnorm_core(
        batch_size, hidden_dim, lane_num, eu_num, dtype_bytes,
        1 if has_scale else 0,
        1 if has_bias else 0,
    )
