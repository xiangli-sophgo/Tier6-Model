"""
Softmax 10步向量操作评估 (Numba JIT 加速)

移植自 llm_simulator/evaluators/softmax_eval.py (DS_TPU_1209)
用于 FA2 评估器的 Softmax 向量操作精确估算

Softmax 被拆分为 10 个向量操作步骤，每步区分：
- theoretical_ops: 对齐到硬件（lane_num/eu_num）后的操作数
- real_ops: 实际有效计算量
"""

import math
import numba
import numpy as np
from typing import Tuple


# Softmax 操作步骤（NumPy 数组，Numba 兼容）
# (shape_type, op_count)
# shape_type: 0 = (QS, 1, KS), 1 = (QS, 1, 1)
SOFTMAX_STEPS_ARRAY = np.array([
    [0, 1],   # add
    [1, 1],   # reduce_max
    [1, 1],   # max
    [0, 35],  # fuse_exp
    [1, 35],  # fuse_exp
    [1, 1],   # reduce_sum
    [1, 1],   # mul
    [1, 1],   # add
    [1, 1],   # copy
    [0, 1],   # data_convert
], dtype=np.int32)


@numba.jit(nopython=True, cache=True)
def _align_up(value: int, alignment: int) -> int:
    """Round up to nearest multiple of alignment."""
    if alignment <= 1:
        return value
    return int(math.ceil(value / alignment)) * alignment


@numba.jit(nopython=True, cache=True)
def _softmax_core(
    QS: int, KS: int, lane_num: int, eu_num: int, dtype_bytes: int,
) -> Tuple[int, int]:
    """Softmax 核心计算（Numba JIT）"""
    vector_theo = 0
    vector_real = 0

    if lane_num > 0 and dtype_bytes > 0:
        eu_block = eu_num // lane_num // dtype_bytes
        eu_block = max(1, eu_block)
    else:
        eu_block = 1

    aligned_qs = _align_up(QS, lane_num)
    aligned_ks = _align_up(KS, eu_block)

    steps = SOFTMAX_STEPS_ARRAY
    for i in range(10):
        shape_type = steps[i, 0]
        op_count = steps[i, 1]

        if shape_type == 0:
            theo = aligned_qs * aligned_ks * op_count
            real = QS * KS * op_count
        else:
            theo = aligned_qs * op_count
            real = QS * op_count

        vector_theo += theo
        vector_real += real

    return vector_theo, vector_real


def softmax_theoretical_and_real(
    QS: int, KS: int, lane_num: int, eu_num: int, dtype_bytes: int,
) -> Tuple[int, int]:
    """
    计算 Softmax 操作的理论和实际操作数

    Args:
        QS: Query 序列长度 (tile 粒度)
        KS: Key 序列长度 (tile 粒度)
        lane_num: SIMD lane 数量 (行维对齐)
        eu_num: 执行单元数量 (列维对齐基数)
        dtype_bytes: 数据类型字节数 (BF16=2)

    Returns:
        (vector_theo, vector_real): 理论操作数和实际操作数
    """
    return _softmax_core(QS, KS, lane_num, eu_num, dtype_bytes)
