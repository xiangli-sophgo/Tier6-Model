"""
RMSNorm 向量操作评估器

RMSNorm 计算步骤:
1. square: x^2
2. reduce_sum: sum(x^2)
3. div_constant: sum / dim (31 cycles, 除法较慢)
4. add_constant: + epsilon
5. rsqrt: 1 / sqrt(...) (30 cycles, 开方较慢)
6. mul: x * rsqrt_result
7. mul_scale: * gamma (可选)
8. add_bias: + beta (可选, LayerNorm)
9. data_convert: 类型转换

每步区分:
- shape_type=0: 2D 操作 (batch x hidden)
- shape_type=1: 1D 操作 (batch x 1)

迁移自 llm_simulator/evaluators/rmsnorm_eval.py
"""

import math
from typing import Tuple, List


# RMSNorm 操作步骤定义
# (操作名, shape_type, op_count)
# shape_type: 0=2D操作(QS x KS), 1=1D操作(QS x 1)
# op_count: 操作周期数
RMSNORM_STEPS: List[Tuple[str, int, int]] = [
    ('square', 0, 1),         # x^2
    ('reduce_sum', 1, 1),     # sum(x^2)
    ('div_constant', 1, 31),  # sum / dim (除法较慢)
    ('add_constant', 1, 1),   # + epsilon
    ('rsqrt', 1, 30),         # 1/sqrt(...) (开方较慢)
    ('mul', 0, 1),            # x * rsqrt_result
    ('mul_scale', 0, 1),      # * gamma (可选)
    ('add_bias', 0, 1),       # + beta (可选)
    ('data_convert', 0, 1),   # 类型转换
]


def _align_up(value: int, alignment: int) -> int:
    """Round up to nearest multiple of alignment."""
    if alignment <= 1:
        return value
    return math.ceil(value / alignment) * alignment


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
        batch_size: 批次大小 (行维, 对应 QS)
        hidden_dim: 隐藏维度 (列维, 对应 KS)
        lane_num: SIMD lane 数量 (行对齐基数)
        eu_num: 执行单元数量 (列对齐基数)
        dtype_bytes: 数据类型字节数 (BF16=2)
        has_scale: 是否包含 scale (gamma) 操作
        has_bias: 是否包含 bias (beta) 操作 (LayerNorm 有, RMSNorm 无)

    Returns:
        (vector_theo, vector_real): 理论操作数, 实际操作数
    """
    QS = batch_size
    KS = hidden_dim

    def calc_step_theo(shape_type: int, op_count: int) -> int:
        if shape_type == 0:
            aligned_qs = _align_up(QS, lane_num)
            eu_block = eu_num // lane_num // dtype_bytes if lane_num > 0 and dtype_bytes > 0 else 1
            eu_block = max(1, eu_block)
            aligned_ks = _align_up(KS, eu_block)
            return aligned_qs * aligned_ks * op_count
        else:
            aligned_qs = _align_up(QS, lane_num)
            return aligned_qs * op_count

    def calc_step_real(shape_type: int, op_count: int) -> int:
        if shape_type == 0:
            return QS * KS * op_count
        else:
            return QS * op_count

    vector_theo = 0
    vector_real = 0

    for name, shape_type, op_count in RMSNORM_STEPS:
        if name == 'mul_scale' and not has_scale:
            continue
        if name == 'add_bias' and not has_bias:
            continue

        vector_theo += calc_step_theo(shape_type, op_count)
        vector_real += calc_step_real(shape_type, op_count)

    return vector_theo, vector_real
