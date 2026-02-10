"""
Softmax 10步向量操作评估

移植自 llm_simulator/evaluators/softmax_eval.py (DS_TPU_1209)
用于 FA2 评估器的 Softmax 向量操作精确估算

Softmax 被拆分为 10 个向量操作步骤，每步区分：
- theoretical_ops: 对齐到硬件（lane_num/eu_num）后的操作数
- real_ops: 实际有效计算量
"""

import math
from typing import Tuple


# Softmax 操作步骤定义
# (操作名, shape_type, op_count)
# shape_type: 0 = (QS, 1, KS), 1 = (QS, 1, 1)
SOFTMAX_STEPS = [
    ('add', 0, 1),
    ('reduce_max', 1, 1),
    ('max', 1, 1),
    ('fuse_exp', 0, 35),
    ('fuse_exp', 1, 35),
    ('reduce_sum', 1, 1),
    ('mul', 1, 1),
    ('add', 1, 1),
    ('copy', 1, 1),
    ('data_convert', 0, 1),
]


def _align_up(value: int, alignment: int) -> int:
    """Round up to nearest multiple of alignment."""
    if alignment <= 1:
        return value
    return math.ceil(value / alignment) * alignment


def softmax_theoretical_and_real(
    QS: int,
    KS: int,
    lane_num: int,
    eu_num: int,
    dtype_bytes: int,
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

    def calc_step_theo(shape_type: int, op_count: int) -> int:
        """计算单步理论操作数（对齐到硬件）"""
        if shape_type == 0:
            # shape = (QS, 1, KS)
            eu_block = eu_num // lane_num // dtype_bytes if lane_num > 0 and dtype_bytes > 0 else 1
            eu_block = max(1, eu_block)
            return (
                _align_up(QS, lane_num)
                * _align_up(KS, eu_block)
                * op_count
            )
        else:
            # shape = (QS, 1, 1)
            return _align_up(QS, lane_num) * op_count

    def calc_step_real(shape_type: int, op_count: int) -> int:
        """计算单步实际操作数"""
        if shape_type == 0:
            return QS * KS * op_count
        else:
            return QS * op_count

    vector_theo = 0
    vector_real = 0

    for _, shape_type, op_count in SOFTMAX_STEPS:
        vector_theo += calc_step_theo(shape_type, op_count)
        vector_real += calc_step_real(shape_type, op_count)

    return vector_theo, vector_real
