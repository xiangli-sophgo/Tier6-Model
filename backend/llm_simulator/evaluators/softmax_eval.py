"""
Softmax 操作估算

移植自 DS_TPU_1209/performance/evaluate/compute/softmax_eval.py
用于 FA2 评估器的 Softmax 向量操作估算
"""

from typing import Tuple
from .utils import align_up


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
        QS: Query 序列长度
        KS: Key 序列长度
        lane_num: SIMD lane 数量
        eu_num: 执行单元数量
        dtype_bytes: 数据类型字节数 (通常 BF16=2)

    Returns:
        (vector_theo, vector_real): 理论操作数和实际操作数
    """

    def calc_step_theo(shape_type: int, op_count: int) -> int:
        """计算单步理论操作数"""
        if shape_type == 0:
            # shape = (QS, 1, KS)
            return (
                align_up(QS, lane_num) *
                align_up(KS, eu_num // lane_num // dtype_bytes) *
                op_count
            )
        else:
            # shape = (QS, 1, 1)
            return align_up(QS, lane_num) * op_count

    def calc_step_real(shape_type: int, op_count: int) -> int:
        """计算单步实际操作数"""
        if shape_type == 0:
            return QS * 1 * KS * op_count
        else:
            return QS * 1 * 1 * op_count

    vector_theo = 0
    vector_real = 0

    for _, shape_type, op_count in SOFTMAX_STEPS:
        vector_theo += calc_step_theo(shape_type, op_count)
        vector_real += calc_step_real(shape_type, op_count)

    return vector_theo, vector_real
