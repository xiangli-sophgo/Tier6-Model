"""
RMSNorm 向量操作评估器

RMSNorm 计算步骤:
1. square: x^2
2. reduce_sum: sum(x^2)
3. div_constant: sum / dim
4. add_constant: + epsilon
5. rsqrt: 1 / sqrt(...)
6. mul: x * rsqrt_result
7. mul_scale: * gamma (可选)
8. add_bias: + beta (可选)
9. data_convert: 类型转换

每个步骤分为:
- shape_type=0: 2D 操作 (batch × hidden)
- shape_type=1: 1D 操作 (batch × 1)
"""

from typing import Tuple, List
from dataclasses import dataclass

from .utils import align_up


# RMSNorm 操作步骤定义
# (操作名, shape_type, op_count)
# shape_type: 0=2D操作(QS×KS), 1=1D操作(QS×1)
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


@dataclass
class RMSNormResult:
    """RMSNorm 评估结果"""

    vector_ops_theoretical: int
    """理论向量操作数 (考虑对齐)"""

    vector_ops_real: int
    """实际向量操作数"""

    utilization: float
    """向量单元利用率"""


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
        batch_size: 批次大小 (对应 QS)
        hidden_dim: 隐藏维度 (对应 KS)
        lane_num: SIMD lane 数量 (行对齐基数)
        eu_num: 执行单元数量
        dtype_bytes: 数据类型字节数 (BF16=2)
        has_scale: 是否包含 scale 操作
        has_bias: 是否包含 bias 操作

    Returns:
        (vector_theo, vector_real): 理论操作数, 实际操作数
    """
    QS = batch_size
    KS = hidden_dim

    def calc_step_theo(shape_type: int, op_count: int) -> int:
        """计算单步理论操作数 (考虑硬件对齐)"""
        if shape_type == 0:
            # 2D 操作: 需要对齐到 lane_num 和 eu 块
            aligned_qs = align_up(QS, lane_num)
            eu_block = eu_num // lane_num // dtype_bytes
            aligned_ks = align_up(KS, eu_block) if eu_block > 0 else KS
            return aligned_qs * aligned_ks * op_count
        else:
            # 1D 操作: 只需对齐 QS
            aligned_qs = align_up(QS, lane_num)
            return aligned_qs * op_count

    def calc_step_real(shape_type: int, op_count: int) -> int:
        """计算单步实际操作数"""
        if shape_type == 0:
            return QS * KS * op_count
        else:
            return QS * 1 * op_count

    vector_theo = 0
    vector_real = 0

    for name, shape_type, op_count in RMSNORM_STEPS:
        # 跳过可选操作
        if name == 'mul_scale' and not has_scale:
            continue
        if name == 'add_bias' and not has_bias:
            continue

        vector_theo += calc_step_theo(shape_type, op_count)
        vector_real += calc_step_real(shape_type, op_count)

    return vector_theo, vector_real


class RMSNormEvaluator:
    """RMSNorm 性能评估器"""

    def __init__(self, arch):
        """
        初始化评估器

        Args:
            arch: AcceleratorMicroArch 配置
        """
        self.arch = arch
        self.lane_num = arch.lane_num
        self.eu_num = arch.eu_num
        self.freq_ghz = arch.freq_ghz

    def evaluate(
        self,
        batch_size: int,
        hidden_dim: int,
        dtype_bytes: int = 2,
        has_scale: bool = True,
        has_bias: bool = False,
    ) -> RMSNormResult:
        """
        评估 RMSNorm 操作

        Args:
            batch_size: 批次大小
            hidden_dim: 隐藏维度
            dtype_bytes: 数据类型字节数
            has_scale: 是否有 scale
            has_bias: 是否有 bias

        Returns:
            RMSNormResult 评估结果
        """
        vector_theo, vector_real = rmsnorm_theoretical_and_real(
            batch_size=batch_size,
            hidden_dim=hidden_dim,
            lane_num=self.lane_num,
            eu_num=self.eu_num,
            dtype_bytes=dtype_bytes,
            has_scale=has_scale,
            has_bias=has_bias,
        )

        utilization = vector_real / vector_theo if vector_theo > 0 else 0.0

        return RMSNormResult(
            vector_ops_theoretical=vector_theo,
            vector_ops_real=vector_real,
            utilization=utilization,
        )

    def estimate_latency_us(
        self,
        batch_size: int,
        hidden_dim: int,
        dtype_bytes: int = 2,
        has_scale: bool = True,
        has_bias: bool = False,
    ) -> float:
        """
        估算 RMSNorm 延迟 (微秒)

        基于向量操作数和执行单元频率计算

        Args:
            batch_size: 批次大小
            hidden_dim: 隐藏维度
            dtype_bytes: 数据类型字节数
            has_scale: 是否有 scale
            has_bias: 是否有 bias

        Returns:
            延迟 (微秒)
        """
        vector_theo, _ = rmsnorm_theoretical_and_real(
            batch_size=batch_size,
            hidden_dim=hidden_dim,
            lane_num=self.lane_num,
            eu_num=self.eu_num,
            dtype_bytes=dtype_bytes,
            has_scale=has_scale,
            has_bias=has_bias,
        )

        # 每周期可执行的向量操作数
        ops_per_cycle = self.eu_num

        # 总周期数
        cycles = vector_theo / ops_per_cycle if ops_per_cycle > 0 else 0

        # 转换为微秒
        latency_us = cycles / (self.freq_ghz * 1e3)  # GHz * 1e3 = cycles/us

        return latency_us


# 模块级别的快捷函数
_rmsnorm_evaluator = None


def get_rmsnorm_evaluator(arch=None):
    """获取 RMSNorm 评估器单例"""
    global _rmsnorm_evaluator
    if _rmsnorm_evaluator is None and arch is not None:
        _rmsnorm_evaluator = RMSNormEvaluator(arch)
    return _rmsnorm_evaluator


def eval_rmsnorm(
    batch_size: int,
    hidden_dim: int,
    lane_num: int = 16,
    eu_num: int = 512,
    dtype_bytes: int = 2,
    has_scale: bool = True,
    has_bias: bool = False,
) -> Tuple[int, int]:
    """
    快捷函数: 计算 RMSNorm 向量操作数

    Returns:
        (vector_theo, vector_real)
    """
    return rmsnorm_theoretical_and_real(
        batch_size=batch_size,
        hidden_dim=hidden_dim,
        lane_num=lane_num,
        eu_num=eu_num,
        dtype_bytes=dtype_bytes,
        has_scale=has_scale,
        has_bias=has_bias,
    )
