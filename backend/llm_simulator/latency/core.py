"""
延迟计算核心模块

职责:
- 评估器初始化与管理
- GEMM/FA2 通用接口
"""

from typing import Optional
from ..types import HardwareConfig, get_bytes_per_element
from ..evaluators import (
    AcceleratorMicroArch,
    GEMMEvaluator,
    GEMMResult,
    FA2Evaluator,
    FA2Result,
    get_arch_preset,
)

# 全局状态
_current_arch: Optional[AcceleratorMicroArch] = None
_gemm_evaluator: Optional[GEMMEvaluator] = None
_fa2_evaluator: Optional[FA2Evaluator] = None


def init_evaluators(hardware: HardwareConfig) -> None:
    """初始化评估器 (模拟开始时调用)"""
    global _current_arch, _gemm_evaluator, _fa2_evaluator
    _current_arch = _create_arch_from_hardware(hardware)
    _gemm_evaluator = GEMMEvaluator(_current_arch)
    _fa2_evaluator = FA2Evaluator(_current_arch)


def _create_arch_from_hardware(hardware: HardwareConfig) -> AcceleratorMicroArch:
    """从硬件配置创建微架构"""
    chip = hardware.chip
    chip_type = getattr(chip, 'chip_type', '').lower()

    # 查找预设
    if 'sg2260' in chip_type:
        arch = get_arch_preset('sg2260e')
    elif 'h100' in chip_type:
        arch = get_arch_preset('h100')
    elif 'a100' in chip_type:
        arch = get_arch_preset('a100')
    else:
        # 默认使用 sg2260e
        arch = get_arch_preset('sg2260e')

    # 更新带宽
    utilization = getattr(chip, 'memory_bandwidth_utilization', 0.85)
    arch.dram_bandwidth_bytes = chip.memory_bandwidth_gbps * 1e9 * utilization
    return arch


def get_arch() -> AcceleratorMicroArch:
    """获取当前微架构"""
    if _current_arch is None:
        raise RuntimeError("评估器未初始化，请先调用 init_evaluators()")
    return _current_arch


def get_evaluator() -> GEMMEvaluator:
    """获取当前 GEMM 评估器"""
    if _gemm_evaluator is None:
        raise RuntimeError("评估器未初始化，请先调用 init_evaluators()")
    return _gemm_evaluator


def calc_gemm_latency(M: int, K: int, N: int, G: int = 1) -> float:
    """
    计算 GEMM 延迟 (ms)

    Args:
        M: 输出行数
        K: 累加维度
        N: 输出列数
        G: Batch 维度

    Returns:
        延迟 (ms)
    """
    evaluator = get_evaluator()
    result = evaluator.evaluate(G, M, K, N)
    return result.latency_us / 1000


def calc_gemm_with_details(M: int, K: int, N: int, G: int = 1) -> GEMMResult:
    """
    计算 GEMM 延迟并返回详细信息

    Args:
        M: 输出行数
        K: 累加维度
        N: 输出列数
        G: Batch 维度

    Returns:
        GEMMResult 详细结果
    """
    evaluator = get_evaluator()
    return evaluator.evaluate(G, M, K, N)


def calc_memory_latency(data_bytes: int) -> float:
    """
    计算内存访问延迟 (ms)

    Args:
        data_bytes: 数据量 (字节)

    Returns:
        延迟 (ms)
    """
    arch = get_arch()
    data_gb = data_bytes / 1e9
    bandwidth_gb_per_s = arch.dram_bandwidth_bytes / 1e9
    return (data_gb / bandwidth_gb_per_s) * 1000 if bandwidth_gb_per_s > 0 else 0.0


# ==================== FA2 接口 ====================

def get_fa2_evaluator() -> FA2Evaluator:
    """获取当前 FA2 评估器"""
    if _fa2_evaluator is None:
        raise RuntimeError("评估器未初始化，请先调用 init_evaluators()")
    return _fa2_evaluator


def calc_fa2_latency(
    B: int,
    QS: int,
    KS: int,
    QD: int,
    VD: int,
) -> float:
    """
    计算 Flash Attention 2 延迟 (ms)

    Args:
        B: Batch size (通常是 num_heads)
        QS: Query 序列长度
        KS: Key/Value 序列长度
        QD: Query/Key 维度 (head_dim)
        VD: Value 维度 (通常等于 QD)

    Returns:
        延迟 (ms)
    """
    evaluator = get_fa2_evaluator()
    result = evaluator.evaluate(B, QS, KS, QD, VD)
    return result.latency_us / 1000


def calc_fa2_with_details(
    B: int,
    QS: int,
    KS: int,
    QD: int,
    VD: int,
) -> FA2Result:
    """
    计算 FA2 延迟并返回详细信息

    Args:
        B: Batch size (num_heads)
        QS: Query 序列长度
        KS: Key/Value 序列长度
        QD: Query/Key 维度
        VD: Value 维度

    Returns:
        FA2Result 详细结果
    """
    evaluator = get_fa2_evaluator()
    return evaluator.evaluate(B, QS, KS, QD, VD)
