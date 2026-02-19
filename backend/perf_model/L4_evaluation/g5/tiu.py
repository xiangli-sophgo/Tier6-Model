"""TIU 计算引擎 (G5 指令级仿真模式)

MM2_NN 延迟公式 (对标 TPUPerf):
    init_cycles = 44
    total_cycles = ceil(tile_m / lane_num) * ceil(tile_n / eu_num)
                 * (ceil(tile_k / ch_per_cyc) + bank_conflict) + init_cycles
    latency_ns = total_cycles / tiu_frequency_ghz

ch_per_cyc 按精度缩放 (对齐 TPUPerf CUBE_IC_*):
    cube_k 配置值 = INT8 通道数 (如 32)
    INT8:  ch_per_cyc = cube_k      = 32
    BF16:  ch_per_cyc = cube_k // 2 = 16
    FP32:  ch_per_cyc = cube_k // 4 = 8

SG2262 参数映射 (对齐 TPUPerf sg2262_template.json):
    lane_num = 16 (lanes_per_core)
    eu_num = cube_n = 8
    tiu_frequency_ghz = 1.0 (clk_tiu = 1000 MHz)

参考设计: docs/plans/G5-instruction-level-simulator.md
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from perf_model.L2_arch.chip import ChipSpecImpl
from perf_model.L3_mapping.g5.program import TIUCommand, TIUOpType

# TIU 初始化 cycles (流水线填充, 对齐 TPUPerf init_cycle=44)
TIU_INIT_CYCLES = 44

# precision -> dtype_bytes 映射
_PRECISION_BYTES = {"INT8": 1, "BF16": 2, "FP16": 2, "FP32": 4, "INT4": 1}


@dataclass
class TIUResult:
    """TIU 指令执行结果

    Attributes:
        latency_ns: 执行延迟 (ns)
        cycles: 总 cycle 数
        flops: 浮点运算量
        init_cycles: 流水线填充周期
        compute_cycles: 纯计算周期
    """
    latency_ns: float
    cycles: int
    flops: int
    init_cycles: int = 0
    compute_cycles: int = 0


def calc_tiu_latency(cmd: TIUCommand, chip: ChipSpecImpl) -> TIUResult:
    """计算 TIU 指令延迟

    Args:
        cmd: TIU 指令
        chip: 芯片规格

    Returns:
        TIUResult 包含延迟、cycle 数、FLOPs
    """
    if cmd.op_type == TIUOpType.MM2_NN:
        return _calc_mm2_nn(cmd, chip)
    raise ValueError(f"Unsupported TIU op_type: {cmd.op_type}")


def _get_ch_per_cyc(cube_k_int8: int, precision: str) -> int:
    """根据精度获取每 cycle 处理的通道数

    cube_k 配置值代表 INT8 的通道数。
    其他精度按元素字节数等比缩放:
        ch_per_cyc = cube_k_int8 // dtype_bytes
    """
    dtype_bytes = _PRECISION_BYTES.get(precision)
    if dtype_bytes is None:
        raise ValueError(f"Unsupported precision '{precision}' for ch_per_cyc calculation")
    ch = cube_k_int8 // dtype_bytes
    if ch <= 0:
        raise ValueError(
            f"ch_per_cyc={ch} for precision={precision}: "
            f"cube_k_int8={cube_k_int8}, dtype_bytes={dtype_bytes}"
        )
    return ch


def _calc_mm2_nn(cmd: TIUCommand, chip: ChipSpecImpl) -> TIUResult:
    """MM2_NN (MatMul) 延迟计算

    total_cycles = ceil(M/lane_num) * ceil(N/eu_num) * ceil(K/ch_per_cyc) + init_cycles
    """
    lane_num = chip.lane_per_core
    eu_num = chip.cube_n
    cube_k_int8 = chip.cube_k
    tiu_freq = chip.get_tiu_frequency()

    if lane_num <= 0:
        raise ValueError(f"Chip '{chip.name}' lane_per_core={lane_num} must be > 0")
    if eu_num <= 0:
        raise ValueError(f"Chip '{chip.name}' cube_n={eu_num} must be > 0")
    if cube_k_int8 <= 0:
        raise ValueError(f"Chip '{chip.name}' cube_k={cube_k_int8} must be > 0")
    if tiu_freq <= 0:
        raise ValueError(f"Chip '{chip.name}' tiu_frequency={tiu_freq} must be > 0")

    ch_per_cyc = _get_ch_per_cyc(cube_k_int8, cmd.precision)

    m_iters = math.ceil(cmd.tile_m / lane_num)
    n_iters = math.ceil(cmd.tile_n / eu_num)
    k_iters = math.ceil(cmd.tile_k / ch_per_cyc)

    compute_cycles = m_iters * n_iters * k_iters
    cycles = compute_cycles + TIU_INIT_CYCLES
    latency_ns = cycles / tiu_freq

    flops = 2 * cmd.tile_m * cmd.tile_n * cmd.tile_k

    return TIUResult(
        latency_ns=latency_ns, cycles=cycles, flops=flops,
        init_cycles=TIU_INIT_CYCLES, compute_cycles=compute_cycles,
    )
