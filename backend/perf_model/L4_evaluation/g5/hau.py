"""HAU 硬件排序引擎 (G5 指令级仿真模式)

延迟公式:
    SORT:   init_cycles + ceil(N / sort_width) * ceil(log2(N)) * compare_cycles
    TOP_K:  init_cycles + ceil(N / sort_width) * ceil(log2(K)) * compare_cycles
    UNIQUE: init_cycles + ceil(N / sort_width) * compare_cycles

SG2262 参数: sort_width=16, compare_cycles=1, init_cycles=20

参考设计: docs/plans/G5-instruction-level-simulator.md
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from perf_model.L2_arch.chip import ChipSpecImpl
from perf_model.L3_mapping.g5.program import HAUCommand, HAUOpType


@dataclass
class HAUResult:
    """HAU 指令执行结果

    Attributes:
        latency_ns: 执行延迟 (ns)
        cycles: 总 cycle 数
    """
    latency_ns: float
    cycles: int


def calc_hau_latency(cmd: HAUCommand, chip: ChipSpecImpl) -> HAUResult:
    """计算 HAU 指令延迟

    Args:
        cmd: HAU 指令
        chip: 芯片规格

    Returns:
        HAUResult 包含延迟和 cycle 数
    """
    hau_cfg = chip.hau_config
    if not hau_cfg:
        raise ValueError(f"Chip '{chip.name}' has no 'hau' config section")
    if "sort_width" not in hau_cfg:
        raise ValueError(f"Missing 'hau.sort_width' in chip config: {chip.name}")
    if "compare_cycles" not in hau_cfg:
        raise ValueError(f"Missing 'hau.compare_cycles' in chip config: {chip.name}")
    if "init_cycles" not in hau_cfg:
        raise ValueError(f"Missing 'hau.init_cycles' in chip config: {chip.name}")

    sort_width = int(hau_cfg["sort_width"])
    compare_cycles = int(hau_cfg["compare_cycles"])
    init_cycles = int(hau_cfg["init_cycles"])

    n = cmd.num_elements
    if n <= 0:
        return HAUResult(latency_ns=0.0, cycles=0)

    passes = math.ceil(n / sort_width)

    if cmd.op_type in (HAUOpType.SORT, HAUOpType.SORT_INDEX):
        # ceil(N/sort_width) * ceil(log2(N)) * compare_cycles + init
        log_n = math.ceil(math.log2(max(n, 2)))
        cycles = passes * log_n * compare_cycles + init_cycles
    elif cmd.op_type == HAUOpType.TOP_K:
        # ceil(N/sort_width) * ceil(log2(K)) * compare_cycles + init
        k = cmd.top_k
        if k <= 0:
            raise ValueError(f"HAU TOP_K requires top_k > 0, got {k}")
        log_k = math.ceil(math.log2(max(k, 2)))
        cycles = passes * log_k * compare_cycles + init_cycles
    elif cmd.op_type == HAUOpType.UNIQUE:
        # ceil(N/sort_width) * compare_cycles + init
        cycles = passes * compare_cycles + init_cycles
    else:
        raise ValueError(f"Unsupported HAU op_type: {cmd.op_type}")

    latency_ns = cycles / chip.get_tiu_frequency()
    return HAUResult(latency_ns=latency_ns, cycles=cycles)
