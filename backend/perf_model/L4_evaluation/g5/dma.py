"""GDMA 搬运引擎 (G5 指令级仿真模式)

简化延迟模型:
    latency_ns = startup_latency_ns + data_bytes / (bandwidth_gbps * efficiency)

SG2262: startup=100ns, bandwidth=68GB/s, efficiency=0.9
注: GB/s = bytes/ns, 所以不需要单位转换

参考设计: docs/plans/G5-instruction-level-simulator.md
"""

from __future__ import annotations

from dataclasses import dataclass

from perf_model.L2_arch.chip import ChipSpecImpl
from perf_model.L3_mapping.g5.program import DMACommand


@dataclass
class DMAResult:
    """DMA 指令执行结果

    Attributes:
        latency_ns: 执行延迟 (ns)
        data_bytes: 搬运数据量 (bytes)
        startup_ns: 启动延迟 (ns)
        transfer_ns: 数据传输延迟 (ns)
    """
    latency_ns: float
    data_bytes: int
    startup_ns: float = 0.0
    transfer_ns: float = 0.0


def calc_dma_latency(cmd: DMACommand, chip: ChipSpecImpl) -> DMAResult:
    """计算 GDMA 搬运延迟

    使用芯片配置中的 GDMA 引擎参数。

    Args:
        cmd: DMA 指令
        chip: 芯片规格

    Returns:
        DMAResult 包含延迟和数据量
    """
    gdma = chip.dma_engines.get("gdma")
    if gdma is None:
        raise ValueError(f"Chip '{chip.name}' has no 'gdma' DMA engine configured")

    latency_ns = gdma.get_transfer_time(cmd.data_bytes)
    startup_ns = gdma.startup_latency_ns
    transfer_ns = latency_ns - startup_ns
    return DMAResult(
        latency_ns=latency_ns, data_bytes=cmd.data_bytes,
        startup_ns=startup_ns, transfer_ns=transfer_ns,
    )
