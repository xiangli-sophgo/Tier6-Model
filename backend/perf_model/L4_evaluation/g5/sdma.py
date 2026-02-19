"""SDMA 核间通信引擎 (G5 指令级仿真模式)

简化延迟模型 (单核版, 无 NoC 距离建模):
    latency_ns = startup_latency_ns + data_bytes / (bandwidth_gbps * efficiency)

复用 DMAEngineImpl.get_transfer_time() 方法。
多核 NoC 距离建模留到 Step 5。

SG2262 参数: bandwidth=64GB/s, startup=120ns, efficiency=0.85

参考设计: docs/plans/G5-instruction-level-simulator.md
"""

from __future__ import annotations

from dataclasses import dataclass

from perf_model.L2_arch.chip import ChipSpecImpl
from perf_model.L3_mapping.g5.program import SDMACommand


@dataclass
class SDMAResult:
    """SDMA 指令执行结果

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


def calc_sdma_latency(cmd: SDMACommand, chip: ChipSpecImpl) -> SDMAResult:
    """计算 SDMA 搬运延迟

    使用芯片配置中的 SDMA 引擎参数。

    Args:
        cmd: SDMA 指令
        chip: 芯片规格

    Returns:
        SDMAResult 包含延迟和数据量
    """
    sdma = chip.dma_engines.get("sdma")
    if sdma is None:
        raise ValueError(f"Chip '{chip.name}' has no 'sdma' DMA engine configured")

    latency_ns = sdma.get_transfer_time(cmd.data_bytes)
    startup_ns = sdma.startup_latency_ns
    transfer_ns = latency_ns - startup_ns
    return SDMAResult(
        latency_ns=latency_ns, data_bytes=cmd.data_bytes,
        startup_ns=startup_ns, transfer_ns=transfer_ns,
    )
