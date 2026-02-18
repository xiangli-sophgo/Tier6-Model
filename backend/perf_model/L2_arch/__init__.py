"""L2: 硬件架构规格层

定义芯片、板卡、Pod 等硬件规格。

层级结构:
    Pod
    └── Rack
        └── Board (板卡)
            └── Chip (芯片)
                ├── Core (核心)
                ├── ComputeUnits (计算单元: Cube/Vector/HAU)
                ├── MemoryHierarchy (存储层级: GMEM/L2M/LMEM)
                ├── DMAEngines (DMA 引擎: GDMA/SDMA/CDMA)
                └── Interconnect (片内互联: NoC)
"""

from perf_model.L2_arch.chip import ChipSpecImpl, chip_registry
from perf_model.L2_arch.core import CoreSpecImpl
from perf_model.L2_arch.compute import CubeSpec, VectorSpec, HAUSpec, ComputeSpec, create_compute_unit
from perf_model.L2_arch.memory import MemoryLevelImpl, MemoryHierarchyImpl, DEFAULT_MEMORY_HIERARCHY
from perf_model.L2_arch.dma import DMAEngineImpl, GDMASpec, SDMASpec, CDMASpec, create_dma_engines
from perf_model.L2_arch.interconnect import (
    InterconnectSpecImpl,
    ChipInterconnectSpecImpl,
    BoardMemorySpecImpl,
    RingAllReduceModel,
    TreeAllReduceModel,
    AllReduceModel,
)
from perf_model.L2_arch.board import BoardSpecImpl, board_registry
from perf_model.L2_arch.rack import RackSpecImpl
from perf_model.L2_arch.pod import PodSpecImpl
from perf_model.L2_arch.topology import TopologySpec, TopologySpecImpl, LinkProfileImpl

__all__ = [
    # 芯片
    "ChipSpecImpl",
    "chip_registry",
    # 核心
    "CoreSpecImpl",
    # 计算单元
    "CubeSpec",
    "VectorSpec",
    "HAUSpec",
    "ComputeSpec",
    "create_compute_unit",
    # 存储
    "MemoryLevelImpl",
    "MemoryHierarchyImpl",
    "DEFAULT_MEMORY_HIERARCHY",
    # DMA
    "DMAEngineImpl",
    "GDMASpec",
    "SDMASpec",
    "CDMASpec",
    "create_dma_engines",
    # 互联
    "InterconnectSpecImpl",
    "ChipInterconnectSpecImpl",
    "BoardMemorySpecImpl",
    "RingAllReduceModel",
    "TreeAllReduceModel",
    "AllReduceModel",
    # 板卡
    "BoardSpecImpl",
    "board_registry",
    # Rack
    "RackSpecImpl",
    # Pod
    "PodSpecImpl",
    # 拓扑
    "TopologySpec",
    "TopologySpecImpl",
    "LinkProfileImpl",
]
