"""L2: 硬件架构规格层

定义芯片、板卡、集群等硬件规格。

层级结构:
    Cluster (集群)
    └── Node (节点)
        └── Board (板卡)
            └── Chip (芯片)
                ├── Core (核心)
                ├── ComputeUnits (计算单元: Cube/Vector/HAU)
                ├── MemoryHierarchy (存储层级: GMEM/L2M/LMEM)
                ├── DMAEngines (DMA 引擎: GDMA/SDMA/CDMA)
                └── Interconnect (片内互联: NoC)
"""

from tier6.L2_arch.chip import ChipSpecImpl, chip_registry
from tier6.L2_arch.core import CoreSpecImpl
from tier6.L2_arch.compute import CubeSpec, VectorSpec, HAUSpec, ComputeSpec, create_compute_unit
from tier6.L2_arch.memory import MemoryLevelImpl, MemoryHierarchyImpl, DEFAULT_MEMORY_HIERARCHY
from tier6.L2_arch.dma import DMAEngineImpl, GDMASpec, SDMASpec, CDMASpec, create_dma_engines
from tier6.L2_arch.interconnect import (
    InterconnectSpecImpl,
    ChipInterconnectSpecImpl,
    BoardMemorySpecImpl,
    RingAllReduceModel,
    TreeAllReduceModel,
    AllReduceModel,
)
from tier6.L2_arch.board import BoardSpecImpl, board_registry
from tier6.L2_arch.node import NodeSpecImpl
from tier6.L2_arch.cluster import ClusterSpecImpl
from tier6.L2_arch.topology import TopologySpec, TopologySpecImpl, LinkProfileImpl

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
    # 节点
    "NodeSpecImpl",
    # 集群
    "ClusterSpecImpl",
    # 拓扑
    "TopologySpec",
    "TopologySpecImpl",
    "LinkProfileImpl",
]
