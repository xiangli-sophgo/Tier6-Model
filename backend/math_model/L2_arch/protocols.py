"""L2 Architecture Spec Protocols

定义硬件规格层的核心协议接口，供 L3/L4 使用。

层级结构:
    Cluster (集群)
    └── Node (节点)
        └── Board (板卡)
            └── Chip (芯片)
                ├── Core (核心)
                ├── ComputeUnits (计算单元: Cube/Vector/HAU)
                ├── MemoryHierarchy (存储层级: GMEM/L2M/LMEM)
                ├── DMAEngines (DMA 引擎: GDMA/SDMA/CDMA/ARE)
                └── Interconnect (片内互联: NoC)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from math_model.L0_entry.types import Bandwidth, Latency, MemorySize, FLOPs


# ============================================================================
# Cluster/Node Level Protocols (集群/节点级)
# ============================================================================


@runtime_checkable
class ClusterSpec(Protocol):
    """集群规格接口"""

    @property
    def num_nodes(self) -> int:
        """节点数量"""
        ...

    @property
    def nodes(self) -> list["NodeSpec"]:
        """节点列表"""
        ...

    @property
    def inter_node_bandwidth_gbps(self) -> "Bandwidth":
        """跨节点带宽 (Gbps)"""
        ...

    @property
    def inter_node_latency_us(self) -> "Latency":
        """跨节点延迟 (us)"""
        ...

    @property
    def topology_ref(self) -> str | None:
        """拓扑引用键"""
        ...

    def get_node(self, node_id: str) -> "NodeSpec":
        """获取指定节点"""
        ...

    def list_nodes(self) -> list["NodeSpec"]:
        """按稳定顺序返回节点"""
        ...


@runtime_checkable
class NodeSpec(Protocol):
    """节点规格接口"""

    @property
    def node_id(self) -> str:
        """节点标识"""
        ...

    @property
    def boards(self) -> list["BoardSpec"]:
        """板卡列表"""
        ...

    @property
    def chips_per_node(self) -> int:
        """每节点芯片数"""
        ...

    @property
    def intra_node_bandwidth_gbps(self) -> "Bandwidth":
        """节点内带宽 (Gbps)"""
        ...

    @property
    def intra_node_latency_us(self) -> "Latency":
        """节点内延迟 (us)"""
        ...

    def list_chips(self) -> list[int]:
        """返回节点内所有芯片 ID"""
        ...


# ============================================================================
# Board Level Protocols (板卡级)
# ============================================================================


@runtime_checkable
class BoardSpec(Protocol):
    """板卡规格接口"""

    @property
    def board_id(self) -> str:
        """板卡标识"""
        ...

    @property
    def name(self) -> str:
        """板卡名称"""
        ...

    @property
    def chip_count(self) -> int:
        """芯片数量"""
        ...

    @property
    def chips(self) -> list["ChipSpec"]:
        """芯片列表"""
        ...

    @property
    def chip_interconnect(self) -> "ChipInterconnectSpec":
        """芯片间互联规格"""
        ...

    @property
    def board_memory(self) -> "BoardMemorySpec | None":
        """板级共享内存（可选）"""
        ...

    @property
    def fabric_tag(self) -> str:
        """互联标签"""
        ...

    def get_chip(self, chip_id: int) -> "ChipSpec":
        """获取指定芯片"""
        ...

    def get_total_compute(self, dtype: str) -> "FLOPs":
        """获取板卡总算力"""
        ...

    def get_total_memory(self) -> "MemorySize":
        """获取板卡总内存"""
        ...

    def get_chip_to_chip_bandwidth(self, src: int, dst: int) -> "Bandwidth":
        """获取芯片间带宽"""
        ...


@runtime_checkable
class ChipInterconnectSpec(Protocol):
    """芯片间互联规格"""

    @property
    def topology(self) -> str:
        """拓扑类型"""
        ...

    @property
    def link_bandwidth_gbps(self) -> "Bandwidth":
        """单链路带宽 (GB/s)"""
        ...

    @property
    def link_count(self) -> int:
        """链路数量"""
        ...

    @property
    def latency_ns(self) -> "Latency":
        """基础延迟 (ns)"""
        ...

    def get_path_bandwidth(self, src: int, dst: int) -> "Bandwidth":
        """获取两芯片间的有效带宽"""
        ...

    def get_path_hops(self, src: int, dst: int) -> int:
        """获取两芯片间的跳数"""
        ...

    def get_allreduce_time(self, data_size: int, chip_count: int) -> "Latency":
        """估算 AllReduce 时间"""
        ...


@runtime_checkable
class BoardMemorySpec(Protocol):
    """板级共享内存规格"""

    @property
    def capacity_bytes(self) -> "MemorySize":
        """容量 (bytes)"""
        ...

    @property
    def bandwidth_gbps(self) -> "Bandwidth":
        """带宽 (GB/s)"""
        ...

    @property
    def latency_ns(self) -> "Latency":
        """延迟 (ns)"""
        ...


# ============================================================================
# Chip Level Protocols (芯片级)
# ============================================================================


@runtime_checkable
class ChipSpec(Protocol):
    """芯片规格接口"""

    @property
    def name(self) -> str:
        """芯片名称"""
        ...

    @property
    def chip_id(self) -> int:
        """在板卡中的编号"""
        ...

    @property
    def core_count(self) -> int:
        """核心数量"""
        ...

    @property
    def lane_per_core(self) -> int:
        """每核心 Lane 数"""
        ...

    @property
    def cores(self) -> list["CoreSpec"]:
        """核心规格列表"""
        ...

    @property
    def frequency_ghz(self) -> float:
        """工作频率 (GHz)"""
        ...

    @property
    def compute_units(self) -> dict[str, "ComputeSpec"]:
        """计算单元 (Cube/Vector/HAU)"""
        ...

    @property
    def memory_hierarchy(self) -> "MemoryHierarchy":
        """存储层级"""
        ...

    @property
    def dma_engines(self) -> dict[str, "DMASpec"]:
        """DMA 引擎"""
        ...

    @property
    def interconnect(self) -> "InterconnectSpec":
        """片内互联"""
        ...

    def get_peak_flops(self, dtype: str, unit: str = "cube") -> "FLOPs":
        """获取峰值算力"""
        ...

    def get_total_sram(self) -> "MemorySize":
        """获取片内 SRAM 总量"""
        ...


# ============================================================================
# Compute Unit Protocols (计算单元)
# ============================================================================


@runtime_checkable
class ComputeSpec(Protocol):
    """计算单元规格接口"""

    @property
    def name(self) -> str:
        """单元名称"""
        ...

    def peak_flops(self, dtype: str) -> "FLOPs":
        """获取峰值算力"""
        ...

    def eu_count(self, dtype: str) -> int:
        """获取执行单元数量"""
        ...


# ============================================================================
# Memory Protocols (存储)
# ============================================================================


@runtime_checkable
class MemoryHierarchy(Protocol):
    """存储层级接口"""

    def get_level(self, name: str) -> "MemoryLevel":
        """获取指定层级"""
        ...

    def list_levels(self) -> list[str]:
        """列出所有层级名称"""
        ...


@runtime_checkable
class MemoryLevel(Protocol):
    """单层存储规格接口"""

    @property
    def name(self) -> str:
        """层级名称"""
        ...

    @property
    def capacity_bytes(self) -> "MemorySize":
        """容量 (bytes)"""
        ...

    @property
    def bandwidth_gbps(self) -> "Bandwidth":
        """带宽 (GB/s)"""
        ...

    @property
    def latency_ns(self) -> "Latency":
        """延迟 (ns)"""
        ...

    @property
    def scope(self) -> str:
        """共享域"""
        ...


# ============================================================================
# DMA Protocols (数据搬运)
# ============================================================================


@runtime_checkable
class DMASpec(Protocol):
    """DMA 引擎规格接口"""

    @property
    def name(self) -> str:
        """引擎名称"""
        ...

    @property
    def bandwidth_gbps(self) -> "Bandwidth":
        """带宽 (GB/s)"""
        ...

    @property
    def startup_latency_ns(self) -> "Latency":
        """启动延迟 (ns)"""
        ...

    @property
    def supported_paths(self) -> list[tuple[str, str]]:
        """支持的源/目的层级组合"""
        ...

    @property
    def scope(self) -> str:
        """作用域"""
        ...

    def get_transfer_time(self, data_size: int, src: str, dst: str) -> "Latency":
        """计算传输时间"""
        ...


# ============================================================================
# Interconnect Protocols (互联)
# ============================================================================


@runtime_checkable
class InterconnectSpec(Protocol):
    """片内互联规格接口"""

    @property
    def noc_topology(self) -> str:
        """NoC 拓扑类型 (mesh/ring/tree)"""
        ...

    @property
    def c2c_links(self) -> int:
        """片间链路数"""
        ...

    @property
    def c2c_bandwidth_gbps(self) -> "Bandwidth":
        """片间带宽 (GB/s)"""
        ...

    def get_hop_latency(self, src: str, dst: str) -> "Latency":
        """获取跳延迟"""
        ...

    def get_bisection_bandwidth(self) -> "Bandwidth":
        """获取对分带宽"""
        ...

    def get_allreduce_model(self, topology: str) -> "AllReduceModel":
        """获取 AllReduce 模型"""
        ...


# ============================================================================
# Core/Topology Protocols (核心/拓扑)
# ============================================================================


@runtime_checkable
class CoreSpec(Protocol):
    """核心规格接口"""

    @property
    def core_id(self) -> int:
        """核心标识"""
        ...

    @property
    def num_lanes(self) -> int:
        """Lane 数量"""
        ...

    @property
    def sram_per_core_kb(self) -> float:
        """每核 SRAM 容量 (KB)"""
        ...

    @property
    def cube_mac_per_lane(self) -> int:
        """每 Lane Cube MAC 数"""
        ...

    @property
    def vector_eu_per_lane(self) -> int:
        """每 Lane Vector EU 数"""
        ...

    @property
    def lmem_per_lane_kb(self) -> float:
        """每 Lane LMEM 容量 (KB)"""
        ...

    @property
    def core_fabric_share(self) -> str:
        """片内互联共享系数/标签"""
        ...


@runtime_checkable
class TopologySpec(Protocol):
    """通信拓扑规格接口"""

    @property
    def nodes(self) -> dict[str, list[str]]:
        """节点与板卡从属关系"""
        ...

    @property
    def boards(self) -> dict[str, list[str]]:
        """板卡与芯片从属关系"""
        ...

    @property
    def chips(self) -> list[str]:
        """芯片清单"""
        ...

    @property
    def rank_map(self) -> dict[str, int]:
        """chip_id -> rank 映射"""
        ...

    @property
    def link_profiles(self) -> dict[str, "LinkProfile"]:
        """链路参数"""
        ...

    @property
    def path_keys(self) -> list[str]:
        """路径键列表"""
        ...

    def get_link_profile(self, path_key: str) -> "LinkProfile":
        """获取链路参数"""
        ...

    def resolve_path(self, src_chip: str | int, dst_chip: str | int) -> tuple[str, int]:
        """解析路径键与估计跳数"""
        ...


@runtime_checkable
class LinkProfile(Protocol):
    """链路参数接口"""

    @property
    def bandwidth_gbps(self) -> "Bandwidth":
        """带宽 (Gbps)"""
        ...

    @property
    def latency_us(self) -> "Latency":
        """延迟 (us)"""
        ...


@runtime_checkable
class AllReduceModel(Protocol):
    """AllReduce 通信模型接口"""

    def estimate_time(self, data_size: int, chip_count: int) -> "Latency":
        """估算 AllReduce 时间"""
        ...

    def get_bandwidth_utilization(self) -> float:
        """获取带宽利用率"""
        ...
