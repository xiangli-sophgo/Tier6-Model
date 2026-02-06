"""互联规格实现

实现 NoC、C2C、芯片间互联等互联规格。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


# ============================================================================
# AllReduce Models
# ============================================================================


@dataclass
class RingAllReduceModel:
    """Ring AllReduce 模型

    时间 = 2 * (n-1) / n * data_size / bandwidth + 2 * (n-1) * latency
    """

    bandwidth_gbps: float
    latency_ns: float = 500.0

    def estimate_time(self, data_size: int, chip_count: int) -> float:
        """估算 AllReduce 时间

        Args:
            data_size: 数据大小 (bytes)
            chip_count: 参与芯片数量

        Returns:
            估算时间 (ns)
        """
        if chip_count <= 1:
            return 0.0

        # 带宽转换: GB/s -> bytes/ns
        bandwidth_bytes_per_ns = self.bandwidth_gbps

        # Ring AllReduce: 2 * (n-1) / n * data_size / bandwidth
        n = chip_count
        comm_time = 2 * (n - 1) / n * data_size / bandwidth_bytes_per_ns

        # 延迟: 2 * (n-1) * latency (reduce-scatter + all-gather)
        latency_time = 2 * (n - 1) * self.latency_ns

        return comm_time + latency_time

    def get_bandwidth_utilization(self) -> float:
        """获取带宽利用率

        Ring AllReduce 的理论利用率为 (n-1)/n

        Returns:
            利用率 (0-1)
        """
        return 0.9  # 典型值


@dataclass
class TreeAllReduceModel:
    """Tree AllReduce 模型

    时间 = 2 * log2(n) * data_size / bandwidth + 2 * log2(n) * latency
    """

    bandwidth_gbps: float
    latency_ns: float = 500.0

    def estimate_time(self, data_size: int, chip_count: int) -> float:
        """估算 AllReduce 时间

        Args:
            data_size: 数据大小 (bytes)
            chip_count: 参与芯片数量

        Returns:
            估算时间 (ns)
        """
        if chip_count <= 1:
            return 0.0

        bandwidth_bytes_per_ns = self.bandwidth_gbps
        log_n = math.ceil(math.log2(chip_count))

        # Tree AllReduce: 2 * log2(n) * data_size / bandwidth
        comm_time = 2 * log_n * data_size / bandwidth_bytes_per_ns

        # 延迟: 2 * log2(n) * latency
        latency_time = 2 * log_n * self.latency_ns

        return comm_time + latency_time

    def get_bandwidth_utilization(self) -> float:
        """获取带宽利用率

        Returns:
            利用率 (0-1)
        """
        return 0.5  # Tree 利用率较低


AllReduceModel = RingAllReduceModel | TreeAllReduceModel


# ============================================================================
# Chip Interconnect (片内互联)
# ============================================================================


@dataclass
class InterconnectSpecImpl:
    """片内互联规格实现

    Attributes:
        noc_topology: NoC 拓扑类型 (mesh/ring/tree)
        noc_bandwidth_gbps: NoC 带宽
        noc_latency_ns: NoC 延迟
        c2c_links: 片间链路数
        c2c_bandwidth_gbps: 片间带宽
    """

    noc_topology: str = "mesh"
    noc_bandwidth_gbps: float = 1000.0
    noc_latency_ns: float = 10.0
    c2c_links: int = 0
    c2c_bandwidth_gbps: float = 0.0

    def get_hop_latency(self, src: str = "", dst: str = "") -> float:
        """获取跳延迟

        Args:
            src: 源位置
            dst: 目标位置

        Returns:
            延迟 (ns)
        """
        return self.noc_latency_ns

    def get_bisection_bandwidth(self) -> float:
        """获取对分带宽

        Returns:
            对分带宽 (GB/s)
        """
        # 简化模型：对分带宽 = NoC 带宽 / 2
        return self.noc_bandwidth_gbps / 2

    def get_allreduce_model(self, topology: str = "ring") -> AllReduceModel:
        """获取 AllReduce 模型

        Args:
            topology: 拓扑类型 (ring/tree)

        Returns:
            AllReduce 模型
        """
        if topology == "tree":
            return TreeAllReduceModel(
                bandwidth_gbps=self.c2c_bandwidth_gbps,
                latency_ns=self.noc_latency_ns,
            )
        else:
            return RingAllReduceModel(
                bandwidth_gbps=self.c2c_bandwidth_gbps,
                latency_ns=self.noc_latency_ns,
            )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "InterconnectSpecImpl":
        """从配置创建

        Args:
            config: 配置字典

        Returns:
            InterconnectSpecImpl 实例
        """
        noc_config = config.get("noc", {})
        c2c_config = config.get("c2c", {})

        return cls(
            noc_topology=noc_config.get("topology", "mesh"),
            noc_bandwidth_gbps=noc_config.get("bandwidth_gbps", 1000.0),
            noc_latency_ns=noc_config.get("latency_ns", 10.0),
            c2c_links=c2c_config.get("links", 0),
            c2c_bandwidth_gbps=c2c_config.get("bandwidth_gbps", 0.0),
        )


# ============================================================================
# Chip-to-Chip Interconnect (芯片间互联)
# ============================================================================


@dataclass
class ChipInterconnectSpecImpl:
    """芯片间互联规格实现

    描述板卡内多芯片之间的互联拓扑。

    Attributes:
        topology: 拓扑类型 (ring/mesh/fully_connected/tree)
        link_bandwidth_gbps: 单链路带宽
        link_count: 链路数量
        latency_ns: 基础延迟
        chip_count: 芯片数量
    """

    topology: str = "ring"
    link_bandwidth_gbps: float = 112.0
    link_count: int = 10
    latency_ns: float = 500.0
    chip_count: int = 8

    def get_path_bandwidth(self, src: int, dst: int) -> float:
        """获取两芯片间的有效带宽

        Args:
            src: 源芯片编号
            dst: 目标芯片编号

        Returns:
            有效带宽 (GB/s)
        """
        if src == dst:
            return float("inf")

        if self.topology == "fully_connected":
            # 全连接：直接使用链路带宽
            return self.link_bandwidth_gbps * self.link_count

        hops = self.get_path_hops(src, dst)
        if hops == 0:
            return float("inf")

        # 带宽按跳数衰减
        return self.link_bandwidth_gbps * self.link_count / hops

    def get_path_hops(self, src: int, dst: int) -> int:
        """获取两芯片间的跳数

        Args:
            src: 源芯片编号
            dst: 目标芯片编号

        Returns:
            跳数
        """
        if src == dst:
            return 0

        if self.topology == "ring":
            # Ring 拓扑：取最短路径
            forward = (dst - src) % self.chip_count
            backward = (src - dst) % self.chip_count
            return min(forward, backward)

        elif self.topology == "fully_connected":
            # 全连接：1 跳
            return 1

        elif self.topology == "mesh":
            # Mesh 拓扑：曼哈顿距离
            side = int(math.sqrt(self.chip_count))
            src_x, src_y = src % side, src // side
            dst_x, dst_y = dst % side, dst // side
            return abs(dst_x - src_x) + abs(dst_y - src_y)

        elif self.topology == "tree":
            # Tree 拓扑：通过根节点
            return 2 * int(math.ceil(math.log2(self.chip_count)))

        return 1

    def get_allreduce_time(self, data_size: int, chip_count: int) -> float:
        """估算 AllReduce 时间

        Args:
            data_size: 数据大小 (bytes)
            chip_count: 参与芯片数量

        Returns:
            估算时间 (ns)
        """
        if self.topology == "ring":
            model = RingAllReduceModel(self.link_bandwidth_gbps, self.latency_ns)
        else:
            model = TreeAllReduceModel(self.link_bandwidth_gbps, self.latency_ns)

        return model.estimate_time(data_size, chip_count)

    @classmethod
    def from_config(
        cls, config: dict[str, Any], chip_count: int = 8
    ) -> "ChipInterconnectSpecImpl":
        """从配置创建

        Args:
            config: 配置字典
            chip_count: 芯片数量

        Returns:
            ChipInterconnectSpecImpl 实例
        """
        return cls(
            topology=config.get("topology", "ring"),
            link_bandwidth_gbps=config.get("link_bandwidth_gbps", 112.0),
            link_count=config.get("link_count", 10),
            latency_ns=config.get("latency_ns", 500.0),
            chip_count=chip_count,
        )


# ============================================================================
# Board Memory (板级共享内存)
# ============================================================================


@dataclass
class BoardMemorySpecImpl:
    """板级共享内存规格实现

    Attributes:
        capacity_bytes: 容量 (bytes)
        bandwidth_gbps: 带宽 (GB/s)
        latency_ns: 延迟 (ns)
    """

    capacity_bytes: int = 0
    bandwidth_gbps: float = 0.0
    latency_ns: float = 0.0

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "BoardMemorySpecImpl":
        """从配置创建

        Args:
            config: 配置字典

        Returns:
            BoardMemorySpecImpl 实例
        """
        capacity = config.get("capacity_bytes", 0)
        if capacity == 0 and "capacity_gb" in config:
            capacity = int(config["capacity_gb"] * 1024**3)

        return cls(
            capacity_bytes=capacity,
            bandwidth_gbps=config.get("bandwidth_gbps", 0.0),
            latency_ns=config.get("latency_ns", 0.0),
        )
