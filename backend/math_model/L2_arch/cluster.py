"""集群规格实现

实现 ClusterSpec。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from math_model.L2_arch.node import NodeSpecImpl
from math_model.L2_arch.board import BoardSpecImpl
from math_model.L2_arch.chip import ChipSpecImpl, chip_registry
from math_model.L2_arch.topology import TopologySpecImpl
from math_model.L2_arch.interconnect import ChipInterconnectSpecImpl


@dataclass
class ClusterSpecImpl:
    """集群规格实现

    Attributes:
        num_nodes: 节点数量
        nodes: 节点列表
        inter_node_bandwidth_gbps: 跨节点带宽 (Gbps)
        inter_node_latency_us: 跨节点延迟 (us)
        topology_ref: 拓扑引用键
        topology: 拓扑规格
    """

    num_nodes: int = 0
    nodes: list[NodeSpecImpl] = field(default_factory=list)
    inter_node_bandwidth_gbps: float = 0.0
    inter_node_latency_us: float = 0.0
    topology_ref: str | None = None
    topology: TopologySpecImpl | None = None

    def __post_init__(self) -> None:
        if self.num_nodes <= 0 and self.nodes:
            self.num_nodes = len(self.nodes)

    def get_node(self, node_id: str) -> NodeSpecImpl:
        """获取指定节点"""
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        raise KeyError(f"Node ID '{node_id}' not found")

    def list_nodes(self) -> list[NodeSpecImpl]:
        """按稳定顺序返回节点"""
        return list(self.nodes)

    def get_total_chips(self) -> int:
        """获取集群总芯片数"""
        return sum(node.chips_per_node for node in self.nodes)

    def get_total_compute(self, dtype: str) -> float:
        """获取集群总算力

        Args:
            dtype: 数据类型

        Returns:
            总算力 (FLOPS)
        """
        return sum(node.get_total_compute(dtype) for node in self.nodes)

    def get_total_memory(self) -> int:
        """获取集群总内存

        Returns:
            总内存 (bytes)
        """
        return sum(node.get_total_memory() for node in self.nodes)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "ClusterSpecImpl":
        """从配置创建集群规格

        支持 Tier6 风格的拓扑配置格式。

        Args:
            config: 配置字典

        Returns:
            ClusterSpecImpl 实例
        """
        # 解析节点配置
        nodes: list[NodeSpecImpl] = []

        # Tier6 风格配置 (grouped_pods 格式)
        if "pods" in config:
            nodes = cls._from_tier6_config(config)
        # CHIPMathica 风格配置
        elif "nodes" in config:
            for node_config in config.get("nodes", []):
                node = cls._parse_node(node_config)
                nodes.append(node)

        return cls(
            num_nodes=len(nodes),
            nodes=nodes,
            inter_node_bandwidth_gbps=config.get("inter_node_bandwidth_gbps", 100.0),
            inter_node_latency_us=config.get("inter_node_latency_us", 5.0),
            topology_ref=config.get("topology_ref"),
        )

    @classmethod
    def _from_tier6_config(cls, config: dict[str, Any]) -> list[NodeSpecImpl]:
        """从 grouped_pods 格式配置创建节点列表

        遍历 pods[].racks[].boards[].chips[], 展开 count 分组,
        每个 rack 实例生成一个 NodeSpecImpl.

        Args:
            config: grouped_pods 格式拓扑配置

        Returns:
            节点列表
        """
        nodes: list[NodeSpecImpl] = []

        pods = config.get("pods", [])
        chips_config = config.get("chips", {})
        interconnect_config = config.get("interconnect", {}).get("links", {})

        node_idx = 0
        pod_global_idx = 0
        for pod_group in pods:
            pod_count = pod_group.get("count", 1)
            rack_groups = pod_group.get("racks", [])

            for _ in range(pod_count):
                rack_global_idx = 0
                for rack_group in rack_groups:
                    rack_count = rack_group.get("count", 1)
                    board_groups = rack_group.get("boards", [])

                    for _ in range(rack_count):
                        boards: list[BoardSpecImpl] = []

                        for board_group in board_groups:
                            board_count = board_group.get("count", 1)
                            chips_in_board = board_group.get("chips", [])

                            for _ in range(board_count):
                                board_chips: list[ChipSpecImpl] = []

                                for chip_entry in chips_in_board:
                                    chip_name = chip_entry.get("name", "unknown")
                                    chip_count = chip_entry.get("count", 1)
                                    preset_id = chip_entry.get("preset_id")

                                    if preset_id and chip_registry.has(preset_id):
                                        base_chip = chip_registry.get(preset_id)
                                    elif chip_name in chips_config:
                                        base_chip = ChipSpecImpl.from_config(
                                            chip_name, chips_config[chip_name]
                                        )
                                    else:
                                        base_chip = ChipSpecImpl(name=chip_name)

                                    for _ in range(chip_count):
                                        chip = base_chip.with_chip_id(len(board_chips))
                                        board_chips.append(chip)

                                c2c_config = interconnect_config.get("c2c", {})
                                chip_interconnect = ChipInterconnectSpecImpl(
                                    topology="ring",
                                    link_bandwidth_gbps=c2c_config.get("bandwidth_gbps", 112.0),
                                    link_count=10,
                                    latency_ns=c2c_config.get("latency_us", 0.5) * 1000,
                                    chip_count=len(board_chips),
                                )

                                board = BoardSpecImpl(
                                    name=f"board_{pod_global_idx}_{rack_global_idx}_{len(boards)}",
                                    chip_count=len(board_chips),
                                    chips=board_chips,
                                    chip_interconnect=chip_interconnect,
                                )
                                boards.append(board)

                        b2b_config = interconnect_config.get("b2b", {})
                        intra_node_bw = b2b_config.get("bandwidth_gbps", 200.0)
                        intra_node_lat = b2b_config.get("latency_us", 2.0)

                        node = NodeSpecImpl(
                            node_id=f"node_{node_idx}",
                            boards=boards,
                            intra_node_bandwidth_gbps=intra_node_bw,
                            intra_node_latency_us=intra_node_lat,
                        )
                        nodes.append(node)
                        node_idx += 1
                        rack_global_idx += 1

                pod_global_idx += 1

        return nodes

    @classmethod
    def _parse_node(cls, node_config: dict[str, Any]) -> NodeSpecImpl:
        """解析节点配置

        Args:
            node_config: 节点配置字典

        Returns:
            NodeSpecImpl 实例
        """
        boards: list[BoardSpecImpl] = []
        for board_config in node_config.get("boards", []):
            board = BoardSpecImpl.from_config(
                board_config.get("name", "board"),
                board_config,
            )
            boards.append(board)

        return NodeSpecImpl(
            node_id=node_config.get("node_id", "node_0"),
            boards=boards,
            intra_node_bandwidth_gbps=node_config.get("intra_node_bandwidth_gbps", 200.0),
            intra_node_latency_us=node_config.get("intra_node_latency_us", 2.0),
        )

    def to_summary(self) -> dict[str, Any]:
        """导出为汇总参数格式

        Returns:
            汇总参数字典
        """
        return {
            "num_nodes": self.num_nodes,
            "total_chips": self.get_total_chips(),
            "total_compute_bf16": self.get_total_compute("BF16"),
            "total_memory_gb": self.get_total_memory() / 1024**3,
            "inter_node_bandwidth_gbps": self.inter_node_bandwidth_gbps,
            "inter_node_latency_us": self.inter_node_latency_us,
        }
