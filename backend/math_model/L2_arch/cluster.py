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

        # Tier6 风格配置
        if "pod_count" in config or "rack_config" in config:
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
        """从 Tier6 风格配置创建节点列表

        Args:
            config: Tier6 拓扑配置

        Returns:
            节点列表
        """
        nodes: list[NodeSpecImpl] = []

        pod_count = config.get("pod_count", 1)
        racks_per_pod = config.get("racks_per_pod", 1)
        rack_config = config.get("rack_config", {})

        # 解析板卡配置
        boards_config = rack_config.get("boards", [])

        # 硬件参数
        hardware_params = config.get("hardware_params", {})
        chips_config = hardware_params.get("chips", {})
        interconnect_config = hardware_params.get("interconnect", {})

        node_idx = 0
        for pod_idx in range(pod_count):
            for rack_idx in range(racks_per_pod):
                # 每个 rack 对应一个 node
                boards: list[BoardSpecImpl] = []

                for board_config in boards_config:
                    board_chips: list[ChipSpecImpl] = []
                    chips_in_board = board_config.get("chips", [])

                    for chip_entry in chips_in_board:
                        chip_name = chip_entry.get("name", "unknown")
                        chip_count = chip_entry.get("count", 1)
                        preset_id = chip_entry.get("preset_id")

                        # 尝试从配置或注册表获取芯片规格
                        if preset_id and chip_registry.has(preset_id):
                            base_chip = chip_registry.get(preset_id)
                        elif chip_name in chips_config:
                            base_chip = ChipSpecImpl.from_config(
                                chip_name, chips_config[chip_name]
                            )
                        else:
                            # 创建默认芯片
                            base_chip = ChipSpecImpl(name=chip_name)

                        for i in range(chip_count):
                            chip = base_chip.with_chip_id(len(board_chips))
                            board_chips.append(chip)

                    # 创建芯片间互联
                    c2c_config = interconnect_config.get("c2c", {})
                    chip_interconnect = ChipInterconnectSpecImpl(
                        topology="ring",
                        link_bandwidth_gbps=c2c_config.get("bandwidth_gbps", 112.0),
                        link_count=10,
                        latency_ns=c2c_config.get("latency_us", 0.5) * 1000,
                        chip_count=len(board_chips),
                    )

                    board = BoardSpecImpl(
                        name=f"board_{pod_idx}_{rack_idx}_{len(boards)}",
                        chip_count=len(board_chips),
                        chips=board_chips,
                        chip_interconnect=chip_interconnect,
                    )
                    boards.append(board)

                # 计算节点内带宽
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
