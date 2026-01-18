"""
拓扑解析模块

解析前端的 HierarchicalTopology 配置，构建芯片互联图，
并根据并行策略将芯片分配到 TP/PP/DP/EP 组。
"""

from typing import Any
from .types import (
    HierarchicalTopology, PodConfig, RackConfig, BoardConfig, ChipConfig,
    ConnectionConfig, ChipNode, ChipLink, InterconnectGraph,
    ParallelismStrategy, ChipAssignment, ParallelGroupAssignment,
    HardwareConfig,
)


class TopologyParser:
    """拓扑解析器"""

    def __init__(self, topology_dict: dict[str, Any], hardware: HardwareConfig):
        """
        初始化解析器

        Args:
            topology_dict: 前端传入的拓扑配置字典
            hardware: 硬件配置
        """
        self.hardware = hardware
        self.topology = self._parse_topology(topology_dict)
        self.interconnect: InterconnectGraph | None = None

    def _parse_topology(self, data: dict[str, Any]) -> HierarchicalTopology:
        """解析拓扑配置"""
        pods = []
        for pod_data in data.get("pods", []):
            racks = []
            for rack_data in pod_data.get("racks", []):
                boards = []
                for board_data in rack_data.get("boards", []):
                    chips = []
                    for chip_data in board_data.get("chips", []):
                        pos = chip_data.get("position", [0, 0])
                        chips.append(ChipConfig(
                            id=chip_data["id"],
                            type=chip_data.get("type", "chip"),
                            position=tuple(pos) if isinstance(pos, list) else pos,
                            label=chip_data.get("label", ""),
                        ))
                    boards.append(BoardConfig(
                        id=board_data["id"],
                        u_position=board_data.get("u_position", 1),
                        u_height=board_data.get("u_height", 1),
                        label=board_data.get("label", ""),
                        chips=chips,
                    ))
                pos = rack_data.get("position", [0, 0])
                racks.append(RackConfig(
                    id=rack_data["id"],
                    position=tuple(pos) if isinstance(pos, list) else pos,
                    label=rack_data.get("label", ""),
                    total_u=rack_data.get("total_u", 42),
                    boards=boards,
                ))
            grid = pod_data.get("grid_size", [1, 1])
            pods.append(PodConfig(
                id=pod_data["id"],
                label=pod_data.get("label", ""),
                grid_size=tuple(grid) if isinstance(grid, list) else grid,
                racks=racks,
            ))

        connections = []
        for conn_data in data.get("connections", []):
            connections.append(ConnectionConfig(
                source=conn_data["source"],
                target=conn_data["target"],
                type=conn_data.get("type", "intra"),
                bandwidth=conn_data.get("bandwidth", 0),
                latency=conn_data.get("latency", 0),
            ))

        return HierarchicalTopology(pods=pods, connections=connections)

    def build_interconnect_graph(self) -> InterconnectGraph:
        """构建芯片互联图"""
        nodes: list[ChipNode] = []
        links: list[ChipLink] = []

        # 收集所有芯片
        chip_to_location: dict[str, ChipNode] = {}
        for pod in self.topology.pods:
            for rack in pod.racks:
                for board in rack.boards:
                    for chip in board.chips:
                        node = ChipNode(
                            chip_id=chip.id,
                            pod_id=pod.id,
                            rack_id=rack.id,
                            board_id=board.id,
                            position=chip.position,
                        )
                        nodes.append(node)
                        chip_to_location[chip.id] = node

        # 解析连接并确定链路参数
        for conn in self.topology.connections:
            src = conn.source
            dst = conn.target

            # 跳过非芯片连接
            if src not in chip_to_location or dst not in chip_to_location:
                continue

            # 确定链路类型和参数
            src_node = chip_to_location[src]
            dst_node = chip_to_location[dst]

            if src_node.board_id == dst_node.board_id:
                # 同板芯片 - 使用节点内带宽 (如NVLink)
                link_type = "nvlink"
                bandwidth = self.hardware.node.intra_node_bandwidth_gbps
                latency = self.hardware.node.intra_node_latency_us
            elif src_node.rack_id == dst_node.rack_id:
                # 同机柜不同板 - 使用PCIe或机柜内网络
                link_type = "pcie"
                bandwidth = self.hardware.chip.pcie_bandwidth_gbps
                latency = self.hardware.chip.pcie_latency_us
            elif src_node.pod_id == dst_node.pod_id:
                # 同Pod不同机柜 - 使用节点间网络
                link_type = "ib"
                bandwidth = self.hardware.cluster.inter_node_bandwidth_gbps
                latency = self.hardware.cluster.inter_node_latency_us
            else:
                # 跨Pod - 使用数据中心网络
                link_type = "ethernet"
                bandwidth = self.hardware.cluster.inter_node_bandwidth_gbps / 2  # 假设跨Pod带宽减半
                latency = self.hardware.cluster.inter_node_latency_us * 2  # 延迟增加

            # 使用连接配置中的显式值覆盖默认值
            if conn.bandwidth > 0:
                bandwidth = conn.bandwidth
            if conn.latency > 0:
                latency = conn.latency / 1000  # ns -> us

            links.append(ChipLink(
                source=src,
                target=dst,
                bandwidth_gbps=bandwidth,
                latency_us=latency,
                link_type=link_type,
            ))

        self.interconnect = InterconnectGraph(nodes=nodes, links=links)
        return self.interconnect

    def get_all_chip_ids(self) -> list[str]:
        """获取所有芯片ID（按层级顺序）"""
        chip_ids = []
        for pod in self.topology.pods:
            for rack in pod.racks:
                for board in rack.boards:
                    for chip in board.chips:
                        chip_ids.append(chip.id)
        return chip_ids

    def map_parallelism(self, strategy: ParallelismStrategy) -> ParallelGroupAssignment:
        """
        根据并行策略将芯片分配到各并行组

        分配顺序（从内到外）：TP -> EP -> PP -> DP
        - TP组优先放在同板芯片（高带宽NVLink）
        - PP组可以跨板（P2P通信）
        - DP组可以跨机柜甚至跨Pod

        Args:
            strategy: 并行策略

        Returns:
            ParallelGroupAssignment: 并行组分配结果
        """
        chip_ids = self.get_all_chip_ids()
        total_chips = strategy.dp * strategy.tp * strategy.pp * strategy.ep

        if len(chip_ids) < total_chips:
            raise ValueError(
                f"芯片数量不足: 需要 {total_chips} 个芯片 "
                f"(DP={strategy.dp} × TP={strategy.tp} × PP={strategy.pp} × EP={strategy.ep})，"
                f"但只有 {len(chip_ids)} 个芯片"
            )

        assignments: list[ChipAssignment] = []
        tp_groups: list[list[str]] = []
        pp_groups: list[list[str]] = []
        dp_groups: list[list[str]] = []
        ep_groups: list[list[str]] = []

        # 初始化组列表
        num_tp_groups = strategy.dp * strategy.pp * strategy.ep
        num_pp_groups = strategy.dp * strategy.tp * strategy.ep
        num_dp_groups = strategy.tp * strategy.pp * strategy.ep
        num_ep_groups = strategy.dp * strategy.tp * strategy.pp

        for _ in range(num_tp_groups):
            tp_groups.append([])
        for _ in range(num_pp_groups):
            pp_groups.append([])
        for _ in range(num_dp_groups):
            dp_groups.append([])
        for _ in range(num_ep_groups):
            ep_groups.append([])

        # 分配芯片到各组
        global_rank = 0
        for dp in range(strategy.dp):
            for pp in range(strategy.pp):
                for ep in range(strategy.ep):
                    for tp in range(strategy.tp):
                        if global_rank >= len(chip_ids):
                            break

                        chip_id = chip_ids[global_rank]

                        # 创建分配记录
                        assignment = ChipAssignment(
                            chip_id=chip_id,
                            global_rank=global_rank,
                            dp_rank=dp,
                            tp_rank=tp,
                            pp_rank=pp,
                            ep_rank=ep,
                            sp_rank=0,  # SP通常与TP绑定
                        )
                        assignments.append(assignment)

                        # 计算组索引
                        tp_group_idx = dp * strategy.pp * strategy.ep + pp * strategy.ep + ep
                        pp_group_idx = dp * strategy.tp * strategy.ep + tp * strategy.ep + ep
                        dp_group_idx = tp * strategy.pp * strategy.ep + pp * strategy.ep + ep
                        ep_group_idx = dp * strategy.tp * strategy.pp + tp * strategy.pp + pp

                        # 添加到对应组
                        tp_groups[tp_group_idx].append(chip_id)
                        pp_groups[pp_group_idx].append(chip_id)
                        dp_groups[dp_group_idx].append(chip_id)
                        if ep_group_idx < len(ep_groups):
                            ep_groups[ep_group_idx].append(chip_id)

                        global_rank += 1

        return ParallelGroupAssignment(
            assignments=assignments,
            tp_groups=tp_groups,
            pp_groups=pp_groups,
            dp_groups=dp_groups,
            ep_groups=ep_groups,
        )

    def get_link_params_for_group(
        self,
        group_chips: list[str],
        comm_type: str  # 'allreduce' | 'p2p' | 'alltoall'
    ) -> tuple[float, float]:
        """
        获取组内通信的链路参数

        对于 AllReduce，使用组内最慢链路的参数
        对于 P2P，使用源目标之间的链路参数

        Args:
            group_chips: 组内芯片ID列表
            comm_type: 通信类型

        Returns:
            (带宽 Gbps, 延迟 us) 元组
        """
        if not self.interconnect:
            self.build_interconnect_graph()

        if len(group_chips) <= 1:
            # 单芯片不需要通信，返回最大带宽和零延迟
            # 避免使用 float('inf')，因为它无法 JSON 序列化
            return self.hardware.node.intra_node_bandwidth_gbps, 0.0

        # 找到组内所有芯片的位置
        chip_locations = {}
        for node in self.interconnect.nodes:
            if node.chip_id in group_chips:
                chip_locations[node.chip_id] = node

        # 确定链路类型
        min_bandwidth = float('inf')
        max_latency = 0.0

        for i, chip1 in enumerate(group_chips):
            for chip2 in group_chips[i + 1:]:
                loc1 = chip_locations.get(chip1)
                loc2 = chip_locations.get(chip2)

                if not loc1 or not loc2:
                    continue

                # 根据位置确定链路参数
                if loc1.board_id == loc2.board_id:
                    bw = self.hardware.node.intra_node_bandwidth_gbps
                    lat = self.hardware.node.intra_node_latency_us
                elif loc1.rack_id == loc2.rack_id:
                    bw = self.hardware.chip.pcie_bandwidth_gbps
                    lat = self.hardware.chip.pcie_latency_us
                elif loc1.pod_id == loc2.pod_id:
                    bw = self.hardware.cluster.inter_node_bandwidth_gbps
                    lat = self.hardware.cluster.inter_node_latency_us
                else:
                    bw = self.hardware.cluster.inter_node_bandwidth_gbps / 2
                    lat = self.hardware.cluster.inter_node_latency_us * 2

                min_bandwidth = min(min_bandwidth, bw)
                max_latency = max(max_latency, lat)

        if min_bandwidth == float('inf'):
            # 如果没有找到链路，使用默认值
            min_bandwidth = self.hardware.node.intra_node_bandwidth_gbps
            max_latency = self.hardware.node.intra_node_latency_us

        return min_bandwidth, max_latency
