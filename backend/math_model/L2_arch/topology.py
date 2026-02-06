"""通信拓扑规格实现

实现 TopologySpec。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TopologySpec:
    """L2 拓扑通信参数口径

    Attributes:
        intra_board_bw_gbps: 板内互联带宽（GB/s）
        inter_board_bw_gbps: 板间互联带宽（GB/s）
        inter_node_bw_gbps: 节点间互联带宽（GB/s）
        c2c_lat_us: C2C 固定延迟（us）
        ddr_r_lat_us: DDR 读延迟（us）
        ddr_w_lat_us: DDR 写延迟（us）
        noc_lat_us: NoC 延迟（us）
        d2d_lat_us: Die-to-Die 延迟（us）
        link_delay_us: 链路附加延迟（us）
        switch_delay_us: 交换机延迟（us）
        cable_delay_us: 线缆延迟（us）
    """

    intra_board_bw_gbps: float = 400.0
    inter_board_bw_gbps: float = 200.0
    inter_node_bw_gbps: float = 100.0
    c2c_lat_us: float = 0.15
    ddr_r_lat_us: float = 0.15
    ddr_w_lat_us: float = 0.01
    noc_lat_us: float = 0.05
    d2d_lat_us: float = 0.04
    link_delay_us: float = 0.0
    switch_delay_us: float = 0.25
    cable_delay_us: float = 0.025

    def to_dict(self) -> dict[str, float]:
        """转换为字典"""
        return {
            "intra_board_bw_gbps": self.intra_board_bw_gbps,
            "inter_board_bw_gbps": self.inter_board_bw_gbps,
            "inter_node_bw_gbps": self.inter_node_bw_gbps,
            "c2c_lat_us": self.c2c_lat_us,
            "ddr_r_lat_us": self.ddr_r_lat_us,
            "ddr_w_lat_us": self.ddr_w_lat_us,
            "noc_lat_us": self.noc_lat_us,
            "d2d_lat_us": self.d2d_lat_us,
            "link_delay_us": self.link_delay_us,
            "switch_delay_us": self.switch_delay_us,
            "cable_delay_us": self.cable_delay_us,
        }


@dataclass
class LinkProfileImpl:
    """链路参数实现

    Attributes:
        bandwidth_gbps: 带宽 (Gbps)
        latency_us: 延迟 (us)
        efficiency: 有效系数 (0-1)
    """

    bandwidth_gbps: float
    latency_us: float
    efficiency: float = 1.0


@dataclass
class TopologySpecImpl:
    """通信拓扑规格实现

    Attributes:
        nodes: node_id -> board_id 列表
        boards: board_id -> chip_id 列表
        chips: chip_id 列表
        rank_map: chip_id -> rank 映射
        link_profiles: 链路参数字典
        path_keys: 路径键列表
    """

    nodes: dict[str, list[str]] = field(default_factory=dict)
    boards: dict[str, list[str]] = field(default_factory=dict)
    chips: list[str] = field(default_factory=list)
    rank_map: dict[str, int] = field(default_factory=dict)
    link_profiles: dict[str, LinkProfileImpl] = field(default_factory=dict)
    path_keys: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.chips and self.boards:
            chip_ids: list[str] = []
            for board_chips in self.boards.values():
                chip_ids.extend(board_chips)
            self.chips = chip_ids

        if not self.rank_map and self.chips:
            self.rank_map = {chip_id: idx for idx, chip_id in enumerate(self.chips)}

        if not self.path_keys and self.link_profiles:
            self.path_keys = list(self.link_profiles.keys())

        self._chip_to_board: dict[str, str] = {}
        self._board_to_node: dict[str, str] = {}
        for board_id, chip_ids in self.boards.items():
            for chip_id in chip_ids:
                self._chip_to_board[str(chip_id)] = board_id
        for node_id, board_ids in self.nodes.items():
            for board_id in board_ids:
                self._board_to_node[board_id] = node_id

    def get_link_profile(self, path_key: str) -> LinkProfileImpl:
        """获取链路参数"""
        if path_key not in self.link_profiles:
            raise KeyError(f"Link profile '{path_key}' not found")
        return self.link_profiles[path_key]

    def resolve_path(self, src_chip: str | int, dst_chip: str | int) -> tuple[str, int]:
        """解析路径键与估计跳数"""
        src = str(src_chip)
        dst = str(dst_chip)
        if src == dst:
            return "intra_board", 0

        src_board = self._chip_to_board.get(src)
        dst_board = self._chip_to_board.get(dst)
        if src_board is None or dst_board is None:
            return "inter_node", 3

        if src_board == dst_board:
            return "intra_board", 1

        src_node = self._board_to_node.get(src_board)
        dst_node = self._board_to_node.get(dst_board)
        if src_node is not None and src_node == dst_node:
            return "inter_board", 2

        return "inter_node", 3
