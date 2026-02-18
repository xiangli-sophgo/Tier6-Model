"""通信拓扑规格实现

实现 TopologySpec。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TopologySpec:
    """L2 拓扑通信参数口径

    每层级包含 bandwidth + latency（从 YAML interconnect.links 来）

    Attributes:
        c2c_bandwidth_gbps: C2C 互联带宽（GB/s）
        c2c_latency_us: C2C 互联延迟（us）
        b2b_bandwidth_gbps: B2B 互联带宽（GB/s）
        b2b_latency_us: B2B 互联延迟（us）
        r2r_bandwidth_gbps: R2R 互联带宽（GB/s）
        r2r_latency_us: R2R 互联延迟（us）
        p2p_bandwidth_gbps: P2P 互联带宽（GB/s）
        p2p_latency_us: P2P 互联延迟（us）
        memory_read_latency_us: DDR 读延迟（us）
        memory_write_latency_us: DDR 写延迟（us）
        noc_latency_us: NoC 延迟（us）
        die_to_die_latency_us: Die-to-Die 延迟（us）
        switch_latency_us: 交换机延迟（us）
        cable_latency_us: 线缆延迟（us）
    """

    c2c_bandwidth_gbps: float = 400.0
    c2c_latency_us: float = 0.15
    b2b_bandwidth_gbps: float = 200.0
    b2b_latency_us: float = 0.35
    r2r_bandwidth_gbps: float = 100.0
    r2r_latency_us: float = 2.0
    p2p_bandwidth_gbps: float = 50.0
    p2p_latency_us: float = 5.0
    memory_read_latency_us: float = 0.15
    memory_write_latency_us: float = 0.01
    noc_latency_us: float = 0.05
    die_to_die_latency_us: float = 0.04
    switch_latency_us: float = 0.25
    cable_latency_us: float = 0.025

    def to_dict(self) -> dict[str, float]:
        """转换为字典"""
        return {
            "c2c_bandwidth_gbps": self.c2c_bandwidth_gbps,
            "c2c_latency_us": self.c2c_latency_us,
            "b2b_bandwidth_gbps": self.b2b_bandwidth_gbps,
            "b2b_latency_us": self.b2b_latency_us,
            "r2r_bandwidth_gbps": self.r2r_bandwidth_gbps,
            "r2r_latency_us": self.r2r_latency_us,
            "p2p_bandwidth_gbps": self.p2p_bandwidth_gbps,
            "p2p_latency_us": self.p2p_latency_us,
            "memory_read_latency_us": self.memory_read_latency_us,
            "memory_write_latency_us": self.memory_write_latency_us,
            "noc_latency_us": self.noc_latency_us,
            "die_to_die_latency_us": self.die_to_die_latency_us,
            "switch_latency_us": self.switch_latency_us,
            "cable_latency_us": self.cable_latency_us,
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
        pods: pod_id -> rack_id 列表
        racks: rack_id -> board_id 列表
        boards: board_id -> chip_id 列表
        chips: chip_id 列表
        rank_map: chip_id -> rank 映射
        link_profiles: 链路参数字典
        path_keys: 路径键列表
    """

    pods: dict[str, list[str]] = field(default_factory=dict)
    racks: dict[str, list[str]] = field(default_factory=dict)
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
        self._board_to_rack: dict[str, str] = {}
        self._rack_to_pod: dict[str, str] = {}
        for board_id, chip_ids in self.boards.items():
            for chip_id in chip_ids:
                self._chip_to_board[str(chip_id)] = board_id
        for rack_id, board_ids in self.racks.items():
            for board_id in board_ids:
                self._board_to_rack[board_id] = rack_id
        for pod_id, rack_ids in self.pods.items():
            for rack_id in rack_ids:
                self._rack_to_pod[rack_id] = pod_id

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
            return "c2c", 0

        src_board = self._chip_to_board.get(src)
        dst_board = self._chip_to_board.get(dst)
        if src_board is None or dst_board is None:
            return "p2p", 4

        if src_board == dst_board:
            return "c2c", 1

        src_rack = self._board_to_rack.get(src_board)
        dst_rack = self._board_to_rack.get(dst_board)
        if src_rack is not None and src_rack == dst_rack:
            return "b2b", 2

        src_pod = self._rack_to_pod.get(src_rack) if src_rack else None
        dst_pod = self._rack_to_pod.get(dst_rack) if dst_rack else None
        if src_pod is not None and src_pod == dst_pod:
            return "r2r", 3

        return "p2p", 4
