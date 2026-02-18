"""Pod 规格实现

实现 PodSpec。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from perf_model.L2_arch.rack import RackSpecImpl
from perf_model.L2_arch.board import BoardSpecImpl
from perf_model.L2_arch.chip import ChipSpecImpl, chip_registry
from perf_model.L2_arch.topology import TopologySpecImpl
from perf_model.L2_arch.interconnect import ChipInterconnectSpecImpl


@dataclass
class PodSpecImpl:
    """Pod 规格实现

    Attributes:
        num_racks: Rack 数量
        racks: Rack 列表
        r2r_bandwidth_gbps: 跨 Rack 带宽 (Gbps)
        r2r_latency_us: 跨 Rack 延迟 (us)
        topology_ref: 拓扑引用键
        topology: 拓扑规格
    """

    num_racks: int = 0
    racks: list[RackSpecImpl] = field(default_factory=list)
    r2r_bandwidth_gbps: float = 0.0
    r2r_latency_us: float = 0.0
    topology_ref: str | None = None
    topology: TopologySpecImpl | None = None

    def __post_init__(self) -> None:
        if self.num_racks <= 0 and self.racks:
            self.num_racks = len(self.racks)

    def get_rack(self, rack_id: str) -> RackSpecImpl:
        """获取指定 Rack"""
        for rack in self.racks:
            if rack.rack_id == rack_id:
                return rack
        raise KeyError(f"Rack ID '{rack_id}' not found")

    def list_racks(self) -> list[RackSpecImpl]:
        """按稳定顺序返回 Rack"""
        return list(self.racks)

    def get_total_chips(self) -> int:
        """获取 Pod 总芯片数"""
        return sum(rack.chips_per_rack for rack in self.racks)

    def get_total_compute(self, dtype: str) -> float:
        """获取 Pod 总算力

        Args:
            dtype: 数据类型

        Returns:
            总算力 (FLOPS)
        """
        return sum(rack.get_total_compute(dtype) for rack in self.racks)

    def get_total_memory(self) -> int:
        """获取 Pod 总内存

        Returns:
            总内存 (bytes)
        """
        return sum(rack.get_total_memory() for rack in self.racks)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "PodSpecImpl":
        """从配置创建 Pod 规格

        支持 Tier6 风格的拓扑配置格式。

        Args:
            config: 配置字典

        Returns:
            PodSpecImpl 实例
        """
        # 解析 Rack 配置
        racks: list[RackSpecImpl] = []

        # Tier6 风格配置 (grouped_pods 格式)
        if "pods" in config:
            racks = cls._from_tier6_config(config)
        # CHIPMathica 风格配置
        elif "racks" in config:
            for rack_config in config.get("racks", []):
                rack = cls._parse_rack(rack_config)
                racks.append(rack)

        # r2r 网络参数（必需）
        if "r2r_bandwidth_gbps" not in config:
            raise ValueError("Missing 'r2r_bandwidth_gbps' in pod config")
        if "r2r_latency_us" not in config:
            raise ValueError("Missing 'r2r_latency_us' in pod config")

        return cls(
            num_racks=len(racks),
            racks=racks,
            r2r_bandwidth_gbps=config["r2r_bandwidth_gbps"],
            r2r_latency_us=config["r2r_latency_us"],
            topology_ref=config.get("topology_ref"),  # 可选
        )

    @classmethod
    def _from_tier6_config(cls, config: dict[str, Any]) -> list[RackSpecImpl]:
        """从 grouped_pods 格式配置创建 Rack 列表

        遍历 pods[].racks[].boards[].chips[], 展开 count 分组,
        每个 rack 实例生成一个 RackSpecImpl.

        Args:
            config: grouped_pods 格式拓扑配置

        Returns:
            Rack 列表
        """
        racks: list[RackSpecImpl] = []

        pods = config.get("pods", [])
        chips_config = config.get("chips", {})
        interconnect_config = config.get("interconnect", {}).get("links", {})

        rack_idx = 0
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
                        b2b_bw = b2b_config.get("bandwidth_gbps", 200.0)
                        b2b_lat = b2b_config.get("latency_us", 2.0)

                        rack = RackSpecImpl(
                            rack_id=f"rack_{rack_idx}",
                            boards=boards,
                            b2b_bandwidth_gbps=b2b_bw,
                            b2b_latency_us=b2b_lat,
                        )
                        racks.append(rack)
                        rack_idx += 1
                        rack_global_idx += 1

                pod_global_idx += 1

        return racks

    @classmethod
    def _parse_rack(cls, rack_config: dict[str, Any]) -> RackSpecImpl:
        """解析 Rack 配置

        Args:
            rack_config: Rack 配置字典

        Returns:
            RackSpecImpl 实例
        """
        boards: list[BoardSpecImpl] = []
        for board_config in rack_config.get("boards", []):
            board = BoardSpecImpl.from_config(
                board_config.get("name", "board"),
                board_config,
            )
            boards.append(board)

        return RackSpecImpl(
            rack_id=rack_config.get("rack_id", "rack_0"),
            boards=boards,
            b2b_bandwidth_gbps=rack_config.get("b2b_bandwidth_gbps", 200.0),
            b2b_latency_us=rack_config.get("b2b_latency_us", 2.0),
        )

    def to_summary(self) -> dict[str, Any]:
        """导出为汇总参数格式

        Returns:
            汇总参数字典
        """
        return {
            "num_racks": self.num_racks,
            "total_chips": self.get_total_chips(),
            "total_compute_bf16": self.get_total_compute("BF16"),
            "total_memory_gb": self.get_total_memory() / 1024**3,
            "r2r_bandwidth_gbps": self.r2r_bandwidth_gbps,
            "r2r_latency_us": self.r2r_latency_us,
        }
