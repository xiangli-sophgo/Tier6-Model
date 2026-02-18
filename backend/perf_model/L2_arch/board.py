"""板卡规格实现

实现 BoardSpec 及其工厂方法。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from perf_model.L2_arch.chip import ChipSpecImpl, chip_registry
from perf_model.L2_arch.interconnect import BoardMemorySpecImpl, ChipInterconnectSpecImpl
from perf_model.L0_entry.registry import InstanceRegistry


# 板卡注册表 (使用 InstanceRegistry 存储单例实例)
board_registry: InstanceRegistry["BoardSpecImpl"] = InstanceRegistry("board")


@dataclass
class BoardSpecImpl:
    """板卡规格实现

    Attributes:
        board_id: 板卡标识
        name: 板卡名称
        chip_count: 芯片数量
        chips: 芯片列表
        chip_interconnect: 芯片间互联规格
        board_memory: 板级共享内存
        fabric_tag: 互联标签
    """

    name: str
    board_id: str | None = None
    chip_count: int = 1
    chips: list[ChipSpecImpl] = field(default_factory=list)
    chip_interconnect: ChipInterconnectSpecImpl = field(
        default_factory=ChipInterconnectSpecImpl
    )
    board_memory: BoardMemorySpecImpl | None = None
    fabric_tag: str = "c2c"

    def __post_init__(self) -> None:
        if self.board_id is None:
            self.board_id = self.name

    def get_chip(self, chip_id: int) -> ChipSpecImpl:
        """获取指定芯片

        Args:
            chip_id: 芯片编号

        Returns:
            芯片规格

        Raises:
            IndexError: 芯片编号越界
        """
        if chip_id < 0 or chip_id >= len(self.chips):
            raise IndexError(f"Chip ID {chip_id} out of range [0, {len(self.chips)})")
        return self.chips[chip_id]

    def get_total_compute(self, dtype: str) -> float:
        """获取板卡总算力

        Args:
            dtype: 数据类型 (BF16/FP16/INT8 等)

        Returns:
            总算力 (FLOPS)
        """
        return sum(chip.get_peak_flops(dtype) for chip in self.chips)

    def get_total_memory(self) -> int:
        """获取板卡总内存

        Returns:
            总内存 (bytes)
        """
        chip_memory = sum(chip.get_gmem_capacity() for chip in self.chips)
        board_memory = self.board_memory.capacity_bytes if self.board_memory else 0
        return chip_memory + board_memory

    def get_chip_to_chip_bandwidth(self, src: int, dst: int) -> float:
        """获取芯片间带宽

        Args:
            src: 源芯片编号
            dst: 目标芯片编号

        Returns:
            带宽 (GB/s)
        """
        return self.chip_interconnect.get_path_bandwidth(src, dst)

    def get_allreduce_time(self, data_size: int) -> float:
        """估算 AllReduce 时间

        Args:
            data_size: 数据大小 (bytes)

        Returns:
            估算时间 (ns)
        """
        return self.chip_interconnect.get_allreduce_time(data_size, self.chip_count)

    @classmethod
    def from_config(cls, name: str, config: dict[str, Any]) -> "BoardSpecImpl":
        """从配置创建板卡规格

        Args:
            name: 板卡名称
            config: 配置字典

        Returns:
            BoardSpecImpl 实例
        """
        # 芯片配置
        chips_config = config.get("chips", {})

        # chip_count: 优先从 chips.count，否则从顶层 chip_count（至少要有一个）
        chip_count = None
        if "count" in chips_config:
            chip_count = chips_config["count"]
        elif "chip_count" in config:
            chip_count = config["chip_count"]
        else:
            raise ValueError(f"Missing 'chips.count' or 'chip_count' in board config '{name}'")

        chip_type = chips_config.get("chip_type", config.get("chip_type"))
        chip_config = chips_config.get("chip_config", {})

        # 创建芯片列表（必须提供 chip_type 或 chip_config）
        chips: list[ChipSpecImpl] = []
        if chip_type:
            # 从注册表或配置创建
            try:
                base_chip = chip_registry.get(chip_type)
                for i in range(chip_count):
                    chips.append(base_chip.with_chip_id(i))
            except KeyError:
                # 芯片类型未注册，尝试从配置创建
                base_chip = ChipSpecImpl.from_config(chip_type, chip_config)
                for i in range(chip_count):
                    chips.append(base_chip.with_chip_id(i))
        elif chip_config:
            # 从内联配置创建
            base_chip = ChipSpecImpl.from_config(name, chip_config)
            for i in range(chip_count):
                chips.append(base_chip.with_chip_id(i))
        else:
            raise ValueError(f"Missing 'chip_type' or 'chip_config' in board config '{name}'")

        # 芯片间互联
        interconnect_config = config.get("chip_interconnect", {})
        chip_interconnect = ChipInterconnectSpecImpl.from_config(
            interconnect_config, chip_count
        )

        # 板级共享内存
        board_memory = None
        board_memory_config = config.get("board_memory", {})
        if board_memory_config and board_memory_config.get("capacity_gb", 0) > 0:
            board_memory = BoardMemorySpecImpl.from_config(board_memory_config)

        return cls(
            board_id=config.get("board_id"),
            name=name,
            chip_count=chip_count,
            chips=chips,
            chip_interconnect=chip_interconnect,
            board_memory=board_memory,
            fabric_tag=config.get("fabric_tag") or "c2c",  # 可选，默认 c2c
        )

    def to_summary(self) -> dict[str, Any]:
        """导出为汇总参数格式

        Returns:
            汇总参数字典
        """
        return {
            "board_id": self.board_id,
            "name": self.name,
            "chip_count": self.chip_count,
            "total_compute_bf16": self.get_total_compute("BF16"),
            "total_memory_gb": self.get_total_memory() / 1024**3,
            "chip_interconnect": {
                "topology": self.chip_interconnect.topology,
                "bandwidth_gbps": self.chip_interconnect.link_bandwidth_gbps,
            },
            "fabric_tag": self.fabric_tag,
        }
