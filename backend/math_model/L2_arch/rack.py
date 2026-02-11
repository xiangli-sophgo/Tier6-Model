"""Rack 规格实现

实现 RackSpec。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from math_model.L2_arch.board import BoardSpecImpl


@dataclass
class RackSpecImpl:
    """Rack 规格实现

    Attributes:
        rack_id: Rack 标识
        boards: 板卡列表
        chips_per_rack: 每 Rack 芯片数
        b2b_bandwidth_gbps: Board 间带宽 (Gbps)
        b2b_latency_us: Board 间延迟 (us)
    """

    rack_id: str
    boards: list[BoardSpecImpl] = field(default_factory=list)
    chips_per_rack: int = 0
    b2b_bandwidth_gbps: float = 0.0
    b2b_latency_us: float = 0.0

    def __post_init__(self) -> None:
        if self.chips_per_rack <= 0 and self.boards:
            self.chips_per_rack = sum(board.chip_count for board in self.boards)

    def list_chips(self) -> list[int]:
        """返回 Rack 内所有芯片 ID"""
        chip_ids: list[int] = []
        for board in self.boards:
            chip_ids.extend([chip.chip_id for chip in board.chips])
        return chip_ids

    def get_total_compute(self, dtype: str) -> float:
        """获取 Rack 总算力

        Args:
            dtype: 数据类型

        Returns:
            总算力 (FLOPS)
        """
        return sum(board.get_total_compute(dtype) for board in self.boards)

    def get_total_memory(self) -> int:
        """获取 Rack 总内存

        Returns:
            总内存 (bytes)
        """
        return sum(board.get_total_memory() for board in self.boards)
