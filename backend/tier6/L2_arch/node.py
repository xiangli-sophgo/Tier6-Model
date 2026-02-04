"""节点规格实现

实现 NodeSpec。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tier6.L2_arch.board import BoardSpecImpl


@dataclass
class NodeSpecImpl:
    """节点规格实现

    Attributes:
        node_id: 节点标识
        boards: 板卡列表
        chips_per_node: 每节点芯片数
        intra_node_bandwidth_gbps: 节点内带宽 (Gbps)
        intra_node_latency_us: 节点内延迟 (us)
    """

    node_id: str
    boards: list[BoardSpecImpl] = field(default_factory=list)
    chips_per_node: int = 0
    intra_node_bandwidth_gbps: float = 0.0
    intra_node_latency_us: float = 0.0

    def __post_init__(self) -> None:
        if self.chips_per_node <= 0 and self.boards:
            self.chips_per_node = sum(board.chip_count for board in self.boards)

    def list_chips(self) -> list[int]:
        """返回节点内所有芯片 ID"""
        chip_ids: list[int] = []
        for board in self.boards:
            chip_ids.extend([chip.chip_id for chip in board.chips])
        return chip_ids

    def get_total_compute(self, dtype: str) -> float:
        """获取节点总算力

        Args:
            dtype: 数据类型

        Returns:
            总算力 (FLOPS)
        """
        return sum(board.get_total_compute(dtype) for board in self.boards)

    def get_total_memory(self) -> int:
        """获取节点总内存

        Returns:
            总内存 (bytes)
        """
        return sum(board.get_total_memory() for board in self.boards)
