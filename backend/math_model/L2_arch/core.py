"""核心规格实现

实现 CoreSpec。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CoreSpecImpl:
    """核心规格实现

    Attributes:
        core_id: 核心标识
        num_lanes: Lane 数量
        sram_per_core_kb: 每核 SRAM 容量 (KB)
        cube_mac_per_lane: 每 Lane Cube MAC 数
        vector_eu_per_lane: 每 Lane Vector EU 数
        lmem_per_lane_kb: 每 Lane LMEM 容量 (KB)
        core_fabric_share: 片内互联共享系数/标签
    """

    core_id: int
    num_lanes: int
    sram_per_core_kb: float = 0.0
    cube_mac_per_lane: int = 0
    vector_eu_per_lane: int = 0
    lmem_per_lane_kb: float = 0.0
    core_fabric_share: str = "default"

    def capacity_summary(self) -> dict[str, float]:
        """返回容量与执行单元摘要"""
        return {
            "sram_per_core_kb": self.sram_per_core_kb,
            "lmem_per_lane_kb": self.lmem_per_lane_kb,
            "cube_mac_per_lane": float(self.cube_mac_per_lane),
            "vector_eu_per_lane": float(self.vector_eu_per_lane),
        }

    def validate(self) -> bool:
        """校验字段一致性"""
        if self.core_id < 0 or self.num_lanes <= 0:
            return False
        if self.sram_per_core_kb < 0 or self.lmem_per_lane_kb < 0:
            return False
        if self.cube_mac_per_lane < 0 or self.vector_eu_per_lane < 0:
            return False
        return True
