"""存储层级实现

实现 MemoryHierarchy 和 MemoryLevel。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MemoryLevelImpl:
    """单层存储规格实现

    Attributes:
        name: 层级名称 (gmem/l2m/lmem/smem)
        capacity_bytes: 容量 (bytes)
        bandwidth_gbps: 带宽 (GB/s)
        latency_ns: 延迟 (ns)
        scope: 共享域
        sram_utilization: SRAM 可用比例 (仅 lmem 使用，0-1)
    """

    name: str
    capacity_bytes: int
    bandwidth_gbps: float
    latency_ns: float = 0.0
    scope: str = "chip"
    sram_utilization: float = 1.0

    @classmethod
    def from_config(cls, name: str, config: dict[str, Any]) -> "MemoryLevelImpl":
        """从配置创建

        Args:
            name: 层级名称
            config: 配置字典

        Returns:
            MemoryLevelImpl 实例
        """
        # 容量单位转换
        capacity = config.get("capacity_bytes", 0)
        if capacity == 0:
            if "capacity_gb" in config:
                capacity = int(config["capacity_gb"] * 1024**3)
            elif "capacity_mb" in config:
                capacity = int(config["capacity_mb"] * 1024**2)
            elif "capacity_kb" in config:
                capacity = int(config["capacity_kb"] * 1024)

        # bandwidth 和 latency 必需（0 会导致除零错误）
        if "bandwidth_gbps" not in config:
            raise ValueError(f"Missing 'bandwidth_gbps' in memory level '{name}'")
        bandwidth = config["bandwidth_gbps"]
        if bandwidth <= 0:
            raise ValueError(f"bandwidth_gbps must be > 0 in memory level '{name}' (got {bandwidth})")

        if "latency_ns" not in config:
            raise ValueError(f"Missing 'latency_ns' in memory level '{name}'")

        return cls(
            name=name,
            capacity_bytes=capacity,
            bandwidth_gbps=bandwidth,
            latency_ns=config["latency_ns"],
            scope=config.get("scope", "chip"),  # 可选，默认 chip
            sram_utilization=float(config.get("sram_utilization", 1.0)),  # 可选，默认 1.0
        )


@dataclass
class MemoryHierarchyImpl:
    """存储层级实现

    管理 GMEM -> L2M -> LMEM -> SMEM 的层级结构。

    Attributes:
        levels: 存储层级字典
    """

    levels: dict[str, MemoryLevelImpl] = field(default_factory=dict)

    def get_level(self, name: str) -> MemoryLevelImpl:
        """获取指定层级

        Args:
            name: 层级名称

        Returns:
            存储层级规格

        Raises:
            KeyError: 层级不存在
        """
        if name not in self.levels:
            raise KeyError(f"Memory level '{name}' not found")
        return self.levels[name]

    def list_levels(self) -> list[str]:
        """列出所有层级名称

        Returns:
            层级名称列表
        """
        return list(self.levels.keys())

    def add_level(self, level: MemoryLevelImpl) -> None:
        """添加存储层级

        Args:
            level: 存储层级
        """
        self.levels[level.name] = level

    def get_total_capacity(self) -> int:
        """获取总容量

        Returns:
            总容量 (bytes)
        """
        return sum(level.capacity_bytes for level in self.levels.values())

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "MemoryHierarchyImpl":
        """从配置创建

        Args:
            config: 配置字典，键为层级名称

        Returns:
            MemoryHierarchyImpl 实例
        """
        hierarchy = cls()
        for name, level_config in config.items():
            if isinstance(level_config, dict):
                level = MemoryLevelImpl.from_config(name, level_config)
                hierarchy.add_level(level)
        return hierarchy


# 默认存储层级模板
DEFAULT_MEMORY_HIERARCHY = {
    "gmem": {"capacity_gb": 32, "bandwidth_gbps": 273, "latency_ns": 100, "scope": "global"},
    "l2m": {"capacity_mb": 16, "bandwidth_gbps": 1000, "latency_ns": 10, "scope": "shared"},
    "lmem": {"capacity_kb": 256, "bandwidth_gbps": 2000, "latency_ns": 1, "scope": "core"},
    "smem": {"capacity_kb": 64, "bandwidth_gbps": 2000, "latency_ns": 1, "scope": "core"},
}
