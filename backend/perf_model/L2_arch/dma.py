"""DMA 引擎实现

实现 GDMA/SDMA/CDMA 等数据搬运引擎。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DMAEngineImpl:
    """DMA 引擎实现

    Attributes:
        name: 引擎名称 (GDMA/SDMA/CDMA)
        bandwidth_gbps: 带宽 (GB/s)
        startup_latency_ns: 启动延迟 (ns)
        efficiency: 带宽利用效率 (0-1)
        supported_paths: 支持的源/目的层级组合
        scope: 引擎作用域
        engine_type: 引擎类型
    """

    name: str
    bandwidth_gbps: float
    startup_latency_ns: float = 100.0
    efficiency: float = 0.9
    supported_paths: list[tuple[str, str]] = field(default_factory=list)
    scope: str = "chip"
    engine_type: str | None = None

    def __post_init__(self) -> None:
        if self.engine_type is None:
            self.engine_type = self.name

    def get_transfer_time(self, data_size: int, src: str = "", dst: str = "") -> float:
        """计算传输时间

        time = startup_latency + data_size / (bandwidth * efficiency)

        Args:
            data_size: 数据大小 (bytes)
            src: 源存储层级 (可选)
            dst: 目标存储层级 (可选)

        Returns:
            传输时间 (ns)
        """
        if data_size <= 0:
            return 0.0

        # 带宽转换: GB/s -> bytes/ns
        bandwidth_bytes_per_ns = self.bandwidth_gbps  # GB/s = bytes/ns

        transfer_time = data_size / (bandwidth_bytes_per_ns * self.efficiency)
        return self.startup_latency_ns + transfer_time

    def get_effective_bandwidth(self) -> float:
        """获取有效带宽

        Returns:
            有效带宽 (GB/s)
        """
        return self.bandwidth_gbps * self.efficiency

    @classmethod
    def from_config(cls, name: str, config: dict[str, Any]) -> "DMAEngineImpl":
        """从配置创建

        Args:
            name: 引擎名称
            config: 配置字典

        Returns:
            DMAEngineImpl 实例
        """
        # bandwidth 必需（0 会导致除零）
        if "bandwidth_gbps" not in config:
            raise ValueError(f"Missing 'bandwidth_gbps' in DMA engine '{name}'")
        bandwidth = config["bandwidth_gbps"]
        if bandwidth <= 0:
            raise ValueError(f"bandwidth_gbps must be > 0 in DMA engine '{name}' (got {bandwidth})")

        return cls(
            name=name,
            bandwidth_gbps=bandwidth,
            startup_latency_ns=config.get("startup_latency_ns", 100.0),  # 可选
            efficiency=config.get("efficiency", 0.9),  # 可选
            supported_paths=config.get("supported_paths", []),  # 可选
            scope=config.get("scope", "chip"),  # 可选
            engine_type=config.get("engine_type"),  # 可选
        )


@dataclass
class GDMASpec(DMAEngineImpl):
    """GDMA (Global DMA) 规格

    用于 GMEM <-> LMEM 数据搬运。
    """

    def __init__(
        self,
        bandwidth_gbps: float,
        startup_latency_ns: float = 100.0,
        efficiency: float = 0.9,
        supported_paths: list[tuple[str, str]] | None = None,
        scope: str = "chip",
    ):
        super().__init__(
            name="GDMA",
            bandwidth_gbps=bandwidth_gbps,
            startup_latency_ns=startup_latency_ns,
            efficiency=efficiency,
            supported_paths=supported_paths or [],
            scope=scope,
            engine_type="GDMA",
        )


@dataclass
class SDMASpec(DMAEngineImpl):
    """SDMA (Shared DMA) 规格

    用于 L2M <-> LMEM 数据搬运。
    """

    def __init__(
        self,
        bandwidth_gbps: float,
        startup_latency_ns: float = 50.0,
        efficiency: float = 0.95,
        supported_paths: list[tuple[str, str]] | None = None,
        scope: str = "chip",
    ):
        super().__init__(
            name="SDMA",
            bandwidth_gbps=bandwidth_gbps,
            startup_latency_ns=startup_latency_ns,
            efficiency=efficiency,
            supported_paths=supported_paths or [],
            scope=scope,
            engine_type="SDMA",
        )


@dataclass
class CDMASpec(DMAEngineImpl):
    """CDMA (Chip-to-Chip DMA) 规格

    用于片间数据搬运。
    """

    def __init__(
        self,
        bandwidth_gbps: float,
        startup_latency_ns: float = 200.0,
        efficiency: float = 0.85,
        supported_paths: list[tuple[str, str]] | None = None,
        scope: str = "inter_chip",
    ):
        super().__init__(
            name="CDMA",
            bandwidth_gbps=bandwidth_gbps,
            startup_latency_ns=startup_latency_ns,
            efficiency=efficiency,
            supported_paths=supported_paths or [],
            scope=scope,
            engine_type="CDMA",
        )


def create_dma_engines(config: dict[str, Any]) -> dict[str, DMAEngineImpl]:
    """从配置创建 DMA 引擎字典

    Args:
        config: DMA 配置字典

    Returns:
        DMA 引擎字典
    """
    engines: dict[str, DMAEngineImpl] = {}

    for name, engine_config in config.items():
        if isinstance(engine_config, dict):
            engines[name] = DMAEngineImpl.from_config(name, engine_config)

    return engines
