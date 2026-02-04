"""芯片规格实现

实现 ChipSpec 及其工厂方法。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tier6.L2_arch.compute import ComputeSpec, CubeSpec, VectorSpec, create_compute_unit
from tier6.L2_arch.core import CoreSpecImpl
from tier6.L2_arch.dma import DMAEngineImpl, create_dma_engines
from tier6.L2_arch.interconnect import InterconnectSpecImpl
from tier6.L2_arch.memory import MemoryHierarchyImpl, MemoryLevelImpl
from tier6.core.registry import InstanceRegistry


# 芯片注册表 (使用 InstanceRegistry 存储单例实例)
chip_registry: InstanceRegistry["ChipSpecImpl"] = InstanceRegistry("chip")


@dataclass
class ChipSpecImpl:
    """芯片规格实现

    Attributes:
        name: 芯片名称
        chip_id: 在板卡中的编号
        core_count: 核心数量
        lane_per_core: 每核心 Lane 数
        cores: 核心规格列表
        frequency_ghz: 工作频率
        compute_units: 计算单元
        memory_hierarchy: 存储层级
        dma_engines: DMA 引擎
        interconnect: 片内互联
    """

    name: str
    chip_id: int = 0
    core_count: int = 1
    lane_per_core: int = 64
    cores: list[CoreSpecImpl] = field(default_factory=list)
    frequency_ghz: float = 1.0
    compute_units: dict[str, ComputeSpec] = field(default_factory=dict)
    memory_hierarchy: MemoryHierarchyImpl = field(default_factory=MemoryHierarchyImpl)
    dma_engines: dict[str, DMAEngineImpl] = field(default_factory=dict)
    interconnect: InterconnectSpecImpl = field(default_factory=InterconnectSpecImpl)

    def __post_init__(self) -> None:
        if not self.cores and self.core_count > 0:
            self.cores = self._derive_core_specs()

    def get_peak_flops(self, dtype: str, unit: str = "cube") -> float:
        """获取峰值算力

        Args:
            dtype: 数据类型
            unit: 计算单元 (cube/vector)

        Returns:
            峰值算力 (FLOPS)
        """
        compute_unit = self.compute_units.get(unit.lower())
        if compute_unit is None:
            return 0.0
        return compute_unit.peak_flops(dtype)

    def get_total_sram(self) -> int:
        """获取片内 SRAM 总量

        包括 L2M、LMEM、SMEM 等片内存储。

        Returns:
            SRAM 总量 (bytes)
        """
        sram_levels = ["l2m", "lmem", "smem"]
        total = 0
        for level_name in sram_levels:
            try:
                level = self.memory_hierarchy.get_level(level_name)
                total += level.capacity_bytes
            except KeyError:
                continue
        return total

    def get_gmem_capacity(self) -> int:
        """获取 GMEM 容量

        Returns:
            GMEM 容量 (bytes)
        """
        try:
            return self.memory_hierarchy.get_level("gmem").capacity_bytes
        except KeyError:
            return 0

    def get_gmem_bandwidth(self) -> float:
        """获取 GMEM 带宽

        Returns:
            GMEM 带宽 (GB/s)
        """
        try:
            return self.memory_hierarchy.get_level("gmem").bandwidth_gbps
        except KeyError:
            return 0.0

    def list_cores(self) -> list[int]:
        """返回核心 ID 顺序"""
        return [core.core_id for core in self.cores]

    def summary(self) -> dict[str, Any]:
        """返回芯片关键指标摘要"""
        return {
            "name": self.name,
            "chip_id": self.chip_id,
            "core_count": self.core_count,
            "lane_per_core": self.lane_per_core,
            "peak_flops_bf16": self.get_peak_flops("BF16"),
            "gmem_capacity_bytes": self.get_gmem_capacity(),
            "gmem_bandwidth_gbps": self.get_gmem_bandwidth(),
        }

    def with_chip_id(self, chip_id: int) -> "ChipSpecImpl":
        """创建新实例并设置 chip_id

        Args:
            chip_id: 芯片编号

        Returns:
            新的 ChipSpecImpl 实例
        """
        return ChipSpecImpl(
            name=self.name,
            chip_id=chip_id,
            core_count=self.core_count,
            lane_per_core=self.lane_per_core,
            cores=self.cores,
            frequency_ghz=self.frequency_ghz,
            compute_units=self.compute_units,
            memory_hierarchy=self.memory_hierarchy,
            dma_engines=self.dma_engines,
            interconnect=self.interconnect,
        )

    @classmethod
    def from_config(cls, name: str, config: dict[str, Any]) -> "ChipSpecImpl":
        """从配置创建芯片规格

        支持两种配置格式：
        1. 新格式 (结构化)
        2. Tier6 兼容格式 (汇总参数)

        Args:
            name: 芯片名称
            config: 配置字典

        Returns:
            ChipSpecImpl 实例
        """
        # 检测配置格式
        if "peak_flops_bf16" in config or "compute_tflops_bf16" in config:
            # Tier6 兼容格式
            return cls._from_tier6_config(name, config)
        else:
            # 新格式
            return cls._from_structured_config(name, config)

    @classmethod
    def _from_structured_config(cls, name: str, config: dict[str, Any]) -> "ChipSpecImpl":
        """从结构化配置创建

        Args:
            name: 芯片名称
            config: 配置字典

        Returns:
            ChipSpecImpl 实例
        """
        # 基础参数
        cores_config = config.get("cores", {})
        core_count = cores_config.get("count", config.get("core_count", 1))
        lane_per_core = cores_config.get("lanes_per_core", config.get("lane_per_core", 64))
        frequency_ghz = config.get("frequency_ghz", 1.0)

        # 计算单元
        compute_units: dict[str, ComputeSpec] = {}
        compute_config = config.get("compute_units", {})
        for unit_type, unit_config in compute_config.items():
            if isinstance(unit_config, dict):
                compute_units[unit_type.lower()] = create_compute_unit(
                    unit_type,
                    unit_config,
                    frequency_ghz,
                    lane_per_core,
                    core_count,
                )

        # 存储层级
        memory_config = config.get("memory", {})
        memory_hierarchy = MemoryHierarchyImpl.from_config(memory_config)

        # DMA 引擎
        dma_config = config.get("dma_engines", config.get("dma", {}))
        dma_engines = create_dma_engines(dma_config)

        # 片内互联
        interconnect_config = config.get("interconnect", {})
        interconnect = InterconnectSpecImpl.from_config(interconnect_config)

        return cls(
            name=name,
            core_count=core_count,
            lane_per_core=lane_per_core,
            frequency_ghz=frequency_ghz,
            compute_units=compute_units,
            memory_hierarchy=memory_hierarchy,
            dma_engines=dma_engines,
            interconnect=interconnect,
        )

    @classmethod
    def _from_tier6_config(cls, name: str, config: dict[str, Any]) -> "ChipSpecImpl":
        """从 Tier6 兼容配置创建

        将 Tier6 的汇总参数转换为结构化规格。

        Args:
            name: 芯片名称
            config: Tier6 格式配置字典

        Returns:
            ChipSpecImpl 实例
        """
        # 基础参数
        core_count = config.get("num_cores", config.get("core_count", 1))
        lane_per_core = config.get("lane_num", config.get("lane_per_core", 64))
        frequency_ghz = config.get("frequency_ghz", 1.0)

        # 从峰值算力反推 MAC
        peak_flops_bf16 = config.get("peak_flops_bf16", 0)
        if peak_flops_bf16 == 0:
            # 尝试从 TFLOPS 转换
            compute_tflops_bf16 = config.get("compute_tflops_bf16", 0)
            peak_flops_bf16 = compute_tflops_bf16 * 1e12

        if peak_flops_bf16 > 0:
            # FLOPs = mac * lanes * cores * freq * 2
            # mac = FLOPs / (lanes * cores * freq * 2)
            mac_per_lane = int(peak_flops_bf16 / (lane_per_core * core_count * frequency_ghz * 1e9 * 2))
        else:
            mac_per_lane = config.get("cube_m", 128)

        # 构建 Cube
        cube = CubeSpec(
            name="Cube",
            mac_per_lane={"BF16": mac_per_lane, "FP16": mac_per_lane, "INT8": mac_per_lane * 2},
            frequency_ghz=frequency_ghz,
            lane_count=lane_per_core,
            core_count=core_count,
        )

        # 构建 Vector
        eu_per_lane = config.get("eu_num", 32)
        vector = VectorSpec(
            name="Vector",
            eu_per_lane={"BF16": eu_per_lane, "FP16": eu_per_lane, "FP32": eu_per_lane // 2},
            frequency_ghz=frequency_ghz,
            lane_count=lane_per_core,
            core_count=core_count,
        )

        compute_units: dict[str, ComputeSpec] = {"cube": cube, "vector": vector}

        # 存储层级
        memory_hierarchy = MemoryHierarchyImpl()

        # GMEM
        gmem_capacity_gb = config.get("memory_capacity_gb", config.get("dram", 32))
        gmem_bandwidth = config.get("memory_bandwidth_gbps", config.get("memory_bandwidth", 0))
        if gmem_bandwidth > 1e9:
            gmem_bandwidth = gmem_bandwidth / 1e9  # 转换为 GB/s
        memory_hierarchy.add_level(
            MemoryLevelImpl(
                name="gmem",
                capacity_bytes=int(gmem_capacity_gb * 1024**3),
                bandwidth_gbps=gmem_bandwidth,
                latency_ns=100.0,
            )
        )

        # SRAM (L2M + LMEM)
        sram_size_mb = config.get("sram_size_mb", config.get("sram_size_kb", 0) / 1024)
        if sram_size_mb == 0:
            sram_size_mb = 128  # 默认值
        memory_hierarchy.add_level(
            MemoryLevelImpl(
                name="l2m",
                capacity_bytes=int(sram_size_mb * 1024**2 * 0.5),  # 假设 50% 为 L2M
                bandwidth_gbps=1000.0,
                latency_ns=10.0,
            )
        )
        memory_hierarchy.add_level(
            MemoryLevelImpl(
                name="lmem",
                capacity_bytes=int(sram_size_mb * 1024**2 * 0.5),  # 假设 50% 为 LMEM
                bandwidth_gbps=2000.0,
                latency_ns=1.0,
            )
        )

        # DMA 引擎
        dma_bandwidth = config.get("dma_bw", gmem_bandwidth / core_count if core_count > 0 else 0)
        dma_engines: dict[str, DMAEngineImpl] = {
            "gdma": DMAEngineImpl(name="GDMA", bandwidth_gbps=dma_bandwidth),
        }

        # 片内互联
        c2c_bandwidth = config.get("interconnect_bandwidth", config.get("inter_bw", 0))
        if c2c_bandwidth > 1e9:
            c2c_bandwidth = c2c_bandwidth / 1e9
        interconnect = InterconnectSpecImpl(
            noc_topology="mesh",
            noc_bandwidth_gbps=1000.0,
            c2c_links=10,
            c2c_bandwidth_gbps=c2c_bandwidth,
        )

        return cls(
            name=name,
            core_count=core_count,
            lane_per_core=lane_per_core,
            frequency_ghz=frequency_ghz,
            compute_units=compute_units,
            memory_hierarchy=memory_hierarchy,
            dma_engines=dma_engines,
            interconnect=interconnect,
        )

    def _derive_core_specs(self) -> list[CoreSpecImpl]:
        lmem_total = 0
        smem_total = 0
        try:
            lmem_total = self.memory_hierarchy.get_level("lmem").capacity_bytes
        except KeyError:
            lmem_total = 0
        try:
            smem_total = self.memory_hierarchy.get_level("smem").capacity_bytes
        except KeyError:
            smem_total = 0

        per_core_lmem = lmem_total / self.core_count if self.core_count > 0 else 0
        per_core_smem = smem_total / self.core_count if self.core_count > 0 else 0
        sram_per_core_kb = (per_core_lmem + per_core_smem) / 1024 if self.core_count > 0 else 0
        lmem_per_lane_kb = (
            per_core_lmem / self.lane_per_core / 1024
            if self.core_count > 0 and self.lane_per_core > 0
            else 0
        )

        cube_mac_per_lane = self._extract_cube_mac_per_lane()
        vector_eu_per_lane = self._extract_vector_eu_per_lane()

        cores: list[CoreSpecImpl] = []
        for core_id in range(self.core_count):
            cores.append(
                CoreSpecImpl(
                    core_id=core_id,
                    num_lanes=self.lane_per_core,
                    sram_per_core_kb=sram_per_core_kb,
                    cube_mac_per_lane=cube_mac_per_lane,
                    vector_eu_per_lane=vector_eu_per_lane,
                    lmem_per_lane_kb=lmem_per_lane_kb,
                    core_fabric_share="default",
                )
            )
        return cores

    def _extract_cube_mac_per_lane(self) -> int:
        cube = self.compute_units.get("cube")
        if cube is None:
            return 0
        mac_map = getattr(cube, "mac_per_lane", {})
        if isinstance(mac_map, dict):
            if "BF16" in mac_map:
                return int(mac_map["BF16"])
            if "FP16" in mac_map:
                return int(mac_map["FP16"])
            if mac_map:
                return int(next(iter(mac_map.values())))
        return 0

    def _extract_vector_eu_per_lane(self) -> int:
        vector = self.compute_units.get("vector")
        if vector is None:
            return 0
        eu_map = getattr(vector, "eu_per_lane", {})
        if isinstance(eu_map, dict):
            if "BF16" in eu_map:
                return int(eu_map["BF16"])
            if "FP16" in eu_map:
                return int(eu_map["FP16"])
            if eu_map:
                return int(next(iter(eu_map.values())))
        return 0

    def to_summary(self) -> dict[str, Any]:
        """导出为汇总参数格式

        用于兼容 Tier6 等工具。

        Returns:
            汇总参数字典
        """
        cube = self.compute_units.get("cube")
        peak_flops_bf16 = cube.peak_flops("BF16") if cube else 0

        return {
            "name": self.name,
            "peak_flops_bf16": peak_flops_bf16,
            "memory_bandwidth": self.get_gmem_bandwidth() * 1e9,
            "memory_capacity_gb": self.get_gmem_capacity() / 1024**3,
            "sram_size_mb": self.get_total_sram() / 1024**2,
            "num_cores": self.core_count,
            "interconnect_bandwidth": self.interconnect.c2c_bandwidth_gbps * 1e9,
        }
