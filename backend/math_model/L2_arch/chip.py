"""芯片规格实现

实现 ChipSpec 及其工厂方法。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from math_model.L2_arch.compute import ComputeSpec, CubeSpec, VectorSpec, create_compute_unit
from math_model.L2_arch.core import CoreSpecImpl
from math_model.L2_arch.dma import DMAEngineImpl, create_dma_engines
from math_model.L2_arch.interconnect import InterconnectSpecImpl
from math_model.L2_arch.memory import MemoryHierarchyImpl, MemoryLevelImpl
from math_model.L0_entry.registry import InstanceRegistry


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
        align_bytes: 内存对齐字节数
        compute_dma_overlap_rate: 计算与 DMA 搬运的重叠率 (0-1)
        compute_efficiency: 计算效率 (0-1)
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
    align_bytes: int = 32
    compute_dma_overlap_rate: float = 0.8
    compute_efficiency: float = 0.9
    compute_units: dict[str, ComputeSpec] = field(default_factory=dict)
    memory_hierarchy: MemoryHierarchyImpl = field(default_factory=MemoryHierarchyImpl)
    dma_engines: dict[str, DMAEngineImpl] = field(default_factory=dict)
    interconnect: InterconnectSpecImpl = field(default_factory=InterconnectSpecImpl)

    def __post_init__(self) -> None:
        if not self.cores and self.core_count > 0:
            self.cores = self._derive_core_specs()

    # ========== Cube 维度便捷属性 ==========

    @property
    def cube_m(self) -> int:
        """矩阵单元 M 维度"""
        cube = self.compute_units.get("cube")
        if cube is not None and hasattr(cube, "dim_m"):
            return cube.dim_m
        return 0

    @property
    def cube_k(self) -> int:
        """矩阵单元 K 维度"""
        cube = self.compute_units.get("cube")
        if cube is not None and hasattr(cube, "dim_k"):
            return cube.dim_k
        return 0

    @property
    def cube_n(self) -> int:
        """矩阵单元 N 维度"""
        cube = self.compute_units.get("cube")
        if cube is not None and hasattr(cube, "dim_n"):
            return cube.dim_n
        return 0

    @property
    def eu_num(self) -> int:
        """向量执行单元总数 (用于 Softmax 等向量操作估算)"""
        vector = self.compute_units.get("vector")
        if vector is not None and hasattr(vector, "eu_count"):
            return vector.eu_count("BF16")
        return 0

    @property
    def sram_utilization(self) -> float:
        """SRAM 可用比例"""
        try:
            lmem = self.memory_hierarchy.get_level("lmem")
            return getattr(lmem, "utilization", 0.45)
        except KeyError:
            return 0.45

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
        """获取片内 SRAM 总量 (lmem)

        Returns:
            SRAM 总量 (bytes)
        """
        try:
            level = self.memory_hierarchy.get_level("lmem")
            return level.capacity_bytes
        except KeyError:
            return 0

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
        """获取 GMEM 有效带宽 (理论峰值 * 利用率)

        Returns:
            GMEM 有效带宽 (GB/s)
        """
        try:
            gmem = self.memory_hierarchy.get_level("gmem")
            return gmem.bandwidth_gbps * gmem.utilization
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
        """创建新实例并设置 chip_id"""
        return ChipSpecImpl(
            name=self.name,
            chip_id=chip_id,
            core_count=self.core_count,
            lane_per_core=self.lane_per_core,
            cores=self.cores,
            frequency_ghz=self.frequency_ghz,
            align_bytes=self.align_bytes,
            compute_dma_overlap_rate=self.compute_dma_overlap_rate,
            compute_efficiency=self.compute_efficiency,
            compute_units=self.compute_units,
            memory_hierarchy=self.memory_hierarchy,
            dma_engines=self.dma_engines,
            interconnect=self.interconnect,
        )

    @classmethod
    def from_config(cls, name: str, config: dict[str, Any]) -> "ChipSpecImpl":
        """从结构化配置创建芯片规格

        Args:
            name: 芯片名称
            config: 配置字典 (结构化格式)

        Returns:
            ChipSpecImpl 实例
        """
        # 基础参数 - 不使用默认值
        cores_config = config.get("cores")
        if cores_config is None:
            raise ValueError(f"Missing 'cores' in chip config: {name}")
        if "count" not in cores_config:
            raise ValueError(f"Missing 'cores.count' in chip config: {name}")
        if "lanes_per_core" not in cores_config:
            raise ValueError(f"Missing 'cores.lanes_per_core' in chip config: {name}")

        core_count = cores_config["count"]
        lane_per_core = cores_config["lanes_per_core"]

        if "frequency_ghz" not in config:
            raise ValueError(f"Missing 'frequency_ghz' in chip config: {name}")
        frequency_ghz = config["frequency_ghz"]

        if "align_bytes" not in config:
            raise ValueError(f"Missing 'align_bytes' in chip config: {name}")
        align_bytes = int(config["align_bytes"])

        if "compute_dma_overlap_rate" not in config:
            raise ValueError(
                f"Missing 'compute_dma_overlap_rate' in chip config: {name}"
            )
        compute_dma_overlap_rate = float(config["compute_dma_overlap_rate"])

        if "compute_efficiency" not in config:
            raise ValueError(f"Missing 'compute_efficiency' in chip config: {name}")
        compute_efficiency = float(config["compute_efficiency"])

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
        memory_config = config.get("memory")
        if memory_config is None:
            raise ValueError(f"Missing 'memory' in chip config: {name}")
        memory_hierarchy = MemoryHierarchyImpl.from_config(memory_config)

        # DMA 引擎
        dma_config = config.get("dma_engines")
        if dma_config is None:
            raise ValueError(f"Missing 'dma_engines' in chip config: {name}")
        dma_engines = create_dma_engines(dma_config)

        # 片内互联
        interconnect_config = config.get("interconnect", {})
        interconnect = InterconnectSpecImpl.from_config(interconnect_config)

        return cls(
            name=name,
            core_count=core_count,
            lane_per_core=lane_per_core,
            frequency_ghz=frequency_ghz,
            align_bytes=align_bytes,
            compute_dma_overlap_rate=compute_dma_overlap_rate,
            compute_efficiency=compute_efficiency,
            compute_units=compute_units,
            memory_hierarchy=memory_hierarchy,
            dma_engines=dma_engines,
            interconnect=interconnect,
        )

    def _derive_core_specs(self) -> list[CoreSpecImpl]:
        lmem_total = 0
        try:
            lmem_total = self.memory_hierarchy.get_level("lmem").capacity_bytes
        except KeyError:
            lmem_total = 0

        per_core_lmem = lmem_total / self.core_count if self.core_count > 0 else 0
        sram_per_core_kb = per_core_lmem / 1024 if self.core_count > 0 else 0
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
        """导出为汇总参数格式"""
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
