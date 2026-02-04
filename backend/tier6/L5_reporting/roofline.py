"""Roofline 分析模块

生成 Roofline 模型数据用于性能可视化。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tier6.L2_arch.protocols import ChipSpec
    from tier6.L4_evaluation.metrics import EngineResult, StepMetrics


@dataclass
class RooflinePoint:
    """Roofline 图上的点

    Attributes:
        name: 点名称
        arithmetic_intensity: 算术强度 (FLOPS/Byte)
        achieved_flops: 实际达到的算力 (GFLOPS)
        peak_flops: 峰值算力 (GFLOPS)
        peak_bandwidth: 峰值带宽 (GB/s)
        is_compute_bound: 是否计算瓶颈
        is_memory_bound: 是否内存瓶颈
        efficiency: 效率 (0-1)
    """

    name: str
    arithmetic_intensity: float
    achieved_flops: float
    peak_flops: float
    peak_bandwidth: float
    is_compute_bound: bool = False
    is_memory_bound: bool = False
    efficiency: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "arithmeticIntensity": self.arithmetic_intensity,
            "achievedFlops": self.achieved_flops,
            "peakFlops": self.peak_flops,
            "peakBandwidth": self.peak_bandwidth,
            "isComputeBound": self.is_compute_bound,
            "isMemoryBound": self.is_memory_bound,
            "efficiency": self.efficiency,
        }


@dataclass
class RooflineData:
    """Roofline 图数据

    Attributes:
        peak_flops: 峰值算力 (GFLOPS)
        peak_bandwidth: 峰值带宽 (GB/s)
        ridge_point: 拐点 (FLOPS/Byte)
        points: 数据点列表
        roofline_x: Roofline 曲线 X 坐标 (AI)
        roofline_y: Roofline 曲线 Y 坐标 (GFLOPS)
    """

    peak_flops: float  # GFLOPS
    peak_bandwidth: float  # GB/s
    ridge_point: float  # FLOPS/Byte
    points: list[RooflinePoint] = field(default_factory=list)
    roofline_x: list[float] = field(default_factory=list)
    roofline_y: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """转换为前端格式"""
        return {
            "peakFlops": self.peak_flops,
            "peakBandwidth": self.peak_bandwidth,
            "ridgePoint": self.ridge_point,
            "points": [p.to_dict() for p in self.points],
            "roofline": {
                "x": self.roofline_x,
                "y": self.roofline_y,
            },
        }


class RooflineAnalyzer:
    """Roofline 分析器

    计算 Roofline 模型数据。
    """

    def __init__(
        self,
        peak_flops_gflops: float,
        peak_bandwidth_gbps: float,
    ) -> None:
        """初始化

        Args:
            peak_flops_gflops: 峰值算力 (GFLOPS)
            peak_bandwidth_gbps: 峰值带宽 (GB/s)
        """
        self.peak_flops = peak_flops_gflops
        self.peak_bandwidth = peak_bandwidth_gbps
        self.ridge_point = self._calculate_ridge_point()

    def _calculate_ridge_point(self) -> float:
        """计算拐点

        Ridge Point = Peak FLOPS / Peak Bandwidth

        Returns:
            float: 拐点 (FLOPS/Byte)
        """
        if self.peak_bandwidth <= 0:
            return 0.0
        return self.peak_flops / self.peak_bandwidth

    def get_attainable_flops(self, arithmetic_intensity: float) -> float:
        """获取给定算术强度的可达算力

        Roofline Model:
        attainable = min(peak_flops, arithmetic_intensity × peak_bandwidth)

        Args:
            arithmetic_intensity: 算术强度 (FLOPS/Byte)

        Returns:
            float: 可达算力 (GFLOPS)
        """
        memory_bound_flops = arithmetic_intensity * self.peak_bandwidth
        return min(self.peak_flops, memory_bound_flops)

    def analyze_point(
        self,
        name: str,
        flops: int,
        bytes_accessed: int,
        time_ns: float,
    ) -> RooflinePoint:
        """分析单个数据点

        Args:
            name: 点名称
            flops: 浮点运算数
            bytes_accessed: 访问字节数
            time_ns: 执行时间 (纳秒)

        Returns:
            RooflinePoint: 数据点
        """
        # 计算算术强度
        ai = flops / bytes_accessed if bytes_accessed > 0 else 0.0

        # 计算实际达到的算力 (GFLOPS)
        achieved = (flops / time_ns) if time_ns > 0 else 0.0  # FLOPS/ns = GFLOPS

        # 计算可达算力
        attainable = self.get_attainable_flops(ai)

        # 判断瓶颈
        is_compute_bound = ai >= self.ridge_point
        is_memory_bound = ai < self.ridge_point

        # 计算效率
        efficiency = achieved / attainable if attainable > 0 else 0.0

        return RooflinePoint(
            name=name,
            arithmetic_intensity=ai,
            achieved_flops=achieved,
            peak_flops=self.peak_flops,
            peak_bandwidth=self.peak_bandwidth,
            is_compute_bound=is_compute_bound,
            is_memory_bound=is_memory_bound,
            efficiency=efficiency,
        )

    def generate_roofline_curve(
        self,
        ai_min: float = 0.01,
        ai_max: float = 1000.0,
        num_points: int = 100,
    ) -> tuple[list[float], list[float]]:
        """生成 Roofline 曲线

        Args:
            ai_min: 最小算术强度
            ai_max: 最大算术强度
            num_points: 采样点数

        Returns:
            tuple[list[float], list[float]]: (X 坐标, Y 坐标)
        """
        import math

        x = []
        y = []

        # 对数采样
        log_min = math.log10(ai_min)
        log_max = math.log10(ai_max)
        log_step = (log_max - log_min) / (num_points - 1)

        for i in range(num_points):
            ai = 10 ** (log_min + i * log_step)
            flops = self.get_attainable_flops(ai)
            x.append(ai)
            y.append(flops)

        return x, y

    def analyze(
        self,
        steps: list["StepMetrics"],
    ) -> RooflineData:
        """分析步骤列表

        Args:
            steps: 步骤指标列表

        Returns:
            RooflineData: Roofline 数据
        """
        points = []
        for step in steps:
            if step.flops > 0 and step.bytes_accessed > 0:
                point = self.analyze_point(
                    name=step.name,
                    flops=step.flops,
                    bytes_accessed=step.bytes_accessed,
                    time_ns=step.total_ns,
                )
                points.append(point)

        # 生成 Roofline 曲线
        roofline_x, roofline_y = self.generate_roofline_curve()

        return RooflineData(
            peak_flops=self.peak_flops,
            peak_bandwidth=self.peak_bandwidth,
            ridge_point=self.ridge_point,
            points=points,
            roofline_x=roofline_x,
            roofline_y=roofline_y,
        )


def build_roofline_from_chip(
    chip: "ChipSpec",
    steps: list["StepMetrics"],
) -> RooflineData:
    """从芯片规格构建 Roofline 数据

    Args:
        chip: 芯片规格
        steps: 步骤指标列表

    Returns:
        RooflineData: Roofline 数据
    """
    # 转换单位: FLOPS -> GFLOPS, B/s -> GB/s
    peak_gflops = chip.peak_flops / 1e9
    peak_gbps = chip.memory_bandwidth / 1e9

    analyzer = RooflineAnalyzer(peak_gflops, peak_gbps)
    return analyzer.analyze(steps)


def build_roofline_from_engine_result(
    result: "EngineResult",
) -> RooflineData:
    """从评估结果构建 Roofline 数据

    Args:
        result: 评估结果

    Returns:
        RooflineData: Roofline 数据
    """
    if not result.hardware_spec:
        raise ValueError("EngineResult 缺少 hardware_spec")

    # 转换单位
    peak_gflops = result.hardware_spec.peak_flops / 1e9
    peak_gbps = result.hardware_spec.memory_bandwidth / 1e9

    analyzer = RooflineAnalyzer(peak_gflops, peak_gbps)

    # 合并 prefill 和 decode 步骤
    all_steps = result.prefill_steps + result.decode_steps

    return analyzer.analyze(all_steps)
