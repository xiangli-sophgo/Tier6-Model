"""校准模块.

提供可选校准参数以修正估时，但不改变指标定义与单位。
只允许乘法/加法修正。
"""

from __future__ import annotations

from dataclasses import dataclass

from math_model.L4_evaluation.metrics import StepMetrics


@dataclass
class CalibrationConfig:
    """校准配置参数

    Attributes:
        effective_bw_factor: 有效带宽系数（0-1），用于修正理论带宽与实际带宽的差异
        congestion_factor: 拥塞系数（>=1），用于模拟通信拥塞
        startup_overhead_ms: 启动开销（ms），加到每个通信操作
        overlap_efficiency: 重叠效率（0-1），用于修正计算/通信重叠
        compute_efficiency: 计算效率系数（0-1），用于修正理论算力与实际算力的差异
    """

    effective_bw_factor: float = 1.0
    congestion_factor: float = 1.0
    startup_overhead_ms: float = 0.0
    overlap_efficiency: float = 1.0
    compute_efficiency: float = 1.0

    def validate(self) -> None:
        """校验参数合法性"""
        if not 0 < self.effective_bw_factor <= 1:
            raise ValueError(
                f"effective_bw_factor 应在 (0, 1] 范围内，实际为 {self.effective_bw_factor}"
            )
        if self.congestion_factor < 1:
            raise ValueError(
                f"congestion_factor 应 >= 1，实际为 {self.congestion_factor}"
            )
        if self.startup_overhead_ms < 0:
            raise ValueError(
                f"startup_overhead_ms 应 >= 0，实际为 {self.startup_overhead_ms}"
            )
        if not 0 <= self.overlap_efficiency <= 1:
            raise ValueError(
                f"overlap_efficiency 应在 [0, 1] 范围内，实际为 {self.overlap_efficiency}"
            )
        if not 0 < self.compute_efficiency <= 1:
            raise ValueError(
                f"compute_efficiency 应在 (0, 1] 范围内，实际为 {self.compute_efficiency}"
            )


class Calibration:
    """校准器

    对 StepMetrics 做系数调整，但不改变指标定义与单位。
    """

    def __init__(self, config: CalibrationConfig | None = None) -> None:
        self.config = config or CalibrationConfig()
        self.config.validate()

    def apply(self, metrics: StepMetrics) -> StepMetrics:
        """应用校准参数

        输入:
            - metrics: 原始 StepMetrics
        输出:
            - 修正后的 StepMetrics（新对象）
        关键步骤:
            - t_compute *= (1 / compute_efficiency)
            - t_comm = t_comm * congestion_factor / effective_bw_factor + startup_overhead
        """
        # 计算时间修正
        t_compute = metrics.t_compute / self.config.compute_efficiency

        # 通信时间修正
        t_comm = (
            metrics.t_comm
            * self.config.congestion_factor
            / self.config.effective_bw_factor
            + self.config.startup_overhead_ms
        )

        # 等待时间保持不变
        t_wait = metrics.t_wait

        # 重新计算总时间
        # 注意：Tile 级 compute-DMA overlap 已在 evaluator (precise.py/compute.py) 中处理
        # Model 级 MoE dispatch/combine overlap 在 engine.py 中处理
        # calibration 层只做系数校准，不做 overlap 计算
        t_total = t_compute + t_comm + t_wait

        return StepMetrics(
            op_id=metrics.op_id,
            t_compute=t_compute,
            t_comm=t_comm,
            t_wait=t_wait,
            t_total=t_total,
            bottleneck_tag=metrics.bottleneck_tag,
            flops=metrics.flops,
            bytes_read=metrics.bytes_read,
            bytes_write=metrics.bytes_write,
            meta={**metrics.meta, "calibrated": True},
        )

    def apply_batch(self, metrics_list: list[StepMetrics]) -> list[StepMetrics]:
        """批量应用校准参数"""
        return [self.apply(m) for m in metrics_list]
