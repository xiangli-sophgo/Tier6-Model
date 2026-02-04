"""L5 Reporting 数据模型（精简版）."""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Any


@dataclass
class OutputConfig:
    """输出配置

    Attributes:
        output_dir: 输出目录
        include_step_metrics: 是否包含 step 级指标
        pretty_print: JSON 格式化输出
    """

    output_dir: str = "./output"
    include_step_metrics: bool = True
    pretty_print: bool = True


@dataclass
class PerformanceSummary:
    """性能摘要

    Attributes:
        total_time_ms: 总执行时间（ms）
        ttft_ms: Time To First Token（ms）
        tpot_ms: Time Per Output Token（ms）
        tps: Tokens Per Second
        mfu: Model FLOPS Utilization
        mbu: Memory Bandwidth Utilization
        memory_peak_mb: 内存峰值（MB）
        compute_time_ms: 计算时间（ms）
        comm_time_ms: 通信时间（ms）
        wait_time_ms: 等待时间（ms）
        total_flops: 总 FLOPs
        total_bytes: 总访存量（bytes）
        num_ops: Op 数量
    """

    total_time_ms: float = 0.0
    ttft_ms: float = 0.0
    tpot_ms: float = 0.0
    tps: float = 0.0
    mfu: float = 0.0
    mbu: float = 0.0
    memory_peak_mb: float = 0.0
    compute_time_ms: float = 0.0
    comm_time_ms: float = 0.0
    wait_time_ms: float = 0.0
    total_flops: int = 0
    total_bytes: int = 0
    num_ops: int = 0


@dataclass
class BottleneckSummary:
    """瓶颈摘要

    Attributes:
        compute_bound_count: 计算瓶颈 Op 数量
        bw_bound_count: 带宽瓶颈 Op 数量
        latency_bound_count: 延迟瓶颈 Op 数量
        unknown_count: 未知瓶颈 Op 数量
        top_ops: Top N 耗时 Op
    """

    compute_bound_count: int = 0
    bw_bound_count: int = 0
    latency_bound_count: int = 0
    unknown_count: int = 0
    top_ops: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ReportingReport:
    """指标报告（L5 核心输出）

    Attributes:
        schema_version: 输出结构版本
        timestamp: 生成时间
        granularity: 评估精度
        performance: 性能摘要
        bottleneck: 瓶颈摘要
        config: 运行配置摘要
        step_metrics: Step 级指标（可选）
    """

    schema_version: str = "1.0.0"
    timestamp: str = ""
    granularity: str = "CHIP"
    performance: PerformanceSummary = field(default_factory=PerformanceSummary)
    bottleneck: BottleneckSummary = field(default_factory=BottleneckSummary)
    config: dict[str, Any] = field(default_factory=dict)
    step_metrics: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.datetime.now().isoformat()


@dataclass
class ReportText:
    """文本报告

    Attributes:
        summary: 总览文本
        full_text: 完整报告文本
    """

    summary: str = ""
    full_text: str = ""
