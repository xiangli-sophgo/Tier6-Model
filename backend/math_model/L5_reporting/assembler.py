"""ReportingAssembler - 指标汇总与视图装配."""

from __future__ import annotations

from typing import TYPE_CHECKING

from math_model.L4_evaluation.metrics import BottleneckTag
from math_model.L5_reporting.models import (
    BottleneckSummary,
    ReportingReport,
    PerformanceSummary,
)
from math_model.L5_reporting.schema import SCHEMA_VERSION

if TYPE_CHECKING:
    from math_model.L4_evaluation.metrics import Aggregates, EngineResult, StepMetrics


class ReportingAssembler:
    """指标汇总与视图装配

    功能：在不改变口径的前提下做指标汇总与视图装配
    """

    def assemble(
        self,
        engine_result: EngineResult,
        config: dict | None = None,
        include_step_metrics: bool = True,
    ) -> ReportingReport:
        """装配指标报告

        输入:
            - engine_result: L4 评估结果
            - config: 运行配置
            - include_step_metrics: 是否包含 step 级指标
        输出:
            - ReportingReport
        关键步骤:
            - 转换 Aggregates 为 PerformanceSummary
            - 汇总瓶颈信息
            - 提取 Top-N 耗时 Op
        """
        agg = engine_result.aggregates
        steps = engine_result.step_metrics

        # 1. 构建性能摘要
        performance = self._build_performance_summary(agg)

        # 2. 构建瓶颈摘要
        bottleneck = self._build_bottleneck_summary(agg, steps)

        # 3. 构建 step_metrics（可选）
        step_metrics_list = []
        if include_step_metrics:
            step_metrics_list = self._build_step_metrics_list(steps)

        return ReportingReport(
            schema_version=SCHEMA_VERSION,
            granularity=engine_result.granularity.name,
            performance=performance,
            bottleneck=bottleneck,
            config=config or {},
            step_metrics=step_metrics_list,
        )

    def _build_performance_summary(self, agg: Aggregates) -> PerformanceSummary:
        """构建性能摘要"""
        return PerformanceSummary(
            total_time_ms=agg.total_time,
            ttft_ms=agg.ttft,
            tpot_ms=agg.tpot,
            tps=agg.tps,
            mfu=agg.mfu,
            mbu=agg.mbu,
            memory_peak_mb=agg.memory_peak / (1024 * 1024),
            compute_time_ms=agg.total_compute_time,
            comm_time_ms=agg.total_comm_time,
            wait_time_ms=agg.total_wait_time,
            total_flops=agg.total_flops,
            total_bytes=agg.total_bytes,
            num_ops=agg.num_steps,
        )

    def _build_bottleneck_summary(
        self,
        agg: Aggregates,
        steps: list[StepMetrics],
    ) -> BottleneckSummary:
        """构建瓶颈摘要"""
        summary = agg.bottleneck_summary

        # 统计瓶颈类型
        compute_bound = summary.get(BottleneckTag.COMPUTE_BOUND.name, 0)
        bw_bound = summary.get(BottleneckTag.BW_BOUND.name, 0)
        latency_bound = summary.get(BottleneckTag.LATENCY_BOUND.name, 0)
        unknown = summary.get(BottleneckTag.UNKNOWN.name, 0)

        # 提取 Top-5 耗时 Op
        sorted_steps = sorted(steps, key=lambda s: s.t_total, reverse=True)
        top_ops = [
            {
                "op_id": s.op_id,
                "t_total_ms": round(s.t_total, 4),
                "bottleneck": s.bottleneck_tag.name,
            }
            for s in sorted_steps[:5]
        ]

        return BottleneckSummary(
            compute_bound_count=compute_bound,
            bw_bound_count=bw_bound,
            latency_bound_count=latency_bound,
            unknown_count=unknown,
            top_ops=top_ops,
        )

    def _build_step_metrics_list(
        self,
        steps: list[StepMetrics],
    ) -> list[dict]:
        """构建 step_metrics 列表"""
        return [
            {
                "op_id": s.op_id,
                "t_compute_ms": round(s.t_compute, 4),
                "t_comm_ms": round(s.t_comm, 4),
                "t_wait_ms": round(s.t_wait, 4),
                "t_total_ms": round(s.t_total, 4),
                "bottleneck": s.bottleneck_tag.name,
                "flops": s.flops,
                "bytes_read": s.bytes_read,
                "bytes_write": s.bytes_write,
            }
            for s in steps
        ]
