"""ReportingEngine - L5 统一入口."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from tier6.L5_reporting.assembler import ReportingAssembler
from tier6.L5_reporting.exporters import ExporterRegistry
from tier6.L5_reporting.models import OutputConfig, ReportText

if TYPE_CHECKING:
    from tier6.L4_evaluation.metrics import EngineResult
    from tier6.L5_reporting.models import ReportingReport


class ReportingEngine:
    """L5 统一入口

    功能：校验、装配、导出与结果落盘

    计算流程:
        - 校验输入：检查 EngineResult 字段完整性
        - 装配视图：调用 ReportingAssembler 生成 ReportingReport
        - 构建报告：生成文本摘要
        - 导出产物：调用 ExporterRegistry 生成 JSON
    """

    def __init__(self) -> None:
        self.assembler = ReportingAssembler()
        self.exporter_registry = ExporterRegistry()

    def run(
        self,
        engine_result: EngineResult,
        config: dict | None = None,
        output_config: OutputConfig | None = None,
    ) -> ReportingReport:
        """执行指标处理

        输入:
            - engine_result: L4 评估结果
            - config: 运行配置（会嵌入输出）
            - output_config: 输出配置
        输出:
            - ReportingReport
        关键步骤:
            - 校验输入 → 装配视图 → 返回报告
        """
        if output_config is None:
            output_config = OutputConfig()

        # 1. 校验输入
        self._validate_input(engine_result)

        # 2. 装配视图
        report = self.assembler.assemble(
            engine_result,
            config=config,
            include_step_metrics=output_config.include_step_metrics,
        )

        return report

    def build_text(self, report: ReportingReport) -> ReportText:
        """构建文本报告

        输入:
            - report: 指标报告
        输出:
            - ReportText
        """
        perf = report.performance

        # 摘要
        summary_lines = [
            f"=== Performance Summary (granularity: {report.granularity}) ===",
            f"Total Time: {perf.total_time_ms:.2f} ms",
            f"TTFT: {perf.ttft_ms:.2f} ms | TPOT: {perf.tpot_ms:.2f} ms | TPS: {perf.tps:.2f}",
            f"MFU: {perf.mfu:.2%} | MBU: {perf.mbu:.2%}",
            f"Memory Peak: {perf.memory_peak_mb:.2f} MB",
        ]
        summary = "\n".join(summary_lines)

        # 完整报告
        full_lines = [
            summary,
            "",
            "--- Time Breakdown ---",
            f"Compute: {perf.compute_time_ms:.2f} ms ({self._ratio(perf.compute_time_ms, perf.total_time_ms):.1%})",
            f"Comm:    {perf.comm_time_ms:.2f} ms ({self._ratio(perf.comm_time_ms, perf.total_time_ms):.1%})",
            f"Wait:    {perf.wait_time_ms:.2f} ms ({self._ratio(perf.wait_time_ms, perf.total_time_ms):.1%})",
            "",
            "--- Bottleneck Summary ---",
            f"Compute Bound: {report.bottleneck.compute_bound_count}",
            f"BW Bound:      {report.bottleneck.bw_bound_count}",
            f"Latency Bound: {report.bottleneck.latency_bound_count}",
            f"Unknown:       {report.bottleneck.unknown_count}",
        ]

        if report.bottleneck.top_ops:
            full_lines.append("")
            full_lines.append("--- Top 5 Ops by Time ---")
            for i, op in enumerate(report.bottleneck.top_ops, 1):
                full_lines.append(
                    f"  {i}. {op['op_id']}: {op['t_total_ms']:.4f} ms ({op['bottleneck']})"
                )

        full_text = "\n".join(full_lines)

        return ReportText(summary=summary, full_text=full_text)

    def export(
        self,
        report: ReportingReport,
        output_config: OutputConfig | None = None,
        filename: str = "reporting_report.json",
    ) -> str:
        """导出报告

        输入:
            - report: 指标报告
            - output_config: 输出配置
            - filename: 输出文件名
        输出:
            - 输出文件路径
        """
        if output_config is None:
            output_config = OutputConfig()

        output_path = os.path.join(output_config.output_dir, filename)
        return self.exporter_registry.export("json", report, output_path, output_config)

    def _validate_input(self, engine_result: EngineResult) -> None:
        """校验输入

        Args:
            engine_result: L4 评估结果

        Raises:
            ValueError: 字段缺失
        """
        if engine_result.aggregates is None:
            raise ValueError("EngineResult.aggregates is required")

    def _ratio(self, part: float, total: float) -> float:
        """计算比例"""
        return part / total if total > 0 else 0.0
