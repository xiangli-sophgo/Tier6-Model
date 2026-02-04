"""导出器 - JSON 导出实现."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from tier6.L5_reporting.models import ReportingReport, OutputConfig


class Exporter(Protocol):
    """导出器协议"""

    def export(
        self,
        report: ReportingReport,
        output_path: str,
        config: OutputConfig,
    ) -> str:
        """导出报告

        Args:
            report: 指标报告
            output_path: 输出路径
            config: 输出配置

        Returns:
            实际输出文件路径
        """
        ...


class JSONExporter:
    """JSON 导出器"""

    def export(
        self,
        report: ReportingReport,
        output_path: str,
        config: OutputConfig,
    ) -> str:
        """导出为 JSON

        输入:
            - report: 指标报告
            - output_path: 输出路径
            - config: 输出配置
        输出:
            - 实际输出文件路径
        """
        # 确保目录存在
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        # 转换为字典
        data = self._to_dict(report)

        # 写入文件
        indent = 2 if config.pretty_print else None
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)

        return output_path

    def _to_dict(self, obj: Any) -> Any:
        """递归转换为字典"""
        if is_dataclass(obj) and not isinstance(obj, type):
            return {k: self._to_dict(v) for k, v in asdict(obj).items()}
        elif isinstance(obj, dict):
            return {k: self._to_dict(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._to_dict(item) for item in obj]
        else:
            return obj


class ExporterRegistry:
    """导出器注册表"""

    def __init__(self) -> None:
        self._exporters: dict[str, Exporter] = {}
        # 注册默认导出器
        self.register("json", JSONExporter())

    def register(self, format_name: str, exporter: Exporter) -> None:
        """注册导出器

        Args:
            format_name: 格式名称
            exporter: 导出器实例
        """
        self._exporters[format_name.lower()] = exporter

    def get(self, format_name: str) -> Exporter | None:
        """获取导出器

        Args:
            format_name: 格式名称

        Returns:
            导出器实例或 None
        """
        return self._exporters.get(format_name.lower())

    def export(
        self,
        format_name: str,
        report: ReportingReport,
        output_path: str,
        config: OutputConfig,
    ) -> str:
        """导出报告

        Args:
            format_name: 格式名称
            report: 指标报告
            output_path: 输出路径
            config: 输出配置

        Returns:
            实际输出文件路径

        Raises:
            ValueError: 未知格式
        """
        exporter = self.get(format_name)
        if exporter is None:
            raise ValueError(f"Unknown export format: {format_name}")
        return exporter.export(report, output_path, config)


# 默认导出器注册表实例（供顶层 API 延迟导入使用）
exporter_registry = ExporterRegistry()
