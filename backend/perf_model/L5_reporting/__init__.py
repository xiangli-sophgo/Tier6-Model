"""L5: 报告与可视化层

生成 Gantt 图、成本分解、内存分析、Roofline 等报告数据。

核心类型:
    - GanttChartData: 甘特图数据
    - GanttChartBuilder: 甘特图构建器
    - CostBreakdown: 成本分解
    - CostAnalyzer: 成本分析器
    - MemoryBreakdown: 内存占用分解
    - MemoryAnalyzer: 内存分析器
    - RooflineData: Roofline 图数据
    - RooflineAnalyzer: Roofline 分析器
    - TrafficReport: 流量报告
    - TrafficAnalyzer: 流量分析器
"""

from perf_model.L5_reporting.gantt import (
    GanttChartBuilder,
    GanttChartData,
    GanttResource,
    GanttTask,
    GanttTaskType,
    InferencePhase,
    TASK_COLORS,
    build_gantt_from_engine_result,
)
from perf_model.L5_reporting.cost_analysis import (
    CHIP_PRICES,
    CostAnalyzer,
    CostBreakdown,
    INTERCONNECT_COST_TIERS,
)
from perf_model.L5_reporting.memory_analysis import (
    MemoryAnalyzer,
    MemoryBreakdown,
)
from perf_model.L5_reporting.roofline import (
    RooflineAnalyzer,
    RooflineData,
    RooflinePoint,
    build_roofline_from_chip,
    build_roofline_from_engine_result,
)
from perf_model.L5_reporting.traffic_analysis import (
    DeviceTraffic,
    LinkTraffic,
    LinkType,
    TrafficAnalyzer,
    TrafficReport,
)

__all__ = [
    # Gantt
    "GanttTaskType",
    "InferencePhase",
    "TASK_COLORS",
    "GanttTask",
    "GanttResource",
    "GanttChartData",
    "GanttChartBuilder",
    "build_gantt_from_engine_result",
    # Cost
    "CHIP_PRICES",
    "INTERCONNECT_COST_TIERS",
    "CostBreakdown",
    "CostAnalyzer",
    # Memory
    "MemoryBreakdown",
    "MemoryAnalyzer",
    # Roofline
    "RooflinePoint",
    "RooflineData",
    "RooflineAnalyzer",
    "build_roofline_from_chip",
    "build_roofline_from_engine_result",
    # Traffic
    "LinkType",
    "LinkTraffic",
    "DeviceTraffic",
    "TrafficReport",
    "TrafficAnalyzer",
]
