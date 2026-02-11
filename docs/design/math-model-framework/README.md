# Math Model Framework 设计文档

本目录包含 `math_model` 框架的分层设计文档，对齐 CHIPMathica 方法论，
描述当前仓库 (`backend/math_model/`) 的实际架构与实现。

## 文档索引

| 文档 | 内容 |
|------|------|
| [00-architecture.md](00-architecture.md) | 总体架构、分层设计、数据流、配置管线 |
| [01-l0-entry.md](01-l0-entry.md) | L0 入口层: API、EvalConfig、配置加载、任务管理 |
| [02-l1-workload.md](02-l1-workload.md) | L1 负载层: WorkloadIR、Layer/Op 模型、DeepSeek V3 |
| [03-l2-arch.md](03-l2-arch.md) | L2 架构层: 5 级硬件层级、ChipSpec、TopologySpec |
| [04-l3-mapping.md](04-l3-mapping.md) | L3 映射层: ParallelismPlanner、TilingPlanner、Scheduler |
| [05-l4-evaluation.md](05-l4-evaluation.md) | L4 评估层: EvaluationEngine、CostModel、CommProtocol |
| [06-l5-reporting.md](06-l5-reporting.md) | L5 报告层: ReportingEngine、CostAnalyzer、可视化 |

## 版本

- 文档版本: v2.2.5
- 对应代码: `backend/math_model/`
- 创建日期: 2026-02-11
