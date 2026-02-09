# SG2262 C2C 互联方案技术分析

本目录包含 SG2262 芯片 Chip-to-Chip 互联方案的详细技术分析，以及在 Tier6+Model 工具中的建模方案。

## 文档结构

| 文件 | 内容 |
|------|------|
| [01_overview.md](01_overview.md) | C2C 方案总览：设计需求、Feature List、拓扑支持 |
| [02_architecture.md](02_architecture.md) | 微架构详解：MAC ID 映射、CLE 路由、CDMA、保序机制 |
| [03_communication.md](03_communication.md) | 通信机制：Send/Receive 流程、报文格式、Datagram |
| [04_modeling_analysis.md](04_modeling_analysis.md) | 建模分析：现有工具能力评估与 C2C 方案建模策略 |

## 信息来源

| 文档 | 版本 | 日期 |
|------|------|------|
| SG2262 C2C 方案 | v1.0.1 | 2025-10-24 |

## 标注约定

- **[DOC]** - 直接引用自 SG2262 C2C 方案文档原文
- **[推导]** - 基于文档信息的合理推导
- **[建模]** - 与 Tier6+Model 工具建模相关的分析
