# PAXI 技术分析文档

本目录包含合见工软(UniVista) PAXI协议及其完整技术栈的详细技术分析。

## 文档结构

| 文件 | 内容 |
|------|------|
| [01_overview.md](01_overview.md) | 协议总览与定位 |
| [02_architecture.md](02_architecture.md) | 微架构与协议栈分层 |
| [03_flit_encoding.md](03_flit_encoding.md) | Flit编码规则与帧格式 |
| [04_virtual_channel.md](04_virtual_channel.md) | 虚拟通道与仲裁机制 |
| [05_flow_control.md](05_flow_control.md) | Credit流控与PFC背压 |
| [06_error_handling.md](06_error_handling.md) | 错误处理与重传机制 |
| [07_rdma_engine.md](07_rdma_engine.md) | RDMA引擎(RoCEv2)详细分析 |
| [08_cesoc_phy.md](08_cesoc_phy.md) | CESOC 800G物理层分析 |
| [09_dcqcn.md](09_dcqcn.md) | DCQCN拥塞控制算法 |
| [10_register_map.md](10_register_map.md) | 寄存器映射与软件接口 |
| [11_operational_flows.md](11_operational_flows.md) | 操作流程(初始化/测量/重配置) |

## 信息来源

| 文档 | 版本 | 日期 |
|------|------|------|
| PAXI_Reference_Guide | v2.0.6 (R0p6) | 2025年4月 |
| CESOC_800G_192B_8CH_X8_TOP Reference Guide | v0.9 | 2024年10月 |
| UniVista RDMA Core Reference Guide | Rev.17 | 2025年11月 |
| UV-RDMA FPGA DEMO REPORT | v0.10 | 2025年4月 |

## 标注约定

- **[DOC]** - 直接引用自官方文档原文
- **[推导]** - 基于文档信息的合理推导
- **[行业]** - 基于行业通用知识的补充说明
