# PAXI 技术分析文档

本目录包含合见工软(UniVista) PAXI (Protocol of Accelerated eXchange Interconnect) 协议及其完整技术栈的详细技术分析。

## 文档结构

| 文件 | 内容 |
|------|------|
| [01_overview.md](01_overview.md) | 协议总览与定位 |
| [02_architecture.md](02_architecture.md) | 微架构与协议栈分层 |
| [03_flit_encoding.md](03_flit_encoding.md) | Flit编码规则与帧格式 |
| [04_virtual_channel.md](04_virtual_channel.md) | 虚拟通道与仲裁机制 |
| [05_flow_control.md](05_flow_control.md) | 流控机制 (CBFC/PFC) |
| [06_error_handling.md](06_error_handling.md) | 错误处理与重传机制 |
| [07_rdma_engine.md](07_rdma_engine.md) | RDMA引擎(RoCEv2) [历史参考] |
| [08_cesoc_phy.md](08_cesoc_phy.md) | CESOC 800G物理层分析 |
| [09_dcqcn.md](09_dcqcn.md) | DCQCN拥塞控制算法 [历史参考] |
| [10_register_map.md](10_register_map.md) | 寄存器映射与软件接口 |
| [11_operational_flows.md](11_operational_flows.md) | 操作流程(初始化/测量/多播/错误恢复) |
| [12_rclink.md](12_rclink.md) | RC Link传输层 (TYPE1/TYPE2/TYPE3) |
| [13_multicast.md](13_multicast.md) | 多播功能 (8组, 每组16设备) |

## 信息来源

| 文档 | 版本 | 日期 | 说明 |
|------|------|------|------|
| UniVista PAXI SUE2.0 Core UserGuide | V2R0P5 | 2026年1月 | PAXI Core SUE2.0架构 (主要参考) |
| RCLINK AFH SPEC | v2.4 | - | RC Link传输层规格 (主要参考) |
| PAXI Reference Guide | v2R0p6 | 2025年4月 | 旧版PAXI参考 (历史) |
| CESOC 800G Reference Guide | v0.9 | 2024年10月 | CESOC物理层参考 |
| UniVista RDMA Core Reference Guide | Rev.17 | 2025年11月 | RDMA引擎参考 (历史) |

## 架构说明

SUE2.0协议栈 (从上到下):

```
Layer 5: Application / NoC (AXI4 / APB3)
Layer 4: PAXI Core (事务层 - AXI <-> Flit编码)
Layer 3: RC Link (传输层 - 可靠传输/Go-Back-N/速率控制)
Layer 2: CESOC CEFEC (MAC/PCS/FEC)
Layer 1: SerDes (112G PAM4)
```

07_rdma_engine.md 和 09_dcqcn.md 描述的是旧版架构, 其功能在SUE2.0中由RC Link (12_rclink.md) 替代。

## 标注约定

- **[DOC]** - 直接引用自官方文档原文
- **[推导]** - 基于文档信息的合理推导
- **[行业]** - 基于行业通用知识的补充说明
