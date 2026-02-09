# 01. PAXI 协议总览与定位

## 1.1 什么是PAXI

**PAXI (Protocol of Accelerated eXchange Interconnect)** 是合见工业软件集团(Shanghai UniVista Industrial Software Group)开发的芯片间互联IP核。

**[DOC]** PAXI SUE2.0 Core的定义:

> "The PAXI (Protocol of Accelerated eXchange Interconnect) core is designed to attach a set of Ethernet cores to the advanced microcontroller bus architecture (AMBA) advanced extensible interface (AXI) bus."

> "Transactions initiated by an AXI Master are handled by the PAXI core AXI Slave interface and generate requests on the RC Link interface. Requests received on the Ethernet Link are handled by the PAXI core AXI Master interface and generate AXI transactions."

简言之, PAXI将片上AXI4总线事务透明地映射到以太网物理链路上, 实现芯片间(Chip-to-Chip)的高速互联。从上层软件/NoC的视角看, 远端芯片的内存空间就像本地AXI可寻址空间的一部分。

**[推导]** SUE2.0相比旧版(Point-to-point AXI Wrapper)的核心变化:
- 名称从"Point-to-point AXI Wrapper"更改为"Protocol of Accelerated eXchange Interconnect", 反映其已从简单封装器演变为完整的互联协议
- PAXI不再直接连接MAC, 而是通过新的RC Link传输层进行连接

## 1.2 技术栈定位

PAXI是合见工软完整智算互联方案(SUE Protocol)中的**事务层(Transaction Layer)**组件:

```
完整技术栈 (SUE2.0):
  Application / NoC         -- AXI4/APB3 接口
      |
  PAXI Core                 -- 事务层: AXI <-> Flit 编码/解码
      |
  RC Link                   -- 传输层: 可靠传输/Go-Back-N/速率控制/CBFC
      |
  CESOC (MAC/PCS/FEC)       -- 数据链路层: 以太网控制器
      |
  SerDes (112G PAM4)        -- 物理层
```

**[推导]** 与旧版架构的对比:

| 层级 | 旧版 (v2R0p6) | SUE2.0 |
|------|---------------|--------|
| 事务层 | PAXI Core (协议转换+流控) | PAXI Core (纯协议转换) |
| 传输层 | RDMA Engine (RoCEv2, 可选) | RC Link (必选, 内置于协议栈) |
| 数据链路层 | CESOC (MAC/PCS/FEC) | CESOC (MAC/PCS/FEC) |
| 物理层 | SerDes (112G PAM4) | SerDes (112G PAM4) |

关键变化: RC Link替代了旧版RDMA Engine的大部分功能, 成为协议栈的必要组件。PAXI不再直接管理per-DA Credit流控, 这些功能下放到RC Link的CBFC机制。

## 1.3 核心特性清单

以下特性引用自PAXI SUE2.0 UserGuide V2R0P5 (1.2.1 General Features):

### 接口与协议

- **[DOC]** Support AXI4 interface for NOC, support all 5 channels
- **[DOC]** Support APB3 interface for NOC
- **[DOC]** Chip-to-Chip AXI interface connection over Ethernet link
- **[DOC]** Compatible with Layer 2 switches
- **[DOC]** Configurable MAC DA and VC through AXI user bus

### 性能

- **[DOC]** Ultra low latency down to 150ns(400G) between AXI-to-AXI
- **[DOC]** Support 200G/400G Ethernet MAC
- **[DOC]** Support overrate mode(220G/440G MAC)
- **[DOC]** Configurable Memory ready latency (1~3)

### 流控与虚拟通道

- **[DOC]** Base-on credit flow control with RC Link
- **[DOC]** Support 8 CBFC and PFC VC
- **[DOC]** Configurable RX AXI Master OST number

### 多播

- **[DOC]** Support Multicast Frame
- **[DOC]** Support maximum 8 multicast group
- **[DOC]** Each multi-cast group support maximum 16 devices

### 可靠性

- **[DOC]** Support error free transmission with RC LINK E2E retry enabled

### 调试与测量

- **[DOC]** Remote chip register access
- **[DOC]** Near-end loopbacks at RC LINK interface
- **[DOC]** Latency measurement(Round-Trip Time include noc)
- **[DOC]** Support Internal pattern generator
- **[DOC]** Various status indicators for debug
- **[DOC]** DFX pin out

### 工艺与集成

- **[DOC]** Commercial simulator and synthesizer supported
- **[DOC]** Synthesizable at 3nm/4nm/5nm/7nm/8nm/12nm/14nm/16nm
- **[DOC]** Configurable synchronizer stage

## 1.4 关键性能指标

| 指标 | 值 | 来源 |
|------|------|------|
| AXI-to-AXI端到端延迟 | 低至150ns @ 400G | [DOC] PAXI Features |
| 支持的MAC带宽 | 200G / 400G | [DOC] PAXI Features |
| 超频模式带宽 | 220G / 440G | [DOC] PAXI Features |
| CBFC/PFC虚拟通道数 | 8 | [DOC] PAXI Features |
| 多播组数 | 最多8组 | [DOC] PAXI Features |
| 每组多播设备数 | 最多16 | [DOC] PAXI Features |
| RX AXI Master OST | 可配置 | [DOC] PAXI Features |

**[推导]** 与旧版指标对比:

| 指标 | 旧版 | SUE2.0 | 变化 |
|------|------|--------|------|
| 虚拟通道 | 3 (Mgmt/Ctrl/Data) | 8 CBFC/PFC VC | 扩展, 改为标准CBFC |
| OST | 128/256 二选一 | 可配置 | 更灵活 |
| DA管理 | 128 DA, Per-DA Credit | 通过Multi DA Enable管理 | 简化 |
| 流控机制 | PAXI内部Credit | RC Link CBFC | 下放到传输层 |
| 重传机制 | MAC L1/L2 Retry | RC Link E2E Retry | 端到端 |
| 多播 | 不支持 | 8组x16设备 | 新增 |

## 1.5 C2C模式 vs Switch模式

**[DOC]** PAXI支持两种组网模式:

### C2C直连模式

```
Chip A                              Chip B
+---------+   SerDes <-> SerDes  +---------+
| PAXI    |                      | PAXI    |
| RC Link |----------------------| RC Link |
| CESOC   |                      | CESOC   |
+---------+                      +---------+
```

- 两个芯片通过以太网PHY直接相连
- 最低延迟: 150ns @ 400G
- 适用场景: 同板/同模块内的芯片互联

### Switch模式

```
Chip A                              Chip B
+---------+      +----------+    +---------+
| PAXI    |------| L2       |----| PAXI    |
| RC Link |      | Ethernet |    | RC Link |
+---------+      | Switch   |    +---------+
                 +----------+
Chip C              |
+---------+         |
| PAXI    |---------+
| RC Link |
+---------+
```

- **[DOC]** Compatible with Layer 2 switches
- 通过标准二层以太网交换机组网
- 延迟增加(交换机转发延迟)
- 适用场景: 多芯片集群, 灵活扩展

**[推导]** SUE2.0通过RC Link层的多种报文格式(Standard/AFH_GEN1/AFH_GEN2_16b/AFH_Lite)支持不同的交换机兼容模式。

## 1.6 序列顺序保证

**[DOC]** 来自2.1 PAXI Partitions:

> "Note PAXI not guarantee the order for same address access of read and write transfer."

> "AXI transaction which send to different DA will not guarantee the order PAXI received on AXI interface."

这意味着:

- 同一地址的读写事务之间**没有**顺序保证
- 发往不同DA的AXI事务之间**没有**顺序保证
- 上层软件/NoC需要自行管理顺序依赖
- **[推导]** 这是为了最大化吞吐量的设计选择, 避免跨DA和读写之间的流水线阻塞

## 1.7 参考文档

**[DOC]** SUE2.0 References:

- MAC 200G Specification (Version 0.3)
- AMBA AXI and ACE Protocol Specification
- AMBA 3 APB Protocol Version 1.0 Specification
