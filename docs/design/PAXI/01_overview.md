# 01. PAXI 协议总览与定位

## 1.1 什么是PAXI

**PAXI (Point-to-point AXI Wrapper)** 是合见工业软件集团(Shanghai UniVista Industrial Software Group)开发的芯片间互联IP核。

**[DOC]** PAXI Core的完整定义:

> "The PAXI (Point-to-point AXI Wrapper) core is designed to attach a set of Ethernet cores to the advanced microcontroller bus architecture (AMBA) advanced extensible interface (AXI) bus."

> "Transactions initiated by an AXI Master are handled by the PAXI core AXI Slave interface and generate requests on the AXI Stream interface of Ethernet Link. Requests received on the Ethernet Link are handled by the PAXI core AXI Master interface and generate AXI transactions."

简言之, PAXI将片上AXI4总线事务透明地映射到以太网物理链路上, 实现芯片间(Chip-to-Chip)的高速互联。从上层软件/NoC的视角看, 远端芯片的内存空间就像本地AXI可寻址空间的一部分。

## 1.2 技术栈定位

PAXI不是一个独立协议, 而是合见工软完整智算互联方案中的**传输层(Transport Layer)**组件:

```
完整技术栈:
  RDMA Engine (RoCEv2)    -- 可靠传输与内存语义
      |
  PAXI Core               -- AXI <-> Ethernet协议转换
      |
  CESOC (MAC/PCS/FEC)     -- 以太网控制器
      |
  SerDes (112G PAM4)      -- 物理层
```

**[DOC]** FPGA Demo报告中明确列出三大IP模块:

| No. | IP/Blocks | Description |
|-----|-----------|-------------|
| 1 | RDMA | RoCEv2 400G, 128 QP, RC/UD, Send/Receive/Write/Read |
| 2 | PAXI | Memory语义支持 |
| 3 | Ethernet MAC | 200G/400G |

## 1.3 核心特性清单

以下所有特性均直接引用自PAXI Reference Guide v2.0.6第186行(1.2.1 General Features):

### 接口与协议

- **[DOC]** Support AXI4 interface for NOC, support all 5 channels
- **[DOC]** Support APB3 interface for NOC
- **[DOC]** Chip-to-Chip AXI interface connection over Ethernet link
- **[DOC]** Support both C2C mode and Switch mode

### 性能

- **[DOC]** Ultra low latency down to 150ns(400G) between AXI-to-AXI
- **[DOC]** Support Simplified MAC framing to achieve >99% Max bandwidth utilization
- **[DOC]** Support 200G/400G Ethernet MAC
- **[DOC]** Support overrate mode(220G/440G MAC)

### 流控

- **[DOC]** Configurable AXI OST number 128 or 256
- **[DOC]** Inside credit function for flow control support
- **[DOC]** Chip-to-Chip flow controller based on AXI OST number
- **[DOC]** Support PFC and Pause flow control
- **[DOC]** Configurable MAC DA through AXI user bus

### 虚拟通道

- **[DOC]** Support 3 type virtual channels, inside management/control/data
- **[DOC]** Independent enablement/reset for 3 virtual channels
- **[DOC]** Support WRR for virtual channel priority

### 可靠性

- **[DOC]** Support error free transmission with L1 retry enabled
- **[DOC]** Support error free transmission with MAC L1/L2 retry enabled

### 调试与测量

- **[DOC]** Remote chip register access
- **[DOC]** Near-end loopbacks at MAC interface
- **[DOC]** Latency measurement(Round-Trip Time include noc)
- **[DOC]** Various status indicators for debug
- **[DOC]** Statistics for MAC framing of virtual channels

### 工艺与集成

- **[DOC]** Configurable Memory ready latency
- **[DOC]** Commercial simulator and synthesizer supported
- **[DOC]** Synthesizable at 3nm/4nm/5nm/7nm/8nm/12nm/14nm/16nm
- **[DOC]** DFX pin out
- **[DOC]** Configurable synchronizer stage

## 1.4 关键性能指标

| 指标 | 值 | 来源 |
|------|------|------|
| AXI-to-AXI端到端延迟 | 低至150ns @ 400G | [DOC] PAXI Features |
| 带宽利用率 | >99% | [DOC] PAXI Features |
| 支持的MAC带宽 | 200G / 400G | [DOC] PAXI Features |
| 超频模式带宽 | 220G / 440G | [DOC] PAXI Features |
| 目的地址(DA)数量 | 128 | [DOC] 2.3 DA insertion |
| Outstanding Transaction | 128 或 256 (可配置) | [DOC] PAXI Features |
| 虚拟通道数 | 3 (Management/Control/Data) | [DOC] PAXI Features |
| WRR权重范围 | 1 ~ 255 | [DOC] 2.4 VC Arbitration |

## 1.5 C2C模式 vs Switch模式

**[DOC]** PAXI支持两种组网模式:

### C2C直连模式

```
Chip A                              Chip B
┌──────┐   SerDes <-> SerDes   ┌──────┐
│ PAXI ├───────────────────────┤ PAXI │
└──────┘                       └──────┘
```

- 两个芯片通过以太网PHY直接相连
- 最低延迟: 150ns @ 400G
- 适用场景: 同板/同模块内的芯片互联

### Switch模式

```
Chip A                              Chip B
┌──────┐         ┌──────────┐  ┌──────┐
│ PAXI ├─────────┤ Ethernet ├──┤ PAXI │
└──────┘         │  Switch  │  └──────┘
                 └──────────┘
Chip C              |
┌──────┐            |
│ PAXI ├────────────┘
└──────┘
```

- 通过标准以太网交换机组网
- 延迟增加(交换机转发延迟)
- 适用场景: 多芯片集群, 灵活扩展

**[DOC]** DA映射机制支持128个目的地址, 配合Switch模式可实现大规模组网。

## 1.6 序列顺序保证

**[DOC]** 来自2.2 AXI Interface Functional Description:

> "Note PAXI not guarantee the order for same address access of read and write transfer."

这意味着:

- 同一地址的读写事务之间**没有**顺序保证
- 上层软件/NoC需要自行管理顺序依赖
- **[推导]** 这是为了最大化吞吐量的设计选择, 避免读写之间的流水线阻塞
