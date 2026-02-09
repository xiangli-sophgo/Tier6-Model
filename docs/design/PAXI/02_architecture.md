# 02. PAXI 微架构与协议栈分层

## 2.1 PAXI Core内部分区

**[DOC]** 来自SUE2.0 2.1 PAXI Partitions:

> "The PAXI core consists of three main blocks: one that propagates transactions from an AXI Slave interface to the Ethernet Link, one that propagates transactions from the Ethernet Link to an AXI Master interface, and one that enables access to the internal registers from an APB interface."

三大功能块:

```
+----------------------------------------------------------+
|                    PAXI Core (SUE2.0)                    |
|                                                          |
|  +--------------------------------------------------+   |
|  | Block 1: AXI Slave -> RC Link TX                  |   |
|  |  - 接收本地AXI事务 (AW/W/AR通道)                   |   |
|  |  - 编码为REQ Flit (AW+W Frame, AR Frame)           |   |
|  |  - 接收远端AXI响应 (R/B通道)                        |   |
|  |  - 编码为RSP Flit (R Frame, B Frame)               |   |
|  |  - 经VC分类和DA仲裁送RC Link                       |   |
|  +--------------------------------------------------+   |
|                                                          |
|  +--------------------------------------------------+   |
|  | Block 2: RC Link RX -> AXI Master                 |   |
|  |  - 接收RC Link数据 (REQ/RSP/MUL VC Buffer)         |   |
|  |  - 解码Flit                                        |   |
|  |  - 重建AXI事务发到AXI Master                        |   |
|  +--------------------------------------------------+   |
|                                                          |
|  +--------------------------------------------------+   |
|  | Block 3: APB Register Interface                   |   |
|  |  - 寄存器配置和状态读取                              |   |
|  |  - 4KB APB地址空间 (12-bit)                         |   |
|  |  - 远程APB访问 (通过APB Flit)                       |   |
|  +--------------------------------------------------+   |
+----------------------------------------------------------+
```

## 2.2 完整协议栈 (SUE2.0)

```
+----------------------------------------------------------+
| Layer 5: Application / NoC                               |
|  - AXI4 Master/Slave (5 channels: AW/W/B/AR/R)          |
|  - APB3 (Register access)                                |
+----------------------------------------------------------+
| Layer 4: PAXI Core (事务层)                               |
|  [DOC] "Protocol of Accelerated eXchange Interconnect"   |
|  - AXI Signal -> REQ/RSP Flit 编码                       |
|  - 8个CBFC/PFC虚拟通道                                    |
|  - 两种VC映射模式 (PAXI默认 / 用户自定义)                  |
|  - DA仲裁与AXI User Field编码                             |
|  - 多播支持 (8组x16设备)                                   |
|  - TX/RX Buffer Management                               |
|  - 远程APB访问、延迟测量、Pattern Generator               |
+----------------------------------------------------------+
| Layer 3: RC Link (传输层)                                 |
|  [DOC] "RCLINK-Stream IP"                                |
|  - TYPE1: 可靠传输, Go-Back-N E2E重传, 最多1024 QP       |
|  - TYPE2: 不可靠传输, 多播支持                             |
|  - TYPE3: 原始以太网报文 (Memory Descriptor)              |
|  - per-QP速率控制 (MB/s精度)                              |
|  - CBFC流量控制 (8个独立VC)                               |
|  - ACK MERGE / CNP MERGE                                 |
|  - 多种报文格式 (Standard/AFH_GEN1/AFH_GEN2/AFH_Lite)    |
+----------------------------------------------------------+
| Layer 2: CESOC (数据链路层)                               |
|  [DOC] "includes CEMAC_800G, CEPCS_800G, CEFEC_800G"    |
|  - CEMAC: MAC层, 200G/400G                               |
|  - CEPCS: PCS层, 编码/解码                                |
|  - CEFEC: FEC层 (RSFEC), 错误率 < 10^-12                 |
+----------------------------------------------------------+
| Layer 1: SerDes (物理层)                                  |
|  [DOC] "8x112G PAM4 SERDES"                             |
|  - 8条Lane, 每条112 Gbps (PAM4调制)                      |
|  - 支持6nm/5nm工艺                                        |
+----------------------------------------------------------+
```

**[推导]** 与旧版协议栈的关键差异:
- RC Link层是全新增加的, 替代了旧版可选的RDMA Engine
- PAXI不再直接管理per-DA Credit和MAC帧封装, 这些职责转移到RC Link
- 错误重传从MAC L2 Retry升级为RC Link端到端(E2E) Go-Back-N重传

## 2.3 TX数据路径

**[DOC]** 基于SUE2.0 2.1.2 AXI Request Handling:

> "The transfers from AW channel, W channel and AR channel on the AXI slave interface were composed as AXI Flit according to the rule in the previous chapter, and then put it on the axi steam tx interface on the ethernet links."

TX路径处理阶段:

```
阶段1: AXI Slave接收
  <- AXI Slave接口接收来自NoC的AXI事务
  <- 5个通道: AW, W, AR (请求方向), R, B (响应方向)
       |
阶段2: Flit编码
  <- REQ: AW+W打包为AW+W Frame, AR打包为AR Frame
  <- RSP: R打包为R Frame, B打包为B Frame
  <- 10-bit Header: {2-bit Flit Type, 2-bit Encode, Length}
  <- WSTRB优化: 全1时省略WSTRB字段 (WSTRB_EN控制)
       |
阶段3: TX Buffer缓冲
  <- AW+W: 缓存一个MPS容量的写burst, 收到WLAST后发出
  <- R: OST个buffer空间, 按积累完成顺序发送
  <- AR/B: 最大深度16, 可打包多个事务为单个Flit
       |
阶段4: VC分类
  <- Mode 0: PAXI自动分配 (REQ->VC0/VC2, RSP->VC1/VC3, MUL->VC4)
  <- Mode 1: 用户通过AXI USER field指定VC
       |
阶段5: DA仲裁与编码
  <- AXI User Field: {n-bit DA, 3-bit VC, 1-bit Multicast}
  <- 多播时: {(n-3)-bit zero, 3-bit multi-grp, 3-bit VC, 1'b1}
       |
  -> 送入RC Link (TYPE1单播 / TYPE2多播)
  -> RC Link封装MAC帧 -> CESOC -> SerDes发送
```

## 2.4 RX数据路径

**[DOC]** 基于SUE2.0 2.1.3 AXI Resp Handling:

> "The transfers from R channel and B channel on the AXI master interface were composed as AXI Flit according to the rule in the previous chapter, and then put it on the axi steam tx interface on the ethernet links."

RX路径处理阶段:

```
  <- SerDes接收
  <- CESOC解码 (FEC纠错 -> PCS -> MAC)
  <- RC Link处理 (E2E重传确认, CBFC Credit管理)
       |
阶段1: VC Buffer接收
  <- REQ VC Buffer: 接收远端请求 (AW+W, AR)
  <- RSP VC Buffer: 接收远端响应 (R, B)
  <- MUL VC Buffer: 接收多播数据
       |
阶段2: 水位线监测
  <- REQ/RSP Buffer: 高水位 32帧+1RTT, 低水位 1RTT
  <- MUL Buffer: 高水位 8帧+1RTT
  <- 超过高水位触发背压 (PFC或CBFC)
       |
阶段3: Flit解码
  <- 从VC Buffer提取Flit
  <- 解析10-bit Header确定类型和编码
       |
阶段4: AXI通道Buffer
  <- 5个独立AXI通道Buffer (AW/W/B/AR/R)
  <- 每通道可缓存最多16个MPS数据单元
  <- 独立握手, 高利用率
       |
  -> AXI Master Interface输出
```

## 2.5 TX Buffer Management

**[DOC]** 来自SUE2.0 2.10 TX Buffer Management:

> "In the TX direction, there are five buffers corresponding to the five AXI channels."

### 5个TX缓冲

| 缓冲 | 容量 | 发出条件 |
|------|------|---------|
| AW+W | 一个MPS容量的写burst | 收到AW请求和当前burst的WLAST |
| R | OST个buffer空间 | 按积累完成顺序发送 |
| AR | 最大深度16 | 见下方三种条件 |
| B | 最大深度16 | 见下方三种条件 |

### AR和B的三种发出条件

**[DOC]**:

1. **达到水位值**: 累积请求达到TX_Buffer_Ctrl寄存器中配置的TX_BUF_WM值
2. **DA或VC切换**: 收到不同DA或VC的B/AR请求时, 先发出之前累积的数据
3. **超时**: TX_BUF_ACC_WT超时到期 -- 如果在超时值内未累积到配置数量的请求, 则发出已有请求

**[DOC]**:

> "The B and AR channel buffers each have a maximum depth of 16, meaning that a PAXI FLIT can contain up to 16 B or AR transactions."

**[推导]** AR和B的打包机制是SUE2.0的重要优化: 将多个小事务合并为一个Flit传输, 显著减少了帧头开销, 提高了带宽利用率。

### R通道特殊处理

**[DOC]**:

> "Considering that the R channel supports interleaving, PAXI allocates OST buffer spaces to accommodate all rdata returned from the OST read requests issued by the PAXI master. The R channel issues data in the order in which the buffers become fully accumulated."

R通道支持interleaving, PAXI为所有已发出的OST读请求分配buffer空间, 按buffer填满的顺序发送。

## 2.6 RX Buffer Management

**[DOC]** 来自SUE2.0 2.11 RX Buffer Management:

> "The RX direction of PAXI comprises a total of 8 memory buffers."

### 8个RX缓冲

| 缓冲类型 | 数量 | 说明 |
|----------|------|------|
| AXI通道Buffer | 5 (AW/W/B/AR/R) | 每通道最多16个MPS数据 |
| REQ VC Buffer | 1 | 请求通道, 32帧+2RTT |
| RSP VC Buffer | 1 | 响应通道, 32帧+2RTT |
| MUL VC Buffer | 1 | 多播通道, 8帧+2RTT |

### VC Buffer容量与水位线

**[DOC]**:

> "Taking the REQ MEM as an example, PAXI supports caching up to 32 maximum-length MAC frames plus 2 RTT worth of data volume by default, placed compactly in memory."

> "the default high watermark for this memory is set to 32 times maximum-length MAC frames plus 1 RTT worth of data volume (as A shown in the figure), leaving the remaining space (B, 1 RTT) to absorb potential incoming MAC frames."

水位线配置:

| VC Buffer | 总容量 | 默认高水位 | 预留空间 |
|-----------|--------|-----------|---------|
| REQ MEM | 32帧 + 2RTT | 32帧 + 1RTT | 1RTT (吸收背压期间数据) |
| RSP MEM | 32帧 + 2RTT | 32帧 + 1RTT | 1RTT |
| MUL MEM | 8帧 + 2RTT | 8帧 + 1RTT | 1RTT |

**[DOC]**:

> "During backpressure, the RClink may output nearly additional 1 RTT worth of data volume."

> "it is recommended to reserve at least space for 1 RTT worth of data volume to ensure data integrity."

用户可灵活调整水位线, 但建议至少预留1RTT的数据空间。

### AXI通道Buffer

**[DOC]**:

> "each channel is designed to buffer up to 16 MPS (Maximum Packet Size) data units. This ensures that the memory of each channel achieves good utilization even under AXI backpressure conditions."

每个AXI通道独立握手, 缓存最多16个MPS数据, 保证在AXI背压条件下仍有良好利用率。

## 2.7 内部缓冲结构

**[DOC]** 来自SUE2.0 3.2.7 FIFO Overflow Register (0x018):

### TX方向缓冲 (Bits[14:10])

| Bit | 缓冲名称 | 说明 |
|-----|---------|------|
| 10 | data B buffer | B响应缓冲 |
| 11 | data AR buffer | 读地址缓冲 |
| 12 | data AW buffer | 写地址缓冲 |
| 13 | data R buffer | 读数据缓冲 |
| 14 | data W buffer | 写数据缓冲 |

### RX方向缓冲 (Bits[7:0])

| Bit | 缓冲名称 | 说明 |
|-----|---------|------|
| 0 | data B buffer | B响应缓冲 |
| 1 | data AR buffer | 读地址缓冲 |
| 2 | data AW buffer | 写地址缓冲 |
| 3 | data R buffer | 读数据缓冲 |
| 4 | data W buffer | 写数据缓冲 |
| 5 | data REQ buffer | REQ VC缓冲 |
| 6 | data RESP buffer | RSP VC缓冲 |
| 7 | data MUL buffer | 多播VC缓冲 |

**[推导]** 与旧版的差异:
- TX: 5个buffer (仅Data通道), 旧版有10个 (Data+Ctrl各5个)
- RX: 8个buffer (5个AXI通道 + 3个VC buffer), 旧版有10个 (Data+Ctrl各5个)
- SUE2.0移除了独立的Ctrl通道缓冲, 统一通过VC机制管理

## 2.8 外部接口汇总

**[DOC]** 基于SUE2.0 第4章 Hardware Interface:

### AXI接口 (面向NoC)

| 接口 | 方向 | 说明 |
|------|------|------|
| Data AXI Master | Output | 接收远端数据后驱动本地AXI Master |
| Data AXI Slave | Input | 接收本地AXI事务 |

**[推导]** SUE2.0移除了旧版的独立Ctrl AXI Master/Slave接口, 控制流量通过VC机制统一管理。

### RC Link接口 (面向传输层)

**[DOC]** 来自SUE2.0 4.5 DATA LINK Interface (RC LINK):

| 接口 | 方向 | 说明 |
|------|------|------|
| TX Data Bus | Output | 发送到RC Link的数据流 |
| RX Data Bus | Input | 从RC Link接收的数据流 |

### 管理接口

| 接口 | 方向 | 说明 |
|------|------|------|
| APB Slave | Input | 本地寄存器访问 (4KB地址空间) |
| APB Master | Output | 远程APB访问 (通过APB Flit) |

### 存储接口

**[DOC]** 来自SUE2.0 4.8 Memory Interface:

| 存储块 | 用途 |
|--------|------|
| Data AR Memory | 读地址缓冲 |
| Data AW Memory | 写地址缓冲 |
| Data W Memory | 写数据缓冲 |
| Data B Memory | 写响应缓冲 |
| Data R Memory | 读数据缓冲 |
| AXI REQ Memory | REQ VC Buffer |
| AXI RSP Memory | RSP VC Buffer |
| Multi-Cast Memory | 多播VC Buffer |

## 2.9 时钟与复位

**[DOC]** PAXI支持软复位 (SUE2.0 3.2.6 Soft Reset Register, 0x014):

> "Write 1 to trigger soft reset. This is a self-clean reset. After reset done, this bit will be clear automatically. User must not write 0 to this field."

**[DOC]** RC Link采用全同步设计 (RCLINK Spec 7.1):

- **CLK_I**: 最大1GHz
- **RST_N_I**: 全局复位信号, 低有效
