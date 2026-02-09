# 13. 多播功能 (Multicast)

## 13.1 概述

PAXI SUE2.0 新增多播(Multicast)支持, 允许一个AXI写操作同时发送到多个目标设备。该功能通过AXI User Field中的Multicast指示位和多播组号来标识多播事务, 由PAXI封装后经RC Link的TYPE2报文类型发送。

**[DOC]** SUE2.0 Features列表中明确的多播特性:

- "Support Multicast Frame."
- "Support maximum 8 multicast group."
- "Each multi-cast group support maximum 16 devices."

**[推导]** 多播功能的典型应用场景包括: 多芯片间的权重广播(Weight Broadcasting)、集合通信中的Broadcast/AllReduce操作的底层支持、以及需要一写多读的共享数据分发。相比逐一单播写入N个目标, 多播可以将发送侧的带宽占用和延迟从O(N)降低到O(1)。

## 13.2 规格总览

| 参数 | 值 | 来源 |
|------|------|------|
| 最大多播组数 | 8 | **[DOC]** |
| 每组最大设备数 | 16 | **[DOC]** |
| 支持操作类型 | 仅AXI写(Write) | **[DOC]** |
| VC分配 | VC4 (Mode 0默认) | **[DOC]** |
| 综合使能宏 | `UVIP_PAXI_MULTI_CAST_EN` | **[DOC]** |
| 超时默认值 | 0x03d0_9000 cycles | **[DOC]** |

**[DOC]** 综合选项:

> `UVIP_PAXI_MULTI_CAST_EN` - "Enable Multi-cast function If the macro is define."

**[推导]** 多播功能为可选综合特性, 未定义该宏时多播相关逻辑不会被综合, 可节省面积。

## 13.3 AXI User Field编码

**[DOC]** (来自SUE2.0 2.1.4节):

> "The PAXI utilizes the higher bit of User field of the AXI interface to transmit DA (Destination Address), VC (Virtual Channel), and broadcast/multicast frame information. The specific bit arrangement is as follows (MSB -> LSB): {n-bit DA, 3-bit VC, 1-bit Multicast}."

> "If the current AXI transmission is a multicast transaction, the lower 3 bits of the DA field serve as multicast group information. E.g. {(n-3)-bit zero, 3-bit multi-grp, 3-bit VC, 1-bit Multicast}"

### 编码格式

```
单播模式 (Multicast=0):
  MSB                                    LSB
  +------------------+--------+-----------+
  |    n-bit DA      | 3-bit  | 1-bit     |
  |  (目标地址)       |  VC    | Multicast |
  |                  |        |   = 0     |
  +------------------+--------+-----------+

多播模式 (Multicast=1):
  MSB                                    LSB
  +------------+-----------+--------+-----------+
  | (n-3)-bit  | 3-bit     | 3-bit  | 1-bit     |
  |   zero     | multi-grp |  VC    | Multicast |
  |   padding  | (组号0~7) |        |   = 1     |
  +------------+-----------+--------+-----------+
```

**[推导]** 最低位Multicast=1标识当前事务为多播。此时DA字段的高(n-3)位填零, 低3位复用为多播组号(0~7), 对应最多8个多播组。3-bit VC字段在多播模式下通常由PAXI内部映射到VC4。

### 与RC Link接口的映射

**[DOC]** (来自PAXI内部接口信号 DL_TX_USER_O[14:0]):

> "Bit[9:0]: DA number. If frame is multi-cast, bit[2:0] indicate multi-cast group."
> "Bit[13:10]: VC number."
> "Bit[14]: multi-cast packet indication."

对应关系:

| DL_TX_USER_O 位域 | 含义 |
|---|---|
| [14] | 多播报文指示 (1=多播) |
| [13:10] | VC号 |
| [9:0] | DA号; 多播时[2:0]为多播组号 |

## 13.4 操作流程

### 发送流程

**[DOC]** (来自SUE2.0 2.9节):

> "PAXI supports only AXI write operations for multi-cast. Users specify multicast transactions and multicast groups through the AXI USER field. PAXI packages the data and multicast information and transmits it to the RC Link."

完整发送路径:

```
AXI Master                    PAXI TX                    RC Link                  远端设备
   |                            |                          |                        |
   |-- AW+W (User: MC=1) ----->|                          |                        |
   |   (multi-grp=G, VC=4)     |                          |                        |
   |                            |-- TYPE2报文 ------------>|                        |
   |                            |   (group_index=G)        |-- 多播分发 ----------->| 设备A
   |                            |                          |                   +--->| 设备B
   |                            |                          |                   +--->| 设备C
   |                            |                          |                        |
   |                            |<-- B resp (合并) --------|<-- B resp (各设备) ----|
   |<-- B resp (单个) ---------|                          |                        |
```

操作步骤:

1. **用户配置**: AXI Master在AW通道的USER field中设置Multicast位=1, 并在DA低3位填入目标多播组号
2. **PAXI封装**: PAXI将写数据和多播信息封装为RC Link TYPE2数据类型报文
3. **RC Link发送**: RC Link通过TYPE2报文发送多播帧, 携带`TX_S_TYPE2_GROUP_INDEX`指示组号
4. **远端分发**: MAC层根据多播组配置将数据分发到组内所有目标设备

### B响应合并

**[DOC]**:

> "Upon receiving multiple B responses from other multicast group devices, they are merged to one B resp and returned to AXI. If PAXI receives an error response from any devices, it will result in returning a B response error."

**[推导]** B响应合并采用"任一错误即整体错误"的策略, 这意味着:
- PAXI内部需要等待多播组内所有目标设备的B响应
- 所有设备成功 -> 返回OKAY
- 任意一个设备返回错误 -> 整体返回SLVERR/DECERR
- 超时未收到全部响应 -> 触发MULTI-CAST TIMEOUT错误处理

### RX侧多播接收

**[DOC]** (来自PAXI RX Data Bus接口信号):

PAXI接收侧通过以下信号处理多播报文:

| 信号 | 方向 | 说明 |
|------|------|------|
| `DL_RX_USER_I[14]` | Input | Multicast Indication |
| `MUL_GRP_HAS_LOCAL_I` | Input | "From MAC. Indicate local Device is in multicast group." |
| `MUL_GRP_BITMAP_I[15:0]` | Input | "From MAC. Valid DA in the current multicast group." |

**[推导]** `MUL_GRP_BITMAP_I[15:0]`为16位位图, 与每组最大16个设备的规格一致。MAC层负责检查本地设备是否属于收到的多播组, 通过`MUL_GRP_HAS_LOCAL_I`通知PAXI是否需要接收该多播数据。

## 13.5 CEMAC侧配置

**[DOC]** (来自SUE2.0 2.7节 Bring-up Flow 和 2.9节):

> "To use multi-cast, user should configure multi-cast group related register in CEMAC Core and follow rules described in CEMAC reference guide."

**[推导]** 多播组的成员关系(哪些设备属于哪个组)在CEMAC Core中配置, 而非在PAXI中。PAXI只负责:
- TX侧: 将AXI多播事务封装并通过RC Link发送TYPE2报文
- RX侧: 根据MAC的指示(`MUL_GRP_HAS_LOCAL_I`)决定是否接收多播数据

CEMAC配置内容(需参考CEMAC Reference Guide):
- 多播组成员列表 (每组最多16个设备DA)
- 多播组号(0~7)与成员DA的映射关系

## 13.6 VC分配

**[DOC]** (来自2.2节 VC映射表, Mode 0):

| VC Number | Application |
|-----------|-------------|
| 0, 2 | Single-cast request |
| 1, 3 | Single-cast response |
| **4** | **Multi-cast request** |
| 0~3 | APB/Msg request/response (可配置) |

**[推导]** 多播请求固定使用VC4, 与单播的REQ(VC0/VC2)和RSP(VC1/VC3)完全分离。这种设计有以下优势:

1. **避免死锁**: 多播和单播使用不同VC, 消除了多播报文阻塞单播请求/响应的风险
2. **独立流控**: VC4有独立的Credit管理, 多播流控不影响单播通道
3. **独立缓冲**: RX方向有专用MUL MEM缓冲区, 与REQ MEM/RSP MEM分离

## 13.7 多播流量控制

多播报文的流量控制涉及PAXI和RC Link两层, 采用双重约束机制。

### PAXI RX侧: MUL MEM缓冲与水位管理

**[DOC]** (来自SUE2.0 2.11节):

RX方向有8个memory buffer, 其中3个VC buffer分别用于request、response和**multicast**缓冲:

> "The remaining 3 VC buffers are dedicated to request, response, and multicast buffering, respectively. They are designed to support internal backpressure mechanisms while absorbing additional output data from the RClink pathway when backpressure occurs."

MUL MEM缓冲区规格:

> "The MUL MEM supports absorbing up to 8 maximum-length MAC frames plus 2 RTT data volume simultaneously, with its default high watermark set to 8 maximum-length MAC frames plus 1 RTT."

**[推导]** 与REQ/RSP MEM(32帧+2RTT)相比, MUL MEM较小(8帧+2RTT), 这与多播通常用于较小数据量的广播操作一致。

**[DOC]** RX MUL Buffer Water Mark Register (0x060):

| 位域 | 名称 | 说明 | 类型 |
|------|------|------|------|
| [31:16] | MUL CH HI WM | 多播通道高水位线 | RW |
| [15:0] | MUL CH LO WM | 多播通道低水位线 | RW |

默认值: 高水位 = 8倍最大帧长 + 1 RTT, 低水位 = 1 RTT。剩余空间(1 RTT)用于在PFC反压期间吸收额外流入的数据。

**[DOC]** 反压信号:

> `DL_RX_MUL_BUF_AFULL_O` - "Indicate RX AXI multicast buffer almost full. RC link should stop transmit multi-cast vc data."

### RC Link侧: 基于Credit的多播流控

**[DOC]** (来自RCLINK 5.6节):

> 在RCLINK中, 对于多播报文有额外的基于Credit的流量管理, 与CBFC流量控制不同, 该Credit仅反映下游MAC TX方向的多播Buffer的空间, 但同样需要配置CBFC流控中的相关参数, CREDIT SIZE和CREDIT LIMIT与Buffer尺寸相关。

关键特性:

| 特性 | 说明 |
|------|------|
| Credit粒度 | 128 Byte (每个Credit cell) |
| 约束关系 | TYPE2多播报文同时受CBFC和该流控的**双重约束** |
| 实现方式 | 复用CBFC控制器模块 |
| 限制 | 不支持软件流控和动态调节下限功能 |
| 使能控制 | 受独立的`multi_credit_ctrl_en`寄存器控制 |
| 复位 | 不受MAC强制复位的影响 |

**[DOC]** RC Link与MAC的Credit接口信号:

| 信号 | 方向 | 说明 |
|------|------|------|
| `MAC2TX_TYPE2_CRD_VLD_I` | Input | MAC释放多播Credit有效指示 |
| `MAC2TX_TYPE2_CRD_NUM_I[CRD_NUM-1:0]` | Input | MAC释放的多播Credit数量, Credit cell size=128Byte |
| `MAC2TX_TYPE2_CREDIT_RST_I` | Input | 恢复多播Credit到初始值(TYPE2_CRD_RST_NUM) |

**[DOC]** PAXI DL_TX_CRD_I信号:

> "DL_TX_CRD_I[1:0] - Credit return indication. RC Link return 1 credit when DL_TX_CRD_I[0] is high. RC Link return 1 multi-cast credit when DL_TX_CRD_I[1] is high."

**[推导]** 多播流控的完整机制:

```
PAXI TX --> RC Link TX --> MAC TX --> 物理链路 --> 远端MAC RX --> 远端PAXI RX
                |                                                    |
                |<--------- 多播Credit返还 (MAC2TX_TYPE2_CRD) -------|
                |<--------- CBFC流控 (VC4) --------------------------|
```

TYPE2多播报文发送前需同时满足:
1. CBFC流控: VC4的Credit可用
2. 多播专用流控: 多播Credit可用 (反映下游多播Buffer空间)

两者任一不满足则暂停发送。

## 13.8 错误处理

**[DOC]** (来自SUE2.0 2.6节):

### 触发条件

> "When PAXI receives a multicast error indication reported by the MAC or when no response is received within the timeout period after PAXI has sent a multicast request, multicast error handling is triggered."

两种多播错误类型:

| 错误类型 | 说明 | 状态位 |
|----------|------|--------|
| **MULTI-CAST RETRY ERR** | MAC上报的多播重传失败 | PAXI STATUS [17] |
| **MULTI-CAST TIMEOUT** | 发送多播请求后超时未收到B响应 | PAXI STATUS [18] |

**[DOC]** 多播重传错误信号:

> `MUL_RETRY_ERR_VLD_I` - "Multi-cast retry fail indication from MAC."

### 恢复流程

**[DOC]** 完整的多播错误恢复步骤:

```
1. 错误检测
   |-- PAXI收到MAC多播错误指示 或 多播B响应超时
   |
2. 中断上报
   |-- "When a Multi-cast retry failure occurs, PAXI reports a retry fail interrupt."
   |-- 用户读取 Int Indicator Register 和 PAXI STATUS Register
   |   的 MULTI-CAST RETRY ERR / MULTI-CAST TIMEOUT 位
   |
3. TX侧处理
   |-- "After the user-side TX logic receives the multi-cast error indication,
   |    it must complete the burst transmission for any AWDATA/WDATA pair."
   |-- RX方向: PAXI完成当前在传输中的AWDATA/WDATA
   |
4. 数据清除
   |-- "Defaultly, The TX and RX internally clear multicast data automatically."
   |-- "However, user can set Ctrl Register to manually control paxi's entry
   |    into error handling flow."
   |
5. 完成确认
   |-- "Once PAXI's error handling is complete, it reports an interrupt."
   |-- 用户读取 PAXI STATUS Register 确认多播错误状态位已清除
   |
6. 恢复传输
   |-- 用户清除 Int Indicator Register 和 PAXI STATUS Register 中的
   |   MULTI-CAST RETRY ERR 位
   |-- AXI数据传输恢复
```

**[推导]** 关键注意点:
- TX侧不能立即中断, 必须完成当前burst的AW+W数据传输, 否则会破坏AXI协议
- 默认自动清除模式简化了错误处理, 手动模式提供了更精细的控制(可能用于调试或自定义恢复策略)
- 需要等待TX和RX两侧的错误状态位都清除后才能恢复

### 相关状态位详解

**[DOC]** PAXI Status Register (0x01C) 多播相关位:

| 位 | 名称 | 说明 | 类型 |
|----|------|------|------|
| 18 | MULTI-CAST TIMEOUT | "Assert when multi-cast b response exceeds timeout threshold." | RW1C |
| 17 | MULTI-CAST RETRY ERR | "Multicast Retry Failure Indication. When bit is set to 1." | RW1C |
| 16 | RX_MULTI_CAST_ERR_STAT | "1: RX multi-cast error handling not completed. 0: RX multi-cast error handle completed. No error DA in RX." | RO |
| 15 | TX_MULTI_CAST_ERR_STAT | "1: TX multi-cast error handling not completed. 0: TX multi-cast error handle completed. No error DA in TX." | RO |

**[DOC]** Int Indicator Register (0x008) 多播相关位:

| 位 | 名称 | 说明 | 类型 |
|----|------|------|------|
| 12 | MULTI_CAST TIMEOUT | "Multi-cast timeout interrupt." | RW |
| 11 | MULTI_CAST RETRY ERR CLR | "Multi-cast Retry error clear interrupt clear." | RW |

**[DOC]** Int Mask Register 多播相关位:

| 位 | 名称 | 说明 | 类型 |
|----|------|------|------|
| 12 | MULTI_CAST Timeout MASK | "Multi-cast timeout interrupt mask." | RW |
| 10 | MULTI_CAST RETRY ERR CLR MASK | "Multi-cast Retry error clear interrupt clear mask." | RW |

## 13.9 多播超时配置

**[DOC]** RX Multicast Timeout Register (0x080):

| 位域 | 名称 | 说明 | 类型 |
|------|------|------|------|
| [31:0] | RX_MUL_TIMEOUT | "Multicast timeout setting, when arrival this set, timeout. This value represents the number of cycle." | RW |

- **地址**: 0x080
- **宽度**: 32 bit
- **默认值**: 0x03d0_9000

**[推导]** 默认值0x03d0_9000 = 64,229,376 cycles。假设500MHz时钟频率, 约128.5ms超时; 假设1GHz, 约64.2ms超时。该超时需要覆盖多播组内所有设备的B响应返回时间, 包括最远路径的往返延迟和多播分发延迟。

## 13.10 相关寄存器汇总

| 寄存器 | 地址 | 默认值 | 说明 |
|--------|------|--------|------|
| Int Indicator Register | 0x008 | - | bit12: MULTI_CAST TIMEOUT, bit11: MULTI_CAST RETRY ERR CLR |
| Int Mask Register | 0x00C区域 | - | bit12/bit10: 多播中断屏蔽 |
| Ctrl Register | 0x00C | - | bit7: CBFC_EN (影响多播VC流控); 手动错误处理控制 |
| PAXI Status Register | 0x01C | - | bit18: MULTI-CAST TIMEOUT, bit17: MULTI-CAST RETRY ERR, bit16: RX_MULTI_CAST_ERR_STAT, bit15: TX_MULTI_CAST_ERR_STAT |
| RX MUL Buffer Water Mark Register | 0x060 | 参数化 | 多播RX缓冲高/低水位线 |
| Channel Weight Register | 0x070 | - | bit[15:0]: MUL CH WEIGHT, 多播通道调度权重 |
| RX Multicast Timeout Register | 0x080 | 0x03d0_9000 | 多播B响应超时阈值(cycle数) |
| Multi DA Enable Register 0~31 | 0x380~0x3FC | - | DA使能位图, 多播目标DA需要在此使能 |

## 13.11 Multi-Cast Memory接口

**[DOC]** (来自PAXI 4.8.9节 Multi-Cast Memory Address):

PAXI提供外部MUL MEM存储接口, 用于多播数据缓存:

| 信号 | 方向 | 时序 | 说明 |
|------|------|------|------|
| `MUL_MEM_WVLD_O` | Output | Late | 多播AXI请求存储写有效 |
| `MUL_MEM_WADDR_O[N-1:0]` | Output | Late | 多播AXI请求存储写地址 |
| `MUL_MEM_WDATA_O[N-1:0]` | Output | Late | 多播AXI请求存储写数据 |
| `MUL_MEM_RVLD_O` | Output | Late | 多播AXI请求存储读有效 |
| `MUL_MEM_RADDR_O[N-1:0]` | Output | Late | 多播AXI请求存储读地址 |
| `MUL_MEM_RDATA_I[N-1:0]` | Input | Middle | 多播AXI请求存储读数据 |

**[推导]** MUL MEM采用外挂SRAM的方式, 与REQ MEM/RSP MEM接口风格一致。Late timing的写端口表明可以在时钟周期后半段写入, Middle timing的读数据表明有半周期的读延迟。

## 13.12 RC Link TYPE2报文

**[DOC]** (来自RC Link接口信号):

RC Link通过TYPE2数据类型承载多播报文, 相关的TX侧MAC接口信号:

| 信号 | 说明 |
|------|------|
| `TX2MAC_MULTICAST_EN_O` | Multicast Packet flag |
| `TX2MAC_TYPE2_UDP_DPORT_O[15:0]` | 多播报文UDP目的端口 |
| `TX2MAC_TYPE2_GROUP_INDEX_O[2:0]` | 多播组号 (0~7) |
| `TX2MAC_TYPE2_ETH_TYPE_O[15:0]` | 多播报文以太网类型 |

**[推导]** TYPE2报文在以太网帧层面使用独立的UDP目的端口和EtherType, 这使得交换机/MAC可以在链路层识别多播报文并进行相应的转发处理。组号通过3位`GROUP_INDEX`携带, 与PAXI侧AXI User Field中的multi-grp字段一致。

## 13.13 设计注意事项

**[行业]** 基于文档信息和行业通用知识的实施建议:

1. **CEMAC配置先行**: 使用多播前必须在CEMAC Core中完成多播组寄存器配置, 否则多播报文将无法正确分发

2. **仅支持写操作**: 多播仅支持AXI Write, 不支持Read。读操作的多播语义(一读多回)在AXI协议中无法自然表达

3. **超时阈值调整**: 默认超时值(0x03d0_9000)适用于典型配置, 若多播组跨越较大物理距离或组内设备数较多, 可能需要增大该值

4. **错误恢复的完整性**: 错误处理时必须等待TX和RX两侧状态位(bit15/bit16)都清除后才能恢复传输, 仅检查一侧可能导致数据不一致

5. **流控双重约束**: TYPE2报文同时受CBFC(VC4)和多播专用Credit两层流控约束, 需确保两层配置协调一致, 避免一层过紧导致性能瓶颈

6. **DA使能**: 多播目标设备的DA需要在Multi DA Enable Register (0x380~0x3FC)中使能, 否则PAXI不会向该DA发送任何报文(包括多播)
