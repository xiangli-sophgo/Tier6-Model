# 10. PAXI寄存器映射

## 10.1 概述

**[DOC]** SUE2.0 Software Interface:

> "Registers are implemented in a 4 KB address space, but not all addresses in this space have a defined register. PAXI Address use 12-bit address space (4K page)."

- 寄存器空间: 4 KB地址空间 (12-bit地址)
- 访问接口: AMBA 3 APB
- 注意: 事务进行中修改寄存器可能产生不可预期的效果, 通常在初始化时配置

## 10.2 寄存器字段类型约定

**[DOC]**:

| 类型 | 说明 |
|------|------|
| RW | 读写。可自由读写 |
| RO | 只读。写操作无效 |
| RW1C | 只读, 写1清除。硬件置1, 软件写1清除, 写0无效 |
| Rsvd | 保留。读返回0, 写无效 |

## 10.3 完整地址映射

**[DOC]** SUE2.0 Address Map:

### 基本控制寄存器 (0x000 ~ 0x080)

| 地址 | 寄存器名称 | 复位值 | 说明 |
|------|-----------|--------|------|
| 0x000 | Watermark Register | Design-defined | Revision ID (RO) |
| 0x004 | Interrupt Mask Register | User-defined | 中断屏蔽寄存器 |
| 0x008 | Interrupt Indicator Register | 0x0000_0000 | 中断状态寄存器 |
| 0x00C | Ctrl Register | 0x0000_0080 | PAXI控制 (CBFC/VC/WSTRB/Loopback) |
| 0x010 | Ethernet Frame Length Register | UVIP_PAXI_MAX_FRAME_LEN | 最大MAC帧长度 (RO) |
| 0x014 | Soft Reset Register | 0x0000_0000 | 软复位触发 (RW1C, 自清除) |
| 0x018 | FIFO Overflow Register | 0x0000_0000 | FIFO溢出指示 |
| 0x01C | PAXI Status Register | 0x0000_0007 | PAXI状态 (IDLE/Pattern/Error) |
| 0x020 | Latency Ctrl Register | 0x0000_0000 | 延迟测量控制 |
| 0x024 | DAXI W Latency Result | 0x0000_0000 | 写延迟结果 |
| 0x028 | DAXI R Latency Result | 0x0000_0000 | 读延迟结果 |
| 0x02C | DAXI W Latency Result ID | 0x0000_0000 | 写延迟结果ID |
| 0x030 | DAXI W Latency Result ADDR LOW | 0x0000_0000 | 写延迟结果地址低位 |
| 0x034 | DAXI W Latency Result ADDR HIGH | 0x0000_0000 | 写延迟结果地址高位 |
| 0x038 | DAXI R Latency Result ID | 0x0000_0000 | 读延迟结果ID |
| 0x03C | DAXI R Latency Result ADDR LOW | 0x0000_0000 | 读延迟结果地址低位 |
| 0x040 | DAXI R Latency Result ADDR HIGH | 0x0000_0000 | 读延迟结果地址高位 |
| 0x044 | Remote APB CTRL Register | 0x0000_0000 | 远程APB控制 |
| 0x048 | Remote APB DATA Register | 0x0000_0000 | 远程APB写数据 |
| 0x04C | Remote APB RDATA Register | 0x0000_0000 | 远程APB读数据 (RO) |
| 0x050 | Message Ctrl Register | 0x0000_0000 | 消息控制 (Linkup/Error) |
| 0x054 | Remote APB Timeout Threshold | 0x0010_0000 | 远程APB超时阈值 |
| 0x058 | RX REQ Buffer Water Mark | 参数化 | REQ通道PFC水位线 |
| 0x05C | RX RSP Buffer Water Mark | 参数化 | RSP通道PFC水位线 |
| 0x060 | RX MUL Buffer Water Mark | 参数化 | MUL通道PFC水位线 |
| 0x064 | Pattern Generator Write Ctrl | 0x0000_0000 | Pattern写控制 |
| 0x068 | Pattern Generator Read Ctrl | 0x0000_0000 | Pattern读控制 |
| 0x06C | Pattern Generator Ctrl Register | 0x0000_1005 | Pattern配置 (MPS/GAP/DA_SEL) |
| 0x070 | Channel Weight Register | 0x0008_0008 | REQ/RSP通道权重 |
| 0x074 | Channel Weight Register2 | 0x0100_0008 | APB/MUL通道权重 |
| 0x078 | Remote APB Higher Address Register | 0x0000_0000 | 远程APB高位地址 |
| 0x07C | TX Buffer Ctrl Register | 0x0000_00FF | TX Buffer AR/B打包控制 |
| 0x080 | RX Multicast Timeout Register | 0x03d0_9000 | 多播B响应超时阈值 |

### 错误/状态寄存器 (0x200 ~ 0x3FC)

| 地址范围 | 寄存器名称 | 数量 | 复位值 | 说明 |
|---------|-----------|------|--------|------|
| 0x200~0x27C | Remote Error Status Register 0~31 | 32 | 0x0000_0000 | 远端DA错误状态位图, 每寄存器覆盖32个DA |
| 0x280~0x2FC | Remote PCS Linkup Register 0~31 | 32 | 0x0000_0000 | 远端PCS Link-up状态位图 |
| 0x300~0x37C | Retry Error Register 0~31 | 32 | 0x0000_0000 | DA重传失败状态位图 |
| 0x380~0x3FC | Multi DA Enable Register 0~31 | 32 | 0xFFFF_FFFF | DA使能位图, 默认全部使能 |

**[推导]** 与旧版对比, SUE2.0的重大变化:
- **移除**: DA MAP寄存器区 (旧版0x200~0x4FC, 128 DA x 2寄存器), 因per-DA Credit机制已被RC Link CBFC替代
- **移除**: Ctrl Channel Credit寄存器区 (旧版0xA80~0xB7C)
- **移除**: Credit Update Status寄存器区 (旧版0xC80~0xE7C)
- **新增**: Multi DA Enable Register (0x380~0x3FC), 控制哪些DA参与Linkup消息发送
- **新增**: Channel Weight Register (0x070/0x074), TX通道仲裁权重
- **新增**: TX Buffer Ctrl Register (0x07C), AR/B打包控制
- **新增**: RX Multicast Timeout Register (0x080), 多播超时配置
- **新增**: 8个Latency Result寄存器 (0x024~0x040), 扩展了延迟测量结果 (ID + 地址)
- **变更**: 水位线从DAXI/CAXI改为REQ/RSP/MUL三通道
- **变更**: 错误/状态区地址从0xA00起移到0x200起, 扩展到32个寄存器覆盖最多1024个DA

## 10.4 重点寄存器详解

### Interrupt Mask Register (0x004) - 中断屏蔽

**[DOC]** SUE2.0 3.2.2:

| Bit | 字段 | 类型 | 说明 |
|-----|------|------|------|
| 31:13 | RSVD | RO | 保留 |
| 12 | MULTI_CAST Timeout MASK | RW | 多播超时中断屏蔽 |
| 11 | MULTI_CAST RETRY ERR CLR MASK | RW | 多播重传错误清除中断屏蔽 |
| 10 | RETRY ERR CLR MASK | RW | 单播重传错误清除中断屏蔽 |
| 9 | PAT DATA ERR MASK | RW | Pattern Generator读写错误屏蔽 |
| 8 | PAT RD DONE MASK | RW | Pattern读完成屏蔽 |
| 7 | PAT WR DONE MASK | RW | Pattern写完成屏蔽 |
| 6 | RSVD | RO | 保留 |
| 5 | APB_Linkup_MSG MASK | RW | PCS Linkup消息屏蔽 |
| 4 | LM timeout MASK | RW | 延迟测量超时屏蔽 |
| 3 | LM done MASK | RW | 延迟测量完成屏蔽 |
| 2 | APB_ERR_MSG MASK | RW | APB消息错误屏蔽 |
| 1 | E2E_RETRY_ERR MASK | RW | RC Link E2E Retry错误屏蔽 |
| 0 | OFLOW_IND MASK | RW | FIFO溢出屏蔽 |

**[推导]** 与旧版对比: bit6从"DA update MASK"变为保留 (Credit Update机制已移除), 新增bit10~12的多播和重传错误清除中断, bit1从"L2 Retry"改为"E2E Retry"。

### Interrupt Indicator Register (0x008) - 中断状态

**[DOC]** SUE2.0 3.2.3:

| Bit | 字段 | 类型 | 说明 |
|-----|------|------|------|
| 31:13 | RSVD | RO | 保留 |
| 12 | MULTI_CAST TIMEOUT | RW1C | 多播B响应超时 |
| 11 | MULTI_CAST RETRY ERR CLR | RW1C | 多播重传错误清除完成 |
| 10 | RETRY ERR CLR | RW1C | 单播重传错误清除完成 |
| 9 | PAT DATA ERR | RW1C | Pattern Generator读写数据不匹配 |
| 8 | PAT RD DONE | RW1C | Pattern Generator读完成 |
| 7 | PAT WR DONE | RW1C | Pattern Generator写完成 |
| 6 | RSVD | RO | 保留 |
| 5 | APB_Linkup_MSG | RW1C | 收到远端PCS link-up消息 |
| 4 | LM timeout | RW1C | 延迟测量超时 |
| 3 | LM done | RW1C | 延迟测量完成 |
| 2 | APB_ERR_MSG | RW1C | APB消息错误 |
| 1 | E2E_RETRY_ERR | RW1C | RC Link E2E Retry失败 |
| 0 | OFLOW_IND | RW1C | FIFO溢出 |

### Ctrl Register (0x00C) - 控制寄存器

**[DOC]** SUE2.0 3.2.4:

| Bit | 字段 | 类型 | 说明 |
|-----|------|------|------|
| 31:15 | Reserved | RO | 保留 |
| 14 | Error Handle Trigger | RW1C | 写1触发错误处理流程 |
| 13 | Self-retry err clr en | RW | 1: PAXI自动进入retry错误处理; 0: 软件介入控制 |
| 12 | rx ost constr en | RW | 1: RX Master在outstanding达限时停止发送; 0: 不限制 |
| 11:9 | APB VC | RW | APB事务的虚拟通道选择 |
| 8 | VC mapping mode | RW | 1: 用户自定义VC映射; 0: PAXI固定映射 |
| 7 | CBFC_EN | RW | 1: 使能CBFC模式; 0: 使能PFC模式 |
| 6 | WSTRB_EN | RW | 1: Flit中携带WSTRB; 0: 不携带 |
| 5 | PAT_EN | RW | Pattern Generator使能 |
| 4:2 | Reserved | RO | 保留 |
| 1 | Remote loopback | RW | AXI接口环回 |
| 0 | loopback | RW | MAC接口环回 (TX out -> RX in) |

**[推导]** 复位值0x0000_0080表示默认CBFC_EN=1 (启用CBFC模式)。与旧版对比:
- 旧版bit4:2为CST (Credit Stop Threshold), 已移除
- 新增: WSTRB_EN (bit6), CBFC_EN (bit7), VC mode (bit8), APB VC (bit11:9), rx ost (bit12), 错误处理控制 (bit13~14)

### Ethernet Frame Length Register (0x010)

**[DOC]** SUE2.0 3.2.5:

| Bit | 字段 | 类型 | 说明 |
|-----|------|------|------|
| 31:16 | Reserved | RO | 保留 |
| 15:0 | Ethernet Frame length | RO | 最大帧长 (字节) |

**[DOC]**: 默认值为REQ和RSP帧中较大者。REQ帧长 = header + AW + (W beat数 x W长度); RSP帧长 = header + (R OST数 x R数据长度)。

### FIFO Overflow Register (0x018)

**[DOC]** SUE2.0 3.2.7:

TX方向 (Bit[14:10]) - 仅Data通道:

| Bit | 说明 |
|-----|------|
| 14 | data W buffer overflow |
| 13 | data R buffer overflow |
| 12 | data AW buffer overflow |
| 11 | data AR buffer overflow |
| 10 | data B buffer overflow |

RX方向 (Bit[7:0]) - Data通道 + VC Buffer:

| Bit | 说明 |
|-----|------|
| 7 | data MUL buffer overflow |
| 6 | data RESP buffer overflow |
| 5 | data REQ buffer overflow |
| 4 | data W buffer overflow |
| 3 | data R buffer overflow |
| 2 | data AW buffer overflow |
| 1 | data AR buffer overflow |
| 0 | data B buffer overflow |

**[推导]** 与旧版对比的重大变化:
- TX: 从10个buffer (Data 5 + Ctrl 5) 减少到5个 (仅Data通道), 反映了SUE2.0中Ctrl通道合并到统一通道
- RX: 从10个buffer变为8个 (Data 5个AXI通道 + REQ/RESP/MUL 3个VC buffer)
- RX新增的bit5~7对应REQ/RESP/MUL三个VC缓冲区的溢出检测

### PAXI Status Register (0x01C)

**[DOC]** SUE2.0 3.2.8:

| Bit | 字段 | 类型 | 说明 |
|-----|------|------|------|
| 31:19 | Reserved | RO | 保留 |
| 18 | MULTI-CAST TIMEOUT | RW1C | 多播B响应超时 |
| 17 | MULTI-CAST RETRY ERR | RW1C | 多播重传失败 |
| 16 | RX_MULTI_CAST_ERR_STAT | RO | 1: RX多播错误处理未完成; 0: 完成 |
| 15 | TX_MULTI_CAST_ERR_STAT | RO | 1: TX多播错误处理未完成; 0: 完成 |
| 14 | RX_ERR_STAT | RO | 1: RX错误处理未完成; 0: 完成 |
| 13 | TX_ERR_STAT | RO | 1: TX错误处理未完成; 0: 完成 |
| 12:9 | PAT_WR_ERR | RW1C | Pattern写错误 (每bit对应1个DA) |
| 8:5 | PAT_RD_ERR | RW1C | Pattern读错误 (每bit对应1个DA) |
| 4 | PAT_WR_DONE | RW1C | Pattern写完成 |
| 3 | PAT_RD_DONE | RW1C | Pattern读完成 |
| 2 | RX_IDLE | RO | RX空闲状态 |
| 1 | TX_IDLE | RO | TX空闲状态 |
| 0 | IDLE | RO | PAXI整体空闲 |

**[推导]** 新增bit13~18用于错误处理状态跟踪:
- TX_ERR_STAT/RX_ERR_STAT: 单播错误处理进度
- TX/RX_MULTI_CAST_ERR_STAT: 多播错误处理进度
- MULTI-CAST RETRY ERR/TIMEOUT: 多播错误类型指示

### Latency Ctrl Register (0x020)

**[DOC]** SUE2.0 3.2.9:

| Bit | 字段 | 类型 | 说明 |
|-----|------|------|------|
| 31:20+DAW | Reserved | RO | 保留 |
| 20+DAW-1:20 | Destination address | RW | DA模式下的目标地址 |
| 19 | Latency measure | RW | 写1使能Pattern Generator总延迟测量 |
| 18 | Latency measure loopback mode | RW | 环回测量模式, 需远端先设loopback=1 |
| 17 | Mode select | RW | 1: DA模式延迟测量; 0: AXI ID模式延迟测量 |
| 16 | Latency mode enable | RW | 延迟测量使能 |
| 15:0 | AXI ID | RW | AXI事务ID |

**[推导]** 与旧版对比新增:
- Mode select (bit17): 新增DA模式, 可按目标地址而非AXI ID进行延迟测量
- Destination address (bit20+): DA模式下指定测量目标
- Latency measure (bit19): 新增Pattern Generator总轮次延迟测量

注: DAW为`UVIP_PAXI_DA_WIDTH`参数。

### Latency Result寄存器 (0x024~0x040)

**[DOC]** SUE2.0 3.2.10~3.2.17:

SUE2.0将延迟结果从4个寄存器扩展到8个:

| 地址 | 名称 | 说明 |
|------|------|------|
| 0x024 | DAXI W Latency Result | bit31: done, bit[30:0]: 写延迟 (cycle) |
| 0x028 | DAXI R Latency Result | bit31: done, bit[30:0]: 读延迟 (cycle) |
| 0x02C | DAXI W Latency Result ID | bit[DAW-1:0]: 捕获的AW ID |
| 0x030 | DAXI W Latency Result ADDR LOW | bit[31:0]: 捕获的AW地址低32位 |
| 0x034 | DAXI W Latency Result ADDR HIGH | bit[DAW-1:0]: 捕获的AW地址高位 |
| 0x038 | DAXI R Latency Result ID | bit[DAW-1:0]: 捕获的AR ID |
| 0x03C | DAXI R Latency Result ADDR LOW | bit[31:0]: 捕获的AR地址低32位 |
| 0x040 | DAXI R Latency Result ADDR HIGH | bit[DAW-1:0]: 捕获的AR地址高位 |

**[推导]** 新增的ID和ADDR寄存器支持DA模式测量 -- 当按目标地址匹配时, 可以精确记录匹配到的事务的ID和完整地址, 便于调试分析。

### Remote APB CTRL Register (0x044)

**[DOC]** SUE2.0 3.2.18:

| Bit | 字段 | 类型 | 说明 |
|-----|------|------|------|
| 31:22 | DA | RW | 远程APB目标DA |
| 21:1 | ADDR | RW | 远程APB地址 |
| 0 | WR | RW | 1=写, 0=读 |

### Message Ctrl Register (0x050)

**[DOC]** SUE2.0 3.2.21:

| Bit | 字段 | 类型 | 说明 |
|-----|------|------|------|
| 31:22 | DA | RW | 远端DA地址 (高位有效) |
| 21:2 | Reserved | RO | 保留 |
| 1 | Link-up | RW | 触发link-up消息 |
| 0 | err | RW | 触发error消息 |

**[推导]** 与旧版对比: bit2~15的Credit Update相关字段 (WR/CH/ACK/REQ/INC/CV) 全部移除, 仅保留Link-up和err触发。这反映了Credit Update协商机制在SUE2.0中被CBFC完全替代。

### Water Mark寄存器 (0x058/0x05C/0x060)

**[DOC]** SUE2.0 3.2.23~3.2.25:

三个水位线寄存器格式相同:

| 位域 | 名称 | 说明 |
|------|------|------|
| [31:16] | HI WM | 高水位线 |
| [15:0] | LO WM | 低水位线 |

| 寄存器 | 地址 | 通道 | 高水位默认 | 低水位默认 |
|--------|------|------|-----------|-----------|
| RX REQ Buffer Water Mark | 0x058 | REQ | 32帧 + 1 RTT | 1 RTT |
| RX RSP Buffer Water Mark | 0x05C | RSP | 32帧 + 1 RTT | 1 RTT |
| RX MUL Buffer Water Mark | 0x060 | MUL | 8帧 + 1 RTT | 1 RTT |

**[DOC]**: 剩余深度(1 RTT)用于在PFC反压期间吸收额外流入的数据。最大帧长度与用户当前配置相关。

**[推导]** 与旧版对比: 从DAXI/CAXI两个水位线寄存器 (0x058/0x05C) 改为REQ/RSP/MUL三个, 新增0x060用于多播通道。

### Pattern Generator Ctrl Register (0x06C)

**[DOC]** SUE2.0 3.2.28:

| Bit | 字段 | 类型 | 说明 |
|-----|------|------|------|
| 31:16 | Reserved | RO | 保留 |
| 15:12 | DA_SEL | RW | 目标DA选择, 每bit控制一个DA通道, 最多4个DA |
| 11:4 | GAP | RW | AR/AW burst之间的间隔 (空闲周期数) |
| 3:0 | MPS | RW | 数据包大小: 0=128B, 1=256B, 2=512B, 3=1KB, 4=2KB, 5=4KB |

**[推导]** 与旧版对比新增:
- MPS: 可配置数据包大小 (旧版固定4KB)
- GAP: burst间空闲周期控制, 用于模拟不同流量模式

### Channel Weight Register (0x070)

**[DOC]** SUE2.0 3.2.29:

| Bit | 字段 | 默认值 | 说明 |
|-----|------|--------|------|
| 31:16 | RSP CH WEIGHT | 0x0008 | RSP通道权重 |
| 15:0 | REQ CH WEIGHT | 0x0008 | REQ通道权重 |

### Channel Weight Register2 (0x074)

**[DOC]** SUE2.0 3.2.30:

| Bit | 字段 | 默认值 | 说明 |
|-----|------|--------|------|
| 31:16 | APB CH WEIGHT | 0x0100 | APB通道权重 |
| 15:0 | MUL CH WEIGHT | 0x0008 | 多播通道权重 |

**[推导]** APB默认权重0x0100远高于其他通道(0x0008), 确保APB管理消息优先传输。

### TX Buffer Ctrl Register (0x07C)

**[DOC]** SUE2.0 3.2.32:

| Bit | 字段 | 默认值 | 说明 |
|-----|------|--------|------|
| 31:14 | Reserved | - | 保留 |
| 13:8 | TX_BUF_WM | 0x00 | AR/B buffer水位线, 单个Flit中AR或B数量不超过此值 |
| 7:0 | TX_BUF_ACC_TW | 0xFF | AR/B buffer累积时间窗口 |

**[DOC]**:
- TX_BUF_WM最小值0表示每个Flit只打包1个AR或B; 最大值15
- TX_BUF_ACC_TW最小值为1
- 如果WM值超过时间窗口, 时间窗口约束失效

### Remote Error Status Register 0~31 (0x200~0x27C)

**[DOC]** SUE2.0 3.2.34:

| 位域 | 说明 | 类型 |
|------|------|------|
| [31:0] | 每bit对应一个DA的远端错误状态 | RW1C |

Register 0 覆盖 DA0~DA31, Register 1~31 覆盖 DA32~DA1023。

### Remote PCS Linkup Register 0~31 (0x280~0x2FC)

**[DOC]** SUE2.0 3.2.35:

| 位域 | 说明 | 类型 |
|------|------|------|
| [31:0] | 每bit对应一个DA的远端PCS link-up状态 | RW1C |

### Retry Error Register 0~31 (0x300~0x37C)

**[DOC]** SUE2.0 3.2.36:

| 位域 | 说明 | 类型 |
|------|------|------|
| [31:0] | 每bit对应一个DA的重传失败状态 | RW1C |

### Multi DA Enable Register 0~31 (0x380~0x3FC)

**[DOC]** SUE2.0 3.2.37:

| 位域 | 说明 | 类型 |
|------|------|------|
| [31:0] | DA使能位图, bit置0的DA不发送link-up消息 | RW |

默认值: 0xFFFF_FFFF (所有DA默认使能)。

## 10.5 寄存器空间布局总览

```
0x000 +--------- 基本控制寄存器区 (34个寄存器)
      |  0x000: Watermark (Revision ID)
      |  0x004: Interrupt Mask
      |  0x008: Interrupt Indicator
      |  0x00C: Ctrl Register
      |  0x010: Ethernet Frame Length
      |  0x014: Soft Reset
      |  0x018: FIFO Overflow
      |  0x01C: PAXI Status
      |  0x020: Latency Ctrl
      |  0x024~0x040: Latency Results (8个)
      |  0x044~0x04C: Remote APB (3个)
      |  0x050: Message Ctrl
      |  0x054: Remote APB Timeout
      |  0x058~0x060: RX Buffer Water Mark (REQ/RSP/MUL)
      |  0x064~0x06C: Pattern Generator (3个)
      |  0x070~0x074: Channel Weight (2个)
      |  0x078: Remote APB Higher Address
      |  0x07C: TX Buffer Ctrl
      |  0x080: RX Multicast Timeout
0x084 +--------- 保留区
0x200 +--------- 远端状态/错误寄存器区
      |  0x200~0x27C: Remote Error Status (32个, DA0~1023)
      |  0x280~0x2FC: Remote PCS Linkup (32个, DA0~1023)
      |  0x300~0x37C: Retry Error (32个, DA0~1023)
      |  0x380~0x3FC: Multi DA Enable (32个, DA0~1023)
0x400 +--------- 保留区
0xFFF +--------- 4KB地址空间结束
```
