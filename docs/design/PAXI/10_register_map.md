# 10. PAXI寄存器映射

## 10.1 概述

**[DOC]** 3. Software Interface:

> "The core implements registers for control and status of various operations. The registers can be accessed through the AMBA 3 APB interface."

- 寄存器空间: 4 KB地址空间
- 访问接口: AMBA 3 APB
- 注意: 事务进行中修改寄存器可能产生不可预期的效果, 通常在初始化时配置

## 10.2 寄存器字段类型约定

**[DOC]** 3.1.1:

| 类型 | 说明 |
|------|------|
| RW | 读写。可自由读写 |
| RO | 只读。写操作无效 |
| RW1C | 只读, 写1清除。硬件置1, 软件写1清除, 写0无效 |
| Rsvd | 保留。读返回0, 写无效 |

## 10.3 完整地址映射

**[DOC]** 3.1.2 Address Map:

### 基本控制寄存器 (0x000 ~ 0x064)

| 地址 | 寄存器名称 | 复位值 | 说明 |
|------|-----------|--------|------|
| 0x000 | REG_WATERMARK | Design-defined | Revision ID (RO) |
| 0x004 | REG_CONTROL | User-defined | 中断屏蔽寄存器 |
| 0x008 | REG_STATUS | 0x0000_0000 | 中断状态寄存器 |
| 0x00C | PAXI Ctrl | 0x0000_001C | PAXI控制 (CST/Loopback/PAT_EN) |
| 0x010 | REG_Ethernet_Flit | 0x0000_0400 | 最大MAC帧长度 |
| 0x014 | REG_Soft_Reset | 0x0000_0000 | 软复位触发 (RW1C, 自清除) |
| 0x018 | Overflow_Info | 0x0000_0000 | FIFO溢出指示 (20个buffer) |
| 0x01C | WEIGHT CFG | 0x0001_0100 | VC仲裁权重 |
| 0x020 | PAXI Status | 0x0000_0007 | PAXI状态 (IDLE/Pattern) |
| 0x024 | Latency Ctrl | 0x0000_0000 | 延迟测量控制 |
| 0x028 | DAXI WLatency Result | 0x0000_0000 | Data AXI写延迟结果 |
| 0x02C | DAXI RLatency Result | 0x0000_0000 | Data AXI读延迟结果 |
| 0x030 | CAXI WLatency Result | 0x0000_0000 | Ctrl AXI写延迟结果 |
| 0x034 | CAXI RLatency Result | 0x0000_0000 | Ctrl AXI读延迟结果 |
| 0x038 | Remote APB CTRL | 0x0000_0000 | 远程APB控制 |
| 0x03C | Remote APB DATA | 0x0000_0000 | 远程APB写数据 |
| 0x040 | Remote APB RDATA | 0x0000_0000 | 远程APB读数据 (RO) |
| 0x044 | Message Ctrl | 0x0000_0000 | 消息控制 (Credit Update/Linkup/Error) |
| 0x048 | MAC LEN_TYPE Field | 0x0000_0000 | MAC帧LEN/TYPE字段 |
| 0x04C | MAC SA Register0 | 0x0000_0000 | 源MAC地址低32位 |
| 0x050 | MAC SA Register1 | 0x0000_0000 | 源MAC地址高16位 |
| 0x054 | Remote APB Timeout | 0x0010_0000 | 远程APB超时阈值 |
| 0x058 | DAXI WaterMarker | 0x1000_0000 | Data通道PFC水位线 |
| 0x05C | CAXI WaterMarker | 0x1000_0000 | Ctrl通道PFC水位线 |
| 0x060 | Pattern Generator Write Ctrl | 0x0000_0000 | Pattern写控制 |
| 0x064 | Pattern Generator Read Ctrl | 0x0000_0000 | Pattern读控制 |

### DA MAP寄存器 (0x200 ~ 0x4FC)

| 地址 | 寄存器名称 | 说明 |
|------|-----------|------|
| 0x200 | DA00 MAP | DA0映射寄存器0 (地址低位+DA使能) |
| 0x204 | DA01 MAP | DA0映射寄存器1 (地址高位+Credit) |
| 0x208 | DA10 MAP | DA1映射寄存器0 |
| 0x20C | DA11 MAP | DA1映射寄存器1 |
| ... | ... | ... |
| 0x4F8 | DA1270 MAP | DA127映射寄存器0 |
| 0x4FC | DA1271 MAP | DA127映射寄存器1 |

每个DA占用8字节 (两个32位寄存器), 128个DA共占用 128 x 8 = 1024字节。

### 错误/状态寄存器 (0xA00 ~ 0xA6C)

| 地址范围 | 寄存器名称 | 数量 | 说明 |
|---------|-----------|------|------|
| 0xA00~0xA0C | Remote Error Status | 4 | 远端错误状态 |
| 0xA20~0xA2C | Remote PCS Linkup | 4 | 远端PCS链路状态 |
| 0xA40~0xA4C | TX MAC L2 Retry Field | 4 | TX方向L2重传状态 |
| 0xA60~0xA6C | RX MAC L2 Retry Field | 4 | RX方向L2重传状态 |

### Ctrl Channel Credit寄存器 (0xA80 ~ 0xB7C)

| 地址范围 | 寄存器名称 | 数量 | 说明 |
|---------|-----------|------|------|
| 0xA80~0xB7C | Ctrl Channel Credit | 64 | Ctrl通道Per-DA Credit (DA0~127, 每寄存器覆盖2个DA) |

### Credit Update Status寄存器 (0xC80 ~ 0xE7C)

| 地址范围 | 寄存器名称 | 数量 | 说明 |
|---------|-----------|------|------|
| 0xC80~0xE7C | Credit Update Status DA0~127 | 128 | 每个DA的Credit更新状态 |

## 10.4 重点寄存器详解

### REG_CONTROL (0x004) - 中断屏蔽

**[DOC]** 3.2.2:

| Bit | 字段 | 类型 | 说明 |
|-----|------|------|------|
| 31:10 | Interrupt MASK TBD | RO | 保留 |
| 9 | PAT_ERR_MASK | WR | Pattern Generator读/写错误屏蔽 |
| 8 | PAT_RD_DONE_MASK | WR | Pattern读完成屏蔽 |
| 7 | PAT_WR_DONE_MASK | WR | Pattern写完成屏蔽 |
| 6 | DA update MASK | RW | DA更新消息屏蔽 |
| 5 | Linkup_MASK | RW | PCS Linkup屏蔽 |
| 4 | LM_TO_MASK | RW | 延迟测量超时屏蔽 |
| 3 | LM_MASK | RW | 延迟测量完成屏蔽 |
| 2 | PM_MASK | RW | APB消息错误屏蔽 |
| 1 | ERR_MASK | RW | L2 Retry错误屏蔽 |
| 0 | OFLOW_MASK | RW | FIFO溢出屏蔽 |

### PAXI Ctrl (0x00C) - 控制寄存器

**[DOC]** 3.2.4:

| Bit | 字段 | 类型 | 说明 |
|-----|------|------|------|
| 5 | PAT_EN | RW | Pattern Generator使能 |
| 4:2 | CST | RW | Credit Stop Threshold |
| 1 | Remote loopback | RW | AXI接口环回 |
| 0 | loopback | RW | MAC接口环回 (TX out -> RX in) |

**[DOC]** CST说明:

> "Credit Stop Threshold. When current credit is reach to (max credit - threshold), the AX_DA_STOP signal will be asserted."

### REG_Ethernet_Flit (0x010) - MAC帧长度

**[DOC]** 3.2.5:

| Bit | 字段 | 类型 | 说明 |
|-----|------|------|------|
| 15:0 | Ethernet Flit length | RW | 最大MAC帧长度 (字节), 必须为128B的整数倍 |

复位值0x0400 = 1024字节。最大值不得超过 `UV_PAXI_MAC_FRAME_LEN/8`。

### Overflow_Info (0x018) - FIFO溢出指示

**[DOC]** 3.2.7:

TX方向 (Bit[19:10]):

| Bit | 说明 |
|-----|------|
| 10 | Data B buffer overflow |
| 11 | Data AR buffer overflow |
| 12 | Data AW buffer overflow |
| 13 | Data R buffer overflow |
| 14 | Data W buffer overflow |
| 15 | Ctrl B buffer overflow |
| 16 | Ctrl AR buffer overflow |
| 17 | Ctrl AW buffer overflow |
| 18 | Ctrl R buffer overflow |
| 19 | Ctrl W buffer overflow |

RX方向 (Bit[9:0]):

| Bit | 说明 |
|-----|------|
| 0 | Data B buffer overflow |
| 1 | Data AR buffer overflow |
| 2 | Data AW buffer overflow |
| 3 | Data R buffer overflow |
| 4 | Data W buffer overflow |
| 5 | Ctrl B buffer overflow |
| 6 | Ctrl AR buffer overflow |
| 7 | Ctrl AW buffer overflow |
| 8 | Ctrl R buffer overflow |
| 9 | Ctrl W buffer overflow |

**[推导]** 从这20个溢出位可以得出PAXI内部有20个独立buffer:
- TX/RX各10个
- 每个方向: Data通道5个 (AW/AR/W/R/B) + Ctrl通道5个 (AW/AR/W/R/B)
- 对应AXI5个通道 x 2个通道类型 x 2个方向

### PAXI Status (0x020) - 状态寄存器

**[DOC]** 3.2.9:

| Bit | 字段 | 类型 | 说明 |
|-----|------|------|------|
| 12:9 | PAT_WR_ERR | RW1C | Pattern写错误 (每bit对应1个DA) |
| 8:5 | PAT_RD_ERR | RW1C | Pattern读错误 (每bit对应1个DA) |
| 4 | PAT_WR_DONE | RW1C | Pattern写完成 |
| 3 | PAT_RD_DONE | RW1C | Pattern读完成 |
| 2 | RX_IDLE | RO | RX空闲状态 |
| 1 | TX_IDLE | RO | TX空闲状态 |
| 0 | IDLE | RO | PAXI整体空闲 |

复位值0x0000_0007表示初始状态下IDLE/TX_IDLE/RX_IDLE均为1。

### Latency Ctrl (0x024) - 延迟测量控制

**[DOC]** 3.2.10:

| Bit | 字段 | 类型 | 说明 |
|-----|------|------|------|
| 17 | Latency measure loopback mode | RW | 环回测量模式, 需远端先设loopback=1 |
| 16 | Latency measure enable | RW | 延迟测量使能 |
| 15:0 | AXI ID | RW | 用于测量的AXI事务ID |

### Latency Result寄存器 (0x028~0x034)

**[DOC]** 3.2.11~3.2.14:

四个延迟结果寄存器格式相同:

| Bit | 字段 | 类型 | 说明 |
|-----|------|------|------|
| 31 | Done | RW1C | 测量完成标志 |
| 30:0 | Latency | RW1C | 延迟计数 (时钟周期) |

**[DOC]**: 超时时, 低31位全为1。

### Remote APB CTRL (0x038)

**[DOC]** 3.2.15:

| Bit | 字段 | 类型 | 说明 |
|-----|------|------|------|
| 27:21 | DA | RW | 远程APB目标DA |
| 20:1 | ADDR | RW | 远程APB地址 |
| 0 | WR | RW | 1=写, 0=读 |

### Message Ctrl (0x044) - 消息控制

**[DOC]** 3.2.18:

| Bit | 字段 | 类型 | 说明 |
|-----|------|------|------|
| 31:24 | DA | RW | 远端DA地址 (高位有效) |
| 15 | WR | RW | 1=Write Credit, 0=Read Credit |
| 14 | CH | RW | 1=Data Channel, 0=Ctrl Channel |
| 13 | ACK | RW | 1=Request ACK, 0=Request NACK |
| 12 | REQ | RW | 1=Request, 0=Response |
| 11 | INC | RW | 1=增加Credit, 0=减少Credit |
| 10:2 | CV | RW | Credit Value (变更量) |
| 1 | Link-up | RW | 触发Link-up消息 |
| 0 | err | RW | 触发Error消息 |

### REG_Soft_Reset (0x014)

**[DOC]** 3.2.6:

| Bit | 字段 | 类型 | 说明 |
|-----|------|------|------|
| 0 | Soft Reset | RW1C | 写1触发软复位, 自清除。禁止写0 |

## 10.5 寄存器空间布局总览

```
0x000 ┬─────── 基本控制寄存器区 (26个寄存器)
      │  0x000: Watermark (Revision ID)
      │  0x004: Int Control (中断屏蔽)
      │  0x008: Int Status (中断状态)
      │  0x00C: PAXI Ctrl
      │  0x010: Ethernet Flit Length
      │  0x014: Soft Reset
      │  0x018: Overflow Info
      │  0x01C: Weight Config
      │  0x020: PAXI Status
      │  0x024: Latency Ctrl
      │  0x028~0x034: Latency Results (4个)
      │  0x038~0x040: Remote APB (3个)
      │  0x044: Message Ctrl
      │  0x048~0x050: MAC SA (3个)
      │  0x054: Remote APB Timeout
      │  0x058~0x05C: Watermark (2个)
      │  0x060~0x064: Pattern Generator (2个)
0x068 ├─────── 保留区
0x200 ├─────── DA MAP寄存器区 (256个寄存器, 128 DA x 2)
      │  每DA: DA*0 (地址+使能), DA*1 (高位+Credit)
0x500 ├─────── 保留区
0xA00 ├─────── 错误/状态寄存器区
      │  0xA00~0xA0C: Remote Error Status (4个)
      │  0xA20~0xA2C: Remote PCS Linkup (4个)
      │  0xA40~0xA4C: TX L2 Retry (4个)
      │  0xA60~0xA6C: RX L2 Retry (4个)
0xA80 ├─────── Ctrl Credit寄存器区 (64个)
0xB80 ├─────── 保留区
0xC80 ├─────── Credit Update Status区 (128个)
0xE80 ├─────── 保留区
0xFFF └─────── 4KB地址空间结束
```
