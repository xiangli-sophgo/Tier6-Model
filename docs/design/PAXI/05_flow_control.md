# 05. Credit流控与PFC背压

## 5.1 Credit-Based流控概述

**[DOC]** PAXI Features:

> "Inside credit function for flow control support"
> "Chip-to-Chip flow controller based on AXI OST number"
> "Configurable AXI OST number 128 or 256"

PAXI使用基于Credit的端到端流控, 核心参数是AXI Outstanding Transaction (OST) 数量。

## 5.2 Per-DA Credit机制

**[DOC]** 2.9 Multi-DA Credit (完整原文):

> "The AXI buffer can accepts max_outstanding_numbers of transactions, which transactions can associated with different DA. But numbers is limited, the total number the all the DA is equal to max_outstanding_numbers. User should set the max limit number of each DA in Credit Threshold Register. For a particular DA, TX will not send transactions exceeds the DA limit until that transactions response returns."

### 关键约束

1. **总OST**: 128或256 (设计时配置)
2. **Per-DA Credit**: 每个DA有独立的Credit Threshold
3. **总量约束**: 所有DA的Credit之和 <= 总OST
4. **阻塞行为**: 当某DA的outstanding事务达到其Credit上限时, 该DA的新事务被阻塞, 直到收到Response释放Credit

### Credit Threshold寄存器

**[DOC]** 3.2.364 Data Channel Credit Threshold Register DA0~127

- 地址范围: DA0~127各有独立寄存器
- 功能: 设置每个DA的最大outstanding事务数

**[DOC]** 3.2.365 Ctrl Channel Credit Threshold Register DA0~127

- Ctrl通道也有独立的Per-DA Credit

### Credit Stop Threshold

**[DOC]** 3.2.4 PAXI CTRL Register, Bits[4:2] CST:

> "Credit Stop Threshold. When current credit is reach to (max credit - threshold), the AX_DA_STOP signal will be asserted."

这提供了一个提前停止机制:
- 不是等到Credit完全耗尽才停止
- 而是在距离上限还有threshold个时就提前断言STOP信号
- **[推导]** 这是为了补偿流水线延迟, 避免因为late-stopping而超发

## 5.3 Buffer Ratio机制

**[DOC]** 2.9 Multi-DA Credit:

> "Buffer ratio in the cfg table is 1 by default. If system side try to control the credit more smoothly, it suggests the buffer ratio can be set to more than 1, that means the physical buffer size is multiplied by the buffer ratio, and the system side has more extra buffers to control. In general, this will help DA credit change more smoothly."

### 含义

- Buffer Ratio = 1 (默认): 物理缓冲 = 逻辑Credit
- Buffer Ratio > 1: 物理缓冲 = 逻辑Credit x Ratio, 提供额外空间

### 优势

**[推导]** Buffer Ratio > 1的好处:

1. **缓冲余量**: 当Credit动态调整时, 物理缓冲有余量, 不会因为瞬时过载而丢数据
2. **更平滑的流控**: 减少因Credit耗尽导致的突然阻塞
3. **支持运行时Credit变更**: 动态调整Credit时, 额外缓冲吸收过渡期的数据

## 5.4 运行时Credit更新协商

**[DOC]** 2.9 Multi-DA Credit, Credit Update Flow:

> 1. "Set Message Ctrl Register to send DA credit update request message to remote PAXI."
> 2. "When remote side received the message, PAXI assert interrupt and system read Credit Update Status Register."
> 3. "User set Message Ctrl Register to send DA credit update response message back to the initiator PAXI."
> 4. "When initiator received the message, PAXI assert interrupt and system read Credit Update Status Register."
> 5. "If the request is acknowledged by the remote side, user can update DA credit by setting DA MAP Register and Ctrl Channel Credit Register."

### 协商流程

```
发起端(Chip A)                        远端(Chip B)
     |                                      |
     |-- Credit Update Request (Mgmt VC) -->|
     |                                      |-- PAXI中断 -->
     |                                      |-- 系统读Status -->
     |                                      |-- 决定ACK/NACK -->
     |<-- Credit Update Response (Mgmt VC)--|
     |-- PAXI中断 -->                       |
     |-- 系统读Status -->                   |
     |-- 如果ACK: 更新DA MAP和Credit Reg -->|
     |                                      |
```

### Message Ctrl Register字段

**[DOC]** 3.2.18 Message Ctrl Register (地址0x044):

| Bits | 字段 | 说明 |
|------|------|------|
| [31:24] | DA | 远端DA地址 (高位有效) |
| [23:16] | Reserved | - |
| [15] | WR | 1=Write Credit, 0=Read Credit |
| [14] | CH | 1=Data Channel, 0=Ctrl Channel |
| [13] | ACK | 1=Request ACK, 0=Request NACK |
| [12] | REQ | 1=Request, 0=Response |
| [11] | INC | 1=增加Credit, 0=减少Credit |
| [10:2] | CV | Credit Value (变更量) |
| [1] | Link-up | 触发Linkup消息 |
| [0] | err | 触发Error消息 |

### Credit Update Status

**[DOC]** 3.2.363 Credit Update Status Register DA0~127

- 地址范围: 0xc80 ~ 0xe7c
- 用途: 记录每个DA的Credit更新状态

## 5.5 PFC (Priority Flow Control) 背压

**[DOC]** 2.13 PFC Frame Creation flow:

> "PAXI supports the generation of PFC (Priority Flow Control) frames by triggering the MAC based on water level control to limit data transmission from other devices."

### 水位线配置

**[DOC]** PFC creation flow (完整原文):

> 1. "Configure DAXI Water Mark Register and CAXI Water Mark Register to set high water mark and low water mark"
> 2. "When the amount of received data exceeds the higher watermark set by PAXI, trigger MAC to send PFC frames to limit data transmission from other DA"
> 3. "Release the limits when data volume drops below the lower water mark"

### 水位线寄存器

**[DOC]** 3.2.23-3.2.26:

| 寄存器 | 地址 | 说明 |
|--------|------|------|
| DAXI Write Channel Water Mark | - | Data通道写方向水位线 |
| DAXI Read Channel Water Mark | - | Data通道读方向水位线 |
| CAXI Write Channel Water Mark | - | Ctrl通道写方向水位线 |
| CAXI Read Channel Water Mark | - | Ctrl通道读方向水位线 |

**[DOC]** 地址:
- DAXI WaterMarker Register: 0x058
- CAXI WaterMarker Register: 0x05C

### PFC行为模型

```
接收端RX Buffer使用量:

  High Watermark ──────────── 触发PFC帧发送, 通知远端暂停
                  |          |
                  |  正常区  |
                  |          |
  Low Watermark ───────────── 解除PFC, 通知远端恢复发送
                  |          |
                  |  空闲区  |
                  |          |
  0 ──────────────────────────
```

**[推导]** High/Low水位线之间的gap提供了迟滞效应(hysteresis), 避免PFC频繁开关。

## 5.6 Credit流控与PFC的关系

**[推导]** 两种机制的协同:

| 机制 | 层级 | 粒度 | 速度 |
|------|------|------|------|
| Credit-based | 端到端 | Per-DA, 事务级 | 响应驱动(RTT延迟) |
| PFC | 链路级 | Per-priority | 快速(几十ns级) |

- **Credit流控**: 精细的端到端流控, 防止远端缓冲溢出, 粒度是单个事务
- **PFC**: 粗粒度的链路级背压, 作为安全网防止突发流量导致的buffer溢出
- **组合效果**: Credit提供精确控制, PFC提供最终安全保障
