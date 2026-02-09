# 06. 错误处理与重传机制

## 6.1 三层错误保护体系

SUE2.0技术栈提供三层错误保护:

| 层级 | 机制 | 位置 | 说明 |
|------|------|------|------|
| L1 | FEC (RSFEC) | CESOC CEFEC | 前向纠错, 自动纠正小错误, 错误率 < 10^-12 |
| L2 | E2E Retry | RC Link | 端到端Go-Back-N重传, 替代旧版MAC L2 Retry |
| L3 | PAXI Error Handling | PAXI Core | 端到端错误处理, DA级隔离与恢复 |

**[推导]** 与旧版的变化:
- L2层从MAC Retry升级为RC Link E2E Retry, 提供端到端可靠性
- Go-Back-N重传由RC Link的TYPE1数据类型实现, 支持最多1024个QP

## 6.2 L1: FEC (RSFEC)

**[DOC]** CESOC Features:

> "Ethernet error rate lower than 10e-12"

- 使用Reed-Solomon FEC (RSFEC)算法
- 自动纠正链路中的少量误码
- 纠错后错误率低于10^-12
- 无法纠正的错误传递给L2层处理

## 6.3 L2: RC Link E2E Retry

**[DOC]** PAXI SUE2.0 Features:

> "Support error free transmission with RC LINK E2E retry enabled"

### Go-Back-N重传

**[DOC]** 来自RCLINK Spec 4.1:

RC Link的TYPE1数据类型支持端到端保护和Go-Back-N重传:
- 最多1024个队列 (QP)
- 512 outstanding
- 最大报文4KB
- 检测到PSN(数据包序列号)错误时触发重传

### TYPE1 ACK机制

**[DOC]** 来自RCLINK Spec RH结构:

| RH字段 | Bits | 说明 |
|--------|------|------|
| Dest_QP | [12:0] | QP序列号 |
| PSN | [24:13] | 数据包序列号 |
| Pkt_len | [31:25] | RH+payload长度 |
| P_key | [39:32] | Partition Key, 隔离机制 |
| Timestamp | [55:40] | 时间戳, RTT测量 |
| Opcode | [63:62] | 00:SEND Req, 01:ACK/NAK, 10:CNP |

### ACK MERGE

**[DOC]** 来自RCLINK Spec 5.8:

ACK MERGE模块合并多个ACK, 减少ACK报文数量, 提高效率。使用深度为2^QP_AW的寄存器Buffer, 以4组为单位轮询检查并发出合并的ACK。

## 6.4 L3: PAXI Error Handling

**[DOC]** 来自SUE2.0 2.6 Error Handle:

> "When a DA's retry fails, rc-link reports an error indication. Upon receiving it, paxi initiates its internal error-handling procedure."

### 单播错误处理

**[DOC]** 恢复步骤:

1. **检测**: DA重传失败时, PAXI上报retry fail中断

2. **诊断**: 用户读取 Int Indicator Register (0x008) 和 Retry Error Register (0x300~0x37C) 确定错误DA和详情

3. **隔离**:
   **[DOC]**: "After the user-side TX logic receives the error indication, it must complete the burst transmission for any AWDATA/WDATA pair and RDATA burst that has already been sent to PAXI TX, and subsequently stop sending data to the faulty DA."

   **[DOC]**: "On the RX direction, PAXI will complete any AWDATA/WDATA and RDATA currently in-transit, after which no new requests or responses related to error DA will be sent."

4. **错误处理**:
   **[DOC]**: "Defaultly, PAXI initiates its internal error-handling and discards the packets of the error DA automatically. However, user can set Ctrl Register to manually control paxi's entry into error handling flow."

5. **确认完成**: 用户读取PAXI STATUS Register (0x01C) 的TX_ERR_STAT (bit13) 和 RX_ERR_STAT (bit14) 确认错误位已清除

6. **恢复**: 清除Int Indicator Register和Retry Error Register, 恢复AXI数据传输

### 自动 vs 手动错误处理

**[DOC]** Ctrl Register (0x00C):

| Bit | 字段 | 说明 |
|-----|------|------|
| 14 | Error Handle Trigger | 写1触发错误处理流程 (RW1C) |
| 13 | Self-retry err clr en | 1: PAXI自动进入错误处理; 0: 软件介入控制 |

**[推导]** 两种模式:
- **自动模式** (bit13=1): PAXI检测到retry fail后自动进入错误处理, 丢弃错误DA的数据包
- **手动模式** (bit13=0): 软件决定是否触发错误处理, 通过写bit14=1手动触发

## 6.5 多播错误处理

**[DOC]** 来自SUE2.0 2.6:

> "When PAXI receives a multicast error indication reported by the MAC or when no response is received within the timeout period after PAXI has sent a multicast request, multicast error handling is triggered."

### 两种多播错误

| 错误类型 | PAXI STATUS位 | Int Indicator位 |
|---------|---------------|----------------|
| MULTI-CAST RETRY ERR | bit17 | bit11 |
| MULTI-CAST TIMEOUT | bit18 | bit12 |

### 恢复步骤

**[DOC]**:

1. 多播重传失败时, PAXI上报retry fail中断
2. 用户读取Int Indicator Register和PAXI STATUS Register的MULTI-CAST RETRY ERR/TIMEOUT位
3. TX侧必须完成当前burst的AW+W数据传输
4. 默认TX和RX内部自动清除多播数据, 也可通过Ctrl Register手动控制
5. PAXI错误处理完成后上报中断
6. 用户确认PAXI STATUS的multi-cast错误状态位清除
7. 清除相关寄存器, 恢复传输

### 多播超时配置

**[DOC]** RX Multicast Timeout Register (0x080):

默认值: 0x03d0_9000

## 6.6 中断系统

### 中断状态寄存器

**[DOC]** Int Indicator Register (0x008):

| Bit | 字段 | 说明 | 类型 |
|-----|------|------|------|
| 12 | MULTI_CAST TIMEOUT | 多播B响应超时 | RW1C |
| 11 | MULTI_CAST RETRY ERR CLR | 多播重传错误清除完成 | RW1C |
| 10 | RETRY ERR CLR | 单播重传错误清除完成 | RW1C |
| 9 | PAT DATA ERR | Pattern Generator读/写数据不匹配 | RW1C |
| 8 | PAT RD DONE | Pattern Generator读完成 | RW1C |
| 7 | PAT WR DONE | Pattern Generator写完成 | RW1C |
| 6 | RSVD | 保留 | RO |
| 5 | APB_Linkup_MSG | 收到远端PCS link-up消息 | RW1C |
| 4 | LM timeout | 延迟测量超时 | RW1C |
| 3 | LM done | 延迟测量完成 | RW1C |
| 2 | APB_ERR_MSG | APB消息错误 | RW1C |
| 1 | E2E_RETRY_ERR | RC Link E2E Retry错误 | RW1C |
| 0 | OFLOW_IND | FIFO溢出 | RW1C |

### 中断屏蔽寄存器

**[DOC]** Interrupt Mask Register (0x004):

| Bit | 字段 | 说明 |
|-----|------|------|
| 12 | MULTI_CAST Timeout MASK | 多播超时屏蔽 |
| 11 | MULTI_CAST RETRY ERR CLR MASK | 多播重传错误清除屏蔽 |
| 10 | RETRY ERR CLR MASK | 重传错误清除屏蔽 |
| 9 | PAT DATA ERR MASK | Pattern错误屏蔽 |
| 8 | PAT RD DONE MASK | Pattern读完成屏蔽 |
| 7 | PAT WR DONE MASK | Pattern写完成屏蔽 |
| 5 | APB_Linkup_MSG MASK | PCS Linkup屏蔽 |
| 4 | LM timeout MASK | 延迟测量超时屏蔽 |
| 3 | LM done MASK | 延迟测量完成屏蔽 |
| 2 | APB_ERR_MSG MASK | APB消息错误屏蔽 |
| 1 | E2E_RETRY_ERR MASK | E2E Retry错误屏蔽 |
| 0 | OFLOW_IND MASK | FIFO溢出屏蔽 |

### PAXI Status Register错误位

**[DOC]** PAXI Status Register (0x01C):

| Bit | 字段 | 说明 | 类型 |
|-----|------|------|------|
| 18 | MULTI-CAST TIMEOUT | 多播超时 | RW1C |
| 17 | MULTI-CAST RETRY ERR | 多播重传失败 | RW1C |
| 16 | RX_MULTI_CAST_ERR_STAT | RX多播错误处理状态 (1:未完成, 0:完成) | RO |
| 15 | TX_MULTI_CAST_ERR_STAT | TX多播错误处理状态 (1:未完成, 0:完成) | RO |
| 14 | RX_ERR_STAT | RX错误处理状态 (1:未完成, 0:完成) | RO |
| 13 | TX_ERR_STAT | TX错误处理状态 (1:未完成, 0:完成) | RO |

## 6.7 远端链路状态检测

**[DOC]** 来自SUE2.0 3.2.35:

### Remote PCS Linkup Register 0~31

- 地址: 0x280~0x2FC
- 当远端芯片PCS重新Link-up时置位

### Remote Error Status Register 0~31

- 地址: 0x200~0x27C
- 记录远端错误状态

### Retry Error Register 0~31

- 地址: 0x300~0x37C
- 记录各DA的重传错误状态

## 6.8 RC Link异常处理

**[DOC]** 来自RCLINK Spec 第10章:

### TX方向异常

| 异常类型 | 说明 | 处理方式 |
|---------|------|---------|
| TX overcredit | 上游下发超过Credit限额 | 丢弃超Credit包, 上报中断 |
| TX oversize | 报文长度超过TYPE1_PKT_LEN | 丢弃超长包, 上报中断 |
| TX retry times over | QP重传次数超过阈值 | 上报中断 |

### 下游数据通路异常

**[DOC]**:

| 场景 | 处理 |
|------|------|
| 短暂断路 | RC Link内部缓冲吸收, 恢复后继续传输 |
| 复位 | 清除所有状态, 需要重新初始化 |
| 长时间断链 | 重传超时, 上报中断, 等待软件处理 |

### TYPE3异常

**[DOC]** TYPE3有独立的异常中断, 通过0x1170寄存器查询:
- rx fifo overflow
- tx fifo underflow
- md axi wr/rd resp abnormal
- data axi resp abnormal
- tx get two sop / tx last eop
- tx/rx buffer oversize
- tx/rx get md size zero
