# 06. 错误处理与重传机制

## 6.1 三层错误保护体系

PAXI技术栈提供三层错误保护:

| 层级 | 机制 | 位置 | 说明 |
|------|------|------|------|
| L1 | FEC (RSFEC) | CESOC CEFEC | 前向纠错, 自动纠正小错误 |
| L2 | MAC Retry | CESOC CEMAC | 链路层重传, 检测坏帧后重传 |
| L3 | PAXI Error Handling | PAXI Core | 端到端错误处理, DA级隔离 |

## 6.2 L1: FEC (RSFEC)

**[DOC]** CESOC Features:

> "Ethernet error rate lower than 10e-12"

**[DOC]** CESOC IP组件:

> "CEFEC_800G: 10G/25G/50G/100G/200G/400G/800G RSFEC"

- 使用Reed-Solomon FEC (RSFEC)算法
- 可以自动纠正链路中的少量误码
- 纠错后错误率低于10^-12
- 无法纠正的错误传递给L2层处理

**[DOC]** PAXI Features:

> "Support error free transmission with L1 retry enabled"

L1 Retry: 当FEC无法纠正时, 在SerDes/FEC层面触发重传。

## 6.3 L2: MAC Retry

**[DOC]** PAXI Features:

> "Support error free transmission with MAC L1/L2 retry enabled"

**[DOC]** RDMA IP Features:

> "Go-Back-N retransmission"

### RX侧帧检测

**[DOC]** 2.10 Error Handle:

> "RX buffer one mac frame. If this frame is bad indicated by mac, drop it. Else process it normally."

行为:
1. RX端缓冲一整个MAC帧
2. MAC指示该帧是否有错
3. 错误帧直接丢弃
4. 正确帧正常处理

### L2 Retry寄存器

**[DOC]** 3.2.361-362:

| 寄存器 | 地址 | 说明 |
|--------|------|------|
| TX MAC L2 Retry Field Register0~3 | 0xa40~0xa4c | TX方向L2重传状态 |
| RX MAC L2 Retry Field Register0~3 | 0xa60~0xa6c | RX方向L2重传状态 |

每个寄存器组有4个(0~3), 可以覆盖多个DA的状态。

### Burst级别的错误处理

**[DOC]** 2.10:

> "If RX can't get the whole data of the burst when the l2 retry occurs, the burst of the error da is drop."

当L2 Retry发生时, 如果RX端还没有接收完整个burst的数据, 则该DA的整个burst被丢弃。

## 6.4 L3: PAXI Error Handling

**[DOC]** 2.10 Error Handle (完整描述):

> "PAXI start error handling process and assert interrupt when it receives one of following:"
> - "TX Retry fatal from mac (TX can't find retry packet)"
> - "RX Retry fatal indication from mac (1. Rx receive retry fail. 2. Rx retry timeout)"
> - "Receive linkup message"

### 错误类型

| 错误类型 | 触发条件 | 说明 |
|----------|---------|------|
| TX Retry Fatal | TX端找不到需要重传的包 | 重传缓冲已被覆盖 |
| RX Retry Fail | RX端重传接收失败 | 数据无法恢复 |
| RX Retry Timeout | RX端重传超时 | 远端未响应 |
| Remote Linkup | 收到远端Linkup消息 | 远端芯片重启/重连 |

### TX方向错误处理

**[DOC]**:

> "For TX the error DA will not be blocked by flow ctrl credit, and the mac frame with error DA is not transmitted downstream. Wait for system error processing to complete and clear interrupt, paxi resumes normal process flow."

行为:
1. 错误DA的帧不再下发到MAC
2. 但错误DA不被Credit机制阻塞(错误帧不消耗credit)
3. 等待系统软件处理完成并清除中断
4. 清除后PAXI恢复正常流程

## 6.5 错误恢复流程

**[DOC]** 2.10 Retry Process Flow:

> 1. "PAXI assert interrupt when get retry error indication from MAC or receive remote pcs linkup."
> 2. "User read Int Indicator Register, Remote PCS Linkup Register, TX MAC L2 Retry Field Register and RX MAC L2 Retry Field Register to know which DA has error and detail error info."
> 3. "System stop sending new request for error DA. Write 1 to Int Indicator Register, Remote PCS Linkup Register, TX MAC L2 Retry Field Register and RX MAC L2 Retry Field Register to clear error status after system clean up done."
> 4. "User can send new request only When all the above registers status are clear. Any of error bit assertion cause transaction drop."

### 恢复步骤

```
步骤1: 检测
  PAXI检测到错误 -> 触发中断

步骤2: 诊断
  系统读取:
  - Int Indicator Register (0x008)  -> 错误类型
  - Remote PCS Linkup Register      -> 哪个远端链路
  - TX MAC L2 Retry Field Register  -> TX重传详情
  - RX MAC L2 Retry Field Register  -> RX重传详情
  确定: 哪个DA出错, 什么类型的错误

步骤3: 隔离
  系统停止向错误DA发送新事务

步骤4: 清除
  写1到所有错误状态寄存器以清除(RW1C)

步骤5: 恢复
  确认所有错误位都被清除后, 恢复该DA的事务发送
```

### 关键约束

**[DOC]**:

> "Any of error bit assertion cause transaction drop."

只要有任何错误位被置起, 该DA的事务就会被丢弃。必须全部清除才能恢复。

## 6.6 中断系统

### 中断状态寄存器

**[DOC]** 3.2.3 Int Indicator Register (REG_STATUS), 地址0x008:

| Bit | 字段 | 说明 | 类型 |
|-----|------|------|------|
| 9 | PAT ERR | Pattern Generator读/写错误 | RW1C |
| 8 | PAT RD DONE | Pattern Generator读完成 | RW1C |
| 7 | PAT WR DONE | Pattern Generator写完成 | RW1C |
| 6 | DA update | DA更新消息已接收 | RW1C |
| 5 | Linkup | 远端PCS已Link-up | RW1C |
| 4 | LM timeout | 延迟测量超时 | RW1C |
| 3 | LM done | 延迟测量完成 | RW1C |
| 2 | ERR_MSG | APB消息错误中断 | RW1C |
| 1 | ERR_IND | L2 Retry错误指示 | RW1C |
| 0 | OFLOW_IND | FIFO溢出中断 | RW1C |

所有状态位都是RW1C (写1清除)。

### 中断屏蔽寄存器

**[DOC]** 3.2.2 Int Control Register (REG_CONTROL), 地址0x004:

| Bit | 字段 | 说明 |
|-----|------|------|
| 9 | PAT_ERR_MASK | Pattern Generator错误屏蔽 |
| 8 | PAT_RD_DONE_MASK | Pattern Generator读完成屏蔽 |
| 7 | PAT_WR_DONE_MASK | Pattern Generator写完成屏蔽 |
| 6 | DA update MASK | DA更新消息屏蔽 |
| 5 | Linkup_MASK | PCS Linkup屏蔽 |
| 4 | LM_TO_MASK | 延迟测量超时屏蔽 |
| 3 | LM_MASK | 延迟测量完成屏蔽 |
| 2 | PM_MASK | APB消息错误屏蔽 |
| 1 | ERR_MASK | L2 Retry错误屏蔽 |
| 0 | OFLOW_MASK | FIFO溢出屏蔽 |

## 6.7 远端链路状态检测

**[DOC]** 3.2.360 Remote PCS Linkup Register0~3:

- 地址: 0xa20~0xa2c
- 4个寄存器覆盖多个DA的PCS Link状态
- 当远端芯片PCS重新Link-up时置位, 表示远端可能重启过

**[DOC]** 3.2.356 Remote Error Status Register0~3:

- 地址: 0xa00~0xa0c
- 记录远端错误状态
