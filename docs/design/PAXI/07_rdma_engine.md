# 07. RDMA引擎 (RoCEv2) 详细分析

> **[历史文档]** 此文档描述旧版独立RDMA Engine (RoCEv2)。在SUE2.0架构中, RC Link替代了其大部分功能, 包括可靠传输(Go-Back-N)、速率控制(per-QP)和拥塞通知(ECN/CNP)。请参见 [12_rclink.md](12_rclink.md) 了解SUE2.0的传输层设计。
>
> 以下内容保留作为历史参考。

## 7.1 概述

**[DOC]** RDMA Reference Guide 1.1:

> "This document describes implementation-specific characteristics of the UniVista RDMA implementation. This IP support RocEv2 RDMA with up to 8192 Queue pair."

**[DOC]** RDMA Reference Guide 2. Function Description:

> "The RDMA IP is an implementation of RDMA over Converged Ethernet (RoCE v2) enabled NIC functionality. This parameterizable soft IP core can work with Orichip MAC, Ethernet Controller IP implementations providing a high throughput, low latency, and completely hardware offloaded reliable data transfer solution over standard Ethernet. The RDMA IP allows simultaneous connections to multiple remote hosts running RoCE v2 traffic."

## 7.2 协议支持

**[DOC]** RDMA Features (1.2):

| 特性 | 规格 |
|------|------|
| 协议 | RoCE v2 (RDMA over Converged Ethernet v2) |
| 网络层 | IPv4 (简化版) |
| 传输层 | UDP (parity bypass) |
| 数据速率 | 200G / 400 Gbps |
| MTU | 最大4KB |

### 传输类型

**[DOC]**:

| 类型 | 操作 |
|------|------|
| RC (Reliable Connection) | Send / Receive(partial) / Write |
| UD (Unreliable Datagram) | Send / Receive |

### Queue Pair规模

**[DOC]**:

- 标准: 4096 QPs (含QP0/QP1管理通道)
- 可选: 8192 QPs
- QP0/QP1: 保留用于管理通道(UD类型)

### Memory Region

**[DOC]**:

- 可选: 最大32K MR
- 仅支持connected VA到PA映射 (连续虚拟地址到物理地址)

## 7.3 内部架构

**[DOC]** RDMA 1.4 Related Cores:

### QP Manager

> "Gets the config from processor by RCFG_AXI_LITE*"
> "Arbitrates the SQ Fetch WQE and launches it to WQE Process Module"
> "Handling error by re-transmission"

行为:
1. 软件通过更新sq_head触发硬件开始工作
2. 收到sq_head更新后, 硬件为该QP生成sq_req
3. QP被授权后更新sq_tail
4. 如果sq_tail != sq_head, 继续生成新的sq_req
5. 如果WQE无效(非UD SEND/RC SEND/WRITE), 仅在该QP上产生fault并触发中断

### WQE Process Unit

**[DOC]**:

> "This module is responsible for:
> 1. receiving WQE info requests (including CNP tags) from the QP-MGR module and ACK requests from the RX_PKT_PROC module.
> 2. constructing corresponding message headers.
> 3. generating DMA read commands to read Write/send type messages to the DMA_READ module (if a message command that requires retry is received, the starting address and length of the retry message need to be recalculated, as retry may start from any position in the WQE slice).
> 4. grouping the header (including MAC/IP/UDP header) and message data (if any) into packets before sending them to the MAC module."

### RX Packet Process Unit

**[DOC]**:

功能:
1. 从MAC接收数据, 进行ICRC和长度验证
2. 根据MAC/IP/UDP header判断是RDMA还是Non-RDMA, 分发到对应模块
3. 检查UD/RC send和RC write包格式

**[DOC]** RC包检查规则 (2.5.2 RC_CHK):

| 检查项 | 期望值 | 不匹配时动作 |
|--------|--------|-------------|
| OPCODE | RC send/write | Fatal.invalid req |
| Length | Send: <= RQ buffer; First/Middle: == MTU; Last/Only: <= MTU | Fatal.invalid req |
| PKEY | PKT.PKEY[14:0] == PKEY[dest_qp][14:0] 且 PKT.PKEY[14] 或 PKEY[dest_qp][14] == 1 | Drop |
| DestQP | dest_qp已使能且 > 1 | Drop |
| RQ almost_full | - | Send RNR |
| WRITE MRT | R_KEY PD_NUM, Memory Region Range Matches | NAK Remote Access Error |
| CQ | CQ未溢出 | Fatal.cq_overflow |
| RX Push Bresp | Bresp == resp_ok | Fatal.berror |
| PSN | RECV_PSN == EPSN | Send Dup/Retry |

**[DOC]** 验证成功后的行为:

1. UD/RC send: 将payload写入RQ, RC send在ack_req=1时立即发ACK, 否则发implicit ACK
2. RC write: 将payload写入Memory Region, 同样的ACK策略
3. ACK response: 释放一个outstanding slot给该QP
4. RETRY NAK: 启动重传
5. Fatal NAK: 停止该QP的新命令发送并触发中断

### Response Unit

**[DOC]**:

> "Manage the outstanding buff for send / write cmd in process"
> "Trigger retransmission when getting NAK or timeout"

### MR Table

**[DOC]** 2.2:

> "1. After receiving a remote write/read request, use R_Key high 16bits as index searching MRT and get the PD information."
> "2. PD information used for authentication, R_key low 8 bits, PD number, access address, access range and type should be checked."
> "3. If all checks are passed, RETH virtual address will be transferred to Physical Address."

## 7.4 Per-QP Outstanding

**[DOC]** 配置参数:

| 参数 | 值 | 说明 |
|------|------|------|
| OSQ_PER_QP | 16 | 每QP最大outstanding数 |
| QP_OST_NUM | = OSQ_PER_QP | QP outstanding number |

**[DOC]** ECN行为:

> "Check ECN packet. For RX side, send the CNP to TX. When TX side gets CNP, it will decrease the outstanding from 16 to 8 and launch interrupt to the App."

当收到CNP(Congestion Notification Packet)时:
1. TX端outstanding从16降低到8
2. 触发中断通知应用
3. 定时器开始计时
4. 计时到CNP_TIMER_THRD后, outstanding恢复到16

## 7.5 可配置参数

**[DOC]** 3.1 Parameters (用户可重配置):

| 参数 | 范围 | 说明 |
|------|------|------|
| QP_NUM | 256~4096 (2的幂) | QP数量 |
| MR_NUM | 256~16384 (2的幂) | MR数量 |
| RX_BUF_AW | 5~13 | RX缓冲深度 = 2^RX_BUF_AW, 宽度128B |
| RAM_RD_LATENCY | 1~3 cycles | 存储读延迟 |
| QPM_AXI_DW | 64~512 | WQE AXI数据宽度 |
| MAC_DW | 1024(400G) / 512(200G) | MAC接口数据宽度 |

## 7.6 数据接口

**[DOC]** 4.3:

| 接口 | 类型 | 方向 | 说明 |
|------|------|------|------|
| WQE_M_AXI | AXI4 Read Only | Master | 从DDR读取WQE |
| TXD_M_AXI | AXI4 Read Only | Master | 从DDR读取发送数据 |
| RX_PUSH_M_AXI | AXI4 Write Only | Master | 将接收数据写入DDR |
| RESP_CQ_M_AXI | AXI4 Write Only | Master | 写入CQ完成通知 |
| CFG_S_AXI_LITE | AXI-Lite 64b | Slave | 配置寄存器 |
| MAC TX/RX | AXI-Stream 1024b | Both | MAC数据接口 |
| NONR_DATA/MD | AXI4 | Both | Non-RDMA数据通道 |

## 7.7 错误处理

**[DOC]** Features:

> "Packet retransmission when detecting error"
> "Each QP will stop independently when gets the error cmd or runs to error branch."

错误处理特点:
1. **QP级隔离**: 一个QP的错误不影响其他QP
2. **自动重传**: Go-Back-N重传策略
3. **Fatal错误**: 停止对应QP并触发中断

## 7.8 RDMA与PAXI的关系

**[DOC]** RDMA IP Brief:

> "When combined with UniVista 200G/400G Ethernet controller and PAXI IP, it can quickly build up 200G/400G connections among chips through Ethernet physical links."

**[DOC]** RDMA Features:

> "P2P AXI bridge for Chip-to-Chip connections"

RDMA引擎可以通过PAXI的AXI接口透明地访问远端芯片内存, PAXI负责AXI-to-Ethernet的协议转换, RDMA负责可靠传输和内存语义。
