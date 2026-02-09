# 08. CESOC 800G 物理层分析

## 8.1 概述

**[DOC]** CESOC Reference Guide 1.1:

> "CESOC_800G_192B is a design that includes CEMAC_800G, CEPCS_800G and CEFEC_800G IP cores to support 16 SERDES lanes application. And it supports 10G/25G/50G/100G/200G/400G/800G Ethernet."

CESOC是合见工软的以太网控制器子系统, 为PAXI提供物理层连接。

## 8.2 IP组成

**[DOC]** 1.4 IP Blocks and Functions:

| No. | IP名称 | 说明 |
|-----|--------|------|
| 1 | CEMAC_800G | 10G/25G/50G/100G/200G/400G/800G Ethernet MAC |
| 2 | CEPCS_800G | 10G/25G/50G/100G/200G/400G/800G Ethernet PCS |
| 3 | CEFEC_800G | 10G/25G/50G/100G/200G/400G/800G RSFEC |
| 4 | CESOC_800G_REG | 配置寄存器 |
| 5 | IF_192B | 192B用户逻辑接口 |

### 层次关系

```
用户逻辑(PAXI)
    |
    |  IF_192B接口 (192B TDM, 8端口)
    v
┌─────────────────────────┐
│ CEMAC_800G              │  ← MAC层: 帧处理、流控
│  - Flow Control         │
│  - PTP 1588             │
│  - L2 Retry             │
├─────────────────────────┤
│ CEPCS_800G              │  ← PCS层: 编码/解码
│  - Lane Swap (8x8)      │
├─────────────────────────┤
│ CEFEC_800G              │  ← FEC层: 前向纠错
│  - RSFEC                │
│  - L1 Retry             │
├─────────────────────────┤
│ SerDes x 8              │  ← 物理层: 112G PAM4
│  - 8 x 112G PAM4       │
└─────────────────────────┘
```

## 8.3 核心特性

**[DOC]** 1.2 Features List:

- **[DOC]** Up to 800G bps MAX bandwidth for Ethernet
- **[DOC]** Support 800G-Ethernet system
- **[DOC]** Support bounding 8x112G PAM4 SERDES for Ethernet protocol
- **[DOC]** Ethernet error rate lower than 10e-12
- **[DOC]** Support 1588 PTP include 1-step and 2-step
- **[DOC]** Support APB interface for Register Access
- **[DOC]** Support loopback modes for diagnosis
- **[DOC]** Support system clock frequency up to 1.35GHz
- **[DOC]** Support 192B TDM interface which shared 8 ports and per-port credit flow control for User logic
- **[DOC]** 800G system supports 8x8 any-to-any SERDES lane swap both on TX/RX
- **[DOC]** Support 6nm/5nm Tech

## 8.4 灵活的端口配置

**[DOC]** 800G系统支持多种客户端口配置:

| 配置 | 通道数 | 每通道速率 | 总带宽 |
|------|--------|-----------|--------|
| 25GE x 8 | 8 | 25 Gbps | 200 Gbps |
| 50GE x 8 | 8 | 50 Gbps | 400 Gbps |
| 100GE x 8 | 8 | 100 Gbps | 800 Gbps |
| 200GE x 4 | 4 | 200 Gbps | 800 Gbps |
| 400GE x 2 | 2 | 400 Gbps | 800 Gbps |
| 800GE x 1 | 1 | 800 Gbps | 800 Gbps |

**[推导]** 这种灵活性意味着:
- 同一IP核可以适配不同的互联需求
- 可以在带宽和端口数之间权衡
- 例如: 需要连接8个芯片时用25GE x 8, 需要最大带宽时用800GE x 1

## 8.5 192B TDM用户接口

**[DOC]** 2.2.4 Interface 192B (For User):

### TX Data Bus

| 信号 | 宽度 | 说明 |
|------|------|------|
| IF_192B_TX_DATA_I | 1536 bits (192B) | 用户发送数据 |
| IF_192B_TX_DATA_VALID_I | 1 | 数据有效指示 |
| IF_192B_TX_DATA_FCS_I | 3 | FCS模式 |
| IF_192B_TX_DATA_BCNT_I | 8 | 有效字节计数, 最大192 |
| IF_192B_TX_DATA_CH_I | 3 | 通道号 (0~7) |
| IF_192B_TX_DATA_SOF_I | 1 | 帧起始指示 |
| IF_192B_TX_DATA_EOF_I | 1 | 帧结束指示 |
| IF_192B_TX_DATA_CREDIT_O | 8 | Per-port credit返回 |

**[DOC]** FCS模式 (IF_192B_TX_DATA_FCS_I):

| 值 | 说明 |
|----|------|
| 3'h0 | 保持FCS |
| 3'h1 | 添加正确FCS |
| 3'h2 | 添加错误FCS |
| 3'h3 | 替换正确FCS |
| 3'h4 | 替换错误FCS |
| 3'h5~7 | 保留 |

### RX Data Bus

| 信号 | 宽度 | 说明 |
|------|------|------|
| IF_192B_RX_DATA_O | 1536 bits (192B) | 接收数据 |
| IF_192B_RX_DATA_VALID_O | 1 | 数据有效 |
| IF_192B_RX_DATA_BCNT_O | 8 | 有效字节计数 |
| IF_192B_RX_DATA_CH_O | 3 | 通道号 |
| IF_192B_RX_DATA_SOF_O | 1 | 帧起始 |
| IF_192B_RX_DATA_EOF_O | 1 | 帧结束 |
| IF_192B_RX_DATA_ERR_O | 2 | 错误指示 |

**关键设计**:
- 数据总线宽度: 192 Bytes = 1536 bits
- 8个端口通过TDM共享
- Per-port credit流控
- 支持SOF/EOF帧界定

## 8.6 SerDes配置

**[DOC]** 每条Lane独立的TX/RX时钟:

- SD0~SD7_TX_CLK_I: 8条TX Lane时钟
- SD0~SD7_RX_CLK_I: 8条RX Lane时钟
- 每条Lane 112G PAM4

**[DOC]** 8x8 Lane Swap:

> "800G system supports 8x8 any-to-any SERDES lane swap both on TX/RX"

- TX和RX方向都支持任意Lane交换
- 8条Lane可以任意映射到另外8条Lane
- **[推导]** 这简化了PCB布线, 不需要物理Lane严格对应

## 8.7 PTP 1588时间同步

**[DOC]** 5. PTP 1588:

支持两种模式:
- **1-step**: 在发送时直接修改报文中的时间戳
  - CorrectionField Update
  - OriginTimestampField Update
- **2-step**: 发送时记录时间戳, 之后通过Follow-up消息传递

PTP相关信号:

| 信号 | 说明 |
|------|------|
| IF_192B_TS_64NS_I | 64-bit纳秒时间戳输入 |
| IF_192B_TS_32S32NS_I | 32秒+32纳秒时间戳 |
| IF_192B_TX_TS_RX_I | TX时间戳接收 |
| IF_192B_TX_TS_CMD_I | 时间戳命令 |

## 8.8 时序约束

**[DOC]** 2. Hardware Interface, Table 2-1 Timing Definition:

| 时序参数 | 定义 |
|----------|------|
| DC | 信号在使用前至少一个周期有效 |
| Begin | 信号在时钟上升沿后8%周期内有效 |
| Early | 信号在时钟上升沿后18%周期内有效 |
| Middle | 信号在时钟上升沿后43%周期内有效 |
| Late | 信号在时钟上升沿后58%周期内有效 |
| End | 信号在时钟上升沿后78%周期内有效 |

大部分用户接口信号的时序要求为 **Middle** (43%周期), 说明有充足的建立时间裕量。
